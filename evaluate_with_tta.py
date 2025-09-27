#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) Evaluation
=======================================

Evaluates trained models using test-time augmentation to boost inference accuracy.
Applies multiple augmentations during inference and averages predictions.

Author: Augment Agent
Date: 2025-09-27
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

# Import our components
sys.path.append('.')
from advanced_training_components import (
    StandardizedPreprocessor,
    ComprehensiveVideoDataset
)
from train_icu_finetune_fixed import EnhancedLightweightCNNLSTM
from models.heads.small_fc import SmallFCHead

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTAEvaluator:
    """Test-Time Augmentation evaluator"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = None
        
        # Load model and config
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load trained model from checkpoint"""
        logger.info(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint.get('config', {})
        
        # Create model
        self.model = EnhancedLightweightCNNLSTM(
            num_classes=4,  # ICU classes
            dropout=self.config.get('dropout', 0.4)
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info("Model loaded successfully")
        
    def create_tta_augmentations(self, num_passes=5):
        """Create different augmentation configurations for TTA"""
        augmentations = []
        
        # Original (no augmentation)
        augmentations.append({
            'brightness_factor': 1.0,
            'contrast_factor': 1.0,
            'horizontal_flip': False,
            'name': 'original'
        })
        
        # Brightness variations
        augmentations.append({
            'brightness_factor': 1.1,
            'contrast_factor': 1.0,
            'horizontal_flip': False,
            'name': 'bright_up'
        })
        
        augmentations.append({
            'brightness_factor': 0.9,
            'contrast_factor': 1.0,
            'horizontal_flip': False,
            'name': 'bright_down'
        })
        
        # Contrast variations
        augmentations.append({
            'brightness_factor': 1.0,
            'contrast_factor': 1.1,
            'horizontal_flip': False,
            'name': 'contrast_up'
        })
        
        augmentations.append({
            'brightness_factor': 1.0,
            'contrast_factor': 0.9,
            'horizontal_flip': False,
            'name': 'contrast_down'
        })
        
        # Horizontal flip
        if num_passes > 5:
            augmentations.append({
                'brightness_factor': 1.0,
                'contrast_factor': 1.0,
                'horizontal_flip': True,
                'name': 'h_flip'
            })
            
        # Combined augmentations
        if num_passes > 6:
            augmentations.append({
                'brightness_factor': 1.05,
                'contrast_factor': 1.05,
                'horizontal_flip': False,
                'name': 'combined_up'
            })
            
        if num_passes > 7:
            augmentations.append({
                'brightness_factor': 0.95,
                'contrast_factor': 0.95,
                'horizontal_flip': False,
                'name': 'combined_down'
            })
            
        return augmentations[:num_passes]
        
    def apply_augmentation(self, video_tensor, aug_config):
        """Apply augmentation to video tensor"""
        # video_tensor shape: [T, H, W] or [C, T, H, W]
        
        # Ensure we have the right shape
        if video_tensor.dim() == 3:
            video_tensor = video_tensor.unsqueeze(0)  # Add channel dim
            
        # Apply brightness
        if aug_config['brightness_factor'] != 1.0:
            video_tensor = video_tensor * aug_config['brightness_factor']
            video_tensor = torch.clamp(video_tensor, 0, 1)
            
        # Apply contrast
        if aug_config['contrast_factor'] != 1.0:
            mean_val = video_tensor.mean()
            video_tensor = (video_tensor - mean_val) * aug_config['contrast_factor'] + mean_val
            video_tensor = torch.clamp(video_tensor, 0, 1)
            
        # Apply horizontal flip
        if aug_config['horizontal_flip']:
            video_tensor = torch.flip(video_tensor, dims=[-1])  # Flip width dimension
            
        return video_tensor
        
    def evaluate_with_tta(self, dataset, num_passes=5, batch_size=8):
        """Evaluate dataset using test-time augmentation"""
        logger.info(f"Starting TTA evaluation with {num_passes} passes")
        
        # Create augmentation configurations
        tta_configs = self.create_tta_augmentations(num_passes)
        logger.info(f"TTA configurations: {[cfg['name'] for cfg in tta_configs]}")
        
        # Create data loader
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        all_predictions = []
        all_labels = []
        all_tta_predictions = []  # Store predictions for each TTA pass
        
        with torch.no_grad():
            for batch_videos, batch_labels in tqdm(data_loader, desc="TTA Evaluation"):
                batch_videos = batch_videos.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Store predictions for each TTA pass
                tta_batch_predictions = []
                
                for aug_config in tta_configs:
                    # Apply augmentation to each video in batch
                    augmented_batch = []
                    for video in batch_videos:
                        aug_video = self.apply_augmentation(video, aug_config)
                        augmented_batch.append(aug_video)
                    
                    augmented_batch = torch.stack(augmented_batch)
                    
                    # Get predictions
                    outputs = self.model(augmented_batch)
                    probabilities = F.softmax(outputs, dim=1)
                    tta_batch_predictions.append(probabilities.cpu())
                
                # Average predictions across TTA passes
                avg_predictions = torch.stack(tta_batch_predictions).mean(dim=0)
                final_predictions = torch.argmax(avg_predictions, dim=1)
                
                # Store results
                all_predictions.extend(final_predictions.numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_tta_predictions.append(tta_batch_predictions)
                
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        
        # Calculate per-class metrics
        cm = confusion_matrix(all_labels, all_predictions)
        class_report = classification_report(
            all_labels, all_predictions, 
            target_names=dataset.classes,
            output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'num_tta_passes': num_passes,
            'tta_configurations': tta_configs,
            'total_samples': len(all_labels)
        }
        
        logger.info(f"TTA Evaluation Results:")
        logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  Macro F1: {f1:.4f}")
        logger.info(f"  Total samples: {len(all_labels)}")
        
        return results
        
    def compare_with_single_pass(self, dataset, num_passes=5, batch_size=8):
        """Compare TTA results with single-pass inference"""
        logger.info("Comparing TTA with single-pass inference")
        
        # Single-pass evaluation
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        single_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_videos, batch_labels in tqdm(data_loader, desc="Single-pass"):
                batch_videos = batch_videos.to(self.device)
                outputs = self.model(batch_videos)
                predictions = torch.argmax(outputs, dim=1)
                
                single_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                
        single_accuracy = accuracy_score(all_labels, single_predictions)
        single_f1 = f1_score(all_labels, single_predictions, average='macro')
        
        # TTA evaluation
        tta_results = self.evaluate_with_tta(dataset, num_passes, batch_size)
        
        # Comparison
        improvement = {
            'accuracy_improvement': tta_results['accuracy'] - single_accuracy,
            'f1_improvement': tta_results['f1_score'] - single_f1,
            'single_pass': {
                'accuracy': single_accuracy,
                'f1_score': single_f1
            },
            'tta_pass': {
                'accuracy': tta_results['accuracy'],
                'f1_score': tta_results['f1_score']
            }
        }
        
        logger.info(f"\n📊 TTA vs Single-pass Comparison:")
        logger.info(f"Single-pass: {single_accuracy:.4f} accuracy, {single_f1:.4f} F1")
        logger.info(f"TTA ({num_passes} passes): {tta_results['accuracy']:.4f} accuracy, {tta_results['f1_score']:.4f} F1")
        logger.info(f"Improvement: +{improvement['accuracy_improvement']:.4f} accuracy, +{improvement['f1_improvement']:.4f} F1")
        
        tta_results['comparison'] = improvement
        return tta_results

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Test-Time Augmentation Evaluation')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', default='data/stabilized_speaker_sets', 
                       help='Directory containing test data')
    parser.add_argument('--manifest', help='Specific manifest file to evaluate')
    parser.add_argument('--tta-passes', type=int, default=5, help='Number of TTA passes')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare with single-pass inference')
    
    args = parser.parse_args()
    
    logger.info("🔍 TEST-TIME AUGMENTATION EVALUATION")
    logger.info("=" * 50)
    
    # Initialize evaluator
    evaluator = TTAEvaluator(args.checkpoint)
    
    # Create dataset
    preprocessor = StandardizedPreprocessor()
    
    if args.manifest:
        dataset = ComprehensiveVideoDataset(
            args.manifest, preprocessor, None, is_training=False
        )
    else:
        # Use a sample validation set (you might want to specify this differently)
        logger.error("Please specify a manifest file with --manifest")
        return
        
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    logger.info(f"Classes: {dataset.classes}")
    
    # Run evaluation
    if args.compare:
        results = evaluator.compare_with_single_pass(
            dataset, args.tta_passes, args.batch_size
        )
    else:
        results = evaluator.evaluate_with_tta(
            dataset, args.tta_passes, args.batch_size
        )
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    logger.info("✅ TTA Evaluation complete")

if __name__ == "__main__":
    main()
