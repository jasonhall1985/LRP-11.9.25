#!/usr/bin/env python3
"""
Enhanced ICU Training with Stabilized ROI
==========================================

Trains ICU lip-reading model using stabilized ROI data with improved preprocessing,
extended training schedule, and optimized hyperparameters for better generalization.

Key Improvements:
- Uses stabilized ROI data from geometric cropping
- Enhanced preprocessing pipeline
- Extended training schedule with cosine annealing
- Improved data augmentation
- Better regularization techniques
- Comprehensive validation tracking

Author: Augment Agent
Date: 2025-09-27
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

# Import training components
from advanced_training_components import (
    EnhancedLightweightCNNLSTM,
    StandardizedPreprocessor,
    ConservativeAugmentation,
    ComprehensiveVideoDataset,
    FocalLoss
)

from loso_cross_validation_framework import LOSODatasetManager

class SimpleEarlyStopping:
    """Simple early stopping implementation."""

    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model=None):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

class EnhancedICUTrainer:
    """Enhanced trainer for ICU lip-reading with stabilized ROI data."""
    
    def __init__(self, config: Dict):
        """Initialize enhanced trainer."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 4  # ICU classes
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training components
        self.early_stopping = None
        
        self.logger.info(f"Enhanced ICU Trainer initialized on {self.device}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    def init_model(self):
        """Initialize model and training components."""
        # Create model
        self.model = EnhancedLightweightCNNLSTM(
            num_classes=self.num_classes,
            input_channels=1,
            dropout=self.config.get('dropout', 0.4)
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model created with {total_params:,} total parameters")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.0005),
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Initialize scheduler - Cosine Annealing with Warm Restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('scheduler_t0', 10),
            T_mult=self.config.get('scheduler_tmult', 2),
            eta_min=self.config.get('min_lr', 1e-6)
        )
        
        # Initialize loss function with class weights
        class_weights = self.calculate_class_weights()
        self.criterion = FocalLoss(
            alpha=class_weights,
            gamma=self.config.get('focal_gamma', 2.0)
        )
        
        # Initialize training components
        self.early_stopping = SimpleEarlyStopping(
            patience=self.config.get('early_stopping_patience', 25),
            min_delta=self.config.get('early_stopping_delta', 0.001),
            restore_best_weights=True
        )
        
        self.logger.info("Model and training components initialized")
    
    def calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        # Default balanced weights - can be updated based on actual data distribution
        weights = torch.ones(self.num_classes, dtype=torch.float32)
        
        # Adjust based on typical ICU class imbalance if known
        # These can be updated based on actual dataset statistics
        class_frequencies = {
            0: 1.0,  # doctor
            1: 1.2,  # i_need_to_move (typically more frequent)
            2: 0.8,  # my_mouth_is_dry (typically less frequent)
            3: 1.0   # pillow
        }
        
        for i, freq in class_frequencies.items():
            weights[i] = 1.0 / freq
        
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        self.logger.info(f"Class weights: {weights.tolist()}")
        return weights.to(self.device)
    
    def create_enhanced_preprocessor(self) -> StandardizedPreprocessor:
        """Create enhanced preprocessor for stabilized ROI data."""
        return StandardizedPreprocessor(
            target_size=(64, 96),  # Keep original aspect ratio from stabilized ROI
            target_frames=32,
            grayscale=True,
            normalize=True
        )
    
    def create_enhanced_augmentation(self) -> ConservativeAugmentation:
        """Create enhanced augmentation for training."""
        return ConservativeAugmentation(
            brightness_range=0.12,  # Slightly reduced for stability
            contrast_range=0.08,    # Slightly reduced for stability
            horizontal_flip_prob=0.5,
            temporal_speed_range=0.03,  # Conservative speed variation
            spatial_translation=0.02,  # Minimal spatial translation
            rotation_degrees=1.0  # Very small rotation
        )
    
    def create_data_loaders(self, train_manifest: str, val_manifest: str) -> Tuple[DataLoader, DataLoader]:
        """Create enhanced data loaders."""
        # Create preprocessor and augmentation
        preprocessor = self.create_enhanced_preprocessor()
        train_augmentation = self.create_enhanced_augmentation()
        
        # Create datasets
        train_dataset = ComprehensiveVideoDataset(
            manifest_path=train_manifest,
            preprocessor=preprocessor,
            augmentation=train_augmentation,
            synthetic_ratio=0.15  # Reduced synthetic ratio for stability
        )

        val_dataset = ComprehensiveVideoDataset(
            manifest_path=val_manifest,
            preprocessor=preprocessor,
            augmentation=None,  # No augmentation for validation
            synthetic_ratio=0.0
        )
        
        # Create data loaders with improved settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True,  # Drop last incomplete batch for stability
            persistent_workers=True if self.config.get('num_workers', 4) > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True if self.config.get('num_workers', 4) > 0 else False
        )
        
        self.logger.info(f"Created data loaders - Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch with enhanced monitoring."""
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (videos, labels) in enumerate(pbar):
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('grad_clip_norm', 1.0)
            )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            current_acc = 100.0 * correct_predictions / total_samples
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc=f"Validation {epoch}"):
                videos = videos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                # Store for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train_fold(self, train_loader: DataLoader, val_loader: DataLoader, 
                   fold_num: int, output_dir: str) -> Dict:
        """Train a single LOSO fold with enhanced monitoring."""
        
        self.logger.info(f"\n🚀 Starting enhanced training for fold {fold_num}")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        # Initialize training components
        self.init_model()
        
        # Training loop
        best_accuracy = 0.0
        training_history = []
        
        for epoch in range(1, self.config.get('max_epochs', 30) + 1):
            epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - epoch_start_time
            
            self.logger.info(
                f"Epoch {epoch:2d}/{self.config.get('max_epochs', 30)} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                f"LR: {train_metrics['learning_rate']:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Update training history
            epoch_data = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'learning_rate': train_metrics['learning_rate'],
                'epoch_time': epoch_time
            }
            training_history.append(epoch_data)
            
            # Check for best model
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                
                # Save best model
                checkpoint_path = Path(output_dir) / f"enhanced_icu_fold_{fold_num}_best.pth"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_accuracy': best_accuracy,
                    'config': self.config,
                    'training_history': training_history
                }, checkpoint_path)
                
                self.logger.info(f"💾 New best model saved: {best_accuracy:.2f}%")
            
            # Early stopping check
            if self.early_stopping(val_metrics['loss'], self.model):
                self.logger.info(f"⏹️  Early stopping triggered at epoch {epoch}")
                # Restore best weights
                self.early_stopping.restore_best(self.model)
                break
        
        # Save training history
        history_path = Path(output_dir) / f"enhanced_icu_fold_{fold_num}_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        self.logger.info(f"✅ Fold {fold_num} training complete. Best accuracy: {best_accuracy:.2f}%")
        
        return {
            'fold': fold_num,
            'best_accuracy': best_accuracy,
            'training_history': training_history,
            'final_epoch': epoch
        }

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Enhanced ICU Training with Stabilized ROI')
    parser.add_argument('--data-dir', default='data/stabilized_speaker_sets',
                       help='Directory containing stabilized speaker sets')
    parser.add_argument('--output-dir', default='checkpoints/enhanced_icu_stabilized',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--max-epochs', type=int, default=35,
                       help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.0003,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay for regularization')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    args = parser.parse_args()
    
    print("🚀 ENHANCED ICU TRAINING WITH STABILIZED ROI")
    print("=" * 60)
    
    # Build configuration
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'max_epochs': args.max_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'num_workers': args.num_workers,
        'early_stopping_patience': 25,
        'early_stopping_delta': 0.001,
        'grad_clip_norm': 1.0,
        'focal_gamma': 2.0,
        'label_smoothing': 0.1,
        'scheduler_t0': 10,
        'scheduler_tmult': 2,
        'min_lr': 1e-6,
        'cache_size': 100
    }
    
    # Check data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset manager
    dataset_manager = LOSODatasetManager(str(data_dir))
    
    # Generate LOSO splits
    print("📊 Generating LOSO splits...")
    loso_splits = dataset_manager.generate_all_loso_splits(
        output_dir=f"{args.output_dir}/loso_splits"
    )
    
    # Initialize trainer
    trainer = EnhancedICUTrainer(config)
    
    # Train all folds
    fold_results = []
    total_start_time = time.time()
    
    for fold_num, (val_speaker, (train_manifest, val_manifest)) in enumerate(loso_splits.items(), 1):
        print(f"\n🎯 FOLD {fold_num}: Validation Speaker = {val_speaker}")
        print(f"Train manifest: {train_manifest}")
        print(f"Val manifest: {val_manifest}")
        
        # Create data loaders
        train_loader, val_loader = trainer.create_data_loaders(train_manifest, val_manifest)
        
        # Train fold
        fold_result = trainer.train_fold(train_loader, val_loader, fold_num, args.output_dir)
        fold_results.append(fold_result)
        
        print(f"✅ Fold {fold_num} complete: {fold_result['best_accuracy']:.2f}% accuracy")
    
    # Calculate overall results
    total_time = time.time() - total_start_time
    avg_accuracy = np.mean([result['best_accuracy'] for result in fold_results])
    std_accuracy = np.std([result['best_accuracy'] for result in fold_results])
    
    # Save final results
    final_results = {
        'config': config,
        'fold_results': fold_results,
        'summary': {
            'avg_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy,
            'total_training_time': total_time,
            'num_folds': len(fold_results)
        }
    }
    
    results_path = output_dir / 'enhanced_icu_final_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Print final summary
    print(f"\n🎉 ENHANCED ICU TRAINING COMPLETE")
    print("=" * 60)
    print(f"Average LOSO Accuracy: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
    print(f"Total Training Time: {total_time/3600:.1f} hours")
    print(f"Results saved to: {results_path}")
    
    # Individual fold results
    print(f"\n📊 Individual Fold Results:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: {result['best_accuracy']:.2f}%")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
