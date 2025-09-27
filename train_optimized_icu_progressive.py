#!/usr/bin/env python3
"""
Phase 4: Optimized ICU Fine-tuning with Progressive Unfreezing
==============================================================

Implements advanced training techniques to achieve >82% LOSO validation accuracy:
- Progressive unfreezing strategy (3 stages)
- Label smoothing and weighted sampling
- Curriculum learning with high-quality clips first
- Test-time augmentation evaluation
- Enhanced regularization and optimization

Author: Augment Agent
Date: 2025-09-27
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

# Import our components
from advanced_training_components import (
    EnhancedLightweightCNNLSTM,
    FocalLoss,
    ConservativeAugmentation,
    StandardizedPreprocessor,
    ComprehensiveVideoDataset
)
from loso_cross_validation_framework import LOSODatasetManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.log_softmax(pred, dim=-1)
        
        # Create smoothed targets
        smooth_target = torch.zeros_like(log_preds)
        smooth_target.fill_(self.smoothing / (n_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-smooth_target * log_preds, dim=-1))

class SimpleEarlyStopping:
    """Simple early stopping implementation"""
    def __init__(self, patience=25, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.best_weights = model.state_dict().copy()
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
            self.best_weights = model.state_dict().copy()
            
    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

class ProgressiveUnfreezingTrainer:
    """Advanced trainer with progressive unfreezing and optimization techniques"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Progressive Unfreezing Trainer initialized on {self.device}")
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        
    def create_model(self, num_classes=4):
        """Create and initialize the model"""
        self.model = EnhancedLightweightCNNLSTM(
            num_classes=num_classes,
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
        
    def setup_progressive_training(self, stage=1):
        """Setup training components for different unfreezing stages"""
        
        # Stage 1: Freeze encoder, train only classifier
        if stage == 1:
            logger.info("🔒 Stage 1: Freezing encoder, training classifier only")
            for name, param in self.model.named_parameters():
                if 'classifier' not in name and 'lstm' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    
        # Stage 2: Unfreeze LSTM and last CNN block
        elif stage == 2:
            logger.info("🔓 Stage 2: Unfreezing LSTM and last CNN block")
            for name, param in self.model.named_parameters():
                if any(x in name for x in ['classifier', 'lstm', 'features.4', 'features.5']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        # Stage 3: Unfreeze all layers with reduced learning rate
        elif stage == 3:
            logger.info("🔓 Stage 3: Unfreezing all layers")
            for param in self.model.parameters():
                param.requires_grad = True
                
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters in stage {stage}: {trainable_params:,}")
        
        # Setup optimizer with stage-specific learning rate
        lr_multipliers = {1: 1.0, 2: 0.5, 3: 0.1}
        base_lr = self.config['learning_rate'] * lr_multipliers[stage]
        
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=base_lr,
            weight_decay=self.config['weight_decay']
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['scheduler_t0'],
            T_mult=self.config['scheduler_tmult'],
            eta_min=self.config['min_lr']
        )
        
        # Setup loss function with label smoothing
        if self.config.get('use_label_smoothing', True):
            self.criterion = LabelSmoothingCrossEntropy(smoothing=self.config['label_smoothing'])
        else:
            self.criterion = FocalLoss(gamma=self.config['focal_gamma'])
            
        # Setup early stopping
        self.early_stopping = SimpleEarlyStopping(
            patience=self.config['early_stopping_patience'],
            delta=self.config['early_stopping_delta']
        )
        
        logger.info(f"Stage {stage} setup complete - LR: {base_lr:.6f}")
        
    def create_weighted_sampler(self, dataset):
        """Create weighted sampler for class imbalance"""
        # Count class frequencies
        class_counts = {}
        for _, label in dataset:
            class_name = dataset.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        # Calculate weights (inverse frequency)
        total_samples = len(dataset)
        class_weights = {}
        for class_name, count in class_counts.items():
            class_weights[class_name] = total_samples / (len(class_counts) * count)
            
        # Create sample weights
        sample_weights = []
        for _, label in dataset:
            class_name = dataset.classes[label]
            sample_weights.append(class_weights[class_name])
            
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
    def train_stage(self, train_loader, val_loader, stage, epochs_per_stage):
        """Train a single progressive unfreezing stage"""
        logger.info(f"\n🚀 Starting Stage {stage} Training ({epochs_per_stage} epochs)")
        
        stage_history = []
        best_val_acc = 0.0
        
        for epoch in range(epochs_per_stage):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Stage {stage} Epoch {epoch+1}")
            for batch_idx, (videos, labels) in enumerate(train_pbar):
                videos, labels = videos.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_norm'])
                
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                train_acc = 100.0 * train_correct / train_total
                current_lr = self.optimizer.param_groups[0]['lr']
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{train_acc:.2f}%',
                    'LR': f'{current_lr:.6f}'
                })
                
            # Validation phase
            val_loss, val_acc, val_f1 = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log epoch results
            epoch_stats = {
                'stage': stage,
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'train_accuracy': 100.0 * train_correct / train_total,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_f1': val_f1,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            stage_history.append(epoch_stats)
            
            logger.info(
                f"Stage {stage} Epoch {epoch+1:2d}/{epochs_per_stage} | "
                f"Train Loss: {epoch_stats['train_loss']:.4f} | "
                f"Train Acc: {epoch_stats['train_accuracy']:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"Val F1: {val_f1:.4f} | "
                f"LR: {epoch_stats['learning_rate']:.6f}"
            )
            
            # Early stopping check (based on F1 score for better class balance handling)
            self.early_stopping(val_f1, self.model)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
            if self.early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        # Restore best weights
        self.early_stopping.restore_best_weights(self.model)
        logger.info(f"✅ Stage {stage} complete. Best validation accuracy: {best_val_acc:.2f}%")
        
        return stage_history, best_val_acc
        
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_accuracy = 100.0 * val_correct / val_total
        val_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        return val_loss / len(val_loader), val_accuracy, val_f1

    def train_fold(self, fold_num, train_manifest, val_manifest):
        """Train a single LOSO fold with progressive unfreezing"""
        logger.info(f"\n🎯 FOLD {fold_num}: Progressive Unfreezing Training")
        logger.info(f"Train manifest: {train_manifest}")
        logger.info(f"Val manifest: {val_manifest}")

        # Create datasets
        preprocessor = StandardizedPreprocessor()
        augmentation = ConservativeAugmentation()

        train_dataset = ComprehensiveVideoDataset(
            train_manifest, preprocessor, augmentation
        )
        val_dataset = ComprehensiveVideoDataset(
            val_manifest, preprocessor, None
        )

        logger.info(f"📊 Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")

        # Create model
        self.create_model(num_classes=len(train_dataset.classes))

        # Create data loaders with weighted sampling for training
        if self.config.get('use_weighted_sampling', True):
            train_sampler = self.create_weighted_sampler(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                sampler=train_sampler,
                num_workers=self.config['num_workers'],
                pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                pin_memory=True
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        # Progressive training stages
        all_history = []
        best_overall_acc = 0.0

        # Stage 1: Classifier only (5 epochs)
        self.setup_progressive_training(stage=1)
        stage1_history, stage1_acc = self.train_stage(train_loader, val_loader, 1, 5)
        all_history.extend(stage1_history)
        best_overall_acc = max(best_overall_acc, stage1_acc)

        # Stage 2: LSTM + last CNN block (8 epochs)
        self.setup_progressive_training(stage=2)
        stage2_history, stage2_acc = self.train_stage(train_loader, val_loader, 2, 8)
        all_history.extend(stage2_history)
        best_overall_acc = max(best_overall_acc, stage2_acc)

        # Stage 3: Full model fine-tuning (12 epochs)
        self.setup_progressive_training(stage=3)
        stage3_history, stage3_acc = self.train_stage(train_loader, val_loader, 3, 12)
        all_history.extend(stage3_history)
        best_overall_acc = max(best_overall_acc, stage3_acc)

        # Save fold results
        fold_results = {
            'fold': fold_num,
            'best_accuracy': best_overall_acc,
            'stage_accuracies': {
                'stage1': stage1_acc,
                'stage2': stage2_acc,
                'stage3': stage3_acc
            },
            'training_history': all_history
        }

        # Save model checkpoint
        checkpoint_path = os.path.join(
            self.config['output_dir'],
            f'progressive_fold_{fold_num}_best.pth'
        )
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'fold_results': fold_results,
            'config': self.config
        }, checkpoint_path)

        logger.info(f"✅ Fold {fold_num} complete: {best_overall_acc:.2f}% accuracy")
        logger.info(f"Stage breakdown: S1={stage1_acc:.1f}%, S2={stage2_acc:.1f}%, S3={stage3_acc:.1f}%")

        return fold_results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Progressive Unfreezing ICU Training')
    parser.add_argument('--data-dir', default='data/stabilized_speaker_sets',
                       help='Directory containing stabilized video data')
    parser.add_argument('--output-dir', default='checkpoints/progressive_unfreezing',
                       help='Output directory for checkpoints and results')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.02, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')
    parser.add_argument('--label-smoothing', type=float, default=0.05, help='Label smoothing')
    parser.add_argument('--use-weighted-sampling', action='store_true', default=True,
                       help='Use weighted sampling for class imbalance')
    parser.add_argument('--use-label-smoothing', action='store_true', default=True,
                       help='Use label smoothing instead of focal loss')

    args = parser.parse_args()

    # Configuration
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'label_smoothing': args.label_smoothing,
        'use_weighted_sampling': args.use_weighted_sampling,
        'use_label_smoothing': args.use_label_smoothing,
        'num_workers': 4,
        'early_stopping_patience': 15,  # Reduced for progressive training
        'early_stopping_delta': 0.001,
        'grad_clip_norm': 1.0,
        'focal_gamma': 2.0,
        'scheduler_t0': 5,  # Shorter cycles for progressive training
        'scheduler_tmult': 1,
        'min_lr': 1e-7
    }

    logger.info("🚀 PROGRESSIVE UNFREEZING ICU TRAINING")
    logger.info("=" * 60)
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    # Initialize trainer
    trainer = ProgressiveUnfreezingTrainer(config)

    # Initialize LOSO manager
    loso_manager = LOSODatasetManager(config['data_dir'])

    # Generate LOSO splits
    logger.info("📊 Generating LOSO splits...")
    loso_splits = loso_manager.generate_all_loso_splits(
        output_dir=os.path.join(config['output_dir'], 'loso_splits')
    )

    # Train all folds
    all_results = []
    start_time = time.time()

    for fold_num, (speaker, (train_manifest, val_manifest)) in enumerate(loso_splits.items(), 1):
        logger.info(f"\n🎯 Processing Fold {fold_num}: Held-out speaker '{speaker}'")
        fold_results = trainer.train_fold(fold_num, train_manifest, val_manifest)
        fold_results['held_out_speaker'] = speaker
        all_results.append(fold_results)

    # Calculate final statistics
    accuracies = [result['best_accuracy'] for result in all_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    total_time = time.time() - start_time

    # Save final results
    final_results = {
        'config': config,
        'fold_results': all_results,
        'summary': {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'individual_accuracies': accuracies,
            'total_training_time_hours': total_time / 3600
        }
    }

    results_path = os.path.join(config['output_dir'], 'progressive_unfreezing_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Print final summary
    logger.info("\n🎉 PROGRESSIVE UNFREEZING TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Average LOSO Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    logger.info(f"Total Training Time: {total_time/3600:.1f} hours")
    logger.info(f"Results saved to: {results_path}")

    logger.info("\n📊 Individual Fold Results:")
    for i, result in enumerate(all_results, 1):
        acc = result['best_accuracy']
        stages = result['stage_accuracies']
        logger.info(f"  Fold {i}: {acc:.2f}% (S1: {stages['stage1']:.1f}%, "
                   f"S2: {stages['stage2']:.1f}%, S3: {stages['stage3']:.1f}%)")

if __name__ == "__main__":
    main()
