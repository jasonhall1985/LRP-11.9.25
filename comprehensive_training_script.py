#!/usr/bin/env python3
"""
🎯 COMPREHENSIVE TRAINING SCRIPT
===============================

Advanced training script for achieving 82% cross-demographic validation accuracy
using the comprehensive speaker-disjoint pipeline.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import our advanced components
from advanced_training_components import (
    EnhancedLightweightCNNLSTM,
    FocalLoss,
    ConservativeAugmentation,
    StandardizedPreprocessor,
    ComprehensiveVideoDataset
)

class ComprehensiveTrainer:
    """Comprehensive trainer for speaker-disjoint training"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set seeds for reproducibility
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        
        print(f"🎯 COMPREHENSIVE TRAINING SCRIPT")
        print(f"=" * 60)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️  Device: {self.device}")
        print(f"🌱 Seed: {config['seed']}")
        print(f"🎯 Target validation accuracy: {config['target_accuracy']}%")
        print("")
        
        # Initialize components
        self._setup_model()
        self._setup_data()
        self._setup_training()
        
    def _setup_model(self):
        """Setup model architecture"""
        print("🏗️  Setting up enhanced model architecture...")
        
        self.model = EnhancedLightweightCNNLSTM(
            num_classes=4,
            dropout=self.config['dropout']
        ).to(self.device)
        
        param_count = self.model.count_parameters()
        print(f"   Model parameters: {param_count:,}")
        print(f"   Architecture: Enhanced 3D CNN-LSTM with temporal attention")
        print("")
        
    def _setup_data(self):
        """Setup data loaders"""
        print("📂 Setting up data loaders...")
        
        # Preprocessor
        preprocessor = StandardizedPreprocessor(
            target_size=(64, 96),
            target_frames=32,
            grayscale=True,
            normalize=True
        )
        
        # Augmentation for training only
        train_augmentation = ConservativeAugmentation(
            brightness_range=0.15,
            contrast_range=0.1,
            horizontal_flip_prob=0.5
        ) if self.config['use_augmentation'] else None
        
        # Training dataset
        self.train_dataset = ComprehensiveVideoDataset(
            manifest_path=self.config['train_manifest'],
            preprocessor=preprocessor,
            augmentation=train_augmentation,
            synthetic_ratio=self.config['synthetic_ratio']
        )
        
        # Validation dataset (no augmentation)
        self.val_dataset = ComprehensiveVideoDataset(
            manifest_path=self.config['val_manifest'],
            preprocessor=preprocessor,
            augmentation=None,
            synthetic_ratio=0.0  # Pure real data for validation
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation batches: {len(self.val_loader)}")
        print(f"   Training samples: {len(self.train_dataset)}")
        print(f"   Validation samples: {len(self.val_dataset)}")
        print("")
        
    def _setup_training(self):
        """Setup training components"""
        print("⚙️  Setting up training components...")
        
        # Calculate class weights for focal loss
        train_df = pd.read_csv(self.config['train_manifest'])
        class_counts = train_df['class'].value_counts()
        total_samples = len(train_df)
        
        class_weights = []
        for cls in self.train_dataset.classes:
            weight = total_samples / (len(self.train_dataset.classes) * class_counts[cls])
            class_weights.append(weight)
        
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        print(f"   Class weights: {class_weights.cpu().numpy()}")
        
        # Loss function
        self.criterion = FocalLoss(
            alpha=class_weights,
            gamma=2.0,
            reduction='mean'
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Training state
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rate': []
        }
        
        print(f"   Optimizer: AdamW (lr={self.config['learning_rate']}, wd={self.config['weight_decay']})")
        print(f"   Loss: Focal Loss (gamma=2.0)")
        print(f"   Scheduler: ReduceLROnPlateau")
        print("")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Progress reporting
            if batch_idx % max(1, len(self.train_loader) // 5) == 0:
                print(f"   Batch {batch_idx}/{len(self.train_loader)}: Loss {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                _, predicted = output.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_targets, all_predictions) * 100
        f1 = f1_score(all_targets, all_predictions, average='macro') * 100
        
        return avg_loss, accuracy, f1, all_predictions, all_targets
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'config': self.config,
            'class_to_idx': self.train_dataset.class_to_idx,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config['output_dir']) / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config['output_dir']) / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✅ New best model saved! Validation accuracy: {self.best_val_acc:.2f}%")
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting comprehensive training...")
        print("=" * 60)
        
        for epoch in range(1, self.config['max_epochs'] + 1):
            print(f"📅 Epoch {epoch}/{self.config['max_epochs']}")
            print("-" * 40)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_f1, val_preds, val_targets = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_f1'].append(val_f1)
            self.training_history['learning_rate'].append(current_lr)
            
            # Print epoch results
            print(f"📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%")
            print(f"📊 Learning Rate: {current_lr:.2e}")
            
            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                self.epochs_without_improvement = 0
                
                # Save detailed validation results
                self._save_validation_analysis(epoch, val_preds, val_targets)
            else:
                self.epochs_without_improvement += 1
                print(f"⏳ No improvement for {self.epochs_without_improvement} epochs")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config['early_stop_patience']:
                print(f"🛑 Early stopping after {self.epochs_without_improvement} epochs without improvement")
                break
            
            # Target accuracy reached
            if val_acc >= self.config['target_accuracy']:
                print(f"🎯 Target accuracy {self.config['target_accuracy']}% reached!")
                break
            
            print("")
        
        print(f"🏁 Training completed!")
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"🏆 Best validation F1-macro: {self.best_val_f1:.2f}%")
        
        # Save final training history
        self._save_training_plots()
        
        return self.best_val_acc >= self.config['target_accuracy']
    
    def _save_validation_analysis(self, epoch, predictions, targets):
        """Save detailed validation analysis"""
        output_dir = Path(self.config['output_dir'])
        
        # Classification report
        class_names = self.val_dataset.classes
        report = classification_report(
            targets, predictions,
            target_names=class_names,
            output_dict=True
        )
        
        with open(output_dir / f"validation_report_epoch_{epoch}.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / f"confusion_matrix_epoch_{epoch}.png", dpi=300)
        plt.close()
    
    def _save_training_plots(self):
        """Save training history plots"""
        output_dir = Path(self.config['output_dir'])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss plot
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.training_history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.training_history['val_acc'], label='Val Acc')
        axes[0, 1].axhline(y=self.config['target_accuracy'], color='r', linestyle='--', label='Target')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 score plot
        axes[1, 0].plot(self.training_history['val_f1'], label='Val F1-macro')
        axes[1, 0].set_title('Validation F1-macro Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        axes[1, 1].plot(self.training_history['learning_rate'])
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "training_history.png", dpi=300)
        plt.close()
        
        # Save history as JSON
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Comprehensive Speaker-Disjoint Training')
    parser.add_argument('--train-manifest', required=True, help='Training manifest CSV')
    parser.add_argument('--val-manifest', required=True, help='Validation manifest CSV')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--target-accuracy', type=float, default=82.0, help='Target validation accuracy')
    parser.add_argument('--max-epochs', type=int, default=80, help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--early-stop-patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--synthetic-ratio', type=float, default=0.25, help='Synthetic augmentation ratio')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = {
        'train_manifest': args.train_manifest,
        'val_manifest': args.val_manifest,
        'output_dir': str(output_dir),
        'target_accuracy': args.target_accuracy,
        'max_epochs': args.max_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'early_stop_patience': args.early_stop_patience,
        'synthetic_ratio': args.synthetic_ratio,
        'seed': args.seed,
        'use_augmentation': not args.no_augmentation
    }
    
    # Save configuration
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize and run trainer
    trainer = ComprehensiveTrainer(config)
    success = trainer.train()
    
    if success:
        print(f"🎉 SUCCESS: Target accuracy {config['target_accuracy']}% achieved!")
    else:
        print(f"⚠️  Target accuracy {config['target_accuracy']}% not reached")
        print(f"   Best achieved: {trainer.best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
