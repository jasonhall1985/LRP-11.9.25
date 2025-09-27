#!/usr/bin/env python3
"""
ICU Fine-tuning with GRID Pretrained Encoder
============================================

Stage 2 of the three-stage training pipeline: ICU fine-tuning with LOSO validation.
Loads GRID pretrained encoder and fine-tunes on ICU data with speaker-disjoint
validation for honest generalization metrics.

Training Strategy:
- Load GRID pretrained encoder weights
- Replace classification head for 4-class ICU classification
- LOSO cross-validation for speaker-disjoint evaluation
- Conservative fine-tuning with frozen encoder initially
- Staged unfreezing for optimal transfer learning

Author: Augment Agent
Date: 2025-09-27
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Import existing components
from advanced_training_components import (
    EnhancedLightweightCNNLSTM,
    ConservativeAugmentation,
    StandardizedPreprocessor,
    FocalLoss
)
from loso_cross_validation_framework import LOSODatasetManager, LOSOTrainer

class ICUFineTuner:
    """Handles ICU fine-tuning with GRID pretrained encoder."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Initialize components
        self.preprocessor = StandardizedPreprocessor()
        self.augmentation = ConservativeAugmentation()
        
        # Model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # ICU classes
        self.icu_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
        self.num_icu_classes = len(self.icu_classes)
        
        # Training state
        self.best_val_acc = 0.0
        self.training_history = []
        self.current_fold = 0
    
    def load_pretrained_encoder(self, pretrained_path: Path) -> bool:
        """Load GRID pretrained encoder weights."""
        if not pretrained_path.exists():
            print(f"⚠️  Pretrained model not found: {pretrained_path}")
            return False
        
        print(f"📥 Loading GRID pretrained encoder: {pretrained_path}")
        
        try:
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            
            # Create model with ICU classes (4 classes)
            self.model = EnhancedLightweightCNNLSTM(
                num_classes=self.num_icu_classes,
                input_channels=1,
                dropout=0.3
            ).to(self.device)
            
            # Load pretrained encoder weights (excluding classifier)
            pretrained_state = checkpoint['model_state_dict']
            model_state = self.model.state_dict()
            
            # Copy encoder weights (exclude classifier layers)
            encoder_keys = [k for k in pretrained_state.keys() if not k.startswith('classifier')]
            
            loaded_keys = []
            for key in encoder_keys:
                if key in model_state and pretrained_state[key].shape == model_state[key].shape:
                    model_state[key] = pretrained_state[key]
                    loaded_keys.append(key)
            
            self.model.load_state_dict(model_state)
            
            print(f"✅ Loaded {len(loaded_keys)} encoder layers from pretrained model")
            print(f"🎯 GRID classes: {checkpoint.get('num_classes', 'unknown')}")
            print(f"🎯 ICU classes: {self.num_icu_classes}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading pretrained model: {e}")
            return False
    
    def init_model_from_scratch(self):
        """Initialize model from scratch if no pretrained model available."""
        print("🏗️  Initializing model from scratch")
        
        self.model = EnhancedLightweightCNNLSTM(
            num_classes=self.num_icu_classes,
            input_channels=1,
            dropout=0.3
        ).to(self.device)
    
    def setup_training_components(self, freeze_encoder: bool = True):
        """Setup optimizer, scheduler, and loss function."""
        
        # Freeze encoder if requested
        if freeze_encoder and self.model is not None:
            print("🧊 Freezing encoder layers for initial fine-tuning")
            for name, param in self.model.named_parameters():
                if not name.startswith('classifier'):
                    param.requires_grad = False
        else:
            print("🔥 All layers unfrozen for training")
            for param in self.model.parameters():
                param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Parameters: {trainable_params:,} trainable / {total_params:,} total")
        
        # Initialize optimizer (only trainable parameters)
        trainable_params_list = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params_list,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=self.config['scheduler_patience'],
            min_lr=1e-6
        )
        
        # Initialize loss function with class weights for ICU data
        self.criterion = FocalLoss(
            alpha=1.0,
            gamma=2.0,
            reduction='mean'
        )
    
    def unfreeze_encoder_layers(self, unfreeze_last_n_blocks: int = 1):
        """Gradually unfreeze encoder layers for fine-tuning."""
        print(f"🔓 Unfreezing last {unfreeze_last_n_blocks} encoder blocks")
        
        # Get all encoder parameters
        encoder_params = [(name, param) for name, param in self.model.named_parameters() 
                         if not name.startswith('classifier')]
        
        # Unfreeze last N blocks (simplified - unfreeze all for now)
        for name, param in encoder_params:
            param.requires_grad = True
        
        # Update optimizer with newly unfrozen parameters
        self.setup_training_components(freeze_encoder=False)
    
    def train_fold(self, train_loader: DataLoader, val_loader: DataLoader, 
                   fold_num: int, output_dir: Path) -> Dict[str, float]:
        """Train one LOSO fold."""
        print(f"\n🎯 Training LOSO Fold {fold_num}")
        
        self.current_fold = fold_num
        fold_history = []
        best_fold_acc = 0.0
        
        # Stage 1: Frozen encoder training
        print("\n📍 Stage 1: Frozen encoder fine-tuning")
        self.setup_training_components(freeze_encoder=True)
        
        frozen_epochs = min(3, self.config['max_epochs'] // 3)
        for epoch in range(frozen_epochs):
            train_metrics = self.train_epoch(train_loader, epoch, stage="frozen")
            val_metrics = self.validate(val_loader)
            
            self.scheduler.step(val_metrics['accuracy'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{frozen_epochs} (Frozen) - "
                  f"Train: {train_metrics['accuracy']:.2f}%, "
                  f"Val: {val_metrics['accuracy']:.2f}%, "
                  f"LR: {current_lr:.6f}")
            
            fold_history.append({
                'epoch': epoch + 1,
                'stage': 'frozen',
                'train_accuracy': train_metrics['accuracy'],
                'val_accuracy': val_metrics['accuracy'],
                'learning_rate': current_lr
            })
            
            if val_metrics['accuracy'] > best_fold_acc:
                best_fold_acc = val_metrics['accuracy']
        
        # Stage 2: Unfrozen encoder training
        if self.config['max_epochs'] > frozen_epochs:
            print("\n📍 Stage 2: Unfrozen encoder fine-tuning")
            self.unfreeze_encoder_layers()
            
            remaining_epochs = self.config['max_epochs'] - frozen_epochs
            for epoch in range(remaining_epochs):
                train_metrics = self.train_epoch(train_loader, epoch + frozen_epochs, stage="unfrozen")
                val_metrics = self.validate(val_loader)
                
                self.scheduler.step(val_metrics['accuracy'])
                current_lr = self.optimizer.param_groups[0]['lr']
                
                print(f"Epoch {epoch+frozen_epochs+1}/{self.config['max_epochs']} (Unfrozen) - "
                      f"Train: {train_metrics['accuracy']:.2f}%, "
                      f"Val: {val_metrics['accuracy']:.2f}%, "
                      f"LR: {current_lr:.6f}")
                
                fold_history.append({
                    'epoch': epoch + frozen_epochs + 1,
                    'stage': 'unfrozen',
                    'train_accuracy': train_metrics['accuracy'],
                    'val_accuracy': val_metrics['accuracy'],
                    'learning_rate': current_lr
                })
                
                if val_metrics['accuracy'] > best_fold_acc:
                    best_fold_acc = val_metrics['accuracy']
                    # Save best model for this fold
                    self.save_fold_checkpoint(fold_num, val_metrics['accuracy'], output_dir)
                
                # Early stopping
                if current_lr < 1e-6:
                    print("🛑 Learning rate too low, stopping fold training")
                    break
        
        # Save fold history
        fold_history_path = output_dir / f"fold_{fold_num}_history.json"
        with open(fold_history_path, 'w') as f:
            json.dump(fold_history, f, indent=2)
        
        return {'best_accuracy': best_fold_acc, 'history': fold_history}
    
    def train_epoch(self, train_loader: DataLoader, epoch: int, stage: str = "") -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} ({stage})")
        
        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = total_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return {'loss': val_loss, 'accuracy': val_acc}
    
    def save_fold_checkpoint(self, fold_num: int, val_acc: float, output_dir: Path):
        """Save checkpoint for specific fold."""
        checkpoint = {
            'fold': fold_num,
            'model_state_dict': self.model.state_dict(),
            'val_accuracy': val_acc,
            'num_classes': self.num_icu_classes,
            'class_names': self.icu_classes,
            'config': self.config
        }
        
        checkpoint_path = output_dir / f"icu_finetune_fold_{fold_num}_best.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Saved fold {fold_num} checkpoint: {checkpoint_path}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='ICU Fine-tuning with GRID Pretrained Encoder')
    parser.add_argument('--pretrained-encoder', default='checkpoints/grid_pretraining/grid_pretrain_best.pth',
                       help='Path to GRID pretrained encoder')
    parser.add_argument('--data-dir', default='data/speaker sets',
                       help='Path to speaker sets directory')
    parser.add_argument('--output-dir', default='checkpoints/icu_finetuning',
                       help='Output directory for checkpoints')
    parser.add_argument('--max-epochs', type=int, default=20,
                       help='Maximum epochs per fold')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                       help='Initial learning rate for fine-tuning')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay for regularization')
    parser.add_argument('--scheduler-patience', type=int, default=5,
                       help='Scheduler patience')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Train from scratch without pretrained encoder')
    
    args = parser.parse_args()
    
    print("🎯 ICU FINE-TUNING - STAGE 2 OF THREE-STAGE PIPELINE")
    print("=" * 60)
    
    # Configuration
    config = {
        'max_epochs': args.max_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'scheduler_patience': args.scheduler_patience
    }
    
    print("⚙️  Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize fine-tuner
    fine_tuner = ICUFineTuner(config)
    
    # Load pretrained encoder or initialize from scratch
    if not args.no_pretrained:
        pretrained_path = Path(args.pretrained_encoder)
        if not fine_tuner.load_pretrained_encoder(pretrained_path):
            print("⚠️  Falling back to training from scratch")
            fine_tuner.init_model_from_scratch()
    else:
        fine_tuner.init_model_from_scratch()
    
    # Setup LOSO cross-validation using existing framework
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return 1
    
    # Use existing LOSO framework
    dataset_manager = LOSODatasetManager(data_dir)
    speakers = dataset_manager.speakers
    
    print(f"👥 Found {len(speakers)} speakers for LOSO validation")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run LOSO cross-validation
    fold_results = []
    
    # Generate LOSO splits for all speakers
    loso_splits = dataset_manager.generate_all_loso_splits(output_dir=f"{args.output_dir}/loso_splits")

    # Import required classes for data loading
    from advanced_training_components import (
        StandardizedPreprocessor, ConservativeAugmentation, ComprehensiveVideoDataset
    )
    from torch.utils.data import DataLoader

    # Create preprocessor and augmentation
    preprocessor = StandardizedPreprocessor(
        target_size=(64, 96),
        target_frames=32,
        grayscale=True,
        normalize=True
    )

    train_augmentation = ConservativeAugmentation(
        brightness_range=0.15,
        contrast_range=0.1,
        horizontal_flip_prob=0.5
    )

    # Process each LOSO fold
    for fold_num, (val_speaker, (train_manifest, val_manifest)) in enumerate(loso_splits.items()):
        print(f"\n🔄 LOSO Fold {fold_num + 1}: Validation speaker = {val_speaker}")
        print(f"Train manifest: {train_manifest}")
        print(f"Val manifest: {val_manifest}")

        # Create datasets
        train_dataset = ComprehensiveVideoDataset(
            manifest_path=train_manifest,
            preprocessor=preprocessor,
            augmentation=train_augmentation,
            synthetic_ratio=0.25
        )

        val_dataset = ComprehensiveVideoDataset(
            manifest_path=val_manifest,
            preprocessor=preprocessor,
            augmentation=None,
            synthetic_ratio=0.0
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # Train fold
        fold_result = fine_tuner.train_fold(train_loader, val_loader, fold_num + 1, output_dir)
        fold_results.append({
            'fold': fold_num + 1,
            'val_speaker': val_speaker,
            'best_accuracy': fold_result['best_accuracy']
        })
        
        print(f"✅ Fold {fold_num + 1} complete: {fold_result['best_accuracy']:.2f}% accuracy")
    
    # Calculate overall LOSO performance
    accuracies = [result['best_accuracy'] for result in fold_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n🏆 LOSO CROSS-VALIDATION RESULTS")
    print("=" * 40)
    for result in fold_results:
        print(f"Fold {result['fold']} ({result['val_speaker']}): {result['best_accuracy']:.2f}%")
    
    print(f"\n📊 Overall Performance:")
    print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    # Save overall results
    results_path = output_dir / "loso_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'fold_results': fold_results,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'config': config
        }, f, indent=2)
    
    print(f"💾 Saved results: {results_path}")
    print(f"\n🎯 ICU fine-tuning complete!")
    print(f"🔄 Next step: Few-shot personalization for bedside deployment")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
