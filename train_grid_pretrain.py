#!/usr/bin/env python3
"""
GRID Pretraining for Three-Stage Pipeline
=========================================

Stage 1 of the three-stage training pipeline: GRID corpus pretraining.
Trains the model on viseme-matched GRID words to learn robust visual features
before fine-tuning on ICU data.

Training Strategy:
- Multi-class word classification on selected GRID subset
- Conservative data augmentation to preserve visual features
- Encoder-focused training with lightweight classification head
- Checkpointing for subsequent ICU fine-tuning

Author: Augment Agent
Date: 2025-09-27
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
from tqdm import tqdm

# Import training components
from advanced_training_components import (
    EnhancedLightweightCNNLSTM,
    ConservativeAugmentation,
    StandardizedPreprocessor,
    FocalLoss
)

class GRIDWordDataset(Dataset):
    """Dataset for GRID word-level pretraining."""
    
    def __init__(self, manifest_df: pd.DataFrame, preprocessor: StandardizedPreprocessor,
                 augmentation: ConservativeAugmentation = None, is_training: bool = True):
        self.manifest = manifest_df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.augmentation = augmentation if is_training else None
        self.is_training = is_training
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.manifest['word'])
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"📊 Dataset: {len(self.manifest)} examples, {self.num_classes} word classes")
        print(f"🏷️  Word classes: {list(self.label_encoder.classes_)}")
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        
        # For GRID, we need to extract word-level clips from full sentences
        # This is a simplified version - in practice, you'd need temporal alignment
        video_path = row['file_path']
        
        try:
            # Load and preprocess video
            frames = self.load_video_frames(video_path)
            
            if frames is None or len(frames) == 0:
                # Return dummy data for failed loads
                frames = np.zeros((32, 64, 48, 1), dtype=np.float32)
            
            # Apply augmentation if training
            if self.augmentation is not None:
                frames = self.augmentation.apply_augmentation(frames)
            
            # Convert to tensor
            frames_tensor = torch.FloatTensor(frames).permute(3, 0, 1, 2)  # (C, T, H, W)
            
            # Get label
            label = self.labels[idx]
            
            return {
                'frames': frames_tensor,
                'label': torch.LongTensor([label])[0],
                'word': row['word'],
                'speaker_id': row['speaker_id'],
                'icu_class': row['icu_class']
            }
            
        except Exception as e:
            print(f"⚠️  Error loading {video_path}: {e}")
            # Return dummy data
            frames = np.zeros((32, 64, 48, 1), dtype=np.float32)
            frames_tensor = torch.FloatTensor(frames).permute(3, 0, 1, 2)
            return {
                'frames': frames_tensor,
                'label': torch.LongTensor([self.labels[idx]])[0],
                'word': row['word'],
                'speaker_id': row['speaker_id'],
                'icu_class': row['icu_class']
            }
    
    def load_video_frames(self, video_path: str) -> Optional[np.ndarray]:
        """Load video frames with basic preprocessing."""
        if not os.path.exists(video_path):
            return None
        
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale and resize
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized_frame = cv2.resize(gray_frame, (48, 64))  # (W, H)
                frames.append(resized_frame)
            
            cap.release()
            
            if len(frames) == 0:
                return None
            
            # Temporal sampling to 32 frames
            frames = np.array(frames)
            frames = self.preprocessor.temporal_sampling(frames, target_length=32)
            
            # Normalize
            frames = frames.astype(np.float32) / 255.0
            frames = np.expand_dims(frames, axis=-1)  # Add channel dimension
            
            return frames
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None

class GRIDPretrainer:
    """Handles GRID pretraining process."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Initialize components
        self.preprocessor = StandardizedPreprocessor()
        self.augmentation = ConservativeAugmentation()
        
        # Model will be initialized after loading data (need num_classes)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.best_val_acc = 0.0
        self.training_history = []
    
    def load_data(self, manifest_path: Path) -> Tuple[DataLoader, DataLoader]:
        """Load and split GRID pretraining data."""
        print(f"📋 Loading GRID pretraining manifest: {manifest_path}")
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        manifest_df = pd.read_csv(manifest_path)
        print(f"📊 Loaded {len(manifest_df)} training examples")
        
        # Split by speaker to ensure speaker-disjoint validation
        speakers = manifest_df['speaker_id'].unique()
        train_speakers, val_speakers = train_test_split(
            speakers, test_size=0.2, random_state=42
        )
        
        train_df = manifest_df[manifest_df['speaker_id'].isin(train_speakers)]
        val_df = manifest_df[manifest_df['speaker_id'].isin(val_speakers)]
        
        print(f"📈 Train: {len(train_df)} examples from {len(train_speakers)} speakers")
        print(f"📉 Val: {len(val_df)} examples from {len(val_speakers)} speakers")
        
        # Create datasets
        train_dataset = GRIDWordDataset(train_df, self.preprocessor, self.augmentation, is_training=True)
        val_dataset = GRIDWordDataset(val_df, self.preprocessor, augmentation=None, is_training=False)
        
        # Initialize model with correct number of classes
        self.num_classes = train_dataset.num_classes
        self.word_classes = train_dataset.label_encoder.classes_
        self.init_model()
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            drop_last=True,
            pin_memory=False  # Disable for MPS compatibility
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            drop_last=False,
            pin_memory=False
        )
        
        return train_loader, val_loader
    
    def init_model(self):
        """Initialize model, optimizer, and loss function."""
        print(f"🏗️  Initializing model for {self.num_classes} word classes")
        
        # Create model
        self.model = EnhancedLightweightCNNLSTM(
            num_classes=self.num_classes,
            input_channels=1,
            dropout=0.3
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
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
        
        # Initialize loss function
        self.criterion = FocalLoss(
            alpha=1.0,
            gamma=2.0,
            reduction='mean'
        )
        
        print(f"✅ Model initialization complete")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
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
            for batch in tqdm(val_loader, desc="Validation"):
                frames = batch['frames'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = total_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return {'loss': val_loss, 'accuracy': val_acc}
    
    def save_checkpoint(self, epoch: int, val_acc: float, output_dir: Path, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_accuracy': val_acc,
            'num_classes': self.num_classes,
            'word_classes': list(self.word_classes),
            'config': self.config,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = output_dir / f"grid_pretrain_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = output_dir / "grid_pretrain_best.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model: {best_path}")
        
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, output_dir: Path):
        """Main training loop."""
        print(f"🚀 Starting GRID pretraining for {self.config['max_epochs']} epochs")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config['max_epochs']):
            print(f"\n📅 Epoch {epoch+1}/{self.config['max_epochs']}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_metrics['accuracy'])
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"📊 Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"📊 Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"📊 Learning Rate: {current_lr:.6f}")
            
            # Save training history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'learning_rate': current_lr
            })
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
            
            self.save_checkpoint(epoch, val_metrics['accuracy'], output_dir, is_best)
            
            # Early stopping check
            if current_lr < 1e-6:
                print("🛑 Learning rate too low, stopping training")
                break
        
        print(f"\n✅ GRID pretraining complete!")
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Save final training history
        history_path = output_dir / "grid_pretraining_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"📊 Saved training history: {history_path}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='GRID Pretraining for Three-Stage Pipeline')
    parser.add_argument('--manifest-path', default='manifests/pretraining/grid_pretraining_manifest.csv',
                       help='Path to GRID pretraining manifest')
    parser.add_argument('--output-dir', default='checkpoints/grid_pretraining',
                       help='Output directory for checkpoints')
    parser.add_argument('--max-epochs', type=int, default=50,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay for regularization')
    parser.add_argument('--scheduler-patience', type=int, default=10,
                       help='Scheduler patience for learning rate reduction')
    
    args = parser.parse_args()
    
    print("🎯 GRID PRETRAINING - STAGE 1 OF THREE-STAGE PIPELINE")
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
    
    # Check manifest
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        print("Please run tools/select_grid_subset.py first")
        return 1
    
    # Initialize trainer
    trainer = GRIDPretrainer(config)
    
    # Load data
    try:
        train_loader, val_loader = trainer.load_data(manifest_path)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1
    
    # Train model
    output_dir = Path(args.output_dir)
    trainer.train(train_loader, val_loader, output_dir)
    
    print(f"\n🎯 GRID pretraining complete!")
    print(f"📁 Checkpoints saved to: {output_dir}")
    print(f"🔄 Next step: ICU fine-tuning with LOSO validation")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
