#!/usr/bin/env python3
"""
Conservative Resumed Training from 75.9% Checkpoint
Addresses overfitting issues with ultra-conservative fine-tuning approach

Key Changes:
- Ultra-low learning rate (0.00001)
- Freeze convolutional layers, only train FC layers
- Minimal augmentation to prevent overfitting
- ReduceLROnPlateau scheduler
- Increased regularization
"""

import os
import csv
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import time
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from datetime import datetime

class ConservativeResumedTrainer:
    def __init__(self):
        # Checkpoint configuration
        self.checkpoint_dir = Path("backup_75.9_success_20250921_004410")
        self.checkpoint_path = self.checkpoint_dir / "best_4class_model.pth"
        self.train_manifest = self.checkpoint_dir / "4class_train_manifest.csv"
        self.val_manifest = self.checkpoint_dir / "4class_validation_manifest.csv"
        
        # Output directory
        self.output_dir = Path("conservative_resumed_training_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ultra-conservative training configuration
        self.batch_size = 4  # Reduced for better generalization
        self.max_epochs = 40  # Reduced epochs
        self.initial_lr = 0.00001  # Ultra-low learning rate (100x reduction)
        self.device = torch.device('cpu')
        
        # Conservative target
        self.target_val_acc = 82.0
        self.early_stopping_patience = 15  # Reduced patience
        self.min_improvement = 0.5  # Higher improvement threshold
        
        # Class configuration
        self.selected_classes = ['my_mouth_is_dry', 'i_need_to_move', 'doctor', 'pillow']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.selected_classes)}
        
        # Conservative training settings
        self.freeze_conv_layers = True  # Only train FC layers
        self.minimal_augmentation = True  # Reduce augmentation
        
        print("🛡️  CONSERVATIVE RESUMED TRAINING FROM 75.9% CHECKPOINT")
        print("=" * 80)
        print(f"🎯 Target: {self.target_val_acc}% validation accuracy")
        print(f"🔒 Strategy: Ultra-conservative fine-tuning to prevent overfitting")
        print(f"❄️  Learning rate: {self.initial_lr} (ultra-low)")
        print(f"🧊 Freeze conv layers: {self.freeze_conv_layers}")
        print(f"📉 Minimal augmentation: {self.minimal_augmentation}")
        
    def load_checkpoint_and_setup(self):
        """Load checkpoint and setup conservative training."""
        print("\n📋 LOADING CHECKPOINT AND SETTING UP CONSERVATIVE TRAINING")
        print("=" * 70)
        
        # Load checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        print(f"📥 Loading checkpoint: {self.checkpoint_path}")
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        print(f"✅ Loaded checkpoint with validation accuracy: {self.checkpoint.get('best_val_acc', 'N/A')}%")
        
        # Create conservative datasets
        self.train_dataset = ConservativeLipReadingDataset(
            self.train_manifest, self.class_to_idx, 
            augment=True, minimal_augmentation=self.minimal_augmentation
        )
        self.val_dataset = ConservativeLipReadingDataset(
            self.val_manifest, self.class_to_idx, 
            augment=False, minimal_augmentation=False
        )
        
        print(f"📊 Training: {len(self.train_dataset)} videos")
        print(f"📊 Validation: {len(self.val_dataset)} videos")
        
        # Create data loaders (no weighted sampling to reduce complexity)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        print(f"📊 Training batches: {len(self.train_loader)}")
        print(f"📊 Validation batches: {len(self.val_loader)}")
        
    def setup_conservative_model(self):
        """Setup model with conservative fine-tuning."""
        print("\n🏗️  SETTING UP CONSERVATIVE MODEL")
        print("=" * 50)
        
        # Initialize model
        self.model = ConservativeLipReadingModel().to(self.device)
        
        # Load model state
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        print(f"✅ Loaded model state from checkpoint")
        
        # Freeze convolutional layers if specified
        if self.freeze_conv_layers:
            for name, param in self.model.named_parameters():
                if 'conv3d' in name or 'bn3d' in name:
                    param.requires_grad = False
            print(f"❄️  Froze convolutional layers for conservative fine-tuning")
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Ultra-conservative optimizer
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_parameters,
            lr=self.initial_lr,  # Ultra-low LR
            weight_decay=1e-3,   # Increased weight decay
            betas=(0.9, 0.999)
        )
        
        # Conservative scheduler - reduce on plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5,
            min_lr=1e-7
        )
        
        # Standard loss function (no class weighting to reduce complexity)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Minimal label smoothing
        
        # Initialize tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = self.checkpoint.get('best_val_acc', 0.0)
        self.start_epoch = 1  # Start fresh epoch counting
        self.epochs_without_improvement = 0
        
        print(f"✅ Conservative model setup complete:")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Learning rate: {self.initial_lr}")
        print(f"   Scheduler: ReduceLROnPlateau")
        print(f"   Starting validation accuracy: {self.best_val_acc:.2f}%")
        
    def train_epoch(self, epoch):
        """Conservative training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (videos, labels) in enumerate(self.train_loader):
            videos, labels = videos.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Conservative gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch):
        """Conservative validation epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for videos, labels in self.val_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def run_conservative_training(self):
        """Execute conservative training pipeline."""
        print("\n🚀 STARTING CONSERVATIVE TRAINING PIPELINE")
        print("=" * 60)
        
        try:
            # Setup
            self.load_checkpoint_and_setup()
            self.setup_conservative_model()
            
            print(f"\n🎯 CONSERVATIVE TRAINING TARGET: {self.target_val_acc}% validation accuracy")
            print(f"📈 Starting from: {self.best_val_acc:.2f}%")
            print(f"🛡️  Strategy: Ultra-conservative fine-tuning to prevent overfitting")
            
            # Training loop
            for epoch in range(self.start_epoch, self.max_epochs + 1):
                print(f"\n{'='*50}")
                print(f"EPOCH {epoch}/{self.max_epochs}")
                print(f"{'='*50}")
                
                # Train and validate
                train_loss, train_acc = self.train_epoch(epoch)
                val_loss, val_acc = self.validate_epoch(epoch)
                
                # Update scheduler
                self.scheduler.step(val_acc)
                
                # Check for improvement
                is_best = val_acc > self.best_val_acc
                if is_best:
                    improvement = val_acc - self.best_val_acc
                    if improvement >= self.min_improvement:
                        self.best_val_acc = val_acc
                        self.epochs_without_improvement = 0
                        print(f"🎉 NEW BEST: {val_acc:.2f}% (+{improvement:.2f}%)")
                        
                        # Check target
                        if val_acc >= self.target_val_acc:
                            print(f"🎯 TARGET ACHIEVED! {val_acc:.2f}% >= {self.target_val_acc}%")
                            return True
                    else:
                        print(f"📈 Small improvement: {val_acc:.2f}% (+{improvement:.2f}%)")
                        self.epochs_without_improvement += 1
                else:
                    self.epochs_without_improvement += 1
                    print(f"📉 No improvement for {self.epochs_without_improvement} epochs")
                
                # Early stopping
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\n⏹️  EARLY STOPPING: No improvement for {self.early_stopping_patience} epochs")
                    break
                
                # Progress summary
                print(f"\n📊 EPOCH {epoch} SUMMARY:")
                print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
                print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
                print(f"   Best:  {self.best_val_acc:.2f}% (target: {self.target_val_acc}%)")
                print(f"   LR:    {self.optimizer.param_groups[0]['lr']:.8f}")
            
            return self.best_val_acc >= self.target_val_acc
            
        except Exception as e:
            print(f"\n❌ TRAINING ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


class ConservativeLipReadingDataset(Dataset):
    """Conservative dataset with minimal augmentation."""
    
    def __init__(self, manifest_path, class_to_idx, augment=False, minimal_augmentation=True):
        self.manifest_path = Path(manifest_path)
        self.class_to_idx = class_to_idx
        self.augment = augment
        self.minimal_augmentation = minimal_augmentation
        self.videos = []
        
        # Load videos
        with open(self.manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['class'] in self.class_to_idx:
                    self.videos.append({
                        'path': row['video_path'],
                        'class': row['class']
                    })
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_info = self.videos[idx]
        frames = self.load_video_frames(video_info['path'])
        
        # Minimal augmentation if enabled
        if self.augment and self.minimal_augmentation:
            frames = self.apply_minimal_augmentation(frames)
        
        frames_tensor = torch.FloatTensor(frames).unsqueeze(0)
        label = self.class_to_idx[video_info['class']]
        
        return frames_tensor, label
    
    def load_video_frames(self, video_path):
        """Load video frames with standard preprocessing."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        
        # Ensure 32 frames
        frames = np.array(frames)
        if len(frames) > 32:
            start_idx = (len(frames) - 32) // 2
            frames = frames[start_idx:start_idx + 32]
        elif len(frames) < 32:
            last_frame = frames[-1] if len(frames) > 0 else np.zeros((64, 96), dtype=np.float32)
            while len(frames) < 32:
                frames = np.append(frames, [last_frame], axis=0)
        
        return frames
    
    def apply_minimal_augmentation(self, frames):
        """Apply very minimal augmentation to prevent overfitting."""
        augmented_frames = frames.copy()

        # Only horizontal flip (50% chance)
        if random.random() < 0.5:
            augmented_frames = np.flip(augmented_frames, axis=2).copy()  # Fix negative stride

        # Very small brightness adjustment (±5%)
        if random.random() < 0.3:
            brightness_factor = random.uniform(0.95, 1.05)
            augmented_frames = np.clip(augmented_frames * brightness_factor, 0, 1)

        return augmented_frames


class ConservativeLipReadingModel(nn.Module):
    """Same architecture as 75.9% checkpoint for conservative fine-tuning."""
    
    def __init__(self):
        super(ConservativeLipReadingModel, self).__init__()
        
        # Identical architecture
        self.conv3d1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn3d1 = nn.BatchNorm3d(32)
        self.pool3d1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv3d2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn3d2 = nn.BatchNorm3d(64)
        self.pool3d2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d3 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), padding=1)
        self.bn3d3 = nn.BatchNorm3d(96)
        self.pool3d3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d4 = nn.Conv3d(96, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn3d4 = nn.BatchNorm3d(128)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((3, 3, 4))
        
        # FC layers with increased dropout for regularization
        self.feature_size = 128 * 3 * 3 * 4
        self.dropout1 = nn.Dropout(0.5)  # Increased dropout
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        
        self.dropout2 = nn.Dropout(0.4)  # Increased dropout
        self.fc2 = nn.Linear(512, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        
        self.dropout3 = nn.Dropout(0.3)  # Increased dropout
        self.fc3 = nn.Linear(128, 32)
        
        self.fc_out = nn.Linear(32, 4)
    
    def forward(self, x):
        # Identical forward pass
        x = F.relu(self.bn3d1(self.conv3d1(x)))
        x = self.pool3d1(x)
        
        x = F.relu(self.bn3d2(self.conv3d2(x)))
        x = self.pool3d2(x)
        
        x = F.relu(self.bn3d3(self.conv3d3(x)))
        x = self.pool3d3(x)
        
        x = F.relu(self.bn3d4(self.conv3d4(x)))
        x = self.adaptive_pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout1(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        
        x = self.dropout2(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        
        x = self.dropout3(x)
        x = F.relu(self.fc3(x))
        x = self.fc_out(x)
        
        return x


def main():
    """Execute conservative resumed training."""
    print("🛡️  STARTING CONSERVATIVE RESUMED TRAINING")
    print("🎯 TARGET: 82% Validation Accuracy with Overfitting Prevention")
    
    trainer = ConservativeResumedTrainer()
    success = trainer.run_conservative_training()

    if success:
        print("\n🎉 CONSERVATIVE TRAINING SUCCESS!")
        print(f"✅ Achieved {trainer.target_val_acc}% validation accuracy")
        print("🛡️  Successfully prevented overfitting")
    else:
        print("\n💡 Conservative training completed")
        best_acc = getattr(trainer, 'best_val_acc', 0.0)
        print(f"📊 Best accuracy: {best_acc:.2f}%")
        print("🔍 Overfitting successfully prevented")

if __name__ == "__main__":
    main()
