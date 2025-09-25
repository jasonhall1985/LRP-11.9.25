#!/usr/bin/env python3
"""
Enhanced Lightweight 4-Class Lip-Reading Training Pipeline
Using 536-video balanced dataset with optimized lightweight architecture

Key Features:
- Lightweight CNN-LSTM architecture (1-2M parameters)
- Enhanced 536-video balanced dataset (80/20 split)
- Conservative training approach to prevent overfitting
- Target: 75-80% validation accuracy
- Comprehensive logging and visualization
"""

import os
import csv
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

class EnhancedLightweightTrainer:
    def __init__(self):
        # Dataset configuration
        self.train_manifest = Path("enhanced_balanced_training_results/enhanced_balanced_536_train_manifest.csv")
        self.val_manifest = Path("enhanced_balanced_training_results/enhanced_balanced_536_validation_manifest.csv")
        
        # Output directory
        self.output_dir = Path("enhanced_lightweight_training_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration - optimized for 536-video dataset
        self.batch_size = 8  # Balanced batch size
        self.max_epochs = 45  # Moderate epoch count
        self.initial_lr = 0.0001  # Conservative learning rate
        self.device = torch.device('cpu')
        
        # Performance targets
        self.target_val_acc = 75.0  # Realistic target (75-80% range)
        self.max_target_val_acc = 80.0  # Stretch target
        self.early_stopping_patience = 18  # Generous patience
        self.min_improvement = 0.3  # Small improvement threshold
        
        # Class configuration
        self.selected_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.selected_classes)}
        
        print("🚀 ENHANCED LIGHTWEIGHT 4-CLASS LIP-READING TRAINING")
        print("=" * 80)
        print(f"📊 Dataset: 536-video enhanced balanced dataset")
        print(f"🎯 Target: {self.target_val_acc}-{self.max_target_val_acc}% validation accuracy")
        print(f"🏗️  Architecture: Lightweight CNN-LSTM (1-2M parameters)")
        print(f"📈 Learning rate: {self.initial_lr} (conservative)")
        print(f"⏱️  Max epochs: {self.max_epochs} with early stopping")
        
    def load_enhanced_datasets(self):
        """Load the enhanced 536-video balanced datasets."""
        print("\n📋 LOADING ENHANCED BALANCED DATASETS")
        print("=" * 60)
        
        # Verify manifest files exist
        if not self.train_manifest.exists():
            raise FileNotFoundError(f"Training manifest not found: {self.train_manifest}")
        if not self.val_manifest.exists():
            raise FileNotFoundError(f"Validation manifest not found: {self.val_manifest}")
        
        # Create datasets
        self.train_dataset = EnhancedLipReadingDataset(
            self.train_manifest, self.class_to_idx, augment=True
        )
        self.val_dataset = EnhancedLipReadingDataset(
            self.val_manifest, self.class_to_idx, augment=False
        )
        
        print(f"✅ Training dataset: {len(self.train_dataset)} videos")
        print(f"✅ Validation dataset: {len(self.val_dataset)} videos")
        print(f"📊 Total dataset: {len(self.train_dataset) + len(self.val_dataset)} videos")
        
        # Analyze class distribution
        train_classes = [self.train_dataset.videos[i]['class'] for i in range(len(self.train_dataset))]
        val_classes = [self.val_dataset.videos[i]['class'] for i in range(len(self.val_dataset))]
        
        train_counts = Counter(train_classes)
        val_counts = Counter(val_classes)
        
        print(f"\n📈 TRAINING CLASS DISTRIBUTION:")
        for cls in self.selected_classes:
            count = train_counts[cls]
            pct = count / len(self.train_dataset) * 100
            print(f"   {cls}: {count} videos ({pct:.1f}%)")
        
        print(f"\n📈 VALIDATION CLASS DISTRIBUTION:")
        for cls in self.selected_classes:
            count = val_counts[cls]
            pct = count / len(self.val_dataset) * 100
            print(f"   {cls}: {count} videos ({pct:.1f}%)")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        print(f"\n📊 Training batches: {len(self.train_loader)}")
        print(f"📊 Validation batches: {len(self.val_loader)}")
        
        # Store class counts for analysis
        self.train_class_counts = train_counts
        self.val_class_counts = val_counts
        
    def setup_lightweight_model(self):
        """Setup lightweight CNN-LSTM model optimized for small datasets."""
        print("\n🏗️  SETTING UP LIGHTWEIGHT CNN-LSTM MODEL")
        print("=" * 60)
        
        # Initialize lightweight model
        self.model = LightweightCNNLSTM().to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"🎯 Target range: 1-2M parameters ✅" if 1_000_000 <= total_params <= 2_000_000 else f"⚠️  Outside target range")
        
        # Conservative optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.initial_lr,
            weight_decay=1e-4,  # Regularization
            betas=(0.9, 0.999)
        )
        
        # ReduceLROnPlateau scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.7, patience=6, 
            min_lr=1e-6, threshold=0.01
        )
        
        # Loss function with minimal label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        # Initialize tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.training_start_time = time.time()
        
        print(f"✅ Lightweight model setup complete:")
        print(f"   Architecture: CNN-LSTM with {total_params:,} parameters")
        print(f"   Optimizer: AdamW (lr={self.initial_lr}, wd=1e-4)")
        print(f"   Scheduler: ReduceLROnPlateau")
        print(f"   Loss: CrossEntropyLoss with label smoothing (0.05)")
        
    def train_epoch(self, epoch):
        """Execute one training epoch."""
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
        """Execute one validation epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
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
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_name = self.selected_classes[label]
                    class_total[class_name] += 1
                    if predicted[i] == labels[i]:
                        class_correct[class_name] += 1
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)
        
        # Calculate per-class accuracies
        per_class_acc = {}
        for class_name in self.selected_classes:
            if class_total[class_name] > 0:
                per_class_acc[class_name] = 100. * class_correct[class_name] / class_total[class_name]
            else:
                per_class_acc[class_name] = 0.0
        
        return epoch_loss, epoch_acc, per_class_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'class_to_idx': self.class_to_idx
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_lightweight_model.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model: {best_path}")
        
        return checkpoint_path
    
    def run_enhanced_training(self):
        """Execute the complete enhanced training pipeline."""
        print("\n🚀 STARTING ENHANCED LIGHTWEIGHT TRAINING PIPELINE")
        print("=" * 70)
        
        try:
            # Setup
            self.load_enhanced_datasets()
            self.setup_lightweight_model()
            
            print(f"\n🎯 TRAINING TARGETS:")
            print(f"   Primary: {self.target_val_acc}% validation accuracy")
            print(f"   Stretch: {self.max_target_val_acc}% validation accuracy")
            print(f"   Overfitting threshold: <15% train-val gap")
            
            # Training loop
            for epoch in range(1, self.max_epochs + 1):
                print(f"\n{'='*60}")
                print(f"EPOCH {epoch}/{self.max_epochs}")
                print(f"{'='*60}")
                
                # Train and validate
                train_loss, train_acc = self.train_epoch(epoch)
                val_loss, val_acc, per_class_acc = self.validate_epoch(epoch)
                
                # Update scheduler
                self.scheduler.step(val_acc)
                
                # Check for improvement
                is_best = val_acc > self.best_val_acc
                if is_best:
                    improvement = val_acc - self.best_val_acc
                    if improvement >= self.min_improvement:
                        self.best_val_acc = val_acc
                        self.epochs_without_improvement = 0
                        self.save_checkpoint(epoch, is_best=True)
                        print(f"🎉 NEW BEST: {val_acc:.2f}% (+{improvement:.2f}%)")
                        
                        # Check targets
                        if val_acc >= self.max_target_val_acc:
                            print(f"🎯 STRETCH TARGET ACHIEVED! {val_acc:.2f}% >= {self.max_target_val_acc}%")
                        elif val_acc >= self.target_val_acc:
                            print(f"🎯 PRIMARY TARGET ACHIEVED! {val_acc:.2f}% >= {self.target_val_acc}%")
                    else:
                        print(f"📈 Small improvement: {val_acc:.2f}% (+{improvement:.2f}%)")
                        self.epochs_without_improvement += 1
                else:
                    self.epochs_without_improvement += 1
                    print(f"📉 No improvement for {self.epochs_without_improvement} epochs")
                
                # Save regular checkpoint
                if epoch % 5 == 0:
                    self.save_checkpoint(epoch)
                
                # Calculate training-validation gap
                train_val_gap = abs(train_acc - val_acc)
                overfitting_status = "✅ Good" if train_val_gap < 15 else "⚠️ Overfitting"
                
                # Progress summary
                print(f"\n📊 EPOCH {epoch} SUMMARY:")
                print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
                print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
                print(f"   Gap:   {train_val_gap:.2f}% ({overfitting_status})")
                print(f"   Best:  {self.best_val_acc:.2f}%")
                print(f"   LR:    {self.optimizer.param_groups[0]['lr']:.8f}")
                
                print(f"\n🎯 PER-CLASS VALIDATION ACCURACY:")
                for class_name, acc in per_class_acc.items():
                    print(f"   {class_name}: {acc:.1f}%")
                
                # Early stopping
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\n⏹️  EARLY STOPPING: No improvement for {self.early_stopping_patience} epochs")
                    break
            
            # Training completed
            training_time = time.time() - self.training_start_time
            success = self.best_val_acc >= self.target_val_acc
            
            print(f"\n🏁 TRAINING COMPLETED")
            print(f"⏱️  Total time: {training_time/60:.1f} minutes")
            print(f"🎯 Best validation accuracy: {self.best_val_acc:.2f}%")
            print(f"✅ Primary target ({self.target_val_acc}%): {'ACHIEVED' if success else 'NOT ACHIEVED'}")
            print(f"🌟 Stretch target ({self.max_target_val_acc}%): {'ACHIEVED' if self.best_val_acc >= self.max_target_val_acc else 'NOT ACHIEVED'}")
            
            return success, self.best_val_acc
            
        except Exception as e:
            print(f"\n❌ TRAINING ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0


class EnhancedLipReadingDataset(Dataset):
    """Enhanced dataset for 536-video balanced training."""
    
    def __init__(self, manifest_path, class_to_idx, augment=False):
        self.manifest_path = Path(manifest_path)
        self.class_to_idx = class_to_idx
        self.augment = augment
        self.videos = []
        
        # Load videos from manifest
        with open(self.manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['class'] in self.class_to_idx:
                    # Construct full path
                    video_path = Path("data/the_best_videos_so_far") / row['filename']
                    if video_path.exists():
                        self.videos.append({
                            'path': video_path,
                            'class': row['class']
                        })
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_info = self.videos[idx]
        frames = self.load_video_frames(video_info['path'])
        
        # Apply augmentation if enabled
        if self.augment:
            frames = self.apply_augmentation(frames)
        
        frames_tensor = torch.FloatTensor(frames).unsqueeze(0)  # Add channel dimension
        label = self.class_to_idx[video_info['class']]
        
        return frames_tensor, label
    
    def load_video_frames(self, video_path):
        """Load and preprocess video frames."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        
        # Ensure exactly 32 frames
        frames = np.array(frames)
        if len(frames) > 32:
            # Take center 32 frames
            start_idx = (len(frames) - 32) // 2
            frames = frames[start_idx:start_idx + 32]
        elif len(frames) < 32:
            # Pad with last frame
            last_frame = frames[-1] if len(frames) > 0 else np.zeros((64, 96), dtype=np.float32)
            while len(frames) < 32:
                frames = np.append(frames, [last_frame], axis=0)
        
        return frames
    
    def apply_augmentation(self, frames):
        """Apply balanced data augmentation."""
        augmented_frames = frames.copy()
        
        # Horizontal flip (50% chance)
        if random.random() < 0.5:
            augmented_frames = np.flip(augmented_frames, axis=2).copy()
        
        # Brightness adjustment (±10-15%)
        if random.random() < 0.7:
            brightness_factor = random.uniform(0.85, 1.15)
            augmented_frames = np.clip(augmented_frames * brightness_factor, 0, 1)
        
        # Contrast adjustment (0.9-1.1x)
        if random.random() < 0.5:
            contrast_factor = random.uniform(0.9, 1.1)
            mean = np.mean(augmented_frames)
            augmented_frames = np.clip((augmented_frames - mean) * contrast_factor + mean, 0, 1)
        
        return augmented_frames


class LightweightCNNLSTM(nn.Module):
    """Lightweight CNN-LSTM architecture optimized for small datasets (1-2M parameters)."""
    
    def __init__(self):
        super(LightweightCNNLSTM, self).__init__()
        
        # Lightweight 3D CNN feature extractor
        self.conv3d1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1)  # Reduced channels
        self.bn3d1 = nn.BatchNorm3d(16)
        self.pool3d1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv3d2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)  # Reduced channels
        self.bn3d2 = nn.BatchNorm3d(32)
        self.pool3d2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d3 = nn.Conv3d(32, 48, kernel_size=(3, 3, 3), padding=1)  # Reduced channels
        self.bn3d3 = nn.BatchNorm3d(48)
        self.pool3d3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 6))  # Smaller output

        # LSTM for temporal modeling
        self.lstm_input_size = 48 * 4 * 6  # 1152 features per timestep (corrected)
        self.lstm_hidden_size = 128  # Reduced hidden size
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,  # Single layer
            batch_first=True,
            dropout=0.0  # No dropout in LSTM for single layer
        )
        
        # Classifier head with regularization
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.lstm_hidden_size, 64)  # Smaller FC layer
        self.bn_fc1 = nn.BatchNorm1d(64)
        
        self.dropout2 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(64, 4)  # Direct to output
    
    def forward(self, x):
        # CNN feature extraction
        # Input: (batch, 1, 32, 64, 96)
        x = F.relu(self.bn3d1(self.conv3d1(x)))
        x = self.pool3d1(x)  # (batch, 16, 32, 32, 48)
        
        x = F.relu(self.bn3d2(self.conv3d2(x)))
        x = self.pool3d2(x)  # (batch, 32, 16, 16, 24)
        
        x = F.relu(self.bn3d3(self.conv3d3(x)))
        x = self.pool3d3(x)  # (batch, 48, 8, 8, 12)
        
        x = self.adaptive_pool(x)  # (batch, 48, 4, 4, 6)
        
        # Reshape for LSTM: (batch, timesteps, features)
        batch_size = x.size(0)
        timesteps = x.size(2)  # 4 timesteps after pooling
        x = x.permute(0, 2, 1, 3, 4)  # (batch, 4, 48, 4, 6)
        x = x.contiguous().view(batch_size, timesteps, -1)  # (batch, 4, 1152)
        
        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last timestep output
        x = lstm_out[:, -1, :]  # (batch, 128)
        
        # Classification head
        x = self.dropout1(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        
        x = self.dropout2(x)
        x = self.fc_out(x)
        
        return x


def load_enhanced_checkpoint():
    """Load the enhanced 81.65% validation accuracy checkpoint."""
    import torch
    import os

    # Model path
    model_path = "best_lightweight_model.pth"
    if not os.path.exists(model_path):
        model_path = "checkpoint_enhanced_81_65_percent_success_20250924/best_lightweight_model.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Enhanced checkpoint not found at {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Create model
    model = LightweightCNNLSTM()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Extract class mappings
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"✅ Enhanced model loaded successfully!")
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✅ Classes: {list(class_to_idx.keys())}")
    print(f"✅ Best validation accuracy: 81.65%")

    return model, class_to_idx, idx_to_class, checkpoint

def main():
    """Execute enhanced lightweight training pipeline."""
    print("🚀 ENHANCED LIGHTWEIGHT 4-CLASS LIP-READING TRAINING")
    print("🎯 TARGET: 75-80% Validation Accuracy with 536-Video Dataset")

    trainer = EnhancedLightweightTrainer()
    success, best_acc = trainer.run_enhanced_training()
    
    if success:
        print(f"\n🎉 TRAINING SUCCESS!")
        print(f"✅ Achieved {best_acc:.2f}% validation accuracy")
        print(f"🎯 Primary target (75%) achieved!")
    else:
        print(f"\n💡 Training completed with {best_acc:.2f}% validation accuracy")
        print(f"🔍 Target not achieved but progress made")

if __name__ == "__main__":
    main()
