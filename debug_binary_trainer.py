#!/usr/bin/env python3
"""
Debug Binary Trainer - Simplified version to identify core issues
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
from collections import defaultdict

class SimplifiedBinaryTrainer:
    def __init__(self):
        self.manifests_dir = Path("data/classifier training 20.9.25/binary_classification")
        self.output_dir = Path("debug_binary_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simplified configuration for debugging
        self.batch_size = 4  # Smaller batch size
        self.max_epochs = 50  # More epochs
        self.learning_rate = 0.01  # Higher learning rate
        self.device = torch.device('cpu')
        
        self.class_to_idx = {'doctor': 0, 'help': 1}
        
        print("🔧 DEBUG BINARY TRAINER")
        print("=" * 50)
        print("🎯 Goal: Identify why cross-demographic training failed")
        print("🔍 Simplified approach: Higher LR, smaller batches, more epochs")
        
    def load_datasets(self):
        """Load datasets with debugging info."""
        print("\n📋 LOADING DATASETS FOR DEBUGGING")
        
        train_manifest = self.manifests_dir / "binary_train_manifest.csv"
        val_manifest = self.manifests_dir / "binary_validation_manifest.csv"
        
        self.train_dataset = SimpleLipReadingDataset(train_manifest, self.class_to_idx)
        self.val_dataset = SimpleLipReadingDataset(val_manifest, self.class_to_idx)
        
        print(f"📊 Training: {len(self.train_dataset)} videos")
        print(f"📊 Validation: {len(self.val_dataset)} videos")
        
        # Check data loading
        print("\n🔍 TESTING DATA LOADING:")
        train_sample = self.train_dataset[0]
        val_sample = self.val_dataset[0]
        
        print(f"   Train sample shape: {train_sample[0].shape}, label: {train_sample[1]}")
        print(f"   Val sample shape: {val_sample[0].shape}, label: {val_sample[1]}")
        
        # Check for data issues
        print(f"   Train tensor range: [{train_sample[0].min():.3f}, {train_sample[0].max():.3f}]")
        print(f"   Val tensor range: [{val_sample[0].min():.3f}, {val_sample[0].max():.3f}]")
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        
    def setup_model(self):
        """Setup simplified model for debugging."""
        print("\n🏗️  SETTING UP SIMPLIFIED MODEL")
        
        # Much simpler model for debugging
        self.model = SimpleBinaryModel().to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Model parameters: {total_params:,}")
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        
        # Track metrics
        self.train_accs = []
        self.val_accs = []
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        """Train one epoch with detailed logging."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (videos, labels) in enumerate(self.train_loader):
            videos, labels = videos.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Log every batch for debugging
            acc = 100.0 * correct / total
            print(f"     Batch {batch_idx+1}: Loss={loss.item():.4f}, Acc={acc:.1f}%")
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for videos, labels in self.val_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc
    
    def debug_train(self):
        """Debug training loop."""
        print("\n🎯 STARTING DEBUG TRAINING")
        print("=" * 50)
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 15
        
        for epoch in range(1, self.max_epochs + 1):
            print(f"\n📅 Epoch {epoch:2d}/{self.max_epochs}")
            print("-" * 30)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate_epoch()
            
            # Track metrics
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Check improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                print(f"   🎉 NEW BEST: {val_acc:.1f}%")
            else:
                patience_counter += 1
            
            print(f"   📊 Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Best: {best_val_acc:.1f}%")
            
            # Early success check
            if train_acc >= 80.0 and val_acc >= 60.0:
                print(f"\n✅ REASONABLE PERFORMANCE ACHIEVED!")
                print(f"   Train: {train_acc:.1f}% ≥ 80%")
                print(f"   Val: {val_acc:.1f}% ≥ 60%")
                break
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping (patience: {patience})")
                break
        
        # Final analysis
        print(f"\n🔍 DEBUG TRAINING COMPLETED")
        print("=" * 50)
        print(f"📊 Final Results:")
        print(f"   Best Training: {max(self.train_accs):.1f}%")
        print(f"   Best Validation: {best_val_acc:.1f}%")
        print(f"   Total Epochs: {len(self.train_accs)}")
        
        # Diagnosis
        if max(self.train_accs) < 70:
            print("❌ DIAGNOSIS: Model cannot learn training data - architecture issue")
        elif best_val_acc < 55:
            print("❌ DIAGNOSIS: Poor cross-demographic generalization - data issue")
        else:
            print("✅ DIAGNOSIS: Model shows learning capability")
        
        return best_val_acc >= 60.0

class SimpleLipReadingDataset(Dataset):
    """Simplified dataset for debugging."""
    
    def __init__(self, manifest_path, class_to_idx):
        self.class_to_idx = class_to_idx
        self.videos = []
        
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['class'] in class_to_idx:
                    self.videos.append({
                        'path': row['video_path'],
                        'class': row['class'],
                        'class_idx': class_to_idx[row['class']]
                    })
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_info = self.videos[idx]
        frames = self._load_video_simple(video_info['path'])
        
        # Simple preprocessing
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        frames_tensor = frames_tensor.unsqueeze(0)  # Add channel
        
        return frames_tensor, video_info['class_idx']
    
    def _load_video_simple(self, video_path):
        """Simplified video loading."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Load exactly 16 frames (simpler)
        frame_count = 0
        while frame_count < 16:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Downsample for debugging
            small_frame = cv2.resize(gray_frame, (48, 32))
            frames.append(small_frame)
            frame_count += 1
        
        cap.release()
        
        # Pad if needed
        while len(frames) < 16:
            frames.append(frames[-1] if frames else np.zeros((32, 48)))
        
        return np.array(frames[:16])  # Shape: (16, 32, 48)

class SimpleBinaryModel(nn.Module):
    """Much simpler model for debugging."""
    
    def __init__(self):
        super(SimpleBinaryModel, self).__init__()
        
        # Simple 3D CNN
        self.conv3d1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1)
        self.pool3d1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
        self.pool3d2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Feature size: 32 * 4 * 8 * 12 = 12,288
        self.feature_size = 32 * 4 * 8 * 12
        
        # Simple classifier
        self.fc1 = nn.Linear(self.feature_size, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)
        
        print(f"🏗️  Simple Binary Model: {self.feature_size} features → 64 → 2 classes")
    
    def forward(self, x):
        # 3D CNN
        x = F.relu(self.conv3d1(x))
        x = self.pool3d1(x)
        
        x = F.relu(self.conv3d2(x))
        x = self.pool3d2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def main():
    """Run debug training."""
    print("🔧 STARTING DEBUG BINARY TRAINING")
    print("💡 Simplified approach to identify core issues")
    
    trainer = SimplifiedBinaryTrainer()
    trainer.load_datasets()
    trainer.setup_model()
    success = trainer.debug_train()
    
    if success:
        print("\n✅ DEBUG SUCCESSFUL - Model can learn!")
        print("💡 Issue likely in original complex architecture")
    else:
        print("\n❌ DEBUG FAILED - Fundamental data/task issue")
        print("💡 Cross-demographic task may be too challenging")

if __name__ == "__main__":
    main()
