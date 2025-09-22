#!/usr/bin/env python3
"""
Lightweight 4-Class Lip-Reading Model Training Pipeline
Optimized for 340-video dataset with ~2-3M parameters targeting ≥82% validation accuracy
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
from datetime import datetime
import random
from collections import defaultdict
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class LightweightLipDataset(Dataset):
    """Lightweight dataset with effective augmentation for small datasets"""
    
    def __init__(self, manifest_path, data_dir, augment=False):
        self.manifest = pd.read_csv(manifest_path)
        self.data_dir = data_dir
        self.augment = augment
        
        # Class to index mapping
        self.class_to_idx = {
            'doctor': 0,
            'i_need_to_move': 1,
            'my_mouth_is_dry': 2,
            'pillow': 3
        }
        
        print(f"Lightweight Dataset: {len(self.manifest)} videos, Augmentation: {'ON' if augment else 'OFF'}")
        
        # Verify perfect balance
        class_counts = self.manifest['class'].value_counts().sort_index()
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} videos")
    
    def __len__(self):
        return len(self.manifest)
    
    def load_video_frames(self, video_path):
        """Load video with consistent preprocessing"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale and resize
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (96, 64))
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        
        # Ensure exactly 32 frames
        if len(frames) >= 32:
            start_idx = (len(frames) - 32) // 2
            frames = frames[start_idx:start_idx + 32]
        else:
            # Pad with repeated frames
            while len(frames) < 32:
                frames.extend(frames[:min(len(frames), 32 - len(frames))])
            frames = frames[:32]
        
        return np.array(frames)
    
    def apply_augmentation(self, frames):
        """Effective augmentation for small datasets"""
        if not self.augment:
            return frames
        
        # Horizontal flipping (50% chance)
        if np.random.random() > 0.5:
            frames = np.flip(frames, axis=2).copy()
        
        # Brightness adjustment (±12%)
        if np.random.random() > 0.3:
            brightness_factor = np.random.uniform(0.88, 1.12)
            frames = np.clip(frames * brightness_factor, 0, 1)
        
        # Slight scale variations (±5%)
        if np.random.random() > 0.6:
            scale_factor = np.random.uniform(0.95, 1.05)
            h, w = frames.shape[1], frames.shape[2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            scaled_frames = []
            for frame in frames:
                scaled = cv2.resize(frame, (new_w, new_h))
                # Crop or pad back to original size
                if scale_factor > 1:
                    # Crop center
                    start_y = (new_h - h) // 2
                    start_x = (new_w - w) // 2
                    scaled = scaled[start_y:start_y+h, start_x:start_x+w]
                else:
                    # Pad to center
                    pad_y = (h - new_h) // 2
                    pad_x = (w - new_w) // 2
                    scaled = np.pad(scaled, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x)), mode='edge')
                scaled_frames.append(scaled)
            frames = np.array(scaled_frames)
        
        # Minor temporal variations (shift by ±1 frame, 30% chance)
        if np.random.random() > 0.7 and len(frames) > 2:
            shift = np.random.choice([-1, 1])
            frames = np.roll(frames, shift, axis=0)
        
        return frames
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        video_path = os.path.join(self.data_dir, row['filename'])
        
        # Load video frames
        frames = self.load_video_frames(video_path)
        
        # Apply augmentation
        frames = self.apply_augmentation(frames)
        
        # Convert to tensor: (T, H, W) -> (1, T, H, W)
        frames_tensor = torch.FloatTensor(frames).unsqueeze(0)
        
        # Get class label
        class_label = self.class_to_idx[row['class']]
        label_tensor = torch.LongTensor([class_label])
        
        return frames_tensor, label_tensor

class LightweightCNN_LSTM(nn.Module):
    """Lightweight CNN-LSTM with ~2-3M parameters optimized for small datasets"""
    
    def __init__(self, num_classes=4, hidden_size=128, num_layers=2, dropout=0.3):
        super(LightweightCNN_LSTM, self).__init__()
        
        # Lightweight CNN feature extractor
        self.cnn = nn.Sequential(
            # First conv block - reduced channels
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 96x64 -> 48x32
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 48x32 -> 24x16
            
            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 24x16 -> 12x8
            
            # Fourth conv block with adaptive pooling
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4))  # Fixed output: 3x4
        )
        
        # CNN output size: 128 * 3 * 4 = 1536
        self.cnn_output_size = 128 * 3 * 4
        
        # Lightweight LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Lightweight classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        batch_size, channels, seq_len, height, width = x.size()
        
        # Process each frame through CNN
        cnn_features = []
        for t in range(seq_len):
            frame = x[:, :, t, :, :]  # (batch, channels, height, width)
            features = self.cnn(frame)  # (batch, 128, 3, 4)
            features = features.view(batch_size, -1)  # (batch, 1536)
            cnn_features.append(features)
        
        # Stack temporal features
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch, seq_len, 1536)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_features)  # (batch, seq_len, hidden_size*2)
        
        # Use last output for classification
        final_features = lstm_out[:, -1, :]  # (batch, hidden_size*2)
        
        # Classification
        output = self.classifier(final_features)  # (batch, num_classes)
        
        return output

def create_lightweight_loaders(train_manifest, val_manifest, data_dir, batch_size=6):
    """Create optimized data loaders for small dataset"""
    print("🔄 Creating lightweight data loaders...")
    
    train_dataset = LightweightLipDataset(train_manifest, data_dir, augment=True)
    val_dataset = LightweightLipDataset(val_manifest, data_dir, augment=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    print(f"✅ Lightweight loaders created:")
    print(f"  Training: {len(train_dataset)} videos, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} videos, {len(val_loader)} batches")
    
    return train_loader, val_loader, train_dataset.class_to_idx

def train_lightweight_model(model, train_loader, val_loader, class_to_idx, num_epochs=40, target_accuracy=0.82):
    """Optimized training for small datasets"""
    print(f"🚀 Lightweight Training (target: {target_accuracy*100:.1f}% validation accuracy)")
    print(f"Max epochs: {num_epochs}, Early stopping patience: 15")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Standard cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Step scheduler - reduce LR when validation plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 15
    
    print(f"\nStarting training...")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device).squeeze()
                outputs = model(videos)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Check for overfitting
        overfitting_gap = train_acc - val_acc
        overfitting_warning = " ⚠️ OVERFITTING" if overfitting_gap > 0.15 else ""
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Train: {train_acc:.4f} ({avg_train_loss:.4f}) | "
              f"Val: {val_acc:.4f} ({avg_val_loss:.4f}) | "
              f"LR: {current_lr:.6f}{overfitting_warning}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Check if target reached
            if val_acc >= target_accuracy:
                print(f"\n🎯 TARGET ACHIEVED! Validation accuracy: {val_acc:.4f} (≥{target_accuracy:.4f})")
                break
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print("=" * 80)
    print(f"✅ Lightweight training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    
    return model, history, best_val_acc

def main():
    """Execute lightweight model training pipeline"""
    print("🎯 LIGHTWEIGHT 4-CLASS LIP-READING MODEL TRAINING")
    print("=" * 70)
    print("Optimized for 340-video dataset targeting ≥82% cross-demographic validation")
    
    # Paths
    train_manifest = "balanced_85_training_results/balanced_340_train_manifest.csv"
    val_manifest = "balanced_85_training_results/balanced_340_validation_manifest.csv"
    data_dir = "data/the_best_videos_so_far"
    
    # Create lightweight data loaders
    train_loader, val_loader, class_to_idx = create_lightweight_loaders(
        train_manifest, val_manifest, data_dir, batch_size=6
    )
    
    # Create lightweight model
    print("\n🏗️  Creating Lightweight CNN-LSTM model...")
    model = LightweightCNN_LSTM(num_classes=4, hidden_size=128, num_layers=2, dropout=0.3)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Target range: 1-3M parameters ✅" if 1_000_000 <= total_params <= 3_000_000 else f"⚠️ Outside target range")
    
    # Train lightweight model
    print("\n🚀 Starting lightweight training...")
    trained_model, history, best_val_acc = train_lightweight_model(
        model, train_loader, val_loader, class_to_idx, 
        num_epochs=40, target_accuracy=0.82
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = "balanced_85_training_results"
    
    # Save lightweight model
    model_path = os.path.join(output_dir, f"lightweight_85_model_{timestamp}.pth")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'class_to_idx': class_to_idx,
        'best_val_acc': best_val_acc,
        'history': history,
        'model_config': {
            'num_classes': 4,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'architecture': 'LightweightCNN_LSTM',
            'total_params': total_params
        }
    }, model_path)
    
    print(f"\n💾 Lightweight model saved: {model_path}")
    print(f"🏆 Final validation accuracy: {best_val_acc:.4f}")
    
    # Check success
    if best_val_acc >= 0.82:
        print("🎉 SUCCESS: Target ≥82% validation accuracy achieved!")
        print("✅ Lightweight model successfully trained with cross-demographic generalization")
        return True, model_path, best_val_acc
    else:
        print(f"📊 Progress made: {best_val_acc:.4f} validation accuracy")
        print("💡 Consider: longer training, different augmentation, or architecture tweaks")
        return False, model_path, best_val_acc

if __name__ == "__main__":
    success, model_path, accuracy = main()
    exit(0 if success else 1)
