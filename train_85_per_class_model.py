#!/usr/bin/env python3
"""
Comprehensive 4-Class Lip-Reading Model Training Pipeline
Target: ≥82% cross-demographic validation accuracy with 85 videos per class
Perfect balance: 340 total videos (272 train + 68 validation)
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import random
from collections import defaultdict
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class LipReadingDataset(Dataset):
    """Dataset class for lip-reading videos with data augmentation"""
    
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
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        print(f"Dataset initialized: {len(self.manifest)} videos")
        print(f"Augmentation: {'ON' if augment else 'OFF'}")
        
        # Verify class distribution
        class_counts = self.manifest['class'].value_counts().sort_index()
        print("Class distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} videos")
    
    def __len__(self):
        return len(self.manifest)
    
    def load_video_frames(self, video_path):
        """Load and preprocess video frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize to target dimensions (96x64) - this ensures consistency
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
        """Apply data augmentation: brightness ±10-15%, contrast 0.9-1.1x, horizontal flipping"""
        if not self.augment:
            return frames

        # Random brightness adjustment (±10-15%)
        brightness_factor = np.random.uniform(0.85, 1.15)
        frames = np.clip(frames * brightness_factor, 0, 1)

        # Random contrast adjustment (0.9-1.1x)
        contrast_factor = np.random.uniform(0.9, 1.1)
        frames = np.clip((frames - 0.5) * contrast_factor + 0.5, 0, 1)

        # Random horizontal flipping (50% chance)
        if np.random.random() > 0.5:
            frames = np.flip(frames, axis=2).copy()  # Flip and copy to fix negative stride

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

class CNN_LSTM_LipReader(nn.Module):
    """CNN-LSTM architecture for lip-reading"""
    
    def __init__(self, num_classes=4, hidden_size=256, num_layers=2, dropout=0.3):
        super(CNN_LSTM_LipReader, self).__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 6))  # Fixed output size
        )
        
        # Calculate CNN output size
        self.cnn_output_size = 256 * 4 * 6  # 6144
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
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
            features = self.cnn(frame)  # (batch, 256, 4, 6)
            features = features.view(batch_size, -1)  # (batch, 6144)
            cnn_features.append(features)
        
        # Stack temporal features
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch, seq_len, 6144)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_features)  # (batch, seq_len, hidden_size*2)
        
        # Use last output for classification
        final_features = lstm_out[:, -1, :]  # (batch, hidden_size*2)
        
        # Classification
        output = self.classifier(final_features)  # (batch, num_classes)
        
        return output

def create_data_loaders(train_manifest, val_manifest, data_dir, batch_size=8):
    """Create training and validation data loaders"""
    print("🔄 Creating data loaders...")
    
    # Create datasets
    train_dataset = LipReadingDataset(train_manifest, data_dir, augment=True)
    val_dataset = LipReadingDataset(val_manifest, data_dir, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing to avoid issues
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing to avoid issues
        pin_memory=False
    )
    
    print(f"✅ Data loaders created:")
    print(f"  Training: {len(train_dataset)} videos, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} videos, {len(val_loader)} batches")
    
    return train_loader, val_loader, train_dataset.class_to_idx

def train_model(model, train_loader, val_loader, class_to_idx, num_epochs=100, target_accuracy=0.82):
    """Train the model with early stopping and learning rate scheduling"""
    print(f"🚀 Starting training (target: {target_accuracy*100:.1f}% validation accuracy)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
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
    patience = 20
    
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
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Check if target reached
            if val_acc >= target_accuracy:
                print(f"🎯 TARGET ACHIEVED! Validation accuracy: {val_acc:.4f} (≥{target_accuracy:.4f})")
                break
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"✅ Training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    
    return model, history, best_val_acc

def main():
    """Execute comprehensive model training pipeline"""
    print("🎯 COMPREHENSIVE 4-CLASS LIP-READING MODEL TRAINING")
    print("=" * 70)
    print("Objective: ≥82% cross-demographic validation accuracy")
    print("Dataset: 340 videos (85 per class, perfectly balanced)")
    
    # Paths
    train_manifest = "balanced_85_training_results/balanced_340_train_manifest.csv"
    val_manifest = "balanced_85_training_results/balanced_340_validation_manifest.csv"
    data_dir = "data/the_best_videos_so_far"
    
    # Create data loaders
    train_loader, val_loader, class_to_idx = create_data_loaders(
        train_manifest, val_manifest, data_dir, batch_size=8
    )
    
    # Create model
    print("\n🏗️  Creating CNN-LSTM model...")
    model = CNN_LSTM_LipReader(num_classes=4, hidden_size=256, num_layers=2, dropout=0.3)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Train model
    print("\n🚀 Starting training...")
    trained_model, history, best_val_acc = train_model(
        model, train_loader, val_loader, class_to_idx, 
        num_epochs=100, target_accuracy=0.82
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"balanced_85_training_results"
    
    # Save model
    model_path = os.path.join(output_dir, f"balanced_85_model_{timestamp}.pth")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'class_to_idx': class_to_idx,
        'best_val_acc': best_val_acc,
        'history': history,
        'model_config': {
            'num_classes': 4,
            'hidden_size': 256,
            'num_layers': 2,
            'dropout': 0.3
        }
    }, model_path)
    
    print(f"\n💾 Model saved: {model_path}")
    print(f"🏆 Final validation accuracy: {best_val_acc:.4f}")
    
    # Check success
    if best_val_acc >= 0.82:
        print("🎉 SUCCESS: Target ≥82% validation accuracy achieved!")
        print("✅ Phase 3 Complete: Model Training Pipeline")
        return True, model_path, best_val_acc
    else:
        print(f"⚠️  Target not reached: {best_val_acc:.4f} < 0.82")
        print("📊 Model trained but requires further optimization")
        return False, model_path, best_val_acc

if __name__ == "__main__":
    success, model_path, accuracy = main()
    exit(0 if success else 1)
