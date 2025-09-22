#!/usr/bin/env python3
"""
Enhanced 4-Class Lip-Reading Model Training Pipeline
Advanced architecture and training strategies for ≥82% cross-demographic validation accuracy
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

class EnhancedLipReadingDataset(Dataset):
    """Enhanced dataset with better preprocessing and augmentation"""
    
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
        
        print(f"Enhanced Dataset: {len(self.manifest)} videos, Augmentation: {'ON' if augment else 'OFF'}")
        
        # Verify class distribution
        class_counts = self.manifest['class'].value_counts().sort_index()
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} videos")
    
    def __len__(self):
        return len(self.manifest)
    
    def load_and_preprocess_video(self, video_path):
        """Enhanced video loading with better preprocessing"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize to consistent dimensions
            frame = cv2.resize(frame, (96, 64))
            
            # Apply Gaussian blur for noise reduction
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            
            # Histogram equalization for better contrast
            frame = cv2.equalizeHist(frame)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        
        # Ensure exactly 32 frames with better temporal sampling
        if len(frames) >= 32:
            # Use center frames for consistency
            start_idx = (len(frames) - 32) // 2
            frames = frames[start_idx:start_idx + 32]
        else:
            # Intelligent padding - repeat frames symmetrically
            while len(frames) < 32:
                if len(frames) == 1:
                    frames = frames * 32
                else:
                    frames.extend(frames[:min(len(frames), 32 - len(frames))])
            frames = frames[:32]
        
        return np.array(frames)
    
    def apply_enhanced_augmentation(self, frames):
        """Enhanced data augmentation with more sophisticated techniques"""
        if not self.augment:
            return frames
        
        # Random brightness (±12%)
        if np.random.random() > 0.3:
            brightness_factor = np.random.uniform(0.88, 1.12)
            frames = np.clip(frames * brightness_factor, 0, 1)
        
        # Random contrast (0.9-1.1x)
        if np.random.random() > 0.3:
            contrast_factor = np.random.uniform(0.9, 1.1)
            frames = np.clip((frames - 0.5) * contrast_factor + 0.5, 0, 1)
        
        # Random gamma correction
        if np.random.random() > 0.5:
            gamma = np.random.uniform(0.95, 1.05)
            frames = np.power(frames, gamma)
        
        # Random horizontal flipping (40% chance)
        if np.random.random() > 0.6:
            frames = np.flip(frames, axis=2).copy()
        
        # Random temporal shift (slight frame offset)
        if np.random.random() > 0.7 and len(frames) > 2:
            shift = np.random.randint(-1, 2)
            if shift != 0:
                frames = np.roll(frames, shift, axis=0)
        
        return frames
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        video_path = os.path.join(self.data_dir, row['filename'])
        
        # Load and preprocess video
        frames = self.load_and_preprocess_video(video_path)
        
        # Apply augmentation
        frames = self.apply_enhanced_augmentation(frames)
        
        # Convert to tensor: (T, H, W) -> (1, T, H, W)
        frames_tensor = torch.FloatTensor(frames).unsqueeze(0)
        
        # Get class label
        class_label = self.class_to_idx[row['class']]
        label_tensor = torch.LongTensor([class_label])
        
        return frames_tensor, label_tensor

class Enhanced3DCNN_LSTM(nn.Module):
    """Enhanced 3D CNN + LSTM architecture for better lip-reading performance"""
    
    def __init__(self, num_classes=4, hidden_size=512, num_layers=3, dropout=0.4):
        super(Enhanced3DCNN_LSTM, self).__init__()
        
        # 3D CNN for spatiotemporal feature extraction
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))  # Don't pool temporally yet
        )
        
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))  # Pool temporally now
        )
        
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )
        
        self.conv3d_4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((8, 4, 6))  # Fixed output size
        )
        
        # Calculate feature size after 3D CNN
        self.feature_size = 256 * 4 * 6  # 6144
        
        # LSTM for temporal sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        batch_size, channels, seq_len, height, width = x.size()
        
        # 3D CNN processing
        # Reshape for 3D conv: (batch, channels, seq_len, height, width)
        conv_out = self.conv3d_1(x)
        conv_out = self.conv3d_2(conv_out)
        conv_out = self.conv3d_3(conv_out)
        conv_out = self.conv3d_4(conv_out)
        
        # Reshape for LSTM: (batch, seq_len, features)
        batch_size, channels, seq_len, height, width = conv_out.size()
        conv_features = conv_out.permute(0, 2, 1, 3, 4).contiguous()
        conv_features = conv_features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(conv_features)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attended_features = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classification
        output = self.classifier(attended_features)
        
        return output

def create_enhanced_data_loaders(train_manifest, val_manifest, data_dir, batch_size=4):
    """Create enhanced data loaders with smaller batch size for stability"""
    print("🔄 Creating enhanced data loaders...")
    
    train_dataset = EnhancedLipReadingDataset(train_manifest, data_dir, augment=True)
    val_dataset = EnhancedLipReadingDataset(val_manifest, data_dir, augment=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    print(f"✅ Enhanced data loaders created:")
    print(f"  Training: {len(train_dataset)} videos, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} videos, {len(val_loader)} batches")
    
    return train_loader, val_loader, train_dataset.class_to_idx

def train_enhanced_model(model, train_loader, val_loader, class_to_idx, num_epochs=150, target_accuracy=0.82):
    """Enhanced training with advanced techniques"""
    print(f"🚀 Enhanced Training (target: {target_accuracy*100:.1f}% validation accuracy)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Enhanced loss function with label smoothing
    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing
        
        def forward(self, pred, target):
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
            one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
            log_prob = torch.log_softmax(pred, dim=1)
            return -(one_hot * log_prob).sum(dim=1).mean()
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # Enhanced optimizer with different learning rates for different parts
    optimizer = optim.AdamW([
        {'params': model.conv3d_1.parameters(), 'lr': 0.0005},
        {'params': model.conv3d_2.parameters(), 'lr': 0.0005},
        {'params': model.conv3d_3.parameters(), 'lr': 0.0005},
        {'params': model.conv3d_4.parameters(), 'lr': 0.0005},
        {'params': model.lstm.parameters(), 'lr': 0.001},
        {'params': model.attention.parameters(), 'lr': 0.001},
        {'params': model.classifier.parameters(), 'lr': 0.001}
    ], weight_decay=1e-4)
    
    # Enhanced scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
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
    patience = 30  # Increased patience for complex model
    
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
        scheduler.step()
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
    
    print(f"✅ Enhanced training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    
    return model, history, best_val_acc

def main():
    """Execute enhanced model training pipeline"""
    print("🎯 ENHANCED 4-CLASS LIP-READING MODEL TRAINING")
    print("=" * 70)
    print("Advanced architecture targeting ≥82% cross-demographic validation accuracy")
    
    # Paths
    train_manifest = "balanced_85_training_results/balanced_340_train_manifest.csv"
    val_manifest = "balanced_85_training_results/balanced_340_validation_manifest.csv"
    data_dir = "data/the_best_videos_so_far"
    
    # Create enhanced data loaders
    train_loader, val_loader, class_to_idx = create_enhanced_data_loaders(
        train_manifest, val_manifest, data_dir, batch_size=4
    )
    
    # Create enhanced model
    print("\n🏗️  Creating Enhanced 3D CNN-LSTM model...")
    model = Enhanced3DCNN_LSTM(num_classes=4, hidden_size=512, num_layers=3, dropout=0.4)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Enhanced model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Train enhanced model
    print("\n🚀 Starting enhanced training...")
    trained_model, history, best_val_acc = train_enhanced_model(
        model, train_loader, val_loader, class_to_idx, 
        num_epochs=150, target_accuracy=0.82
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = "balanced_85_training_results"
    
    # Save enhanced model
    model_path = os.path.join(output_dir, f"enhanced_85_model_{timestamp}.pth")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'class_to_idx': class_to_idx,
        'best_val_acc': best_val_acc,
        'history': history,
        'model_config': {
            'num_classes': 4,
            'hidden_size': 512,
            'num_layers': 3,
            'dropout': 0.4,
            'architecture': 'Enhanced3DCNN_LSTM'
        }
    }, model_path)
    
    print(f"\n💾 Enhanced model saved: {model_path}")
    print(f"🏆 Final validation accuracy: {best_val_acc:.4f}")
    
    # Check success
    if best_val_acc >= 0.82:
        print("🎉 SUCCESS: Target ≥82% validation accuracy achieved!")
        print("✅ Phase 4 Complete: Cross-Demographic Validation")
        return True, model_path, best_val_acc
    else:
        print(f"⚠️  Target not reached: {best_val_acc:.4f} < 0.82")
        print("📊 Enhanced model trained but requires further optimization")
        return False, model_path, best_val_acc

if __name__ == "__main__":
    success, model_path, accuracy = main()
    exit(0 if success else 1)
