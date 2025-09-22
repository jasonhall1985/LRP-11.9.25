#!/usr/bin/env python3
"""
Optimized 4-Class Lip-Reading Model Training Pipeline
Ultra-lightweight with aggressive regularization for small datasets
Target: ≥82% validation accuracy with ~1-2M parameters
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

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class OptimizedLipDataset(Dataset):
    """Optimized dataset with stronger augmentation for overfitting prevention"""
    
    def __init__(self, manifest_path, data_dir, augment=False):
        self.manifest = pd.read_csv(manifest_path)
        self.data_dir = data_dir
        self.augment = augment
        
        self.class_to_idx = {
            'doctor': 0,
            'i_need_to_move': 1,
            'my_mouth_is_dry': 2,
            'pillow': 3
        }
        
        print(f"Optimized Dataset: {len(self.manifest)} videos, Strong Augmentation: {'ON' if augment else 'OFF'}")
        
        class_counts = self.manifest['class'].value_counts().sort_index()
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} videos")
    
    def __len__(self):
        return len(self.manifest)
    
    def load_video_frames(self, video_path):
        """Load video with enhanced preprocessing"""
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
            
            # Apply slight Gaussian blur to reduce overfitting to noise
            frame = cv2.GaussianBlur(frame, (3, 3), 0.5)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        
        # Ensure exactly 32 frames
        if len(frames) >= 32:
            start_idx = (len(frames) - 32) // 2
            frames = frames[start_idx:start_idx + 32]
        else:
            while len(frames) < 32:
                frames.extend(frames[:min(len(frames), 32 - len(frames))])
            frames = frames[:32]
        
        return np.array(frames)
    
    def apply_strong_augmentation(self, frames):
        """Strong augmentation to prevent overfitting on small dataset"""
        if not self.augment:
            return frames
        
        # Horizontal flipping (60% chance)
        if np.random.random() > 0.4:
            frames = np.flip(frames, axis=2).copy()
        
        # Brightness adjustment (±15%, 70% chance)
        if np.random.random() > 0.3:
            brightness_factor = np.random.uniform(0.85, 1.15)
            frames = np.clip(frames * brightness_factor, 0, 1)
        
        # Contrast adjustment (0.85-1.15x, 60% chance)
        if np.random.random() > 0.4:
            contrast_factor = np.random.uniform(0.85, 1.15)
            frames = np.clip((frames - 0.5) * contrast_factor + 0.5, 0, 1)
        
        # Random noise injection (40% chance, very light)
        if np.random.random() > 0.6:
            noise = np.random.normal(0, 0.02, frames.shape)
            frames = np.clip(frames + noise, 0, 1)
        
        # Temporal shift (50% chance)
        if np.random.random() > 0.5:
            shift = np.random.randint(-2, 3)
            if shift != 0:
                frames = np.roll(frames, shift, axis=0)
        
        # Random frame dropout (30% chance, replace 1-2 frames with adjacent)
        if np.random.random() > 0.7:
            num_drops = np.random.randint(1, 3)
            for _ in range(num_drops):
                drop_idx = np.random.randint(1, len(frames)-1)
                frames[drop_idx] = frames[drop_idx-1]  # Replace with previous frame
        
        return frames
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        video_path = os.path.join(self.data_dir, row['filename'])
        
        frames = self.load_video_frames(video_path)
        frames = self.apply_strong_augmentation(frames)
        
        frames_tensor = torch.FloatTensor(frames).unsqueeze(0)
        class_label = self.class_to_idx[row['class']]
        label_tensor = torch.LongTensor([class_label])
        
        return frames_tensor, label_tensor

class UltraLightCNN_LSTM(nn.Module):
    """Ultra-lightweight CNN-LSTM with ~1-2M parameters and strong regularization"""
    
    def __init__(self, num_classes=4, hidden_size=96, num_layers=2, dropout=0.5):
        super(UltraLightCNN_LSTM, self).__init__()
        
        # Ultra-lightweight CNN with fewer channels
        self.cnn = nn.Sequential(
            # First conv block - very small
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),  # 96x64 -> 48x32
            
            # Second conv block
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout2d(0.15),
            nn.MaxPool2d(2, 2),  # 48x32 -> 24x16
            
            # Third conv block
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2),  # 24x16 -> 12x8
            
            # Final conv with adaptive pooling
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.AdaptiveAvgPool2d((2, 3))  # Fixed output: 2x3
        )
        
        # CNN output size: 64 * 2 * 3 = 384
        self.cnn_output_size = 64 * 2 * 3
        
        # Compact LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Compact classifier with heavy regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 48),
            nn.ReLU(),
            nn.BatchNorm1d(48),
            nn.Dropout(dropout * 0.7),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.BatchNorm1d(24),
            nn.Dropout(dropout * 0.5),
            nn.Linear(24, num_classes)
        )
        
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
            frame = x[:, :, t, :, :]
            features = self.cnn(frame)
            features = features.view(batch_size, -1)
            cnn_features.append(features)
        
        # Stack temporal features
        cnn_features = torch.stack(cnn_features, dim=1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_features)
        
        # Use last output for classification
        final_features = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(final_features)
        
        return output

def train_optimized_model(model, train_loader, val_loader, num_epochs=50, target_accuracy=0.82):
    """Optimized training with advanced regularization techniques"""
    print(f"🚀 Optimized Training (target: {target_accuracy*100:.1f}% validation accuracy)")
    print(f"Max epochs: {num_epochs}, Early stopping patience: 20")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Label smoothing cross-entropy
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
    
    # AdamW with higher weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 20
    
    print(f"\nStarting optimized training...")
    print("=" * 85)
    
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
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
        
        # Check for overfitting
        overfitting_gap = train_acc - val_acc
        overfitting_warning = " ⚠️ OVERFITTING" if overfitting_gap > 0.12 else ""
        
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
    
    print("=" * 85)
    print(f"✅ Optimized training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    
    return model, best_val_acc

def main():
    """Execute optimized model training pipeline"""
    print("🎯 OPTIMIZED 4-CLASS LIP-READING MODEL TRAINING")
    print("=" * 70)
    print("Ultra-lightweight with aggressive regularization for small datasets")
    
    # Paths
    train_manifest = "balanced_85_training_results/balanced_340_train_manifest.csv"
    val_manifest = "balanced_85_training_results/balanced_340_validation_manifest.csv"
    data_dir = "data/the_best_videos_so_far"
    
    # Create optimized data loaders
    print("\n🔄 Creating optimized data loaders...")
    train_dataset = OptimizedLipDataset(train_manifest, data_dir, augment=True)
    val_dataset = OptimizedLipDataset(val_manifest, data_dir, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"✅ Optimized loaders: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Create ultra-light model
    print("\n🏗️  Creating Ultra-Light CNN-LSTM model...")
    model = UltraLightCNN_LSTM(num_classes=4, hidden_size=96, num_layers=2, dropout=0.5)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} total")
    print(f"Target range: 1-2M parameters ✅" if 1_000_000 <= total_params <= 2_000_000 else f"⚠️ Outside target range")
    
    # Train optimized model
    print("\n🚀 Starting optimized training...")
    trained_model, best_val_acc = train_optimized_model(
        model, train_loader, val_loader, num_epochs=50, target_accuracy=0.82
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = "balanced_85_training_results"
    
    model_path = os.path.join(output_dir, f"optimized_85_model_{timestamp}.pth")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx,
        'best_val_acc': best_val_acc,
        'model_config': {
            'num_classes': 4,
            'hidden_size': 96,
            'num_layers': 2,
            'dropout': 0.5,
            'architecture': 'UltraLightCNN_LSTM',
            'total_params': total_params
        }
    }, model_path)
    
    print(f"\n💾 Optimized model saved: {model_path}")
    print(f"🏆 Final validation accuracy: {best_val_acc:.4f}")
    
    if best_val_acc >= 0.82:
        print("🎉 SUCCESS: Target ≥82% validation accuracy achieved!")
        return True, model_path, best_val_acc
    else:
        print(f"📊 Best achieved: {best_val_acc:.4f} validation accuracy")
        print("💡 Model optimized but target not reached - dataset may need expansion")
        return False, model_path, best_val_acc

if __name__ == "__main__":
    success, model_path, accuracy = main()
    exit(0 if success else 1)
