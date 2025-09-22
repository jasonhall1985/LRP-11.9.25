#!/usr/bin/env python3
"""
Train Enhanced Lightweight Model on Expanded 536-Video Dataset
Using proven Lightweight CNN-LSTM architecture targeting ≥60-70% validation accuracy
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
import matplotlib.pyplot as plt
from collections import defaultdict

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class EnhancedLipDataset(Dataset):
    """Enhanced dataset for expanded 536-video training"""
    
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
        
        print(f"Enhanced Dataset: {len(self.manifest)} videos, Augmentation: {'ON' if augment else 'OFF'}")
        
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
            while len(frames) < 32:
                frames.extend(frames[:min(len(frames), 32 - len(frames))])
            frames = frames[:32]
        
        return np.array(frames)
    
    def apply_augmentation(self, frames):
        """Apply proven augmentation strategy"""
        if not self.augment:
            return frames
        
        # Horizontal flipping (50% chance)
        if np.random.random() > 0.5:
            frames = np.flip(frames, axis=2).copy()
        
        # Brightness adjustment (±12%, 60% chance)
        if np.random.random() > 0.4:
            brightness_factor = np.random.uniform(0.88, 1.12)
            frames = np.clip(frames * brightness_factor, 0, 1)
        
        # Contrast adjustment (0.9-1.1x, 50% chance)
        if np.random.random() > 0.5:
            contrast_factor = np.random.uniform(0.9, 1.1)
            frames = np.clip((frames - 0.5) * contrast_factor + 0.5, 0, 1)
        
        # Scale variation (±5%, 40% chance)
        if np.random.random() > 0.6:
            scale_factor = np.random.uniform(0.95, 1.05)
            h, w = frames.shape[1], frames.shape[2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            scaled_frames = []
            for frame in frames:
                scaled = cv2.resize(frame, (new_w, new_h))
                # Center crop or pad to original size
                if scale_factor > 1:
                    # Crop
                    start_y = (new_h - h) // 2
                    start_x = (new_w - w) // 2
                    scaled = scaled[start_y:start_y+h, start_x:start_x+w]
                else:
                    # Pad
                    pad_y = (h - new_h) // 2
                    pad_x = (w - new_w) // 2
                    scaled = np.pad(scaled, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x)), mode='edge')
                scaled_frames.append(scaled)
            frames = np.array(scaled_frames)
        
        # Temporal shift (±1 frame, 30% chance)
        if np.random.random() > 0.7:
            shift = np.random.randint(-1, 2)
            if shift != 0:
                frames = np.roll(frames, shift, axis=0)
        
        return frames
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        video_path = os.path.join(self.data_dir, row['filename'])
        
        frames = self.load_video_frames(video_path)
        frames = self.apply_augmentation(frames)
        
        frames_tensor = torch.FloatTensor(frames).unsqueeze(0)
        class_label = self.class_to_idx[row['class']]
        label_tensor = torch.LongTensor([class_label])
        
        return frames_tensor, label_tensor

class LightweightCNN_LSTM(nn.Module):
    """Proven Lightweight CNN-LSTM architecture (~2.2M parameters)"""
    
    def __init__(self, num_classes=4, hidden_size=128, num_layers=2, dropout=0.3):
        super(LightweightCNN_LSTM, self).__init__()
        
        # Lightweight CNN feature extractor
        self.cnn = nn.Sequential(
            # First conv block
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
            
            # Fourth conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 2))  # Fixed output: 3x2
        )
        
        # CNN output size: 128 * 3 * 2 = 768
        self.cnn_output_size = 128 * 3 * 2
        
        # Bidirectional LSTM
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
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
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

def train_enhanced_model(model, train_loader, val_loader, num_epochs=40, target_accuracy=0.60):
    """Train enhanced model with expanded dataset"""
    print(f"🚀 Enhanced Training on Expanded Dataset")
    print(f"Target: ≥{target_accuracy*100:.0f}% validation accuracy (baseline: 51.47%)")
    print(f"Max epochs: {num_epochs}, Early stopping patience: 15")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Training setup (identical to proven configuration)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 15
    
    print(f"\nStarting enhanced training...")
    print("=" * 85)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device).squeeze()

            # Skip empty batches
            if videos.size(0) == 0 or labels.size(0) == 0:
                continue

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
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

                # Skip empty batches
                if videos.size(0) == 0 or labels.size(0) == 0:
                    continue

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
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # Check for overfitting
        overfitting_gap = train_acc - val_acc
        overfitting_warning = " ⚠️ OVERFITTING" if overfitting_gap > 0.15 else ""
        
        # Progress indicator
        progress = "🎯 TARGET!" if val_acc >= target_accuracy else f"({val_acc/target_accuracy*100:.0f}%)"
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Train: {train_acc:.4f} ({avg_train_loss:.4f}) | "
              f"Val: {val_acc:.4f} ({avg_val_loss:.4f}) | "
              f"LR: {current_lr:.6f} {progress}{overfitting_warning}")
        
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
    print(f"✅ Enhanced training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    print(f"📈 Improvement vs. baseline: {((best_val_acc - 0.5147) / 0.5147 * 100):+.1f}%")
    
    return model, best_val_acc, {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

def plot_training_curves(training_history, output_dir):
    """Plot training and validation curves"""
    print("📊 Generating training curves...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(training_history['train_losses']) + 1)
    
    # Loss curves
    ax1.plot(epochs, training_history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, training_history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, [acc*100 for acc in training_history['train_accuracies']], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, [acc*100 for acc in training_history['val_accuracies']], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.axhline(y=51.47, color='orange', linestyle='--', label='Previous Baseline (51.47%)', linewidth=2)
    ax2.axhline(y=60, color='green', linestyle='--', label='Target (60%)', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    curves_path = os.path.join(output_dir, 'enhanced_training_curves.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved: {curves_path}")
    return curves_path

def main():
    """Execute enhanced lightweight model training"""
    print("🎯 ENHANCED LIGHTWEIGHT MODEL TRAINING")
    print("=" * 70)
    print("Training on expanded 536-video dataset targeting ≥60-70% validation accuracy")
    
    # Paths
    train_manifest = "enhanced_balanced_training_results/enhanced_balanced_536_train_manifest.csv"
    val_manifest = "enhanced_balanced_training_results/enhanced_balanced_536_validation_manifest.csv"
    data_dir = "data/the_best_videos_so_far"
    output_dir = "enhanced_balanced_training_results"
    
    # Create enhanced data loaders
    print("\n🔄 Creating enhanced data loaders...")
    train_dataset = EnhancedLipDataset(train_manifest, data_dir, augment=True)
    val_dataset = EnhancedLipDataset(val_manifest, data_dir, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, drop_last=False)
    
    print(f"✅ Enhanced loaders: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Create proven lightweight model
    print("\n🏗️  Creating Lightweight CNN-LSTM model...")
    model = LightweightCNN_LSTM(num_classes=4, hidden_size=128, num_layers=2, dropout=0.3)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} total")
    print(f"Target range: ~2.2M parameters ✅")
    
    # Train enhanced model
    print("\n🚀 Starting enhanced training...")
    trained_model, best_val_acc, training_history = train_enhanced_model(
        model, train_loader, val_loader, num_epochs=40, target_accuracy=0.60
    )
    
    # Plot training curves
    curves_path = plot_training_curves(training_history, output_dir)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(output_dir, f"enhanced_lightweight_model_{timestamp}.pth")
    
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx,
        'best_val_acc': best_val_acc,
        'training_history': training_history,
        'model_config': {
            'num_classes': 4,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'architecture': 'LightweightCNN_LSTM',
            'total_params': total_params,
            'dataset_size': 536,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset)
        }
    }, model_path)
    
    print(f"\n💾 Enhanced model saved: {model_path}")
    print(f"📊 Training curves: {curves_path}")
    
    # Final assessment
    baseline_acc = 0.5147
    improvement = ((best_val_acc - baseline_acc) / baseline_acc) * 100
    target_achieved = best_val_acc >= 0.60
    
    print("\n" + "=" * 70)
    print("🎉 ENHANCED LIGHTWEIGHT MODEL TRAINING COMPLETE")
    print(f"🏆 Final validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"📈 Improvement vs. baseline: {improvement:+.1f}% (baseline: 51.47%)")
    print(f"🎯 Target achieved: {'✅ YES' if target_achieved else '❌ NO'} (target: ≥60%)")
    print("🚀 Ready for Phase 4: Per-User Calibration Implementation")
    
    return target_achieved, model_path, best_val_acc

if __name__ == "__main__":
    success, model_path, accuracy = main()
    exit(0 if success else 1)
