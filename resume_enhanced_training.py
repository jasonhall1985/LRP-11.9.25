#!/usr/bin/env python3
"""
Resume Enhanced Training: Continue from 62.39% Validation Accuracy Checkpoint
Fine-tune with reduced learning rate and extended training schedule
Target: 65%+ validation accuracy improvement
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import cv2
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchvision import transforms
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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
        final_features = lstm_out[:, -1]

        # Classification
        output = self.classifier(final_features)
        return output

class LipReadingDataset(Dataset):
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
        
        # Enhanced data augmentation for 82% target
        if self.augment:
            self.brightness_range = (-0.2, 0.2)   # ±20% (more aggressive)
            self.contrast_range = (0.8, 1.2)      # 0.8-1.2x (wider range)
            self.gamma_range = (0.8, 1.2)         # Gamma correction
            self.flip_prob = 0.5                  # 50% horizontal flip
            self.noise_prob = 0.3                 # 30% Gaussian noise
            self.noise_std = 0.02                 # Noise standard deviation
        
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

        # Convert to numpy array and ensure 32 frames
        frames = np.array(frames)

        # Handle frame count - take center 32 frames
        if len(frames) >= 32:
            start_idx = (len(frames) - 32) // 2
            frames = frames[start_idx:start_idx + 32]
        else:
            # Pad with last frame if too short
            while len(frames) < 32:
                frames = np.append(frames, [frames[-1]], axis=0)

        return frames

    def apply_augmentation(self, frames):
        """Apply enhanced data augmentation for 82% target"""
        if not self.augment:
            return frames

        # Horizontal flip
        if random.random() < self.flip_prob:
            frames = np.flip(frames, axis=2).copy()

        # Brightness adjustment (more aggressive)
        brightness_delta = random.uniform(*self.brightness_range)
        frames = np.clip(frames + brightness_delta, 0, 1)

        # Contrast adjustment (wider range)
        contrast_factor = random.uniform(*self.contrast_range)
        frames = np.clip(frames * contrast_factor, 0, 1)

        # Gamma correction for lighting variations
        gamma = random.uniform(*self.gamma_range)
        frames = np.power(frames, gamma)

        # Gaussian noise for robustness
        if random.random() < self.noise_prob:
            noise = np.random.normal(0, self.noise_std, frames.shape)
            frames = np.clip(frames + noise, 0, 1)

        return frames

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        filename = row['filename']
        video_path = os.path.join(self.data_dir, filename)

        frames = self.load_video_frames(video_path)
        frames = self.apply_augmentation(frames)

        frames_tensor = torch.FloatTensor(frames).unsqueeze(0)
        class_label = self.class_to_idx[row['class']]
        label_tensor = torch.LongTensor([class_label])

        return frames_tensor, label_tensor
    


def load_checkpoint_and_resume_training():
    """Load checkpoint and resume training with fine-tuned parameters"""
    print("🔄 RESUMING ENHANCED TRAINING FROM 62.39% CHECKPOINT")
    print("=" * 70)
    print("Loading checkpoint: enhanced_lightweight_model_20250923_000053.pth")
    print("Target: 82% validation accuracy improvement")
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print(f"\n📊 Loading enhanced balanced datasets...")
    train_manifest = "enhanced_balanced_training_results/enhanced_balanced_536_train_manifest.csv"
    val_manifest = "enhanced_balanced_training_results/enhanced_balanced_536_validation_manifest.csv"
    
    data_dir = "data/the_best_videos_so_far"
    train_dataset = LipReadingDataset(train_manifest, data_dir, augment=True)
    val_dataset = LipReadingDataset(val_manifest, data_dir, augment=False)
    
    print(f"Enhanced Dataset: {len(train_dataset)} videos, Augmentation: ON")
    train_class_counts = train_dataset.manifest['class'].value_counts().sort_index()
    for class_name, count in train_class_counts.items():
        print(f"  {class_name}: {count} videos")
    
    print(f"Enhanced Dataset: {len(val_dataset)} videos, Augmentation: OFF")
    val_class_counts = val_dataset.manifest['class'].value_counts().sort_index()
    for class_name, count in val_class_counts.items():
        print(f"  {class_name}: {count} videos")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, drop_last=False)
    print(f"✅ Enhanced loaders: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Load model and checkpoint
    print(f"\n🏗️  Loading model from checkpoint...")
    model = LightweightCNN_LSTM(num_classes=4, hidden_size=128, num_layers=2, dropout=0.3)
    
    checkpoint_path = "enhanced_balanced_training_results/enhanced_lightweight_model_20250923_000053.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} total")
    
    # Setup optimizer with aggressive optimization for 82% target
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-3, betas=(0.9, 0.999))  # AdamW with stronger regularization

    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Update learning rate and parameters in loaded optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003
            param_group['weight_decay'] = 1e-3

    # Label smoothing for better generalization
    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing

        def forward(self, pred, target):
            n_classes = pred.size(-1)
            one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
            smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
            return -(smooth_one_hot * pred.log_softmax(dim=-1)).sum(dim=-1).mean()

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    # Cosine annealing with warm restarts for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Resume training configuration
    start_epoch = checkpoint.get('epoch', 0) + 1  # Continue from next epoch
    best_val_acc = checkpoint.get('best_val_acc', 0.6239)  # 62.39% baseline
    max_epochs = 60  # Extended from 40
    patience = 30    # Extended from 15
    epochs_without_improvement = 0
    
    print(f"\n🚀 Aggressive optimization configuration for 82% target:")
    print(f"Starting epoch: {start_epoch}")
    print(f"Best validation accuracy: {best_val_acc:.4f} (62.39% baseline)")
    print(f"Optimizer: AdamW with lr=0.0003, weight_decay=1e-3")
    print(f"Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")
    print(f"Loss: Label Smoothing CrossEntropy (smoothing=0.1)")
    print(f"Augmentation: Enhanced (brightness ±20%, contrast 0.8-1.2x, gamma, noise)")
    print(f"Max epochs: {max_epochs} (extended from 40)")
    print(f"Early stopping patience: {patience} (extended from 15)")
    print(f"Target: ≥82% validation accuracy")
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    learning_rates = []
    
    print(f"\nStarting resumed training...")
    print("=" * 85)
    
    for epoch in range(start_epoch, max_epochs + 1):
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
        
        train_acc = train_correct / train_total if train_total > 0 else 0
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
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling (cosine annealing)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record metrics
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        learning_rates.append(current_lr)
        
        # Progress display
        val_acc_pct = int(val_acc * 100)
        target_indicator = "🎯 TARGET!" if val_acc >= 0.82 else ""
        print(f"Epoch {epoch:2d}/{max_epochs}: Train: {train_acc:.4f} ({avg_train_loss:.4f}) | "
              f"Val: {val_acc:.4f} ({avg_val_loss:.4f}) | LR: {current_lr:.6f} ({val_acc_pct}%) {target_indicator}")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            
            # Save best model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            best_model_path = f"enhanced_balanced_training_results/resumed_best_model_{timestamp}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'learning_rates': learning_rates
            }, best_model_path)
            
            print(f"💾 New best model saved: {best_model_path}")
            
        else:
            epochs_without_improvement += 1
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\n⏹️  Early stopping triggered after {patience} epochs without improvement")
            break
        
        # Target achievement check
        if val_acc >= 0.82:
            print(f"\n🎯 TARGET ACHIEVED! Validation accuracy: {val_acc:.4f} (≥82%)")
            break
    
    print("=" * 85)
    print("✅ Resumed training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    improvement = ((best_val_acc - 0.6239) / 0.6239) * 100
    print(f"📈 Improvement vs. 62.39% baseline: +{improvement:.1f}%")
    
    # Generate training curves
    print("📊 Generating extended training curves...")
    plot_extended_training_curves(train_losses, train_accs, val_losses, val_accs, learning_rates, start_epoch)
    
    return True, best_val_acc

def plot_extended_training_curves(train_losses, train_accs, val_losses, val_accs, learning_rates, start_epoch):
    """Plot extended training curves showing resumed training progression"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(start_epoch, start_epoch + len(train_losses))
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.axvline(x=35, color='gray', linestyle='--', alpha=0.7, label='Original Training End')
    ax1.set_title('Extended Training: Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, [acc * 100 for acc in train_accs], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, [acc * 100 for acc in val_accs], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.axhline(y=62.39, color='orange', linestyle='--', alpha=0.7, label='62.39% Baseline')
    ax2.axhline(y=82.0, color='green', linestyle='--', alpha=0.7, label='82% Target')
    ax2.axvline(x=35, color='gray', linestyle='--', alpha=0.7, label='Original Training End')
    ax2.set_title('Extended Training: Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate
    ax3.plot(epochs, learning_rates, 'g-', linewidth=2)
    ax3.axvline(x=35, color='gray', linestyle='--', alpha=0.7, label='Original Training End')
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Overfitting gap
    overfitting_gap = [(train_acc - val_acc) * 100 for train_acc, val_acc in zip(train_accs, val_accs)]
    ax4.plot(epochs, overfitting_gap, 'purple', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.axvline(x=35, color='gray', linestyle='--', alpha=0.7, label='Original Training End')
    ax4.set_title('Overfitting Gap (Train - Val Accuracy)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Gap (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f"enhanced_balanced_training_results/extended_training_curves_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Extended training curves saved: {plot_path}")
    
    plt.show()

def main():
    """Execute resumed training from 62.39% checkpoint"""
    print("🎯 ENHANCED MODEL FINE-TUNING")
    print("=" * 50)
    print("Resuming from 62.39% validation accuracy checkpoint")
    print("Target: 82% validation accuracy with aggressive optimization")
    
    success, final_acc = load_checkpoint_and_resume_training()
    
    if success:
        print(f"\n🎉 RESUMED TRAINING COMPLETE")
        print(f"🏆 Final validation accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
        target_achieved = "✅ YES" if final_acc >= 0.82 else "❌ NO"
        print(f"🎯 82% target achieved: {target_achieved}")
        print("📊 Extended training curves generated")
        print("💾 Best model checkpoint saved")
    else:
        print("❌ Resumed training failed")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
