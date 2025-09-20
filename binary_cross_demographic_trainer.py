#!/usr/bin/env python3
"""
Binary Classification Cross-Demographic Training Pipeline
Validates lip-reading model architecture with help vs doctor classification.
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
import matplotlib.pyplot as plt

class BinaryCrossDemographicTrainer:
    def __init__(self, manifests_dir="data/classifier training 20.9.25/binary_classification", 
                 output_dir="binary_classification_training_results"):
        self.manifests_dir = Path(manifests_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.batch_size = 8
        self.max_epochs = 30
        self.patience = 10
        self.learning_rate = 0.001
        self.weight_decay = 0.01
        self.target_train_acc = 90.0
        self.target_val_acc = 70.0
        
        # Device setup (CPU due to MPS 3D pooling issues)
        self.device = torch.device('cpu')
        print(f"🔥 Using device: {self.device}")
        print("💻 Using CPU for reliable 3D CNN operations")
        
        # Class mapping
        self.class_to_idx = {'doctor': 0, 'help': 1}
        self.idx_to_class = {0: 'doctor', 1: 'help'}
        
        print("🎯 BINARY CROSS-DEMOGRAPHIC TRAINING PIPELINE")
        print("=" * 80)
        print(f"📁 Manifests: {self.manifests_dir}")
        print(f"📁 Output: {self.output_dir}")
        print(f"🎯 Classes: help vs doctor (2 classes)")
        print(f"🎯 Cross-demographic: 65+ female Caucasian → 18-39 male not_specified")
        
    def load_datasets(self):
        """Load training and validation datasets from manifests."""
        print("\n📋 LOADING CROSS-DEMOGRAPHIC DATASETS")
        print("=" * 60)
        
        # Load manifests
        train_manifest = self.manifests_dir / "binary_train_manifest.csv"
        val_manifest = self.manifests_dir / "binary_validation_manifest.csv"
        
        self.train_dataset = LipReadingDataset(train_manifest, self.class_to_idx)
        self.val_dataset = LipReadingDataset(val_manifest, self.class_to_idx)
        
        print(f"📊 Training dataset: {len(self.train_dataset)} videos")
        print(f"   Demographics: {self.train_dataset.get_demographics()}")
        print(f"   Classes: {self.train_dataset.get_class_distribution()}")
        
        print(f"📊 Validation dataset: {len(self.val_dataset)} videos")
        print(f"   Demographics: {self.val_dataset.get_demographics()}")
        print(f"   Classes: {self.val_dataset.get_class_distribution()}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        print(f"📊 Batch configuration:")
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation batches: {len(self.val_loader)}")
        print(f"   Batch size: {self.batch_size}")
        
        # Verify zero demographic overlap
        train_demos = self.train_dataset.get_unique_demographics()
        val_demos = self.val_dataset.get_unique_demographics()
        overlap = train_demos.intersection(val_demos)
        
        if overlap:
            raise ValueError(f"Demographic overlap detected: {overlap}")
        
        print(f"✅ ZERO DEMOGRAPHIC OVERLAP CONFIRMED")
        print(f"   Training: {train_demos}")
        print(f"   Validation: {val_demos}")

class LipReadingDataset(Dataset):
    """Dataset for lip-reading videos with demographic information."""
    
    def __init__(self, manifest_path, class_to_idx):
        self.class_to_idx = class_to_idx
        self.videos = []
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['class'] in class_to_idx:
                    self.videos.append({
                        'path': row['video_path'],
                        'class': row['class'],
                        'class_idx': class_to_idx[row['class']],
                        'demographic_group': row['demographic_group'],
                        'age_group': row['age_group'],
                        'gender': row['gender'],
                        'ethnicity': row['ethnicity']
                    })
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_info = self.videos[idx]
        
        # Load video
        frames = self._load_video(video_info['path'])
        
        # Convert to tensor: (C, T, H, W) = (1, 32, 64, 96)
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        frames_tensor = frames_tensor.unsqueeze(0)  # Add channel dimension
        
        return frames_tensor, video_info['class_idx']
    
    def _load_video(self, video_path):
        """Load video frames as numpy array."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < 32:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)
        
        cap.release()
        
        # Ensure exactly 32 frames
        while len(frames) < 32:
            frames.append(frames[-1])  # Repeat last frame
        
        frames = frames[:32]  # Truncate if too many
        
        return np.array(frames)  # Shape: (32, 64, 96)
    
    def get_demographics(self):
        """Get demographic distribution."""
        demographics = defaultdict(int)
        for video in self.videos:
            demographics[video['demographic_group']] += 1
        return dict(demographics)
    
    def get_class_distribution(self):
        """Get class distribution."""
        classes = defaultdict(int)
        for video in self.videos:
            classes[video['class']] += 1
        return dict(classes)
    
    def get_unique_demographics(self):
        """Get set of unique demographic groups."""
        return set(video['demographic_group'] for video in self.videos)

class BinaryCNN_LSTM(nn.Module):
    """Binary classification CNN-LSTM model with corrected dimensions."""
    
    def __init__(self, num_classes=2, input_channels=1, hidden_size=256, num_lstm_layers=2):
        super(BinaryCNN_LSTM, self).__init__()
        
        # 3D CNN layers for spatiotemporal feature extraction
        self.conv3d1 = nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn3d1 = nn.BatchNorm3d(32)
        self.pool3d1 = nn.MaxPool3d(kernel_size=(1, 2, 2))  # Only spatial pooling
        
        self.conv3d2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn3d2 = nn.BatchNorm3d(64)
        self.pool3d2 = nn.MaxPool3d(kernel_size=(2, 2, 2))  # Temporal + spatial pooling
        
        self.conv3d3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn3d3 = nn.BatchNorm3d(128)
        self.pool3d3 = nn.MaxPool3d(kernel_size=(2, 2, 2))  # Temporal + spatial pooling
        
        # Feature size after pooling: 128 * 8 * 12 = 12,288
        self.feature_size = 128 * 8 * 12
        
        # LSTM layers for temporal modeling
        self.lstm1 = nn.LSTM(self.feature_size, hidden_size, num_lstm_layers, 
                            batch_first=True, dropout=0.3, bidirectional=True)
        
        # Classification layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # *2 for bidirectional
        self.fc2 = nn.Linear(128, num_classes)
        
        print(f"🏗️  Binary CNN-LSTM Model initialized:")
        print(f"   - Input: (B, 1, 32, 64, 96)")
        print(f"   - Feature size: {self.feature_size}")
        print(f"   - LSTM hidden size: {hidden_size}")
        print(f"   - Output classes: {num_classes}")
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 3D CNN feature extraction
        x = F.relu(self.bn3d1(self.conv3d1(x)))
        x = self.pool3d1(x)
        
        x = F.relu(self.bn3d2(self.conv3d2(x)))
        x = self.pool3d2(x)
        
        x = F.relu(self.bn3d3(self.conv3d3(x)))
        x = self.pool3d3(x)
        # x shape: (B, 128, 8, 8, 12)
        
        # Reshape for LSTM: (B, T, Features)
        x = x.permute(0, 2, 1, 3, 4)  # (B, T=8, C=128, H=8, W=12)
        x = x.contiguous().view(batch_size, x.size(1), -1)  # (B, T=8, Features=12288)
        
        # LSTM processing
        lstm_out, _ = self.lstm1(x)  # (B, T, hidden_size*2)
        
        # Use last time step
        x = lstm_out[:, -1, :]  # (B, hidden_size*2)
        
        # Classification
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    def setup_training(self):
        """Setup model, optimizer, and training components."""
        print("\n🏗️  SETTING UP TRAINING COMPONENTS")
        print("=" * 60)

        # Initialize model
        self.model = BinaryCNN_LSTM(num_classes=2).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"📊 Model parameters:")
        print(f"   Total: {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")

        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

        print(f"✅ Training setup complete:")
        print(f"   Optimizer: AdamW (lr={self.learning_rate}, weight_decay={self.weight_decay})")
        print(f"   Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
        print(f"   Loss: CrossEntropyLoss")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (videos, labels) in enumerate(self.train_loader):
            videos, labels = videos.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Progress logging
            if batch_idx % 2 == 0:  # Log every 2 batches
                acc = 100.0 * correct / total
                print(f"   Batch {batch_idx:2d}/{len(self.train_loader):2d} | "
                      f"Loss: {loss.item():.4f} | Acc: {acc:.2f}%")

        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """Validate for one epoch."""
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
            'val_accuracies': self.val_accuracies
        }

        # Save latest checkpoint
        torch.save(checkpoint, self.output_dir / 'latest_checkpoint.pth')

        # Save best model
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
            print(f"💾 Best model saved: {self.output_dir / 'best_model.pth'}")

    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.axhline(y=self.target_train_acc, color='b', linestyle='--', alpha=0.7, label=f'Target Train Acc ({self.target_train_acc}%)')
        ax2.axhline(y=self.target_val_acc, color='r', linestyle='--', alpha=0.7, label=f'Target Val Acc ({self.target_val_acc}%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Training curves saved: {self.output_dir / 'training_curves.png'}")

    def train_model(self):
        """Execute complete training loop."""
        print("\n🎯 STARTING BINARY CROSS-DEMOGRAPHIC TRAINING")
        print("=" * 80)
        print(f"Target: {self.target_train_acc}% training accuracy, {self.target_val_acc}% cross-demographic validation")

        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            print(f"\n📅 Epoch {epoch:2d}/{self.max_epochs}")
            print("-" * 50)

            # Training
            train_loss, train_acc = self.train_epoch()

            # Validation
            val_loss, val_acc = self.validate_epoch()

            # Update tracking
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Update learning rate
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Epoch summary
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"   Learning Rate: {current_lr:.2e}")
            print(f"   Best Val Acc: {self.best_val_acc:.2f}% (Epoch {epoch - self.epochs_without_improvement})")
            print(f"   Time: {time.time() - start_time:.1f}s")

            # Check success criteria
            if train_acc >= self.target_train_acc and val_acc >= self.target_val_acc:
                print(f"\n🎉 SUCCESS CRITERIA MET!")
                print(f"   ✅ Training accuracy: {train_acc:.2f}% ≥ {self.target_train_acc}%")
                print(f"   ✅ Cross-demographic validation: {val_acc:.2f}% ≥ {self.target_val_acc}%")
                break

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\n⏹️  Early stopping triggered (patience: {self.patience})")
                break

        # Final results
        total_time = time.time() - start_time
        self.generate_final_report(total_time)
        self.plot_training_curves()

    def generate_final_report(self, training_time):
        """Generate comprehensive training report."""
        report_path = self.output_dir / 'training_report.txt'

        # Determine success status
        final_train_acc = self.train_accuracies[-1] if self.train_accuracies else 0
        final_val_acc = self.val_accuracies[-1] if self.val_accuracies else 0

        train_success = final_train_acc >= self.target_train_acc
        val_success = final_val_acc >= self.target_val_acc
        overall_success = train_success and val_success

        with open(report_path, 'w') as f:
            f.write("BINARY CROSS-DEMOGRAPHIC TRAINING REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Classes: help vs doctor (2 classes)\n")
            f.write(f"Training demographic: 65+ female Caucasian\n")
            f.write(f"Validation demographic: 18-39 male not_specified\n")
            f.write(f"Training videos: {len(self.train_dataset)}\n")
            f.write(f"Validation videos: {len(self.val_dataset)}\n")
            f.write(f"Batch size: {self.batch_size}\n")
            f.write(f"Max epochs: {self.max_epochs}\n")
            f.write(f"Training time: {training_time:.1f}s\n\n")

            f.write("FINAL RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Final training accuracy: {final_train_acc:.2f}%\n")
            f.write(f"Final validation accuracy: {final_val_acc:.2f}%\n")
            f.write(f"Best validation accuracy: {self.best_val_acc:.2f}%\n")
            f.write(f"Total epochs: {len(self.train_accuracies)}\n\n")

            f.write("SUCCESS CRITERIA EVALUATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Training accuracy ≥ {self.target_train_acc}%: {'✅ PASS' if train_success else '❌ FAIL'} ({final_train_acc:.2f}%)\n")
            f.write(f"Cross-demographic validation ≥ {self.target_val_acc}%: {'✅ PASS' if val_success else '❌ FAIL'} ({final_val_acc:.2f}%)\n")
            f.write(f"Overall success: {'✅ PASS' if overall_success else '❌ FAIL'}\n\n")

            f.write("NEXT STEPS:\n")
            f.write("-" * 40 + "\n")
            if overall_success:
                f.write("✅ Pipeline validated - proceed to full 7-class cross-demographic training\n")
            else:
                f.write("❌ Pipeline validation failed - debug issues before 7-class training\n")
                if not train_success:
                    f.write("   - Training accuracy insufficient - check model architecture/learning rate\n")
                if not val_success:
                    f.write("   - Cross-demographic generalization poor - consider data augmentation\n")

        print(f"📄 Training report saved: {report_path}")

        # Print final summary
        print(f"\n🎯 BINARY CROSS-DEMOGRAPHIC TRAINING COMPLETED")
        print("=" * 80)
        print(f"📊 Final Results:")
        print(f"   Training Accuracy: {final_train_acc:.2f}% ({'✅ PASS' if train_success else '❌ FAIL'})")
        print(f"   Cross-Demographic Validation: {final_val_acc:.2f}% ({'✅ PASS' if val_success else '❌ FAIL'})")
        print(f"   Overall Success: {'✅ PASS' if overall_success else '❌ FAIL'}")
        print(f"   Training Time: {training_time:.1f}s")

        return overall_success

    def run_complete_pipeline(self):
        """Execute the complete binary training pipeline."""
        try:
            # Load datasets
            self.load_datasets()

            # Setup training
            self.setup_training()

            # Train model
            success = self.train_model()

            return success

        except Exception as e:
            print(f"\n❌ TRAINING PIPELINE FAILED: {e}")
            raise

def main():
    """Main execution with Mac caffeinate support."""
    print("🚀 STARTING BINARY CROSS-DEMOGRAPHIC TRAINING PIPELINE")
    print("💻 Mac caffeinate enabled for uninterrupted training")

    trainer = BinaryCrossDemographicTrainer()
    success = trainer.run_complete_pipeline()

    if success:
        print("\n🎉 READY TO PROCEED TO FULL 7-CLASS TRAINING!")
    else:
        print("\n⚠️  PIPELINE VALIDATION FAILED - DEBUG REQUIRED")

if __name__ == "__main__":
    main()
