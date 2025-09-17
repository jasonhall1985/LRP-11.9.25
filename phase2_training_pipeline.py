#!/usr/bin/env python3
"""
Phase 2: Comprehensive Training Pipeline for Lip-Reading Classifier

This script implements the complete training pipeline using the corrected balanced dataset
with R2Plus1D architecture, proper data splits, and comprehensive monitoring.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LipReadingDataset(Dataset):
    """Dataset class for lip-reading videos with proper splits and augmentation."""
    
    def __init__(self, video_paths, labels, split='train', augment=True):
        self.video_paths = video_paths
        self.labels = labels
        self.split = split
        self.augment = augment and (split == 'train')
        
        # Class mapping
        self.class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        print(f"📊 {split.upper()} Dataset: {len(self.video_paths)} videos")
        
    def __len__(self):
        return len(self.video_paths)
    
    def load_video(self, video_path):
        """Load video frames with error handling."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                raise ValueError(f"No frames loaded from: {video_path}")
            
            return np.array(frames)
            
        except Exception as e:
            print(f"❌ Error loading video {video_path}: {str(e)}")
            # Return dummy frames as fallback
            return np.zeros((32, 432, 640), dtype=np.uint8)
    
    def apply_augmentation(self, frames):
        """Apply minimal lip-reading appropriate augmentations."""
        if not self.augment:
            return frames
        
        # Horizontal flip (50% chance)
        if random.random() < 0.5:
            frames = np.flip(frames, axis=2).copy()
        
        # Slight brightness adjustment (±10-15%)
        if random.random() < 0.3:
            brightness_factor = random.uniform(0.85, 1.15)
            frames = np.clip(frames * brightness_factor, 0, 255).astype(np.uint8)
        
        # Minor contrast variation (0.9-1.1x)
        if random.random() < 0.3:
            contrast_factor = random.uniform(0.9, 1.1)
            frames = np.clip((frames - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
        
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video frames
        frames = self.load_video(video_path)
        
        # Ensure exactly 32 frames
        if len(frames) != 32:
            if len(frames) > 32:
                # Sample 32 frames uniformly
                indices = np.linspace(0, len(frames) - 1, 32, dtype=int)
                frames = frames[indices]
            else:
                # Pad with last frame
                padding_needed = 32 - len(frames)
                last_frame = frames[-1] if len(frames) > 0 else np.zeros((432, 640), dtype=np.uint8)
                padding = np.repeat(last_frame[np.newaxis], padding_needed, axis=0)
                frames = np.concatenate([frames, padding], axis=0)
        
        # Apply augmentation
        frames = self.apply_augmentation(frames)
        
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor: (C, T, H, W) format for R2Plus1D
        frames = torch.from_numpy(frames).unsqueeze(0)  # Add channel dimension
        
        return frames, label

class R2Plus1DModel(nn.Module):
    """R2Plus1D model for lip-reading classification."""
    
    def __init__(self, num_classes=5):
        super(R2Plus1DModel, self).__init__()
        
        # Import torchvision R2Plus1D
        try:
            from torchvision.models.video import r2plus1d_18
            self.backbone = r2plus1d_18(pretrained=False)
            
            # Modify first conv layer for grayscale input
            self.backbone.stem[0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), 
                                            stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
            
            # Replace classifier
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
            
        except ImportError:
            print("⚠️  torchvision R2Plus1D not available, using simplified 3D CNN")
            self.backbone = self._create_simple_3d_cnn(num_classes)
    
    def _create_simple_3d_cnn(self, num_classes):
        """Fallback 3D CNN if R2Plus1D not available."""
        return nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class TrainingManager:
    """Comprehensive training manager with monitoring and recovery."""
    
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Training configuration
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Training state
        self.epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []
        self.training_start_time = None
        
        # Setup logging and directories
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging and output directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"training_experiment_{timestamp}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.experiment_dir}/training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"🚀 Training experiment started: {self.experiment_dir}")
        
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies
        }
        
        checkpoint_path = f"{self.experiment_dir}/checkpoint_epoch_{self.epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = f"{self.experiment_dir}/best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 Best model saved: {best_path}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 5 == 0:
                self.logger.info(f"Epoch {self.epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                               f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
        
        avg_loss = total_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        return avg_loss, train_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        val_acc = 100. * correct / total
        self.val_accuracies.append(val_acc)
        
        return val_acc, all_preds, all_targets
    
    def train(self, num_epochs=50, target_accuracy=80.0):
        """Main training loop."""
        self.training_start_time = time.time()
        self.logger.info(f"🎯 Starting training for {num_epochs} epochs, target accuracy: {target_accuracy}%")
        
        patience_counter = 0
        max_patience = 15
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # Train epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_acc, val_preds, val_targets = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                patience_counter += 1
            
            # Regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint()
            
            # Log progress
            elapsed_time = time.time() - self.training_start_time
            self.logger.info(f"Epoch {self.epoch}/{num_epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                           f"Val Acc: {val_acc:.2f}%, Best Val Acc: {self.best_val_acc:.2f}%, "
                           f"Time: {elapsed_time/60:.1f}min")
            
            # Early stopping
            if patience_counter >= max_patience:
                self.logger.info(f"⏹️  Early stopping triggered after {patience_counter} epochs without improvement")
                break
            
            # Target accuracy reached
            if val_acc >= target_accuracy:
                self.logger.info(f"🎉 Target accuracy {target_accuracy}% reached! Val Acc: {val_acc:.2f}%")
                break
        
        # Final evaluation
        self.final_evaluation()
        
    def final_evaluation(self):
        """Perform final evaluation on test set."""
        self.logger.info("🔍 Performing final evaluation on test set...")
        
        # Load best model
        best_model_path = f"{self.experiment_dir}/best_model.pth"
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"📥 Loaded best model from epoch {checkpoint['epoch']}")
        
        # Test evaluation
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        test_acc = 100. * correct / total
        
        # Generate classification report
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        report = classification_report(all_targets, all_preds, target_names=class_names)
        
        self.logger.info(f"🎯 FINAL TEST ACCURACY: {test_acc:.2f}%")
        self.logger.info(f"📊 Classification Report:\n{report}")
        
        # Save results
        results = {
            'test_accuracy': test_acc,
            'best_val_accuracy': self.best_val_acc,
            'total_epochs': self.epoch,
            'training_time_minutes': (time.time() - self.training_start_time) / 60,
            'classification_report': report
        }
        
        with open(f"{self.experiment_dir}/final_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot training curves
        self.plot_training_curves()
        
        return test_acc
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training loss
        ax1.plot(self.train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Validation accuracy
        ax2.plot(self.val_accuracies)
        ax2.axhline(y=80, color='r', linestyle='--', label='Target (80%)')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.experiment_dir}/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📈 Training curves saved: {self.experiment_dir}/training_curves.png")

def create_data_splits(dataset_path="corrected_balanced_dataset"):
    """Create train/validation/test splits from balanced dataset."""
    print("📊 Creating data splits from corrected balanced dataset...")
    
    # Get all video files
    video_files = list(Path(dataset_path).glob("*.mp4"))
    
    if len(video_files) == 0:
        raise ValueError(f"No video files found in {dataset_path}")
    
    print(f"Found {len(video_files)} videos")
    
    # Organize by class
    class_videos = {'doctor': [], 'glasses': [], 'help': [], 'phone': [], 'pillow': []}
    
    for video_file in video_files:
        class_name = video_file.stem.split('_')[0]
        if class_name in class_videos:
            class_videos[class_name].append(str(video_file))
    
    # Create splits (8 train, 1 val, 1 test per class)
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []
    
    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
    
    for class_name, videos in class_videos.items():
        if len(videos) < 10:
            print(f"⚠️  Warning: {class_name} has only {len(videos)} videos")
        
        # Shuffle videos for this class
        random.shuffle(videos)
        
        # Split: 8 train, 1 val, 1 test
        train_videos.extend(videos[:8])
        train_labels.extend([class_to_idx[class_name]] * 8)
        
        if len(videos) > 8:
            val_videos.append(videos[8])
            val_labels.append(class_to_idx[class_name])
        
        if len(videos) > 9:
            test_videos.append(videos[9])
            test_labels.append(class_to_idx[class_name])
    
    print(f"📊 Data splits created:")
    print(f"   • Training: {len(train_videos)} videos")
    print(f"   • Validation: {len(val_videos)} videos")
    print(f"   • Test: {len(test_videos)} videos")
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def main():
    """Main training execution."""
    print("🚀 PHASE 2: LIP-READING CLASSIFIER TRAINING")
    print("=" * 80)
    
    # Set random seeds
    set_random_seeds(42)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create data splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_data_splits()
    
    # Create datasets
    train_dataset = LipReadingDataset(train_videos, train_labels, split='train', augment=True)
    val_dataset = LipReadingDataset(val_videos, val_labels, split='val', augment=False)
    test_dataset = LipReadingDataset(test_videos, test_labels, split='test', augment=False)
    
    # Create data loaders with smaller batch size for memory efficiency
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Create model
    model = R2Plus1DModel(num_classes=5).to(device)
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create training manager
    trainer = TrainingManager(model, train_loader, val_loader, test_loader, device)
    
    # Start training
    final_accuracy = trainer.train(num_epochs=50, target_accuracy=80.0)
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"📊 Final Test Accuracy: {final_accuracy:.2f}%")
    print(f"📁 Results saved in: {trainer.experiment_dir}")
    
    return final_accuracy

if __name__ == "__main__":
    main()
