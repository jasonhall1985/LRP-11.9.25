#!/usr/bin/env python3
"""
Phase 2: Memory-Efficient Training Pipeline for Lip-Reading Classifier

This version uses smaller models and optimized memory usage to prevent crashes.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import json
import time
from datetime import datetime
from pathlib import Path
import logging
import psutil

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MemoryEfficientLipDataset(Dataset):
    """Memory-efficient dataset for lip-reading videos."""
    
    def __init__(self, video_paths, labels, split='train', augment=True):
        self.video_paths = video_paths
        self.labels = labels
        self.split = split
        self.augment = augment and (split == 'train')
        
        self.class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
        print(f"📊 {split.upper()} Dataset: {len(self.video_paths)} videos")
        
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_efficient(self, video_path):
        """Load video with memory-efficient processing."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            frames = []
            while len(frames) < 32:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Resize to smaller size for memory efficiency
                frame = cv2.resize(frame, (224, 224))  # Smaller than original 640x432
                frames.append(frame)
            
            cap.release()
            
            # Ensure exactly 32 frames
            while len(frames) < 32:
                frames.append(frames[-1] if frames else np.zeros((224, 224), dtype=np.uint8))
            
            return np.array(frames[:32])
            
        except Exception as e:
            print(f"❌ Error loading video {video_path}: {str(e)}")
            return np.zeros((32, 224, 224), dtype=np.uint8)
    
    def apply_augmentation(self, frames):
        """Apply minimal augmentations."""
        if not self.augment:
            return frames
        
        # Horizontal flip (50% chance)
        if random.random() < 0.5:
            frames = np.flip(frames, axis=2).copy()
        
        # Slight brightness adjustment (30% chance)
        if random.random() < 0.3:
            brightness_factor = random.uniform(0.9, 1.1)
            frames = np.clip(frames * brightness_factor, 0, 255).astype(np.uint8)
        
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video frames
        frames = self.load_video_efficient(video_path)
        
        # Apply augmentation
        frames = self.apply_augmentation(frames)
        
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor: (C, T, H, W) format
        frames = torch.from_numpy(frames).unsqueeze(0)
        
        return frames, label

class EfficientLipModel(nn.Module):
    """Memory-efficient 3D CNN for lip-reading."""
    
    def __init__(self, num_classes=5):
        super(EfficientLipModel, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv3d(1, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # Second conv block
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Third conv block
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MemoryEfficientTrainer:
    """Memory-efficient training manager."""
    
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Training configuration
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Training state
        self.epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"efficient_training_{timestamp}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.experiment_dir}/training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 Efficient training started: {self.experiment_dir}")
        
    def train_epoch(self):
        """Train for one epoch with memory monitoring."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Memory check
            if batch_idx % 5 == 0:
                memory_mb = get_memory_usage()
                if memory_mb > 2000:  # If memory > 2GB, force garbage collection
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
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
                memory_mb = get_memory_usage()
                self.logger.info(f"Epoch {self.epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                               f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%, "
                               f"Memory: {memory_mb:.1f}MB")
        
        avg_loss = total_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        self.train_losses.append(avg_loss)
        
        return avg_loss, train_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_acc = 100. * correct / total
        self.val_accuracies.append(val_acc)
        return val_acc
    
    def train(self, num_epochs=30, target_accuracy=80.0):
        """Main training loop."""
        self.logger.info(f"🎯 Starting training for {num_epochs} epochs, target: {target_accuracy}%")
        
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # Train epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f"{self.experiment_dir}/best_model.pth")
            else:
                patience_counter += 1
            
            # Log progress
            memory_mb = get_memory_usage()
            self.logger.info(f"Epoch {self.epoch}/{num_epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                           f"Val Acc: {val_acc:.2f}%, Best: {self.best_val_acc:.2f}%, "
                           f"Memory: {memory_mb:.1f}MB")
            
            # Early stopping
            if patience_counter >= max_patience:
                self.logger.info(f"⏹️  Early stopping after {patience_counter} epochs without improvement")
                break
            
            # Target accuracy reached
            if val_acc >= target_accuracy:
                self.logger.info(f"🎉 Target accuracy {target_accuracy}% reached! Val Acc: {val_acc:.2f}%")
                break
        
        # Final test evaluation
        return self.final_test()
    
    def final_test(self):
        """Final test evaluation."""
        self.logger.info("🔍 Final test evaluation...")
        
        # Load best model
        best_model_path = f"{self.experiment_dir}/best_model.pth"
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_acc = 100. * correct / total
        self.logger.info(f"🎯 FINAL TEST ACCURACY: {test_acc:.2f}%")
        
        # Save results
        results = {
            'test_accuracy': test_acc,
            'best_val_accuracy': self.best_val_acc,
            'total_epochs': self.epoch
        }
        
        with open(f"{self.experiment_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return test_acc

def create_data_splits(dataset_path="corrected_balanced_dataset"):
    """Create train/validation/test splits."""
    print("📊 Creating data splits...")
    
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
        random.shuffle(videos)
        
        train_videos.extend(videos[:8])
        train_labels.extend([class_to_idx[class_name]] * 8)
        
        if len(videos) > 8:
            val_videos.append(videos[8])
            val_labels.append(class_to_idx[class_name])
        
        if len(videos) > 9:
            test_videos.append(videos[9])
            test_labels.append(class_to_idx[class_name])
    
    print(f"📊 Splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def main():
    """Main training execution."""
    print("🚀 MEMORY-EFFICIENT LIP-READING TRAINING")
    print("=" * 60)
    print(f"💾 Initial memory: {get_memory_usage():.1f} MB")
    
    # Set seeds
    set_random_seeds(42)
    
    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")
    
    # Create data splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_data_splits()
    
    # Create datasets
    train_dataset = MemoryEfficientLipDataset(train_videos, train_labels, split='train', augment=True)
    val_dataset = MemoryEfficientLipDataset(val_videos, val_labels, split='val', augment=False)
    test_dataset = MemoryEfficientLipDataset(test_videos, test_labels, split='test', augment=False)
    
    # Create data loaders (batch_size=1 for memory efficiency)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"💾 After data loading: {get_memory_usage():.1f} MB")
    
    # Create model
    model = EfficientLipModel(num_classes=5).to(device)
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"💾 After model: {get_memory_usage():.1f} MB")
    
    # Create trainer
    trainer = MemoryEfficientTrainer(model, train_loader, val_loader, test_loader, device)
    
    # Start training
    final_accuracy = trainer.train(num_epochs=30, target_accuracy=80.0)
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"📊 Final Test Accuracy: {final_accuracy:.2f}%")
    print(f"💾 Final memory: {get_memory_usage():.1f} MB")
    
    return final_accuracy

if __name__ == "__main__":
    main()
