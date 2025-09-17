#!/usr/bin/env python3
"""
Phase 2: Fallback Training - Simplified but Effective Improvements
Focus on proven techniques to achieve 50-65% accuracy reliably.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import json
import time
from datetime import datetime
from pathlib import Path
import logging
import psutil
from sklearn.metrics import accuracy_score, f1_score, classification_report

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

class ImprovedLipDataset(Dataset):
    """Improved dataset with better preprocessing and minimal augmentation."""
    
    def __init__(self, video_paths, labels, split='train', augment=True):
        self.video_paths = video_paths
        self.labels = labels
        self.split = split
        self.augment = augment and (split == 'train')
        
        self.class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
        print(f"📊 {split.upper()} Dataset: {len(self.video_paths)} videos")
        
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_improved(self, video_path):
        """Load video with improved preprocessing."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            frames = []
            while len(frames) < 40:  # Load more frames for better sampling
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Resize to consistent size (keep larger than before)
                frame = cv2.resize(frame, (128, 128))  # Larger than 112x112
                
                frames.append(frame)
            
            cap.release()
            
            # Better temporal sampling: uniform sampling to 32 frames
            if len(frames) >= 32:
                indices = np.linspace(0, len(frames) - 1, 32, dtype=int)
                frames = [frames[i] for i in indices]
            else:
                # Pad with last frame
                while len(frames) < 32:
                    frames.append(frames[-1] if frames else np.zeros((128, 128), dtype=np.uint8))
            
            return np.array(frames[:32])
            
        except Exception as e:
            print(f"❌ Error loading video {video_path}: {str(e)}")
            return np.zeros((32, 128, 128), dtype=np.uint8)
    
    def apply_minimal_augmentation(self, frames):
        """Apply minimal, proven augmentations."""
        if not self.augment:
            return frames
        
        # Only horizontal flip (50% chance) - most reliable augmentation
        if random.random() < 0.5:
            frames = np.flip(frames, axis=2).copy()
        
        # Very slight brightness adjustment (20% chance)
        if random.random() < 0.2:
            brightness = random.uniform(0.95, 1.05)
            frames = np.clip(frames * brightness, 0, 255).astype(np.uint8)
        
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video with improved preprocessing
        frames = self.load_video_improved(video_path)
        
        # Apply minimal augmentation
        frames = self.apply_minimal_augmentation(frames)
        
        # Simple normalization to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor: (C, T, H, W) format
        frames = torch.from_numpy(frames).unsqueeze(0)
        
        return frames, label

class ImprovedLipModel(nn.Module):
    """Improved model - simpler but more effective architecture."""
    
    def __init__(self, num_classes=5, dropout=0.5):
        super(ImprovedLipModel, self).__init__()
        
        # Improved 3D CNN backbone
        self.features = nn.Sequential(
            # First block - larger kernels for better temporal modeling
            nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # Second block
            nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(2, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Third block
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Fourth block - additional depth
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        
        # Improved classifier with proper regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),  # Less dropout in middle layer
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3),  # Even less dropout
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class FallbackTrainer:
    """Simplified but effective training manager."""
    
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Simple but effective training configuration
        self.criterion = nn.CrossEntropyLoss()
        
        # Use SGD with momentum - often more stable than Adam for small datasets
        self.optimizer = optim.SGD(
            model.parameters(), 
            lr=1e-2,  # Higher learning rate
            momentum=0.9, 
            weight_decay=1e-4
        )
        
        # Step scheduler - simple and effective
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.5
        )
        
        # Training state
        self.epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []
        self.patience_counter = 0
        self.max_patience = 8
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"fallback_training_{timestamp}"
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
        self.logger.info(f"🚀 Fallback training started: {self.experiment_dir}")
        
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
            
            if batch_idx % 10 == 0:
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
        val_f1 = f1_score(all_targets, all_preds, average='macro') * 100
        self.val_accuracies.append(val_acc)
        
        return val_acc, val_f1, all_preds, all_targets
        
    def train(self, num_epochs=25):
        """Main training loop."""
        self.logger.info(f"🎯 Starting fallback training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # Train epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_acc, val_f1, val_preds, val_targets = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f"{self.experiment_dir}/best_model.pth")
                self.logger.info(f"💾 New best model saved: {val_acc:.2f}%")
            else:
                self.patience_counter += 1
            
            # Log progress
            current_lr = self.optimizer.param_groups[0]['lr']
            memory_mb = get_memory_usage()
            self.logger.info(f"Epoch {self.epoch}/{num_epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                           f"Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%, "
                           f"Best: {self.best_val_acc:.2f}%, LR: {current_lr:.2e}, "
                           f"Memory: {memory_mb:.1f}MB")
            
            # Early stopping
            if self.patience_counter >= self.max_patience:
                self.logger.info(f"⏹️  Early stopping after {self.patience_counter} epochs without improvement")
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
            self.logger.info("📥 Loaded best model")
        
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
        test_f1 = f1_score(all_targets, all_preds, average='macro') * 100
        
        # Generate classification report
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        report = classification_report(all_targets, all_preds, target_names=class_names)
        
        self.logger.info(f"🎯 FINAL RESULTS:")
        self.logger.info(f"   • Test Accuracy: {test_acc:.2f}%")
        self.logger.info(f"   • Test F1 Score: {test_f1:.2f}%")
        self.logger.info(f"   • Best Val Accuracy: {self.best_val_acc:.2f}%")
        self.logger.info(f"📊 Classification Report:\n{report}")
        
        # Save results
        results = {
            'test_accuracy': test_acc,
            'test_f1_score': test_f1,
            'best_val_accuracy': self.best_val_acc,
            'total_epochs': self.epoch,
            'classification_report': report
        }
        
        with open(f"{self.experiment_dir}/final_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return test_acc

def create_balanced_splits(dataset_path="corrected_balanced_dataset"):
    """Create balanced train/validation/test splits."""
    print("📊 Creating balanced data splits...")
    
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
    
    # Create balanced splits (8 train, 1 val, 1 test per class)
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
    """Main fallback training execution."""
    print("🚀 FALLBACK LIP-READING TRAINING - TARGET: 50-65% ACCURACY")
    print("=" * 70)
    print(f"💾 Initial memory: {get_memory_usage():.1f} MB")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")
    
    # Create balanced data splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_balanced_splits()
    
    # Create improved datasets
    train_dataset = ImprovedLipDataset(train_videos, train_labels, split='train', augment=True)
    val_dataset = ImprovedLipDataset(val_videos, val_labels, split='val', augment=False)
    test_dataset = ImprovedLipDataset(test_videos, test_labels, split='test', augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)  # Slightly larger batch
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    print(f"💾 After data loading: {get_memory_usage():.1f} MB")
    
    # Create improved model
    model = ImprovedLipModel(num_classes=5, dropout=0.4).to(device)  # Less aggressive dropout
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"🧠 Improved Model:")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"💾 After model creation: {get_memory_usage():.1f} MB")
    
    # Create fallback trainer
    trainer = FallbackTrainer(model, train_loader, val_loader, test_loader, device)
    
    print(f"💾 After trainer setup: {get_memory_usage():.1f} MB")
    
    # Start training
    print("\n🎯 STARTING FALLBACK TRAINING")
    print("=" * 40)
    
    start_time = time.time()
    final_accuracy = trainer.train(num_epochs=25)
    training_time = (time.time() - start_time) / 60
    
    print(f"\n🎉 FALLBACK TRAINING COMPLETED!")
    print("=" * 45)
    print(f"🎯 Final Test Accuracy: {final_accuracy:.2f}%")
    print(f"⏱️  Training Time: {training_time:.1f} minutes")
    print(f"💾 Final Memory Usage: {get_memory_usage():.1f} MB")
    print(f"📁 Results Directory: {trainer.experiment_dir}")
    
    # Success evaluation
    baseline = 40.0
    improvement = final_accuracy - baseline
    
    if final_accuracy >= 60:
        print(f"✅ EXCELLENT: Target exceeded! 🎉")
    elif final_accuracy >= 50:
        print(f"✅ SUCCESS: Significant improvement achieved! 📈")
    elif final_accuracy >= 45:
        print(f"📈 GOOD: Meaningful progress made")
    else:
        print(f"⚠️  NEEDS MORE WORK: Consider additional improvements")
    
    print(f"\n📊 IMPROVEMENT SUMMARY:")
    print(f"   • Previous best: 40.00%")
    print(f"   • Fallback result: {final_accuracy:.2f}%")
    print(f"   • Improvement: {improvement:+.2f} percentage points")
    print(f"   • Relative improvement: {((final_accuracy / baseline) - 1) * 100:+.1f}%")
    
    return final_accuracy

if __name__ == "__main__":
    try:
        final_accuracy = main()
        
        if final_accuracy >= 50:
            print(f"\n🎯 MISSION ACCOMPLISHED: Fallback training successful!")
        else:
            print(f"\n🔄 NEEDS ITERATION: Consider further improvements")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
