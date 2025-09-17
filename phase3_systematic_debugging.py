#!/usr/bin/env python3
"""
Phase 3: Systematic Pipeline Debugging
Goal: Identify what broke between 40% baseline and current 20% approaches
Strategy: Recreate exact working conditions step by step
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

class OriginalLipDataset(Dataset):
    """Recreate the EXACT original dataset that achieved 40%."""
    
    def __init__(self, video_paths, labels, split='train'):
        self.video_paths = video_paths
        self.labels = labels
        self.split = split
        
        self.class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
        print(f"📊 {split.upper()} Dataset: {len(self.video_paths)} videos")
        
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_original(self, video_path):
        """Load video with ORIGINAL preprocessing that worked."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # ORIGINAL: Simple BGR to grayscale conversion
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # ORIGINAL: Resize to 112x112 (this was the working size)
                frame = cv2.resize(frame, (112, 112))
                
                frames.append(frame)
            
            cap.release()
            
            # ORIGINAL: Simple temporal sampling to 32 frames
            if len(frames) >= 32:
                # Use uniform sampling like original
                indices = np.linspace(0, len(frames) - 1, 32, dtype=int)
                frames = [frames[i] for i in indices]
            else:
                # Pad with last frame
                while len(frames) < 32:
                    frames.append(frames[-1] if frames else np.zeros((112, 112), dtype=np.uint8))
            
            return np.array(frames[:32])
            
        except Exception as e:
            print(f"❌ Error loading video {video_path}: {str(e)}")
            return np.zeros((32, 112, 112), dtype=np.uint8)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video with original preprocessing
        frames = self.load_video_original(video_path)
        
        # ORIGINAL: Simple normalization to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor: (C, T, H, W) format
        frames = torch.from_numpy(frames).unsqueeze(0)
        
        return frames, label

class OriginalLipModel(nn.Module):
    """Recreate the EXACT original model that achieved 40%."""
    
    def __init__(self, num_classes=5):
        super(OriginalLipModel, self).__init__()
        
        # ORIGINAL: Simple 3D CNN architecture (290K parameters)
        self.features = nn.Sequential(
            # First conv block
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # Second conv block
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Third conv block
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        
        # ORIGINAL: Simple classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def create_original_splits(dataset_path="corrected_balanced_dataset"):
    """Create the EXACT same splits as the working 40% approach."""
    print("📊 Creating original data splits...")
    
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
    
    # ORIGINAL: Use the EXACT same random seed and split logic
    random.seed(42)  # Same seed as original
    
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []
    
    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
    
    for class_name, videos in class_videos.items():
        random.shuffle(videos)  # Same shuffle as original
        
        # ORIGINAL: 8 train, 1 val, 1 test per class
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

class OriginalTrainer:
    """Recreate the EXACT original training that achieved 40%."""
    
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # ORIGINAL: Simple training configuration
        self.criterion = nn.CrossEntropyLoss()
        
        # ORIGINAL: Adam optimizer with original settings
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=1e-3,  # Original learning rate
            weight_decay=1e-4
        )
        
        # ORIGINAL: Step scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.5
        )
        
        # Training state
        self.epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.max_patience = 5  # Original patience
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"debug_original_{timestamp}"
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
        self.logger.info(f"🚀 Original debugging started: {self.experiment_dir}")
        
    def train_epoch(self):
        """Train for one epoch - ORIGINAL approach."""
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
            
            # ORIGINAL: Gradient clipping
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
        
        return avg_loss, train_acc
        
    def validate(self):
        """Validate the model - ORIGINAL approach."""
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
        
        return val_acc, val_f1, all_preds, all_targets
        
    def train(self, num_epochs=20):
        """Main training loop - ORIGINAL approach."""
        self.logger.info(f"🎯 Starting original debugging training for {num_epochs} epochs")
        
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
        """Final test evaluation - ORIGINAL approach."""
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

def main():
    """Main debugging execution - recreate original 40% approach."""
    print("🔍 PHASE 3: SYSTEMATIC PIPELINE DEBUGGING")
    print("=" * 60)
    print("Goal: Recreate EXACT original conditions that achieved 40%")
    print("=" * 60)
    print(f"💾 Initial memory: {get_memory_usage():.1f} MB")
    
    # Set EXACT same random seeds as original
    set_random_seeds(42)
    
    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")
    
    # Create ORIGINAL data splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_original_splits()
    
    # Create ORIGINAL datasets
    train_dataset = OriginalLipDataset(train_videos, train_labels, split='train')
    val_dataset = OriginalLipDataset(val_videos, val_labels, split='val')
    test_dataset = OriginalLipDataset(test_videos, test_labels, split='test')
    
    # Create ORIGINAL data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)  # ORIGINAL settings
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"💾 After data loading: {get_memory_usage():.1f} MB")
    
    # Create ORIGINAL model
    model = OriginalLipModel(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"🧠 Original Model (Recreation):")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Expected: ~290K parameters")
    print(f"   • Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"💾 After model creation: {get_memory_usage():.1f} MB")
    
    # Create ORIGINAL trainer
    trainer = OriginalTrainer(model, train_loader, val_loader, test_loader, device)
    
    print(f"💾 After trainer setup: {get_memory_usage():.1f} MB")
    
    # Start ORIGINAL training
    print("\n🎯 STARTING ORIGINAL RECREATION TRAINING")
    print("=" * 50)
    print("Expected result: ~40% accuracy (if pipeline is correct)")
    print("=" * 50)
    
    start_time = time.time()
    final_accuracy = trainer.train(num_epochs=20)
    training_time = (time.time() - start_time) / 60
    
    print(f"\n🎉 ORIGINAL RECREATION COMPLETED!")
    print("=" * 50)
    print(f"🎯 Final Test Accuracy: {final_accuracy:.2f}%")
    print(f"⏱️  Training Time: {training_time:.1f} minutes")
    print(f"💾 Final Memory Usage: {get_memory_usage():.1f} MB")
    print(f"📁 Results Directory: {trainer.experiment_dir}")
    
    # Diagnostic evaluation
    print(f"\n🔍 DIAGNOSTIC EVALUATION:")
    if final_accuracy >= 35:
        print(f"✅ SUCCESS: Pipeline is working correctly! ({final_accuracy:.1f}% ≈ 40%)")
        print(f"   🎯 Ready to proceed with optimizations")
        print(f"   📈 Next: Apply incremental improvements")
    elif final_accuracy >= 25:
        print(f"⚠️  PARTIAL SUCCESS: Close to working ({final_accuracy:.1f}%)")
        print(f"   🔍 Minor issues in pipeline - investigate small differences")
    else:
        print(f"❌ PIPELINE BROKEN: Still getting {final_accuracy:.1f}%")
        print(f"   🔍 Major issue remains - need deeper debugging")
    
    print(f"\n📊 COMPARISON:")
    print(f"   • Original working: 40.0%")
    print(f"   • Recreation result: {final_accuracy:.1f}%")
    print(f"   • Difference: {final_accuracy - 40:.1f} percentage points")
    
    return final_accuracy

if __name__ == "__main__":
    try:
        final_accuracy = main()
        
        if final_accuracy >= 35:
            print(f"\n🎯 PIPELINE DEBUGGING SUCCESS!")
            print(f"   Ready to proceed with Phase 4: Optimization")
        else:
            print(f"\n🔄 CONTINUE DEBUGGING:")
            print(f"   Need to investigate remaining differences")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
