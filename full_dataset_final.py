#!/usr/bin/env python3
"""
Full Dataset Training - FINAL VERSION
Force class diversity with specialized techniques
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
from pathlib import Path
from collections import Counter
from sklearn.metrics import classification_report

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class FinalDataset(Dataset):
    def __init__(self, video_paths, labels, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        
        print(f"📊 Dataset: {len(video_paths)} videos, Augment: {augment}")
        label_counts = Counter(labels)
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        for i, name in enumerate(class_names):
            print(f"   {name}: {label_counts.get(i, 0)} videos")
        
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_simple(self, path):
        """Simple, consistent video loading."""
        cap = cv2.VideoCapture(path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale and resize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))  # Even smaller for stability
            frames.append(resized)
        
        cap.release()
        
        # Take exactly 16 frames (shorter sequences)
        if len(frames) >= 16:
            indices = np.linspace(0, len(frames)-1, 16, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < 16:
                frames.append(frames[-1] if frames else np.zeros((64, 64), dtype=np.uint8))
        
        return np.array(frames[:16])
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video
        frames = self.load_video_simple(video_path)
        
        # Minimal augmentation
        if self.augment and random.random() < 0.3:
            # Only horizontal flip
            frames = np.flip(frames, axis=2).copy()
        
        # Simple normalization
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor (C, T, H, W)
        frames = torch.from_numpy(frames).unsqueeze(0)
        
        return frames, label

class FinalModel(nn.Module):
    def __init__(self, num_classes=5):
        super(FinalModel, self).__init__()
        
        # Very simple 3D CNN
        self.conv1 = nn.Conv3d(1, 8, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        
        self.conv2 = nn.Conv3d(8, 16, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        
        self.conv3 = nn.Conv3d(16, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Simple classifier
        self.classifier = nn.Linear(32, num_classes)
        
        # Proper initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Critical: Initialize classifier with small, balanced weights
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class BalancedBatchSampler:
    """Custom sampler that ensures each batch has all classes."""
    def __init__(self, labels, batch_size=5):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_classes = len(set(labels))
        
        # Group indices by class
        self.class_indices = {}
        for i, label in enumerate(labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(i)
        
        # Shuffle within each class
        for class_label in self.class_indices:
            random.shuffle(self.class_indices[class_label])
        
        self.class_pointers = {label: 0 for label in self.class_indices}
    
    def __iter__(self):
        while True:
            batch = []
            
            # Get one sample from each class
            for class_label in sorted(self.class_indices.keys()):
                indices = self.class_indices[class_label]
                pointer = self.class_pointers[class_label]
                
                # Get next index, wrap around if needed
                idx = indices[pointer % len(indices)]
                batch.append(idx)
                
                # Update pointer
                self.class_pointers[class_label] = (pointer + 1) % len(indices)
            
            yield batch
    
    def __len__(self):
        return max(len(indices) for indices in self.class_indices.values())

def create_final_splits(dataset_path="the_best_videos_so_far"):
    """Create final splits from full dataset."""
    print("📊 Creating final splits from FULL dataset...")
    
    video_files = list(Path(dataset_path).glob("*.mp4"))
    video_files = [f for f in video_files if "copy" not in f.name]
    
    print(f"Found {len(video_files)} videos (after removing duplicates)")
    
    # Group by class
    class_videos = {'doctor': [], 'glasses': [], 'help': [], 'phone': [], 'pillow': []}
    
    for video_file in video_files:
        filename = video_file.stem
        if filename.startswith('doctor'):
            class_name = 'doctor'
        elif filename.startswith('glasses'):
            class_name = 'glasses'
        elif filename.startswith('help'):
            class_name = 'help'
        elif filename.startswith('phone'):
            class_name = 'phone'
        elif filename.startswith('pillow'):
            class_name = 'pillow'
        else:
            continue
        
        class_videos[class_name].append(str(video_file))
    
    # Print class distribution
    for class_name, videos in class_videos.items():
        print(f"   {class_name}: {len(videos)} videos")
    
    # Create splits
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []
    
    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
    
    for class_name, videos in class_videos.items():
        random.shuffle(videos)
        n_videos = len(videos)
        
        # 70% train, 15% val, 15% test
        n_train = max(1, int(0.7 * n_videos))
        n_val = max(1, int(0.15 * n_videos))
        
        train_videos.extend(videos[:n_train])
        train_labels.extend([class_to_idx[class_name]] * n_train)
        
        val_videos.extend(videos[n_train:n_train+n_val])
        val_labels.extend([class_to_idx[class_name]] * n_val)
        
        test_videos.extend(videos[n_train+n_val:])
        test_labels.extend([class_to_idx[class_name]] * (len(videos) - n_train - n_val))
    
    print(f"📊 Final splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def train_final_model(model, train_dataset, val_dataset, device, num_epochs=25):
    """Train with forced class diversity."""
    
    # Use balanced batch sampler
    batch_sampler = BalancedBatchSampler(train_dataset.labels, batch_size=5)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Simple training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print(f"\n🚀 Final training for {num_epochs} epochs with balanced batches...")
    
    best_val_acc = 0.0
    patience = 0
    max_patience = 8
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_correct = 0
        train_total = 0
        train_preds = []
        train_targets = []
        
        # Train for limited batches per epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 20:  # Limit batches per epoch
                break
                
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            train_preds.extend(pred.cpu().numpy())
            train_targets.extend(target.cpu().numpy())
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # Check class diversity
        unique_train_preds = len(set(train_preds))
        unique_val_preds = len(set(val_preds))
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train: {train_acc:.1f}% ({unique_train_preds}/5), "
              f"Val: {val_acc:.1f}% ({unique_val_preds}/5)")
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), 'best_final_model.pth')
            print(f"  💾 New best: {val_acc:.1f}%")
        else:
            patience += 1
        
        # Early stopping
        if patience >= max_patience:
            print(f"  ⏹️  Early stopping")
            break
        
        # Success check
        if unique_val_preds >= 4 and val_acc >= 35:
            print(f"  🎉 GOOD PROGRESS!")
            if val_acc >= 45:
                print(f"  🏆 EXCELLENT!")
                break
    
    return best_val_acc

def main():
    """Final full dataset training."""
    print("🎯 FINAL FULL DATASET TRAINING")
    print("=" * 60)
    print("FINAL APPROACH:")
    print("• Balanced batch sampling (1 per class per batch)")
    print("• Simple, stable architecture")
    print("• Minimal augmentation")
    print("• All 91 videos from full dataset")
    print("• Target: Consistent multi-class prediction")
    print("=" * 60)
    
    # Set seeds
    set_seeds(42)
    
    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")
    
    # Create final dataset splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_final_splits()
    
    # Create datasets
    train_dataset = FinalDataset(train_videos, train_labels, augment=True)
    val_dataset = FinalDataset(val_videos, val_labels, augment=False)
    test_dataset = FinalDataset(test_videos, test_labels, augment=False)
    
    # Create final model
    model = FinalModel(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Final Model: {total_params:,} parameters")
    
    # Train
    best_val_acc = train_final_model(model, train_dataset, val_dataset, device, num_epochs=20)
    
    # Test
    print(f"\n🔍 Testing final model...")
    
    if os.path.exists('best_final_model.pth'):
        model.load_state_dict(torch.load('best_final_model.pth', map_location=device))
        print("📥 Loaded best model")
    
    model.eval()
    test_correct = 0
    test_total = 0
    test_preds = []
    test_targets = []
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += target.size(0)
            test_preds.extend(pred.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    unique_test_preds = len(set(test_preds))
    
    # Classification report
    if len(set(test_targets)) > 1:
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        report = classification_report(test_targets, test_preds, target_names=class_names, zero_division=0)
        print(f"📊 Classification Report:\n{report}")
    
    print(f"\n🎯 FINAL RESULTS WITH FULL DATASET")
    print("=" * 50)
    print(f"🎯 Test Accuracy: {test_acc:.1f}%")
    print(f"🎯 Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"🎯 Test Predictions: {sorted(set(test_preds))}")
    print(f"🎯 Test Targets: {sorted(set(test_targets))}")
    print(f"🎯 Unique Predictions: {unique_test_preds}/5 classes")
    print(f"🎯 Total Training Videos: {len(train_videos)}")
    print(f"🎯 Dataset Size: {len(train_videos) + len(val_videos) + len(test_videos)} videos")
    
    if test_acc >= 50:
        print("🏆 EXCELLENT: 50%+ accuracy achieved with full dataset!")
    elif test_acc >= 40:
        print("✅ SUCCESS: 40%+ accuracy achieved with full dataset!")
    elif test_acc >= 30:
        print("📈 GOOD: Solid improvement with full dataset!")
    elif unique_test_preds >= 4:
        print("📊 PROGRESS: Multi-class prediction working with full dataset!")
    else:
        print("⚠️  Need advanced techniques")
    
    return test_acc

if __name__ == "__main__":
    try:
        final_accuracy = main()
        print(f"\n🏁 Final full dataset training completed: {final_accuracy:.1f}% accuracy")
        print(f"📊 Using complete dataset of 91 videos vs previous 50 videos")
        
        if final_accuracy >= 40:
            print("🚀 Ready for advanced techniques with full dataset!")
        else:
            print("🔄 Full dataset provides foundation for further optimization")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
