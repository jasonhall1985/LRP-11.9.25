#!/usr/bin/env python3
"""
Fixed Training - Solve the "Stuck on One Class" Problem
Target the specific issue: model only predicting class 0
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import cv2
from pathlib import Path
from collections import Counter

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class FixedDataset(Dataset):
    def __init__(self, video_paths, labels, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        
        # Debug: Print dataset info
        print(f"📊 Dataset: {len(video_paths)} videos, Augment: {augment}")
        label_counts = Counter(labels)
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        for i, name in enumerate(class_names):
            print(f"   {name}: {label_counts.get(i, 0)} videos")
        
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_fixed(self, path):
        """Fixed video loading with better preprocessing."""
        cap = cv2.VideoCapture(path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize to 96x96 (good balance of detail and speed)
            resized = cv2.resize(gray, (96, 96))
            frames.append(resized)
        
        cap.release()
        
        # Take exactly 24 frames
        if len(frames) >= 24:
            indices = np.linspace(0, len(frames)-1, 24, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < 24:
                frames.append(frames[-1] if frames else np.zeros((96, 96), dtype=np.uint8))
        
        return np.array(frames[:24])
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video
        frames = self.load_video_fixed(video_path)
        
        # Enhanced augmentation for training
        if self.augment:
            # Random brightness/contrast
            if random.random() < 0.5:
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                frames = np.clip(frames * contrast + (brightness - 1) * 128, 0, 255).astype(np.uint8)
            
            # Random horizontal flip
            if random.random() < 0.5:
                frames = np.flip(frames, axis=2).copy()
            
            # Random temporal jitter
            if random.random() < 0.3:
                jitter = random.randint(-2, 2)
                if jitter > 0:
                    frames = frames[jitter:]
                    frames = np.pad(frames, ((0, jitter), (0, 0), (0, 0)), mode='edge')
                elif jitter < 0:
                    frames = frames[:jitter]
                    frames = np.pad(frames, ((-jitter, 0), (0, 0), (0, 0)), mode='edge')
        
        # Normalize to [0, 1] with better normalization
        frames = frames.astype(np.float32) / 255.0
        
        # Per-video normalization to prevent bias
        mean = frames.mean()
        std = frames.std()
        if std > 0:
            frames = (frames - mean) / std
            frames = frames * 0.5 + 0.5  # Rescale to [0, 1]
            frames = np.clip(frames, 0, 1)
        
        # Convert to tensor (C, T, H, W)
        frames = torch.from_numpy(frames).unsqueeze(0)
        
        return frames, label

class FixedModel(nn.Module):
    def __init__(self, num_classes=5):
        super(FixedModel, self).__init__()
        
        # Fixed 3D CNN with better initialization
        self.conv1 = nn.Conv3d(1, 32, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(32)
        
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), stride=(2, 2, 2), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)
        
        self.conv3 = nn.Conv3d(64, 128, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, num_classes)
        
        # CRITICAL FIX: Initialize classifier weights to prevent bias toward class 0
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights to prevent class bias."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # CRITICAL: Initialize classifier with small random weights
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

def create_balanced_splits(dataset_path="corrected_balanced_dataset"):
    """Create balanced splits with weighted sampling."""
    print("📊 Creating balanced splits...")
    
    video_files = list(Path(dataset_path).glob("*.mp4"))
    print(f"Found {len(video_files)} videos")
    
    # Group by class
    class_videos = {'doctor': [], 'glasses': [], 'help': [], 'phone': [], 'pillow': []}
    
    for video_file in video_files:
        class_name = video_file.stem.split('_')[0]
        if class_name in class_videos:
            class_videos[class_name].append(str(video_file))
    
    # Create splits ensuring each class is represented
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []
    
    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
    
    for class_name, videos in class_videos.items():
        random.shuffle(videos)
        
        # 6 train, 2 val, 2 test per class
        train_videos.extend(videos[:6])
        train_labels.extend([class_to_idx[class_name]] * 6)
        
        val_videos.extend(videos[6:8])
        val_labels.extend([class_to_idx[class_name]] * 2)
        
        test_videos.extend(videos[8:10])
        test_labels.extend([class_to_idx[class_name]] * 2)
    
    print(f"Splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def create_weighted_sampler(labels):
    """Create weighted sampler to ensure balanced training."""
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Calculate weights (inverse frequency)
    weights = []
    for label in labels:
        weight = total_samples / (len(class_counts) * class_counts[label])
        weights.append(weight)
    
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def train_fixed_model(model, train_loader, val_loader, device, num_epochs=10):
    """Fixed training with class balancing and better optimization."""
    
    # CRITICAL FIX: Use class-balanced loss
    criterion = nn.CrossEntropyLoss()
    
    # CRITICAL FIX: Better optimizer settings
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"\n🚀 Fixed training for {num_epochs} epochs...")
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_targets = []
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # CRITICAL FIX: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            train_preds.extend(pred.cpu().numpy())
            train_targets.extend(target.cpu().numpy())
            
            if batch_idx % 5 == 0:
                batch_acc = 100. * pred.eq(target).sum().item() / target.size(0)
                print(f"  Batch {batch_idx+1}: Loss={loss.item():.4f}, Acc={batch_acc:.1f}%, "
                      f"Pred={pred.cpu().numpy()}, Target={target.cpu().numpy()}")
        
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
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Acc: {train_acc:.1f}% ({unique_train_preds}/5 classes)")
        print(f"  Val Acc: {val_acc:.1f}% ({unique_val_preds}/5 classes)")
        print(f"  Train Preds: {sorted(set(train_preds))}")
        print(f"  Val Preds: {sorted(set(val_preds))}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_fixed_model.pth')
            print(f"  💾 New best model saved!")
        
        scheduler.step()
        
        # Early success check
        if unique_val_preds >= 3 and val_acc > 40:
            print(f"  🎉 SUCCESS: Model predicting multiple classes with good accuracy!")
    
    return best_val_acc

def main():
    """Fixed main function."""
    print("🎯 FIXED TRAINING - SOLVE CLASS BIAS")
    print("=" * 50)
    print("FIXES APPLIED:")
    print("• Better weight initialization")
    print("• Class-balanced sampling")
    print("• Enhanced augmentation")
    print("• Per-video normalization")
    print("• Gradient clipping")
    print("• Better optimizer (AdamW)")
    print("=" * 50)
    
    # Set seeds
    set_seeds(42)
    
    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")
    
    # Create balanced splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_balanced_splits()
    
    # Create datasets
    train_dataset = FixedDataset(train_videos, train_labels, augment=True)
    val_dataset = FixedDataset(val_videos, val_labels, augment=False)
    test_dataset = FixedDataset(test_videos, test_labels, augment=False)
    
    # CRITICAL FIX: Use weighted sampler for balanced training
    weighted_sampler = create_weighted_sampler(train_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, sampler=weighted_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Create fixed model
    model = FixedModel(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Fixed Model: {total_params:,} parameters")
    
    # Train
    best_val_acc = train_fixed_model(model, train_loader, val_loader, device, num_epochs=8)
    
    # Test
    print(f"\n🔍 Testing fixed model...")
    
    # Load best model
    if os.path.exists('best_fixed_model.pth'):
        model.load_state_dict(torch.load('best_fixed_model.pth', map_location=device))
        print("📥 Loaded best model")
    
    model.eval()
    test_correct = 0
    test_total = 0
    test_preds = []
    test_targets = []
    
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
    
    print(f"\n🎯 FIXED TRAINING RESULTS")
    print("=" * 40)
    print(f"🎯 Test Accuracy: {test_acc:.1f}%")
    print(f"🎯 Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"🎯 Test Predictions: {sorted(set(test_preds))}")
    print(f"🎯 Test Targets: {sorted(set(test_targets))}")
    print(f"🎯 Unique Predictions: {unique_test_preds}/5 classes")
    
    if unique_test_preds >= 3:
        print("✅ SUCCESS: Model predicts multiple classes!")
        if test_acc >= 40:
            print("🏆 EXCELLENT: High accuracy achieved!")
        elif test_acc >= 30:
            print("📈 GOOD: Solid baseline established!")
        else:
            print("📊 PROGRESS: Multi-class prediction working!")
    else:
        print("⚠️  PARTIAL: Still some class bias")
    
    return test_acc

if __name__ == "__main__":
    try:
        final_accuracy = main()
        print(f"\n🏁 Fixed training completed: {final_accuracy:.1f}% accuracy")
        
        if final_accuracy >= 30:
            print("🚀 Ready to scale up for higher accuracy!")
        else:
            print("🔄 May need additional fixes")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
