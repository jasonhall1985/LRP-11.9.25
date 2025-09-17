#!/usr/bin/env python3
"""
Full Dataset Training - Use ALL 100 Videos
Finally use the complete dataset for maximum performance
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
from sklearn.metrics import classification_report

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class FullDataset(Dataset):
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
    
    def load_video_full(self, path):
        """Load video with full preprocessing."""
        cap = cv2.VideoCapture(path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize to 112x112
            resized = cv2.resize(gray, (112, 112))
            frames.append(resized)
        
        cap.release()
        
        # Take exactly 32 frames
        if len(frames) >= 32:
            indices = np.linspace(0, len(frames)-1, 32, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < 32:
                frames.append(frames[-1] if frames else np.zeros((112, 112), dtype=np.uint8))
        
        return np.array(frames[:32])
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video
        frames = self.load_video_full(video_path)
        
        # Strong augmentation for training
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
            
            # Random spatial translation
            if random.random() < 0.2:
                dx, dy = random.randint(-3, 3), random.randint(-3, 3)
                frames = np.roll(frames, (dy, dx), axis=(1, 2))
        
        # Normalize
        frames = frames.astype(np.float32) / 255.0
        
        # Per-video normalization
        mean = frames.mean()
        std = frames.std()
        if std > 0.01:
            frames = (frames - mean) / std
            frames = frames * 0.3 + 0.5
            frames = np.clip(frames, 0, 1)
        
        # Convert to tensor (C, T, H, W)
        frames = torch.from_numpy(frames).unsqueeze(0)
        
        return frames, label

class FullModel(nn.Module):
    def __init__(self, num_classes=5):
        super(FullModel, self).__init__()
        
        # Enhanced 3D CNN for full dataset
        self.conv1 = nn.Conv3d(1, 32, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(32)
        
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), stride=(2, 2, 2), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)
        
        self.conv3 = nn.Conv3d(64, 128, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        
        self.conv4 = nn.Conv3d(128, 256, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(256)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Enhanced classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Better initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def create_full_dataset_splits(dataset_path="the_best_videos_so_far"):
    """Create splits using the FULL dataset."""
    print("📊 Creating splits from FULL dataset...")
    
    video_files = list(Path(dataset_path).glob("*.mp4"))
    
    # Remove duplicate files (those with "copy" in name)
    video_files = [f for f in video_files if "copy" not in f.name]
    
    print(f"Found {len(video_files)} videos (after removing duplicates)")
    
    # Group by class
    class_videos = {'doctor': [], 'glasses': [], 'help': [], 'phone': [], 'pillow': []}
    
    for video_file in video_files:
        # Extract class name from filename
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
    
    # Create splits: 70% train, 15% val, 15% test
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []
    
    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
    
    for class_name, videos in class_videos.items():
        random.shuffle(videos)
        n_videos = len(videos)
        
        # 70% train, 15% val, 15% test
        n_train = int(0.7 * n_videos)
        n_val = int(0.15 * n_videos)
        
        train_videos.extend(videos[:n_train])
        train_labels.extend([class_to_idx[class_name]] * n_train)
        
        val_videos.extend(videos[n_train:n_train+n_val])
        val_labels.extend([class_to_idx[class_name]] * n_val)
        
        test_videos.extend(videos[n_train+n_val:])
        test_labels.extend([class_to_idx[class_name]] * (len(videos) - n_train - n_val))
    
    print(f"📊 Final splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def train_full_model(model, train_loader, val_loader, device, num_epochs=15):
    """Train with full dataset."""
    
    # Advanced training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=num_epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    
    print(f"\n🚀 Full dataset training for {num_epochs} epochs...")
    
    best_val_acc = 0.0
    patience = 0
    max_patience = 5
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            train_preds.extend(pred.cpu().numpy())
            train_targets.extend(target.cpu().numpy())
            
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
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
              f"Val: {val_acc:.1f}% ({unique_val_preds}/5), "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), 'best_full_model.pth')
            print(f"  💾 New best: {val_acc:.1f}%")
        else:
            patience += 1
        
        # Early stopping
        if patience >= max_patience:
            print(f"  ⏹️  Early stopping")
            break
        
        # Success check
        if unique_val_preds >= 4 and val_acc >= 60:
            print(f"  🎉 TARGET ACHIEVED!")
            break
    
    return best_val_acc

def main():
    """Full dataset training."""
    print("🎯 FULL DATASET TRAINING - ALL 100 VIDEOS")
    print("=" * 60)
    print("USING COMPLETE DATASET:")
    print("• ~100 videos total (20 per class)")
    print("• Enhanced 4-layer 3D CNN")
    print("• Advanced training techniques")
    print("• Target: 50%+ accuracy")
    print("=" * 60)
    
    # Set seeds
    set_seeds(42)
    
    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")
    
    # Create full dataset splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_full_dataset_splits()
    
    # Create datasets
    train_dataset = FullDataset(train_videos, train_labels, augment=True)
    val_dataset = FullDataset(val_videos, val_labels, augment=False)
    test_dataset = FullDataset(test_videos, test_labels, augment=False)
    
    # Weighted sampler for balanced training
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    # Create data loaders with larger batch size (more data!)
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Create full model
    model = FullModel(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Full Model: {total_params:,} parameters")
    
    # Train
    best_val_acc = train_full_model(model, train_loader, val_loader, device, num_epochs=12)
    
    # Test
    print(f"\n🔍 Testing full model...")
    
    if os.path.exists('best_full_model.pth'):
        model.load_state_dict(torch.load('best_full_model.pth', map_location=device))
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
    
    # Classification report
    if len(set(test_targets)) > 1:
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        report = classification_report(test_targets, test_preds, target_names=class_names, zero_division=0)
        print(f"📊 Classification Report:\n{report}")
    
    print(f"\n🎯 FULL DATASET RESULTS")
    print("=" * 50)
    print(f"🎯 Test Accuracy: {test_acc:.1f}%")
    print(f"🎯 Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"🎯 Test Predictions: {sorted(set(test_preds))}")
    print(f"🎯 Test Targets: {sorted(set(test_targets))}")
    print(f"🎯 Unique Predictions: {unique_test_preds}/5 classes")
    print(f"🎯 Total Training Videos: {len(train_videos)}")
    
    if test_acc >= 60:
        print("🏆 EXCELLENT: 60%+ accuracy achieved!")
    elif test_acc >= 50:
        print("✅ SUCCESS: 50%+ accuracy achieved!")
    elif test_acc >= 40:
        print("📈 GOOD: Strong improvement with full dataset!")
    elif unique_test_preds >= 4:
        print("📊 PROGRESS: Multi-class prediction working!")
    else:
        print("⚠️  NEEDS MORE WORK")
    
    return test_acc

if __name__ == "__main__":
    try:
        final_accuracy = main()
        print(f"\n🏁 Full dataset training completed: {final_accuracy:.1f}% accuracy")
        
        if final_accuracy >= 50:
            print("🚀 Excellent! Ready for advanced techniques!")
        else:
            print("🔄 Continue optimization with full dataset")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
