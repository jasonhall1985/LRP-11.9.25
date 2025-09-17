#!/usr/bin/env python3
"""
Full Dataset Training - IMPROVED VERSION
Better regularization and training strategies
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

class ImprovedDataset(Dataset):
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
    
    def load_video_robust(self, path):
        """Load video with robust preprocessing."""
        cap = cv2.VideoCapture(path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize to 96x96 (smaller for better generalization)
            resized = cv2.resize(gray, (96, 96))
            frames.append(resized)
        
        cap.release()
        
        # Take exactly 32 frames
        if len(frames) >= 32:
            indices = np.linspace(0, len(frames)-1, 32, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < 32:
                frames.append(frames[-1] if frames else np.zeros((96, 96), dtype=np.uint8))
        
        return np.array(frames[:32])
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video
        frames = self.load_video_robust(video_path)
        
        # Conservative augmentation to prevent overfitting
        if self.augment:
            # Light brightness/contrast (reduced range)
            if random.random() < 0.3:
                brightness = random.uniform(0.9, 1.1)
                contrast = random.uniform(0.9, 1.1)
                frames = np.clip(frames * contrast + (brightness - 1) * 128, 0, 255).astype(np.uint8)
            
            # Random horizontal flip
            if random.random() < 0.5:
                frames = np.flip(frames, axis=2).copy()
            
            # Very light temporal jitter
            if random.random() < 0.2:
                jitter = random.randint(-1, 1)
                if jitter > 0:
                    frames = frames[jitter:]
                    frames = np.pad(frames, ((0, jitter), (0, 0), (0, 0)), mode='edge')
                elif jitter < 0:
                    frames = frames[:jitter]
                    frames = np.pad(frames, ((-jitter, 0), (0, 0), (0, 0)), mode='edge')
        
        # Normalize
        frames = frames.astype(np.float32) / 255.0
        
        # Global normalization (not per-video to prevent overfitting)
        frames = (frames - 0.5) / 0.5  # [-1, 1] range
        
        # Convert to tensor (C, T, H, W)
        frames = torch.from_numpy(frames).unsqueeze(0)
        
        return frames, label

class RobustModel(nn.Module):
    def __init__(self, num_classes=5):
        super(RobustModel, self).__init__()
        
        # Smaller, more regularized model
        self.conv1 = nn.Conv3d(1, 16, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(16)
        self.dropout1 = nn.Dropout3d(0.1)
        
        self.conv2 = nn.Conv3d(16, 32, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)
        self.dropout2 = nn.Dropout3d(0.2)
        
        self.conv3 = nn.Conv3d(32, 64, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(64)
        self.dropout3 = nn.Dropout3d(0.3)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Simple classifier to prevent overfitting
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
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
                nn.init.normal_(m.weight, 0, 0.01)  # Smaller weights
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def create_balanced_splits(dataset_path="the_best_videos_so_far"):
    """Create balanced splits from full dataset."""
    print("📊 Creating balanced splits from FULL dataset...")
    
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
    
    # Create balanced splits: use minimum class size for balance
    min_class_size = min(len(videos) for videos in class_videos.values())
    print(f"📊 Balancing to {min_class_size} videos per class")
    
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []
    
    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
    
    for class_name, videos in class_videos.items():
        random.shuffle(videos)
        # Use all available videos, not just min_class_size
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

def train_robust_model(model, train_loader, val_loader, device, num_epochs=20):
    """Train with robust techniques."""
    
    # Conservative training setup
    criterion = nn.CrossEntropyLoss()  # No label smoothing initially
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)  # Lower LR, higher weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    print(f"\n🚀 Robust training for {num_epochs} epochs...")
    
    best_val_acc = 0.0
    patience = 0
    max_patience = 7
    
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
            
            # Light gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            train_loss += loss.item()
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
        
        # Update scheduler
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), 'best_robust_model.pth')
            print(f"  💾 New best: {val_acc:.1f}%")
        else:
            patience += 1
        
        # Early stopping
        if patience >= max_patience:
            print(f"  ⏹️  Early stopping")
            break
        
        # Success check
        if unique_val_preds >= 4 and val_acc >= 40:
            print(f"  🎉 GOOD PROGRESS!")
            if val_acc >= 50:
                print(f"  🏆 TARGET ACHIEVED!")
                break
    
    return best_val_acc

def main():
    """Robust full dataset training."""
    print("🎯 IMPROVED FULL DATASET TRAINING")
    print("=" * 60)
    print("IMPROVEMENTS:")
    print("• Better regularization (dropout, weight decay)")
    print("• Smaller model to prevent overfitting")
    print("• Conservative augmentation")
    print("• Balanced training approach")
    print("• Target: 40%+ accuracy with good generalization")
    print("=" * 60)
    
    # Set seeds
    set_seeds(42)
    
    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")
    
    # Create balanced dataset splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_balanced_splits()
    
    # Create datasets
    train_dataset = ImprovedDataset(train_videos, train_labels, augment=True)
    val_dataset = ImprovedDataset(val_videos, val_labels, augment=False)
    test_dataset = ImprovedDataset(test_videos, test_labels, augment=False)
    
    # Weighted sampler for balanced training
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Create robust model
    model = RobustModel(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Robust Model: {total_params:,} parameters")
    
    # Train
    best_val_acc = train_robust_model(model, train_loader, val_loader, device, num_epochs=15)
    
    # Test
    print(f"\n🔍 Testing robust model...")
    
    if os.path.exists('best_robust_model.pth'):
        model.load_state_dict(torch.load('best_robust_model.pth', map_location=device))
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
    
    print(f"\n🎯 IMPROVED RESULTS")
    print("=" * 50)
    print(f"🎯 Test Accuracy: {test_acc:.1f}%")
    print(f"🎯 Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"🎯 Test Predictions: {sorted(set(test_preds))}")
    print(f"🎯 Test Targets: {sorted(set(test_targets))}")
    print(f"🎯 Unique Predictions: {unique_test_preds}/5 classes")
    print(f"🎯 Total Training Videos: {len(train_videos)}")
    
    if test_acc >= 50:
        print("🏆 EXCELLENT: 50%+ accuracy achieved!")
    elif test_acc >= 40:
        print("✅ SUCCESS: 40%+ accuracy achieved!")
    elif test_acc >= 30:
        print("📈 GOOD: Solid improvement!")
    elif unique_test_preds >= 4:
        print("📊 PROGRESS: Multi-class prediction working!")
    else:
        print("⚠️  Continue optimization")
    
    return test_acc

if __name__ == "__main__":
    try:
        final_accuracy = main()
        print(f"\n🏁 Improved training completed: {final_accuracy:.1f}% accuracy")
        
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
