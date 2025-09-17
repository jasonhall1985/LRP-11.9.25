#!/usr/bin/env python3
"""
Breakthrough System V10 Hybrid - Best of All Systems for 80% Target
Combines the best insights from V5, V6, V7, V8, V9 for maximum performance
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
from sklearn.model_selection import StratifiedKFold

def set_seeds(seed=53):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class HybridDataset(Dataset):
    """Hybrid dataset combining best practices from all systems."""
    def __init__(self, video_paths, labels, augment=False, phase='train'):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        self.phase = phase
        
        print(f"📊 Hybrid Dataset ({phase}): {len(video_paths)} videos, Augment: {augment}")
        label_counts = Counter(labels)
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        for i, name in enumerate(class_names):
            print(f"   {name}: {label_counts.get(i, 0)} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_hybrid(self, path):
        """Hybrid video loading with best practices."""
        cap = cv2.VideoCapture(path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # V5's proven ICU-style crop
            h, w = gray.shape
            crop_h = int(0.5 * h)
            crop_w_start = int(0.335 * w)
            crop_w_end = int(0.665 * w)
            
            cropped = gray[0:crop_h, crop_w_start:crop_w_end]
            # Optimal resolution from V6/V7 success
            resized = cv2.resize(cropped, (112, 112))  # Between 96 and 128
            frames.append(resized)
        
        cap.release()
        
        if len(frames) == 0:
            frames = [np.zeros((112, 112), dtype=np.uint8)]
        
        # V5's proven 32-frame sampling
        target_frames = 32
        if len(frames) >= target_frames:
            indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < target_frames:
                frames.extend(frames[:min(len(frames), target_frames - len(frames))])
        
        return np.array(frames[:target_frames])
    
    def apply_hybrid_augmentation(self, frames):
        """Hybrid augmentation combining best techniques."""
        if not self.augment:
            return frames
        
        # V5's proven core augmentations
        if random.random() < 0.5:
            frames = np.flip(frames, axis=2).copy()
        
        if random.random() < 0.3:
            brightness_factor = random.uniform(0.85, 1.15)
            frames = np.clip(frames * brightness_factor, 0, 255).astype(np.uint8)
        
        if random.random() < 0.2:
            contrast_factor = random.uniform(0.9, 1.1)
            frames = np.clip((frames - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
        
        # V9's geometric augmentations (reduced probability)
        if random.random() < 0.15:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            h, w = frames.shape[1], frames.shape[2]
            translated_frames = []
            for frame in frames:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                translated = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                translated_frames.append(translated)
            frames = np.array(translated_frames)
        
        return frames
    
    def apply_v5_preprocessing(self, frames):
        """V5's exact preprocessing - the proven foundation."""
        frames = frames.astype(np.float32) / 255.0
        
        processed_frames = []
        for frame in frames:
            frame_uint8 = (frame * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(frame_uint8).astype(np.float32) / 255.0
            
            p2, p98 = np.percentile(enhanced, [2, 98])
            if p98 > p2:
                enhanced = np.clip((enhanced - p2) / (p98 - p2), 0, 1)
            
            gamma = 1.2
            enhanced = np.power(enhanced, 1.0 / gamma)
            
            target_brightness = 0.5
            current_brightness = np.mean(enhanced)
            if current_brightness > 0:
                brightness_factor = target_brightness / current_brightness
                enhanced = np.clip(enhanced * brightness_factor, 0, 1)
            
            processed_frames.append(enhanced)
        
        frames = np.array(processed_frames)
        frames = (frames - 0.5) / 0.5
        
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self.load_video_hybrid(video_path)
        frames = self.apply_hybrid_augmentation(frames)
        frames = self.apply_v5_preprocessing(frames)
        frames = torch.from_numpy(frames).float()
        
        return frames, label

class HybridModel(nn.Module):
    """Hybrid model combining best architectures from successful systems."""
    def __init__(self, num_classes=5):
        super(HybridModel, self).__init__()
        
        # V6/V7 proven architecture with optimal size
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3 - V6 Fold 2 success pattern
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 4 - V7 Specialist 4 success pattern
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4)
        )
        
        # Hybrid temporal processing
        self.temporal = nn.Sequential(
            nn.Linear(512 * 16, 1024),  # Large but not ultra-large
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # V5 proven classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B, T, H, W = x.shape
        
        # Process each frame
        x = x.view(B * T, 1, H, W)
        features = self.features(x)  # (B*T, 512, 4, 4)
        features = features.view(B * T, -1)  # (B*T, 8192)
        features = features.view(B, T, -1)  # (B, T, 8192)
        
        # Temporal aggregation
        temporal_features = features.mean(dim=1)  # (B, 8192)
        
        # Process temporally
        processed = self.temporal(temporal_features)  # (B, 256)
        
        # Classify
        output = self.classifier(processed)  # (B, num_classes)
        
        return output

def create_hybrid_splits(dataset_path="the_best_videos_so_far"):
    """Create splits for hybrid training."""
    print("📊 Creating hybrid splits from FULL dataset...")
    
    video_files = list(Path(dataset_path).glob("*.mp4"))
    video_files = [f for f in video_files if "copy" not in f.name]
    
    print(f"Found {len(video_files)} videos (after removing duplicates)")
    
    video_paths = []
    labels = []
    
    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
    
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
        
        video_paths.append(str(video_file))
        labels.append(class_to_idx[class_name])
    
    # Print class distribution
    label_counts = Counter(labels)
    class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
    for i, name in enumerate(class_names):
        print(f"   {name}: {label_counts.get(i, 0)} videos")
    
    video_paths = np.array(video_paths)
    labels = np.array(labels)
    
    # V5's proven stratified split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=53)
    train_val_idx, test_idx = next(skf.split(video_paths, labels))
    
    test_videos = video_paths[test_idx].tolist()
    test_labels = labels[test_idx].tolist()
    
    train_val_videos = video_paths[train_val_idx]
    train_val_labels = labels[train_val_idx]
    
    # Split train+val
    skf2 = StratifiedKFold(n_splits=4, shuffle=True, random_state=53)
    train_idx, val_idx = next(skf2.split(train_val_videos, train_val_labels))
    
    train_videos = train_val_videos[train_idx].tolist()
    train_labels = train_val_labels[train_idx].tolist()
    
    val_videos = train_val_videos[val_idx].tolist()
    val_labels = train_val_labels[val_idx].tolist()
    
    print(f"📊 Hybrid splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def train_hybrid_model(model, train_loader, val_loader, device, num_epochs=40):
    """Train hybrid model with best practices from all systems."""

    # V6/V7 proven optimization
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=8e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.65, patience=6)

    # V8 proven loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n🚀 Hybrid training for {num_epochs} epochs...")

    best_val_acc = 0.0
    patience = 0
    max_patience = 12

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

            # V5 proven gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        unique_train_preds = len(set(train_preds))
        unique_val_preds = len(set(val_preds))

        print(f"Hybrid E{epoch+1}/{num_epochs} - "
              f"Train: {train_acc:.1f}% ({unique_train_preds}/5), "
              f"Val: {val_acc:.1f}% ({unique_val_preds}/5), "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), 'breakthrough_v10_hybrid.pth')
            print(f"  💾 Hybrid New best: {val_acc:.1f}%")
        else:
            patience += 1

        if patience >= max_patience:
            print(f"  ⏹️  Hybrid Early stopping")
            break

        # Success milestones
        if unique_val_preds >= 4 and val_acc >= 60:
            print(f"  🎉 HYBRID BREAKTHROUGH: 60%+ with 4+ classes!")
            if val_acc >= 70:
                print(f"  🏆 HYBRID EXCELLENT: 70%+ achieved!")
                if val_acc >= 80:
                    print(f"  🌟 HYBRID TARGET: 80%+!")
                    break

    return best_val_acc

def main():
    """Hybrid system combining best of all approaches for 80% target."""
    print("🎯 BREAKTHROUGH SYSTEM V10 - HYBRID BEST-OF-ALL FOR 80% TARGET")
    print("=" * 80)
    print("HYBRID TECHNIQUES:")
    print("• V5's proven preprocessing and stratified splits")
    print("• V6 Fold 2's successful architecture (1.86M params)")
    print("• V7 Specialist 4's optimization strategies")
    print("• V8's label smoothing and learning rates")
    print("• V9's geometric augmentations (reduced)")
    print("• Optimal 112x112 resolution and 32-frame sampling")
    print("• Best practices from all breakthrough systems")
    print("• Relentless pursuit of 80% generalization")
    print("=" * 80)

    set_seeds(53)
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")

    # Create hybrid splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_hybrid_splits()

    # Create datasets
    train_dataset = HybridDataset(train_videos, train_labels, augment=True, phase='train')
    val_dataset = HybridDataset(val_videos, val_labels, augment=False, phase='val')
    test_dataset = HybridDataset(test_videos, test_labels, augment=False, phase='test')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Create hybrid model
    model = HybridModel(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Hybrid Model: {total_params:,} parameters")

    # Train hybrid model
    best_val_acc = train_hybrid_model(model, train_loader, val_loader, device, num_epochs=40)

    # Test hybrid model
    if os.path.exists('breakthrough_v10_hybrid.pth'):
        model.load_state_dict(torch.load('breakthrough_v10_hybrid.pth', map_location=device))

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

    print(f"\n🎯 BREAKTHROUGH SYSTEM V10 HYBRID RESULTS")
    print("=" * 70)
    print(f"🎯 Hybrid Best-of-All Performance:")
    print(f"   Parameters: {total_params:,}")
    print(f"   Validation: {best_val_acc:.1f}%")
    print(f"   Test: {test_acc:.1f}% ({unique_test_preds}/5 classes)")

    # Compare with previous best
    previous_best = 36.8
    if test_acc > previous_best:
        improvement = test_acc - previous_best
        print(f"🏆 NEW RECORD: +{improvement:.1f}% improvement over previous best!")

    if test_acc >= 80:
        print("🌟 HYBRID TARGET ACHIEVED: 80%+ generalization!")
    elif test_acc >= 70:
        print("🏆 HYBRID EXCELLENT: 70%+ achieved!")
    elif test_acc >= 60:
        print("🎉 HYBRID GREAT: 60%+ achieved!")
    elif test_acc >= 40:
        print("✅ HYBRID GOOD: 40%+ achieved!")
    else:
        print("🔄 Continue toward 80% target...")

    return test_acc, best_val_acc

if __name__ == "__main__":
    try:
        test_accuracy, val_accuracy = main()
        print(f"\n🏁 Breakthrough V10 Hybrid completed:")
        print(f"   Test: {test_accuracy:.1f}%")
        print(f"   Validation: {val_accuracy:.1f}%")

        if test_accuracy >= 80:
            print("🎯 80% TARGET ACHIEVED!")
        else:
            print("🚀 Continue relentless pursuit of 80%...")

    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
