#!/usr/bin/env python3
"""
Breakthrough System V9 Ultra-Large - Maximum Parameters for 80% Target
Ultra-large model with maximum capacity for complex lip-reading patterns
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

def set_seeds(seed=52):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class UltraDataset(Dataset):
    """Ultra dataset with V5 foundation + maximum augmentation."""
    def __init__(self, video_paths, labels, augment=False, phase='train'):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        self.phase = phase
        
        print(f"📊 Ultra Dataset ({phase}): {len(video_paths)} videos, Augment: {augment}")
        label_counts = Counter(labels)
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        for i, name in enumerate(class_names):
            print(f"   {name}: {label_counts.get(i, 0)} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_ultra(self, path):
        """Ultra video loading with V5 foundation."""
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
            # Larger resolution for ultra model
            resized = cv2.resize(cropped, (128, 128))  # Increased from 96x96
            frames.append(resized)
        
        cap.release()
        
        if len(frames) == 0:
            frames = [np.zeros((128, 128), dtype=np.uint8)]
        
        # V5's proven 32-frame sampling
        target_frames = 32
        if len(frames) >= target_frames:
            indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < target_frames:
                frames.extend(frames[:min(len(frames), target_frames - len(frames))])
        
        return np.array(frames[:target_frames])
    
    def apply_ultra_augmentation(self, frames):
        """Ultra augmentation for maximum data diversity."""
        if not self.augment:
            return frames
        
        # V5's proven augmentations
        if random.random() < 0.5:
            frames = np.flip(frames, axis=2).copy()
        
        if random.random() < 0.3:
            brightness_factor = random.uniform(0.85, 1.15)
            frames = np.clip(frames * brightness_factor, 0, 255).astype(np.uint8)
        
        if random.random() < 0.2:
            contrast_factor = random.uniform(0.9, 1.1)
            frames = np.clip((frames - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
        
        # Ultra augmentations
        if random.random() < 0.2:
            # Translation
            dx = random.randint(-4, 4)
            dy = random.randint(-4, 4)
            h, w = frames.shape[1], frames.shape[2]
            translated_frames = []
            for frame in frames:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                translated = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                translated_frames.append(translated)
            frames = np.array(translated_frames)
        
        if random.random() < 0.15:
            # Rotation
            angle = random.uniform(-4, 4)
            h, w = frames.shape[1], frames.shape[2]
            center = (w // 2, h // 2)
            rotated_frames = []
            for frame in frames:
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                rotated_frames.append(rotated)
            frames = np.array(rotated_frames)
        
        if random.random() < 0.1:
            # Scaling
            scale_factor = random.uniform(0.95, 1.05)
            h, w = frames.shape[1], frames.shape[2]
            center = (w // 2, h // 2)
            scaled_frames = []
            for frame in frames:
                M = cv2.getRotationMatrix2D(center, 0, scale_factor)
                scaled = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                scaled_frames.append(scaled)
            frames = np.array(scaled_frames)
        
        return frames
    
    def apply_v5_preprocessing(self, frames):
        """V5's exact preprocessing."""
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
        
        frames = self.load_video_ultra(video_path)
        frames = self.apply_ultra_augmentation(frames)
        frames = self.apply_v5_preprocessing(frames)
        frames = torch.from_numpy(frames).float()
        
        return frames, label

class UltraLargeModel(nn.Module):
    """Ultra-large model with maximum parameters for complex patterns."""
    def __init__(self, num_classes=5):
        super(UltraLargeModel, self).__init__()
        
        # Ultra-large feature extraction
        self.features = nn.Sequential(
            # Block 1 - Large initial features
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2 - Deep feature extraction
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3 - Ultra-wide features
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 4 - Maximum capacity
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Block 5 - Final features
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4)  # 4x4 spatial features
        )
        
        # Ultra-large temporal processing
        self.temporal = nn.Sequential(
            nn.Linear(1024 * 16, 2048),  # Ultra-large hidden layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Ultra classifier
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
        features = self.features(x)  # (B*T, 1024, 4, 4)
        features = features.view(B * T, -1)  # (B*T, 16384)
        features = features.view(B, T, -1)  # (B, T, 16384)
        
        # Temporal aggregation
        temporal_features = features.mean(dim=1)  # (B, 16384)
        
        # Ultra temporal processing
        processed = self.temporal(temporal_features)  # (B, 256)
        
        # Classify
        output = self.classifier(processed)  # (B, num_classes)
        
        return output

def create_ultra_splits(dataset_path="the_best_videos_so_far"):
    """Create splits for ultra training."""
    print("📊 Creating ultra splits from FULL dataset...")
    
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
    
    # Stratified split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=52)
    train_val_idx, test_idx = next(skf.split(video_paths, labels))
    
    test_videos = video_paths[test_idx].tolist()
    test_labels = labels[test_idx].tolist()
    
    train_val_videos = video_paths[train_val_idx]
    train_val_labels = labels[train_val_idx]
    
    # Split train+val
    skf2 = StratifiedKFold(n_splits=4, shuffle=True, random_state=52)
    train_idx, val_idx = next(skf2.split(train_val_videos, train_val_labels))
    
    train_videos = train_val_videos[train_idx].tolist()
    train_labels = train_val_labels[train_idx].tolist()
    
    val_videos = train_val_videos[val_idx].tolist()
    val_labels = train_val_labels[val_idx].tolist()
    
    print(f"📊 Ultra splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def train_ultra_model(model, train_loader, val_loader, device, num_epochs=35):
    """Train ultra-large model with advanced optimization."""

    # Ultra optimization
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=5)

    # Advanced loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

    print(f"\n🚀 Ultra training for {num_epochs} epochs...")

    best_val_acc = 0.0
    patience = 0
    max_patience = 10

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

            # Ultra gradient clipping
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

        unique_train_preds = len(set(train_preds))
        unique_val_preds = len(set(val_preds))

        print(f"Ultra E{epoch+1}/{num_epochs} - "
              f"Train: {train_acc:.1f}% ({unique_train_preds}/5), "
              f"Val: {val_acc:.1f}% ({unique_val_preds}/5), "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), 'breakthrough_v9_ultra.pth')
            print(f"  💾 Ultra New best: {val_acc:.1f}%")
        else:
            patience += 1

        if patience >= max_patience:
            print(f"  ⏹️  Ultra Early stopping")
            break

        # Success milestones
        if unique_val_preds >= 4 and val_acc >= 60:
            print(f"  🎉 ULTRA BREAKTHROUGH: 60%+ with 4+ classes!")
            if val_acc >= 70:
                print(f"  🏆 ULTRA EXCELLENT: 70%+ achieved!")
                if val_acc >= 80:
                    print(f"  🌟 ULTRA TARGET: 80%+!")
                    break

    return best_val_acc

def main():
    """Ultra-large system for 80% target."""
    print("🎯 BREAKTHROUGH SYSTEM V9 - ULTRA-LARGE FOR 80% TARGET")
    print("=" * 80)
    print("ULTRA-LARGE TECHNIQUES:")
    print("• Maximum parameter count (5M+ parameters)")
    print("• Ultra-deep CNN with 1024 feature channels")
    print("• 128x128 resolution for maximum detail")
    print("• Ultra-large temporal processing (2048 hidden)")
    print("• Advanced augmentation with geometric transforms")
    print("• V5's proven preprocessing foundation")
    print("• Relentless pursuit of 80% generalization")
    print("=" * 80)

    set_seeds(52)
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")

    # Create ultra splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_ultra_splits()

    # Create datasets
    train_dataset = UltraDataset(train_videos, train_labels, augment=True, phase='train')
    val_dataset = UltraDataset(val_videos, val_labels, augment=False, phase='val')
    test_dataset = UltraDataset(test_videos, test_labels, augment=False, phase='test')

    # Create data loaders (batch_size=1 for ultra-large model)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Create ultra-large model
    model = UltraLargeModel(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Ultra-Large Model: {total_params:,} parameters")

    # Train ultra model
    best_val_acc = train_ultra_model(model, train_loader, val_loader, device, num_epochs=35)

    # Test ultra model
    if os.path.exists('breakthrough_v9_ultra.pth'):
        model.load_state_dict(torch.load('breakthrough_v9_ultra.pth', map_location=device))

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

    print(f"\n🎯 BREAKTHROUGH SYSTEM V9 ULTRA-LARGE RESULTS")
    print("=" * 70)
    print(f"🎯 Ultra-Large Model Performance:")
    print(f"   Parameters: {total_params:,}")
    print(f"   Validation: {best_val_acc:.1f}%")
    print(f"   Test: {test_acc:.1f}% ({unique_test_preds}/5 classes)")

    if test_acc >= 80:
        print("🌟 ULTRA TARGET ACHIEVED: 80%+ generalization!")
    elif test_acc >= 70:
        print("🏆 ULTRA EXCELLENT: 70%+ achieved!")
    elif test_acc >= 60:
        print("🎉 ULTRA GREAT: 60%+ achieved!")
    else:
        print("🔄 Continue toward 80% target...")

    return test_acc, best_val_acc

if __name__ == "__main__":
    try:
        test_accuracy, val_accuracy = main()
        print(f"\n🏁 Breakthrough V9 Ultra-Large completed:")
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
