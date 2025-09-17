#!/usr/bin/env python3
"""
Breakthrough System V16 Extreme - Maximum Aggressive Push to 80% Target
The most extreme and aggressive approach combining all techniques with maximum intensity
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
from sklearn.model_selection import StratifiedKFold

def set_seeds(seed=59):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class ExtremeDataset(Dataset):
    """Extreme dataset with maximum aggressive augmentation."""
    def __init__(self, video_paths, labels, augment=False, phase='train'):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        self.phase = phase
        
        print(f"📊 Extreme Dataset ({phase}): {len(video_paths)} videos, Augment: {augment}")
        label_counts = Counter(labels)
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        for i, name in enumerate(class_names):
            print(f"   {name}: {label_counts.get(i, 0)} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_extreme(self, path):
        """Extreme video loading with V5 foundation."""
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
            # Extreme resolution - maximum detail
            resized = cv2.resize(cropped, (128, 128))  # Largest resolution yet
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
    
    def apply_extreme_augmentation(self, frames):
        """EXTREME augmentation - maximum intensity for robustness."""
        if not self.augment:
            return frames
        
        # V5's proven core augmentations - EXTREME INTENSITY
        if random.random() < 0.7:  # Higher probability
            frames = np.flip(frames, axis=2).copy()
        
        if random.random() < 0.6:  # EXTREME brightness variation
            brightness_factor = random.uniform(0.7, 1.3)
            frames = np.clip(frames * brightness_factor, 0, 255).astype(np.uint8)
        
        if random.random() < 0.5:  # EXTREME contrast variation
            contrast_factor = random.uniform(0.8, 1.2)
            frames = np.clip((frames - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
        
        # EXTREME geometric augmentations
        if random.random() < 0.4:
            dx = random.randint(-5, 5)  # Larger shifts
            dy = random.randint(-5, 5)
            h, w = frames.shape[1], frames.shape[2]
            translated_frames = []
            for frame in frames:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                translated = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                translated_frames.append(translated)
            frames = np.array(translated_frames)
        
        # EXTREME rotation
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)  # Larger angles
            h, w = frames.shape[1], frames.shape[2]
            center = (w // 2, h // 2)
            rotated_frames = []
            for frame in frames:
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                rotated_frames.append(rotated)
            frames = np.array(rotated_frames)
        
        # EXTREME scaling
        if random.random() < 0.25:
            scale = random.uniform(0.9, 1.1)  # Larger scale variation
            h, w = frames.shape[1], frames.shape[2]
            scaled_frames = []
            for frame in frames:
                M = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
                scaled = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                scaled_frames.append(scaled)
            frames = np.array(scaled_frames)
        
        # EXTREME noise injection
        if random.random() < 0.2:
            noise = np.random.normal(0, 5, frames.shape).astype(np.int16)
            frames = np.clip(frames.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # EXTREME blur/sharpen
        if random.random() < 0.15:
            blurred_frames = []
            for frame in frames:
                if random.random() < 0.5:
                    # Slight blur
                    blurred = cv2.GaussianBlur(frame, (3, 3), 0.5)
                else:
                    # Slight sharpen
                    kernel = np.array([[-0.1, -0.1, -0.1], [-0.1, 1.8, -0.1], [-0.1, -0.1, -0.1]])
                    blurred = cv2.filter2D(frame, -1, kernel)
                blurred_frames.append(blurred)
            frames = np.array(blurred_frames)
        
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
        
        frames = self.load_video_extreme(video_path)
        frames = self.apply_extreme_augmentation(frames)
        frames = self.apply_v5_preprocessing(frames)
        frames = torch.from_numpy(frames).float()
        
        return frames, label

class ExtremeModel(nn.Module):
    """EXTREME model with maximum capacity and sophistication."""
    def __init__(self, num_classes=5):
        super(ExtremeModel, self).__init__()
        
        # EXTREME feature extraction - maximum capacity
        self.features = nn.Sequential(
            # Block 1 - EXTREME start
            nn.Conv2d(1, 96, 7, stride=2, padding=3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.MaxPool2d(2),
            
            # Block 2 - EXTREME expansion
            nn.Conv2d(96, 192, 5, stride=2, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3 - EXTREME depth
            nn.Conv2d(192, 384, 3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(384, 384, 3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Block 4 - EXTREME capacity
            nn.Conv2d(384, 768, 3, stride=1, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(768, 768, 3, stride=1, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            
            # Block 5 - EXTREME final
            nn.Conv2d(768, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.AdaptiveAvgPool2d(4)
        )
        
        # EXTREME temporal processing - maximum sophistication
        self.temporal = nn.Sequential(
            nn.Linear(1024 * 16, 2048),  # EXTREME capacity
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
        
        # EXTREME classifier
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
        features = self.features(x)
        features = features.view(B * T, -1)
        features = features.view(B, T, -1)
        
        # EXTREME temporal aggregation with attention-like mechanism
        # Simple attention weights based on feature magnitude
        attention_weights = torch.softmax(features.mean(dim=2), dim=1)
        weighted_features = (features * attention_weights.unsqueeze(2)).sum(dim=1)
        
        # Add noise during training for EXTREME generalization
        if self.training:
            noise = torch.randn_like(weighted_features) * 0.02
            weighted_features = weighted_features + noise
        
        # Process temporally
        processed = self.temporal(weighted_features)
        
        # Classify
        output = self.classifier(processed)
        
        return output

def create_extreme_splits(dataset_path="the_best_videos_so_far"):
    """Create extreme splits with different strategy."""
    print("📊 Creating EXTREME splits from FULL dataset...")
    
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
    
    # EXTREME stratified split (different seed for maximum diversity)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=59)
    train_val_idx, test_idx = next(skf.split(video_paths, labels))
    
    test_videos = video_paths[test_idx].tolist()
    test_labels = labels[test_idx].tolist()
    
    train_val_videos = video_paths[train_val_idx]
    train_val_labels = labels[train_val_idx]
    
    # Split train+val
    skf2 = StratifiedKFold(n_splits=4, shuffle=True, random_state=59)
    train_idx, val_idx = next(skf2.split(train_val_videos, train_val_labels))
    
    train_videos = train_val_videos[train_idx].tolist()
    train_labels = train_val_labels[train_idx].tolist()
    
    val_videos = train_val_videos[val_idx].tolist()
    val_labels = train_val_labels[val_idx].tolist()
    
    print(f"📊 EXTREME splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def create_weighted_sampler(labels):
    """Create weighted sampler for balanced training."""
    class_counts = Counter(labels)
    total_samples = len(labels)

    # Calculate weights inversely proportional to class frequency
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (len(class_counts) * count)

    # Create sample weights
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

def train_extreme_model(model, train_loader, val_loader, device, num_epochs=60):
    """EXTREME training with maximum intensity and sophistication."""

    # EXTREME optimization
    optimizer = optim.AdamW(model.parameters(), lr=8e-5, weight_decay=3e-4)  # Conservative but strong
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=4)

    # EXTREME loss with maximum label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.25)

    print(f"\n🚀 EXTREME training for {num_epochs} epochs...")

    best_val_acc = 0.0
    patience = 0
    max_patience = 20  # EXTREME patience

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

            # EXTREME L2 regularization
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss += 2e-5 * l2_reg

            # EXTREME L1 regularization for sparsity
            l1_reg = 0
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)
            loss += 1e-6 * l1_reg

            loss.backward()

            # EXTREME gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)

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

        # EXTREME generalization gap monitoring
        gen_gap = train_acc - val_acc

        print(f"Extreme E{epoch+1}/{num_epochs} - "
              f"Train: {train_acc:.1f}% ({unique_train_preds}/5), "
              f"Val: {val_acc:.1f}% ({unique_val_preds}/5), "
              f"Gap: {gen_gap:.1f}%, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), 'breakthrough_v16_extreme.pth')
            print(f"  💾 EXTREME New best: {val_acc:.1f}% (Gap: {gen_gap:.1f}%)")
        else:
            patience += 1

        if patience >= max_patience:
            print(f"  ⏹️  EXTREME Early stopping")
            break

        # EXTREME success milestones
        if unique_val_preds >= 4 and val_acc >= 60:
            print(f"  🎉 EXTREME BREAKTHROUGH: 60%+ with 4+ classes!")
            if val_acc >= 70:
                print(f"  🏆 EXTREME EXCELLENT: 70%+ achieved!")
                if val_acc >= 80:
                    print(f"  🌟 EXTREME TARGET: 80%+!")
                    break

    return best_val_acc

def main():
    """EXTREME system for 80% target - maximum intensity."""
    print("🎯 BREAKTHROUGH SYSTEM V16 - EXTREME MAXIMUM INTENSITY FOR 80% TARGET")
    print("=" * 80)
    print("EXTREME TECHNIQUES:")
    print("• MAXIMUM aggressive augmentation (blur, noise, extreme transforms)")
    print("• EXTREME model capacity (50M+ parameters)")
    print("• EXTREME regularization (L1+L2, dropout, label smoothing)")
    print("• EXTREME resolution (128x128)")
    print("• EXTREME attention-like temporal aggregation")
    print("• EXTREME training duration (60 epochs)")
    print("• EXTREME patience and conservative optimization")
    print("• V5's proven preprocessing foundation")
    print("• RELENTLESS PURSUIT OF 80% GENERALIZATION")
    print("=" * 80)

    set_seeds(59)
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")

    # Create extreme splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_extreme_splits()

    # Create datasets
    train_dataset = ExtremeDataset(train_videos, train_labels, augment=True, phase='train')
    val_dataset = ExtremeDataset(val_videos, val_labels, augment=False, phase='val')
    test_dataset = ExtremeDataset(test_videos, test_labels, augment=False, phase='test')

    # Create weighted sampler for perfect balance
    weighted_sampler = create_weighted_sampler(train_labels)

    # Create data loaders with weighted sampling
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=weighted_sampler, num_workers=0)  # Batch size 1 for extreme model
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Create extreme model
    model = ExtremeModel(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 EXTREME Model: {total_params:,} parameters")

    # Train extreme model
    best_val_acc = train_extreme_model(model, train_loader, val_loader, device, num_epochs=60)

    # Test extreme model
    if os.path.exists('breakthrough_v16_extreme.pth'):
        model.load_state_dict(torch.load('breakthrough_v16_extreme.pth', map_location=device))

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

    # Calculate final generalization gap
    final_gap = best_val_acc - test_acc

    print(f"\n🎯 BREAKTHROUGH SYSTEM V16 EXTREME RESULTS")
    print("=" * 70)
    print(f"🎯 EXTREME Maximum Intensity Performance:")
    print(f"   Parameters: {total_params:,}")
    print(f"   Validation: {best_val_acc:.1f}%")
    print(f"   Test: {test_acc:.1f}% ({unique_test_preds}/5 classes)")
    print(f"   Generalization Gap: {final_gap:.1f}%")

    # Compare with previous best
    previous_best = 36.8
    if test_acc > previous_best:
        improvement = test_acc - previous_best
        print(f"🏆 NEW EXTREME RECORD: +{improvement:.1f}% improvement!")

    if test_acc >= 80:
        print("🌟 EXTREME TARGET ACHIEVED: 80%+ GENERALIZATION!")
        print("🎯 MISSION ACCOMPLISHED WITH EXTREME INTENSITY!")
    elif test_acc >= 70:
        print("🏆 EXTREME EXCELLENT: 70%+ achieved!")
    elif test_acc >= 60:
        print("🎉 EXTREME GREAT: 60%+ achieved!")
    elif test_acc >= 50:
        print("✅ EXTREME GOOD: 50%+ achieved!")
    elif test_acc >= 40:
        print("🔄 EXTREME PROGRESS: 40%+ achieved!")
    else:
        print("🚀 Continue relentless pursuit of 80%...")

    return test_acc, best_val_acc, final_gap

if __name__ == "__main__":
    try:
        test_accuracy, val_accuracy, gen_gap = main()
        print(f"\n🏁 Breakthrough V16 EXTREME completed:")
        print(f"   Test: {test_accuracy:.1f}%")
        print(f"   Validation: {val_accuracy:.1f}%")
        print(f"   Gap: {gen_gap:.1f}%")

        if test_accuracy >= 80:
            print("🎯 80% TARGET ACHIEVED!")
            print("🌟 EXTREME SUCCESS!")
        else:
            print("🚀 Continue relentless pursuit of 80%...")

    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
