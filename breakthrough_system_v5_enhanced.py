#!/usr/bin/env python3
"""
Breakthrough System V5 Enhanced - Advanced Temporal-Coherent Augmentation
Building on V5's success (44.4% validation, 5/5 classes) with enhanced augmentations
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
from scipy.ndimage import map_coordinates

def set_seeds(seed=47):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class EnhancedDataCentricDataset(Dataset):
    """Enhanced dataset with temporal-coherent augmentations for lip-reading."""
    def __init__(self, video_paths, labels, augment=False, phase='train'):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        self.phase = phase
        
        print(f"📊 Enhanced Data-Centric Dataset ({phase}): {len(video_paths)} videos, Augment: {augment}")
        label_counts = Counter(labels)
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        for i, name in enumerate(class_names):
            print(f"   {name}: {label_counts.get(i, 0)} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_consistent(self, path):
        """Load video with consistent preprocessing (same as V5)."""
        cap = cv2.VideoCapture(path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Consistent ICU-style crop based on user preferences
            h, w = gray.shape
            # Top 50% height, middle 33% width (user's geometric preference)
            crop_h = int(0.5 * h)
            crop_w_start = int(0.335 * w)  # Middle 33%
            crop_w_end = int(0.665 * w)
            
            cropped = gray[0:crop_h, crop_w_start:crop_w_end]
            
            # Resize to 96x96 for model compatibility
            resized = cv2.resize(cropped, (96, 96))
            frames.append(resized)
        
        cap.release()
        
        if len(frames) == 0:
            frames = [np.zeros((96, 96), dtype=np.uint8)]
        
        # Use 32 frames (user's temporal sampling preference)
        target_frames = 32
        if len(frames) >= target_frames:
            # Uniform sampling (user preference)
            indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            # Repeat frames to reach target
            while len(frames) < target_frames:
                frames.extend(frames[:min(len(frames), target_frames - len(frames))])
        
        return np.array(frames[:target_frames])
    
    def apply_temporal_coherent_spatial_augmentation(self, frames):
        """Apply spatial augmentations consistently across all frames."""
        if not self.augment:
            return frames
        
        # Generate augmentation parameters once for the entire sequence
        h, w = frames.shape[1], frames.shape[2]
        
        # 1. Random translation (±5 pixels) - consistent across frames
        if random.random() < 0.4:
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            
            # Apply translation to all frames
            translated_frames = []
            for frame in frames:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                translated = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                translated_frames.append(translated)
            frames = np.array(translated_frames)
        
        # 2. Slight scaling (0.95-1.05x) - consistent across frames
        if random.random() < 0.3:
            scale_factor = random.uniform(0.95, 1.05)
            center = (w // 2, h // 2)
            
            # Apply scaling to all frames
            scaled_frames = []
            for frame in frames:
                M = cv2.getRotationMatrix2D(center, 0, scale_factor)
                scaled = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                scaled_frames.append(scaled)
            frames = np.array(scaled_frames)
        
        # 3. Minor elastic deformation - consistent across frames
        if random.random() < 0.2:
            # Generate deformation field once
            alpha = random.uniform(5, 15)  # Deformation strength
            sigma = random.uniform(2, 4)   # Smoothness
            
            # Create random displacement fields
            dx = np.random.randn(h, w) * alpha
            dy = np.random.randn(h, w) * alpha
            
            # Smooth the displacement fields
            dx = cv2.GaussianBlur(dx, (0, 0), sigma)
            dy = cv2.GaussianBlur(dy, (0, 0), sigma)
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x_new = x + dx
            y_new = y + dy
            
            # Apply deformation to all frames
            deformed_frames = []
            for frame in frames:
                deformed = map_coordinates(frame, [y_new, x_new], order=1, mode='reflect')
                deformed_frames.append(deformed.astype(np.uint8))
            frames = np.array(deformed_frames)
        
        return frames
    
    def apply_temporal_coherent_photometric_augmentation(self, frames):
        """Apply photometric augmentations consistently across all frames."""
        if not self.augment:
            return frames
        
        # Generate photometric parameters once for the entire sequence
        
        # 1. Random gamma correction (0.8-1.2) - consistent across frames
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.2)
            # Build lookup table for gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            
            # Apply gamma correction to all frames
            gamma_frames = []
            for frame in frames:
                gamma_corrected = cv2.LUT(frame, table)
                gamma_frames.append(gamma_corrected)
            frames = np.array(gamma_frames)
        
        # 2. Controlled shadow/highlight variations - consistent across frames
        if random.random() < 0.2:
            # Shadow/highlight adjustment parameters
            shadow_factor = random.uniform(0.8, 1.0)
            highlight_factor = random.uniform(1.0, 1.2)
            
            # Apply shadow/highlight adjustment to all frames
            adjusted_frames = []
            for frame in frames:
                # Separate shadows (dark regions) and highlights (bright regions)
                shadows = frame < 128
                highlights = frame >= 128
                
                adjusted = frame.astype(np.float32)
                adjusted[shadows] *= shadow_factor
                adjusted[highlights] *= highlight_factor
                
                adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
                adjusted_frames.append(adjusted)
            frames = np.array(adjusted_frames)
        
        return frames
    
    def apply_existing_minimal_augmentation(self, frames):
        """Apply user's preferred minimal augmentations (from V5)."""
        if not self.augment:
            return frames
        
        # User's preferred minimal augmentations for lip-reading
        if random.random() < 0.5:
            # Horizontal flipping - consistent across frames
            frames = np.flip(frames, axis=2).copy()
        
        if random.random() < 0.3:
            # Slight brightness adjustments (±10-15%) - consistent across frames
            brightness_factor = random.uniform(0.85, 1.15)
            frames = np.clip(frames * brightness_factor, 0, 255).astype(np.uint8)
        
        if random.random() < 0.2:
            # Minor contrast variations (0.9-1.1x) - consistent across frames
            contrast_factor = random.uniform(0.9, 1.1)
            frames = np.clip((frames - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
        
        # Note: Removed temporal speed variations to maintain sequence integrity
        
        return frames
    
    def apply_standardized_preprocessing(self, frames):
        """Apply user's preferred standardized preprocessing (same as V5)."""
        # Convert to float
        frames = frames.astype(np.float32) / 255.0
        
        # User's preferred grayscale normalization (5-step pipeline)
        processed_frames = []
        for frame in frames:
            # Step 1: Already converted to grayscale
            # Step 2: CLAHE enhancement
            frame_uint8 = (frame * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(frame_uint8).astype(np.float32) / 255.0
            
            # Step 3: Robust percentile normalization
            p2, p98 = np.percentile(enhanced, [2, 98])
            if p98 > p2:
                enhanced = np.clip((enhanced - p2) / (p98 - p2), 0, 1)
            
            # Step 4: Gamma correction
            gamma = 1.2
            enhanced = np.power(enhanced, 1.0 / gamma)
            
            # Step 5: Target brightness standardization
            target_brightness = 0.5
            current_brightness = np.mean(enhanced)
            if current_brightness > 0:
                brightness_factor = target_brightness / current_brightness
                enhanced = np.clip(enhanced * brightness_factor, 0, 1)
            
            processed_frames.append(enhanced)
        
        frames = np.array(processed_frames)
        
        # Final normalization to [-1, 1] range
        frames = (frames - 0.5) / 0.5
        
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video with consistent preprocessing
        frames = self.load_video_consistent(video_path)
        
        # Apply enhanced augmentations in sequence
        frames = self.apply_temporal_coherent_spatial_augmentation(frames)
        frames = self.apply_temporal_coherent_photometric_augmentation(frames)
        frames = self.apply_existing_minimal_augmentation(frames)
        
        # Apply standardized preprocessing
        frames = self.apply_standardized_preprocessing(frames)
        
        # Convert to tensor (T, H, W) with correct dtype
        frames = torch.from_numpy(frames).float()
        
        return frames, label

class OptimalModel(nn.Module):
    """Optimal model based on V5's successful architecture."""
    def __init__(self, num_classes=5):
        super(OptimalModel, self).__init__()
        
        # Based on V5's successful architecture
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(3)  # 3x3 spatial features
        )
        
        # Temporal processing
        self.temporal = nn.Sequential(
            nn.Linear(128 * 9, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly."""
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
        # x shape: (B, T, H, W)
        B, T, H, W = x.shape
        
        # Process each frame
        x = x.view(B * T, 1, H, W)
        features = self.features(x)  # (B*T, 128, 3, 3)
        features = features.view(B * T, -1)  # (B*T, 1152)
        features = features.view(B, T, -1)  # (B, T, 1152)
        
        # Temporal pooling
        temporal_features = features.mean(dim=1)  # (B, 1152)
        
        # Process temporally
        processed = self.temporal(temporal_features)  # (B, 128)
        
        # Classify
        output = self.classifier(processed)  # (B, num_classes)
        
        return output

def create_stratified_splits(dataset_path="the_best_videos_so_far"):
    """Create stratified splits (same as V5's successful approach)."""
    print("📊 Creating stratified splits from FULL dataset...")

    video_files = list(Path(dataset_path).glob("*.mp4"))
    video_files = [f for f in video_files if "copy" not in f.name]

    print(f"Found {len(video_files)} videos (after removing duplicates)")

    # Create video-label pairs
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

    # Use stratified split to ensure balanced representation
    video_paths = np.array(video_paths)
    labels = np.array(labels)

    # First split: 80% train+val, 20% test
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)
    train_val_idx, test_idx = next(skf.split(video_paths, labels))

    test_videos = video_paths[test_idx].tolist()
    test_labels = labels[test_idx].tolist()

    train_val_videos = video_paths[train_val_idx]
    train_val_labels = labels[train_val_idx]

    # Second split: 75% train, 25% val from remaining data
    skf2 = StratifiedKFold(n_splits=4, shuffle=True, random_state=47)
    train_idx, val_idx = next(skf2.split(train_val_videos, train_val_labels))

    train_videos = train_val_videos[train_idx].tolist()
    train_labels = train_val_labels[train_idx].tolist()

    val_videos = train_val_videos[val_idx].tolist()
    val_labels = train_val_labels[val_idx].tolist()

    print(f"📊 Stratified splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")

    # Verify stratification
    print("📊 Train distribution:", Counter(train_labels))
    print("📊 Val distribution:", Counter(val_labels))
    print("📊 Test distribution:", Counter(test_labels))

    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def train_enhanced_model(model, train_loader, val_loader, device, num_epochs=60):
    """Train enhanced model with optimal settings from V5."""

    # Optimal optimizer settings from V5
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)

    # Standard cross entropy
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 Enhanced training for {num_epochs} epochs...")

    best_val_acc = 0.0
    patience = 0
    max_patience = 20

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

            # Moderate gradient clipping
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

        # Check class diversity
        unique_train_preds = len(set(train_preds))
        unique_val_preds = len(set(val_preds))

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train: {train_acc:.1f}% ({unique_train_preds}/5), "
              f"Val: {val_acc:.1f}% ({unique_val_preds}/5), "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), 'breakthrough_enhanced_v5.pth')
            print(f"  💾 New best: {val_acc:.1f}%")
        else:
            patience += 1

        # Early stopping
        if patience >= max_patience:
            print(f"  ⏹️  Early stopping")
            break

        # Success milestones
        if unique_val_preds >= 4 and val_acc >= 60:
            print(f"  🎉 BREAKTHROUGH: 60%+ with 4+ classes!")
            if val_acc >= 70:
                print(f"  🏆 EXCELLENT: 70%+ achieved!")
                if val_acc >= 80:
                    print(f"  🌟 TARGET ACHIEVED: 80%+!")
                    break

    return best_val_acc

def main():
    """Enhanced data-centric breakthrough system main function."""
    print("🎯 BREAKTHROUGH SYSTEM V5 ENHANCED - TEMPORAL-COHERENT AUGMENTATION")
    print("=" * 80)
    print("ENHANCED TECHNIQUES (Building on V5's 44.4% validation success):")
    print("• Stratified splits for proper generalization")
    print("• User's preferred preprocessing pipeline")
    print("• ENHANCED: Temporal-coherent spatial augmentations")
    print("  - Random translation (±5px), scaling (0.95-1.05x)")
    print("  - Minor elastic deformation with smoothing")
    print("• ENHANCED: Temporal-coherent photometric augmentations")
    print("  - Gamma correction (0.8-1.2), shadow/highlight variations")
    print("• Existing minimal augmentations (flip, brightness, contrast)")
    print("• 5-step grayscale normalization")
    print("• ICU-style geometric cropping (top 50%, middle 33%)")
    print("• 32-frame temporal sampling")
    print("=" * 80)

    # Set seeds
    set_seeds(47)

    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")

    # Create stratified splits (same as V5)
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_stratified_splits()

    # Create enhanced datasets
    train_dataset = EnhancedDataCentricDataset(train_videos, train_labels, augment=True, phase='train')
    val_dataset = EnhancedDataCentricDataset(val_videos, val_labels, augment=False, phase='val')
    test_dataset = EnhancedDataCentricDataset(test_videos, test_labels, augment=False, phase='test')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Create optimal model (same as V5)
    model = OptimalModel(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Enhanced Model: {total_params:,} parameters")

    # Train
    best_val_acc = train_enhanced_model(model, train_loader, val_loader, device, num_epochs=60)

    # Test
    print(f"\n🔍 Testing enhanced model...")

    if os.path.exists('breakthrough_enhanced_v5.pth'):
        model.load_state_dict(torch.load('breakthrough_enhanced_v5.pth', map_location=device))
        print("📥 Loaded best enhanced model")

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
        print(f"📊 Enhanced Classification Report:\n{report}")

    print(f"\n🎯 BREAKTHROUGH SYSTEM V5 ENHANCED RESULTS")
    print("=" * 60)
    print(f"🎯 Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"🎯 Test Accuracy: {test_acc:.1f}%")
    print(f"🎯 Test Predictions: {sorted(set(test_preds))}")
    print(f"🎯 Test Targets: {sorted(set(test_targets))}")
    print(f"🎯 Unique Predictions: {unique_test_preds}/5 classes")
    print(f"🎯 Enhancement: Temporal-Coherent Augmentation + V5 Foundation")

    if test_acc >= 80:
        print("🌟 TARGET ACHIEVED: 80%+ accuracy!")
    elif test_acc >= 70:
        print("🏆 EXCELLENT: 70%+ accuracy!")
    elif test_acc >= 60:
        print("🎉 GREAT: 60%+ accuracy!")
    elif test_acc >= 50:
        print("✅ GOOD: 50%+ accuracy!")
    elif test_acc >= 40:
        print("📈 PROGRESS: 40%+ accuracy!")
    elif unique_test_preds >= 4:
        print("📊 IMPROVEMENT: Multi-class prediction!")
    else:
        print("🔄 Continue enhancement...")

    return test_acc, best_val_acc

if __name__ == "__main__":
    try:
        test_accuracy, val_accuracy = main()
        print(f"\n🏁 Breakthrough V5 Enhanced completed:")
        print(f"   Validation: {val_accuracy:.1f}%")
        print(f"   Test: {test_accuracy:.1f}%")

        if test_accuracy >= 80:
            print("🎯 TARGET ACHIEVED!")
        else:
            print("🚀 Continue toward 80% target...")

    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
