#!/usr/bin/env python3
"""
Breakthrough System V3 - Data Quality + Preprocessing Focus
Advanced preprocessing and data quality improvements for 80% accuracy
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
# import mediapipe as mp  # Not available, using alternative approach

def set_seeds(seed=44):  # Different seed for diversity
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class AdvancedPreprocessingDataset(Dataset):
    """Dataset with advanced preprocessing and quality filtering."""
    def __init__(self, video_paths, labels, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        
        # Use advanced geometric mouth detection instead of MediaPipe
        self.use_advanced_roi = True
        
        print(f"📊 Advanced Preprocessing Dataset: {len(video_paths)} videos, Augment: {augment}")
        label_counts = Counter(labels)
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        for i, name in enumerate(class_names):
            print(f"   {name}: {label_counts.get(i, 0)} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def extract_mouth_roi_advanced(self, frame):
        """Extract mouth ROI using advanced geometric approach."""
        h, w = frame.shape[:2]

        # Advanced geometric approach for ICU-style cropped faces
        # The videos show lower half of faces with lips in top-middle portion

        # Method 1: Adaptive mouth region detection
        if self.use_advanced_roi:
            # Convert to grayscale for processing
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Apply edge detection to find mouth region
            edges = cv2.Canny(gray, 50, 150)

            # Focus on the expected mouth region (top 60% of frame, middle 60% width)
            roi_h_start = 0
            roi_h_end = int(0.6 * h)
            roi_w_start = int(0.2 * w)
            roi_w_end = int(0.8 * w)

            mouth_region = edges[roi_h_start:roi_h_end, roi_w_start:roi_w_end]

            # Find contours in the mouth region
            contours, _ = cv2.findContours(mouth_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find the largest contour (likely mouth)
                largest_contour = max(contours, key=cv2.contourArea)

                if cv2.contourArea(largest_contour) > 100:  # Quality threshold
                    # Get bounding box
                    x, y, w_box, h_box = cv2.boundingRect(largest_contour)

                    # Adjust coordinates back to full frame
                    x += roi_w_start
                    y += roi_h_start

                    # Add padding
                    padding = max(w_box, h_box) // 4
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w_box = min(w - x, w_box + 2 * padding)
                    h_box = min(h - y, h_box + 2 * padding)

                    # Extract ROI
                    mouth_roi = frame[y:y+h_box, x:x+w_box]

                    if mouth_roi.size > 0:
                        return mouth_roi, True  # High quality

        # Fallback: Optimized geometric crop for ICU dataset
        # Top 50% height, middle 60% width (based on user preferences)
        crop_h = int(0.5 * h)
        crop_w_start = int(0.2 * w)
        crop_w_end = int(0.8 * w)

        fallback_roi = frame[0:crop_h, crop_w_start:crop_w_end]
        return fallback_roi, False  # Lower quality
    
    def load_video_with_quality_assessment(self, path):
        """Load video with quality assessment and advanced preprocessing."""
        cap = cv2.VideoCapture(path)
        frames = []
        quality_scores = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract mouth ROI
            mouth_roi, is_high_quality = self.extract_mouth_roi_advanced(frame)
            
            if mouth_roi.size > 0:
                # Convert to grayscale
                if len(mouth_roi.shape) == 3:
                    gray_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
                else:
                    gray_roi = mouth_roi
                
                # Resize to consistent size
                resized_roi = cv2.resize(gray_roi, (160, 160))
                
                # Quality assessment
                quality_score = self.assess_frame_quality(resized_roi)
                
                frames.append(resized_roi)
                quality_scores.append(quality_score * (1.5 if is_high_quality else 1.0))
        
        cap.release()
        
        if len(frames) == 0:
            # Create dummy frame
            frames = [np.zeros((160, 160), dtype=np.uint8)]
            quality_scores = [0.1]
        
        # Select best frames based on quality
        frames = np.array(frames)
        quality_scores = np.array(quality_scores)
        
        # Target 40 frames
        target_frames = 40
        if len(frames) >= target_frames:
            # Select frames with highest quality scores
            top_indices = np.argsort(quality_scores)[-target_frames:]
            top_indices = np.sort(top_indices)  # Maintain temporal order
            frames = frames[top_indices]
        else:
            # Duplicate high-quality frames
            while len(frames) < target_frames:
                best_idx = np.argmax(quality_scores)
                frames = np.append(frames, [frames[best_idx]], axis=0)
                quality_scores = np.append(quality_scores, quality_scores[best_idx])
        
        return frames[:target_frames]
    
    def assess_frame_quality(self, frame):
        """Assess frame quality based on multiple metrics."""
        # Variance (sharpness)
        variance = cv2.Laplacian(frame, cv2.CV_64F).var()
        
        # Contrast
        contrast = frame.std()
        
        # Brightness distribution
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
        brightness_score = 1.0 - abs(0.5 - np.mean(frame) / 255.0)
        
        # Edge density
        edges = cv2.Canny(frame, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Combine metrics
        quality_score = (
            0.3 * min(variance / 1000.0, 1.0) +  # Sharpness
            0.3 * min(contrast / 50.0, 1.0) +    # Contrast
            0.2 * brightness_score +              # Brightness
            0.2 * min(edge_density * 10, 1.0)    # Edge density
        )
        
        return quality_score
    
    def apply_quality_augmentation(self, frames):
        """Apply augmentation that preserves quality."""
        if not self.augment:
            return frames
        
        # Conservative augmentations to preserve lip-reading quality
        if random.random() < 0.5:
            # Horizontal flip
            frames = np.flip(frames, axis=2).copy()
        
        if random.random() < 0.3:
            # Slight brightness adjustment
            brightness = random.uniform(0.9, 1.1)
            frames = np.clip(frames * brightness, 0, 255).astype(np.uint8)
        
        if random.random() < 0.2:
            # Slight contrast adjustment
            contrast = random.uniform(0.95, 1.05)
            frames = np.clip((frames - 128) * contrast + 128, 0, 255).astype(np.uint8)
        
        if random.random() < 0.1:
            # Very slight rotation
            angle = random.uniform(-3, 3)
            h, w = frames.shape[1], frames.shape[2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            rotated_frames = []
            for frame in frames:
                rotated = cv2.warpAffine(frame, M, (w, h))
                rotated_frames.append(rotated)
            frames = np.array(rotated_frames)
        
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video with quality assessment
        frames = self.load_video_with_quality_assessment(video_path)
        frames = self.apply_quality_augmentation(frames)
        
        # Advanced normalization
        frames = frames.astype(np.float32) / 255.0
        
        # Histogram equalization in normalized space
        frames_eq = []
        for frame in frames:
            # Convert back to uint8 for CLAHE
            frame_uint8 = (frame * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            frame_eq = clahe.apply(frame_uint8)
            frames_eq.append(frame_eq.astype(np.float32) / 255.0)
        
        frames = np.array(frames_eq)
        
        # Standardization
        frames = (frames - 0.5) / 0.5  # [-1, 1] range
        
        # Convert to tensor (T, H, W)
        frames = torch.from_numpy(frames)
        
        return frames, label

class QualityAwareModel(nn.Module):
    """Model designed for high-quality preprocessed data."""
    def __init__(self, num_classes=5):
        super(QualityAwareModel, self).__init__()
        
        # Feature extractor optimized for mouth ROI
        self.feature_extractor = nn.Sequential(
            # First block - capture fine details
            nn.Conv2d(1, 64, 7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second block - spatial patterns
            nn.Conv2d(64, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block - complex features
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth block - high-level features
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4)  # 4x4 spatial features
        )
        
        # Temporal modeling with GRU
        self.temporal_model = nn.GRU(
            input_size=512 * 16,  # 512 channels * 4*4 spatial
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
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
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x):
        # x shape: (B, T, H, W)
        B, T, H, W = x.shape
        
        # Extract features for each frame
        x = x.view(B * T, 1, H, W)  # (B*T, 1, H, W)
        features = self.feature_extractor(x)  # (B*T, 512, 4, 4)
        features = features.view(B * T, -1)  # (B*T, 512*16)
        features = features.view(B, T, -1)  # (B, T, 512*16)
        
        # Temporal modeling
        gru_out, _ = self.temporal_model(features)  # (B, T, 512)
        
        # Attention mechanism
        attention_weights = self.attention(gru_out)  # (B, T, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended = torch.sum(gru_out * attention_weights, dim=1)  # (B, 512)
        
        # Classification
        output = self.classifier(attended)  # (B, num_classes)
        
        return output

def create_quality_splits(dataset_path="the_best_videos_so_far"):
    """Create splits with quality assessment."""
    print("📊 Creating quality-aware splits from FULL dataset...")

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

    # Create quality-focused splits: 85% train, 7.5% val, 7.5% test
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []

    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}

    for class_name, videos in class_videos.items():
        random.shuffle(videos)
        n_videos = len(videos)

        # 85% train, 7.5% val, 7.5% test (maximize training data)
        n_train = max(1, int(0.85 * n_videos))
        n_val = max(1, int(0.075 * n_videos))

        train_videos.extend(videos[:n_train])
        train_labels.extend([class_to_idx[class_name]] * n_train)

        val_videos.extend(videos[n_train:n_train+n_val])
        val_labels.extend([class_to_idx[class_name]] * n_val)

        test_videos.extend(videos[n_train+n_val:])
        test_labels.extend([class_to_idx[class_name]] * (len(videos) - n_train - n_val))

    print(f"📊 Quality splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")

    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def train_quality_model(model, train_loader, val_loader, device, num_epochs=80):
    """Train quality-aware model with extended training."""

    # Advanced optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Scheduler with multiple phases
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.5)

    # Focal loss for better class separation
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            return focal_loss.mean()

    criterion = FocalLoss(alpha=1, gamma=2)

    print(f"\n🚀 Quality-aware training for {num_epochs} epochs...")

    best_val_acc = 0.0
    patience = 0
    max_patience = 25

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
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), 'breakthrough_quality_v3.pth')
            print(f"  💾 New best: {val_acc:.1f}%")
        else:
            patience += 1

        # Early stopping
        if patience >= max_patience:
            print(f"  ⏹️  Early stopping")
            break

        # Success milestones
        if unique_val_preds >= 4 and val_acc >= 70:
            print(f"  🎉 BREAKTHROUGH: 70%+ with 4+ classes!")
            if val_acc >= 80:
                print(f"  🌟 TARGET ACHIEVED: 80%+!")
                break

    return best_val_acc

def main():
    """Quality-focused breakthrough system main function."""
    print("🎯 BREAKTHROUGH SYSTEM V3 - DATA QUALITY FOCUS")
    print("=" * 80)
    print("QUALITY TECHNIQUES:")
    print("• Advanced geometric mouth ROI extraction")
    print("• Frame quality assessment and selection")
    print("• Advanced CLAHE preprocessing")
    print("• Quality-preserving augmentation")
    print("• GRU + Attention temporal modeling")
    print("• Extended training (80 epochs)")
    print("• Focal loss for class separation")
    print("=" * 80)

    # Set seeds
    set_seeds(44)

    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")

    # Create quality splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_quality_splits()

    # Create datasets
    train_dataset = AdvancedPreprocessingDataset(train_videos, train_labels, augment=True)
    val_dataset = AdvancedPreprocessingDataset(val_videos, val_labels, augment=False)
    test_dataset = AdvancedPreprocessingDataset(test_videos, test_labels, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Create quality model
    model = QualityAwareModel(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Quality Model: {total_params:,} parameters")

    # Train
    best_val_acc = train_quality_model(model, train_loader, val_loader, device, num_epochs=60)

    # Test
    print(f"\n🔍 Testing quality model...")

    if os.path.exists('breakthrough_quality_v3.pth'):
        model.load_state_dict(torch.load('breakthrough_quality_v3.pth', map_location=device))
        print("📥 Loaded best quality model")

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
        print(f"📊 Quality Classification Report:\n{report}")

    print(f"\n🎯 BREAKTHROUGH SYSTEM V3 RESULTS")
    print("=" * 60)
    print(f"🎯 Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"🎯 Test Accuracy: {test_acc:.1f}%")
    print(f"🎯 Test Predictions: {sorted(set(test_preds))}")
    print(f"🎯 Test Targets: {sorted(set(test_targets))}")
    print(f"🎯 Unique Predictions: {unique_test_preds}/5 classes")
    print(f"🎯 Focus: Data Quality + Geometric ROI + Advanced Preprocessing")

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
        print("🔄 Continue to V4...")

    return test_acc, best_val_acc

if __name__ == "__main__":
    try:
        test_accuracy, val_accuracy = main()
        print(f"\n🏁 Breakthrough V3 completed:")
        print(f"   Validation: {val_accuracy:.1f}%")
        print(f"   Test: {test_accuracy:.1f}%")

        if test_accuracy >= 80:
            print("🎯 TARGET ACHIEVED!")
        else:
            print("🚀 Moving to next breakthrough iteration...")

    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
