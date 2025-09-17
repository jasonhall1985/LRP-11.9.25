#!/usr/bin/env python3
"""
Breakthrough System V4 - Ensemble + Extreme Regularization
Addressing the fundamental generalization gap with ensemble methods
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
import copy

def set_seeds(seed=45):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class EnsembleVideoDataset(Dataset):
    """Dataset with extreme augmentation for ensemble training."""
    def __init__(self, video_paths, labels, augment=False, augment_strength=1.0):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        self.augment_strength = augment_strength
        
        print(f"📊 Ensemble Dataset: {len(video_paths)} videos, Augment: {augment}, Strength: {augment_strength}")
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
            
            # ICU-style crop: top 50% height, middle 60% width
            h, w = gray.shape
            crop_h = int(0.5 * h)
            crop_w_start = int(0.2 * w)
            crop_w_end = int(0.8 * w)
            
            cropped = gray[0:crop_h, crop_w_start:crop_w_end]
            
            # Resize to consistent size
            resized = cv2.resize(cropped, (112, 112))
            frames.append(resized)
        
        cap.release()
        
        if len(frames) == 0:
            frames = [np.zeros((112, 112), dtype=np.uint8)]
        
        # Sample 32 frames
        target_frames = 32
        if len(frames) >= target_frames:
            indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < target_frames:
                frames.extend(frames[:min(len(frames), target_frames - len(frames))])
        
        return np.array(frames[:target_frames])
    
    def apply_extreme_augmentation(self, frames):
        """Apply extreme augmentation to force generalization."""
        if not self.augment:
            return frames
        
        strength = self.augment_strength
        
        # Temporal augmentations
        if random.random() < 0.6 * strength:
            # Random temporal shift
            shift = random.randint(-5, 5)
            if shift > 0:
                frames = np.concatenate([frames[shift:], frames[:shift]])
            elif shift < 0:
                frames = np.concatenate([frames[shift:], frames[:shift]])
        
        if random.random() < 0.4 * strength:
            # Random frame dropout
            dropout_indices = random.sample(range(len(frames)), random.randint(1, 3))
            for idx in sorted(dropout_indices, reverse=True):
                if len(frames) > 16:  # Keep minimum frames
                    frames = np.delete(frames, idx, axis=0)
            
            # Pad back
            while len(frames) < 32:
                frames = np.append(frames, [frames[-1]], axis=0)
        
        # Spatial augmentations
        if random.random() < 0.7 * strength:
            # Horizontal flip
            frames = np.flip(frames, axis=2).copy()
        
        if random.random() < 0.5 * strength:
            # Random brightness/contrast
            brightness = random.uniform(0.7, 1.3)
            contrast = random.uniform(0.7, 1.3)
            frames = np.clip(frames * contrast + (brightness - 1) * 128, 0, 255).astype(np.uint8)
        
        if random.random() < 0.4 * strength:
            # Random rotation
            angle = random.uniform(-15, 15)
            h, w = frames.shape[1], frames.shape[2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            rotated_frames = []
            for frame in frames:
                rotated = cv2.warpAffine(frame, M, (w, h))
                rotated_frames.append(rotated)
            frames = np.array(rotated_frames)
        
        if random.random() < 0.3 * strength:
            # Random crop and resize
            h, w = frames.shape[1], frames.shape[2]
            crop_size = random.randint(int(0.7 * min(h, w)), min(h, w))
            start_h = random.randint(0, h - crop_size)
            start_w = random.randint(0, w - crop_size)
            
            cropped_frames = []
            for frame in frames:
                cropped = frame[start_h:start_h+crop_size, start_w:start_w+crop_size]
                resized = cv2.resize(cropped, (112, 112))
                cropped_frames.append(resized)
            frames = np.array(cropped_frames)
        
        if random.random() < 0.2 * strength:
            # Add noise
            noise_std = random.uniform(5, 15)
            noise = np.random.normal(0, noise_std, frames.shape).astype(np.int16)
            frames = np.clip(frames.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        if random.random() < 0.1 * strength:
            # Random blur
            kernel_size = random.choice([3, 5])
            blurred_frames = []
            for frame in frames:
                blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
                blurred_frames.append(blurred)
            frames = np.array(blurred_frames)
        
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load and augment video
        frames = self.load_video_robust(video_path)
        frames = self.apply_extreme_augmentation(frames)
        
        # Normalize
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - 0.5) / 0.5  # [-1, 1] range
        
        # Convert to tensor (T, H, W)
        frames = torch.from_numpy(frames)
        
        return frames, label

class SimpleRobustModel(nn.Module):
    """Simple but robust model for ensemble."""
    def __init__(self, num_classes=5, dropout_rate=0.7):
        super(SimpleRobustModel, self).__init__()
        
        # Simple CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.3),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.4),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5),
            nn.AdaptiveAvgPool2d(2)  # 2x2 spatial features
        )
        
        # Temporal processing
        self.temporal = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8)
        )
        
        # Classifier with extreme regularization
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with small values."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, 0, 0.01)  # Small weights
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # Very small weights
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (B, T, H, W)
        B, T, H, W = x.shape
        
        # Process each frame
        x = x.view(B * T, 1, H, W)
        features = self.features(x)  # (B*T, 128, 2, 2)
        features = features.view(B * T, -1)  # (B*T, 512)
        features = features.view(B, T, -1)  # (B, T, 512)
        
        # Temporal pooling
        temporal_features = features.mean(dim=1)  # (B, 512)
        
        # Process temporally
        processed = self.temporal(temporal_features)  # (B, 128)
        
        # Classify
        output = self.classifier(processed)  # (B, num_classes)
        
        return output

class EnsembleModel(nn.Module):
    """Ensemble of multiple models."""
    def __init__(self, num_models=5, num_classes=5):
        super(EnsembleModel, self).__init__()
        
        # Create diverse models with different dropout rates
        self.models = nn.ModuleList([
            SimpleRobustModel(num_classes, dropout_rate=0.5 + 0.1 * i)
            for i in range(num_models)
        ])
        
        self.num_models = num_models
    
    def forward(self, x):
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        return ensemble_pred
    
    def forward_individual(self, x, model_idx):
        """Forward pass for individual model."""
        return self.models[model_idx](x)

def create_ensemble_splits(dataset_path="the_best_videos_so_far"):
    """Create splits optimized for ensemble training."""
    print("📊 Creating ensemble splits from FULL dataset...")
    
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
    
    # Create ensemble splits: 70% train, 15% val, 15% test
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
    
    print(f"📊 Ensemble splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def create_balanced_sampler(labels):
    """Create weighted sampler for balanced training."""
    class_counts = Counter(labels)
    total_samples = len(labels)

    # Calculate weights for each class (inverse frequency)
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (len(class_counts) * count)

    # Create sample weights
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(sample_weights, total_samples, replacement=True)

def train_ensemble_model(model, train_loader, val_loader, device, num_epochs=100):
    """Train ensemble model with extreme regularization."""

    # Different optimizers for each model in ensemble
    optimizers = []
    schedulers = []

    for i in range(model.num_models):
        # Different learning rates for diversity
        lr = 1e-4 * (0.5 + 0.5 * i / model.num_models)
        optimizer = optim.AdamW(model.models[i].parameters(), lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        optimizers.append(optimizer)
        schedulers.append(scheduler)

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    print(f"\n🚀 Ensemble training for {num_epochs} epochs...")

    best_val_acc = 0.0
    patience = 0
    max_patience = 30

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

            # Train each model in ensemble
            total_loss = 0
            ensemble_output = None

            for i in range(model.num_models):
                optimizers[i].zero_grad()

                # Individual model prediction
                output = model.forward_individual(data, i)
                loss = criterion(output, target)

                # Add regularization loss
                l2_reg = 0
                for param in model.models[i].parameters():
                    l2_reg += torch.norm(param, 2)
                loss += 1e-5 * l2_reg

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.models[i].parameters(), max_norm=0.5)

                optimizers[i].step()

                total_loss += loss.item()

                # Accumulate for ensemble prediction
                if ensemble_output is None:
                    ensemble_output = output.detach()
                else:
                    ensemble_output += output.detach()

            # Average ensemble prediction
            ensemble_output /= model.num_models

            train_loss += total_loss / model.num_models
            pred = ensemble_output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)

            train_preds.extend(pred.cpu().numpy())
            train_targets.extend(target.cpu().numpy())

        # Update schedulers
        for scheduler in schedulers:
            scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)  # Ensemble prediction
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
              f"LR: {optimizers[0].param_groups[0]['lr']:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), 'breakthrough_ensemble_v4.pth')
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
            if val_acc >= 80:
                print(f"  🌟 TARGET ACHIEVED: 80%+!")
                break

    return best_val_acc

def main():
    """Ensemble breakthrough system main function."""
    print("🎯 BREAKTHROUGH SYSTEM V4 - ENSEMBLE + EXTREME REGULARIZATION")
    print("=" * 80)
    print("ENSEMBLE TECHNIQUES:")
    print("• 5-model ensemble with different dropout rates")
    print("• Extreme augmentation (rotation, noise, blur)")
    print("• Weighted random sampling for balance")
    print("• Individual model training with diversity")
    print("• Label smoothing + L2 regularization")
    print("• Extended training (100 epochs)")
    print("• Cosine annealing scheduling")
    print("=" * 80)

    # Set seeds
    set_seeds(45)

    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")

    # Create ensemble splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_ensemble_splits()

    # Create datasets with different augmentation strengths
    train_dataset = EnsembleVideoDataset(train_videos, train_labels, augment=True, augment_strength=1.5)
    val_dataset = EnsembleVideoDataset(val_videos, val_labels, augment=False)
    test_dataset = EnsembleVideoDataset(test_videos, test_labels, augment=False)

    # Create balanced sampler
    sampler = create_balanced_sampler(train_labels)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Create ensemble model
    model = EnsembleModel(num_models=5, num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Ensemble Model: {total_params:,} parameters (5 models)")

    # Train
    best_val_acc = train_ensemble_model(model, train_loader, val_loader, device, num_epochs=80)

    # Test
    print(f"\n🔍 Testing ensemble model...")

    if os.path.exists('breakthrough_ensemble_v4.pth'):
        model.load_state_dict(torch.load('breakthrough_ensemble_v4.pth', map_location=device))
        print("📥 Loaded best ensemble model")

    model.eval()
    test_correct = 0
    test_total = 0
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # Ensemble prediction
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
        print(f"📊 Ensemble Classification Report:\n{report}")

    print(f"\n🎯 BREAKTHROUGH SYSTEM V4 RESULTS")
    print("=" * 60)
    print(f"🎯 Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"🎯 Test Accuracy: {test_acc:.1f}%")
    print(f"🎯 Test Predictions: {sorted(set(test_preds))}")
    print(f"🎯 Test Targets: {sorted(set(test_targets))}")
    print(f"🎯 Unique Predictions: {unique_test_preds}/5 classes")
    print(f"🎯 Architecture: 5-Model Ensemble + Extreme Regularization")

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
        print("🔄 Continue to V5...")

    return test_acc, best_val_acc

if __name__ == "__main__":
    try:
        test_accuracy, val_accuracy = main()
        print(f"\n🏁 Breakthrough V4 completed:")
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
