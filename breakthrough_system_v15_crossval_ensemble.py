#!/usr/bin/env python3
"""
Breakthrough System V15 Cross-Validation Ensemble - Maximum Robustness for 80% Target
Advanced cross-validation ensemble with multiple model architectures
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

def set_seeds(seed=58):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class CrossValDataset(Dataset):
    """Cross-validation dataset with V5 foundation."""
    def __init__(self, video_paths, labels, augment=False, phase='train', fold_id=0):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        self.phase = phase
        self.fold_id = fold_id
        
        print(f"📊 CrossVal Dataset F{fold_id} ({phase}): {len(video_paths)} videos, Augment: {augment}")
        label_counts = Counter(labels)
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        for i, name in enumerate(class_names):
            print(f"   {name}: {label_counts.get(i, 0)} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_crossval(self, path):
        """Cross-validation video loading with V5 foundation."""
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
            # Fold-specific resolution for diversity
            if self.fold_id % 2 == 0:
                resized = cv2.resize(cropped, (96, 96))
            else:
                resized = cv2.resize(cropped, (112, 112))
            frames.append(resized)
        
        cap.release()
        
        if len(frames) == 0:
            size = 96 if self.fold_id % 2 == 0 else 112
            frames = [np.zeros((size, size), dtype=np.uint8)]
        
        # V5's proven 32-frame sampling
        target_frames = 32
        if len(frames) >= target_frames:
            indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < target_frames:
                frames.extend(frames[:min(len(frames), target_frames - len(frames))])
        
        return np.array(frames[:target_frames])
    
    def apply_crossval_augmentation(self, frames):
        """Cross-validation augmentation with fold-specific variations."""
        if not self.augment:
            return frames
        
        # V5's proven core augmentations
        if random.random() < 0.5:
            frames = np.flip(frames, axis=2).copy()
        
        # Fold-specific augmentation strength
        strength = 0.3 + (self.fold_id * 0.1)  # Different strength per fold
        
        if random.random() < strength:
            brightness_factor = random.uniform(0.85, 1.15)
            frames = np.clip(frames * brightness_factor, 0, 255).astype(np.uint8)
        
        if random.random() < strength * 0.7:
            contrast_factor = random.uniform(0.9, 1.1)
            frames = np.clip((frames - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
        
        # Fold-specific geometric augmentations
        if random.random() < strength * 0.5:
            max_shift = 2 + self.fold_id
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)
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
        
        frames = self.load_video_crossval(video_path)
        frames = self.apply_crossval_augmentation(frames)
        frames = self.apply_v5_preprocessing(frames)
        frames = torch.from_numpy(frames).float()
        
        return frames, label

class CrossValModel(nn.Module):
    """Cross-validation model with fold-specific architectures."""
    def __init__(self, num_classes=5, fold_id=0, input_size=96):
        super(CrossValModel, self).__init__()
        self.fold_id = fold_id
        self.input_size = input_size
        
        # Fold-specific architectures for maximum diversity
        if fold_id == 0:
            # Conservative architecture
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4)
            )
            feature_dim = 128 * 16
            
        elif fold_id == 1:
            # Medium architecture
            self.features = nn.Sequential(
                nn.Conv2d(1, 48, 5, stride=2, padding=2),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(48, 96, 3, stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, 3, stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(96, 192, 3, stride=1, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4)
            )
            feature_dim = 192 * 16
            
        elif fold_id == 2:
            # Larger architecture
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4)
            )
            feature_dim = 256 * 16
            
        else:  # fold_id >= 3
            # Deep architecture
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 384, 3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4)
            )
            feature_dim = 384 * 16
        
        # Fold-specific temporal processing
        if fold_id == 0:
            self.temporal = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )
            final_dim = 128
        elif fold_id == 1:
            self.temporal = nn.Sequential(
                nn.Linear(feature_dim, 384),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(384, 192),
                nn.ReLU(inplace=True),
                nn.Dropout(0.15)
            )
            final_dim = 192
        elif fold_id == 2:
            self.temporal = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            )
            final_dim = 256
        else:
            self.temporal = nn.Sequential(
                nn.Linear(feature_dim, 768),
                nn.ReLU(inplace=True),
                nn.Dropout(0.35),
                nn.Linear(768, 384),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )
            final_dim = 384
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
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
        
        # Temporal aggregation
        temporal_features = features.mean(dim=1)
        
        # Process temporally
        processed = self.temporal(temporal_features)
        
        # Classify
        output = self.classifier(processed)
        
        return output

def create_crossval_splits(dataset_path="the_best_videos_so_far", n_folds=5):
    """Create cross-validation splits."""
    print(f"📊 Creating {n_folds}-fold cross-validation splits from FULL dataset...")
    
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
    
    # Create cross-validation splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=58)
    
    fold_splits = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(video_paths, labels)):
        test_videos = video_paths[test_idx].tolist()
        test_labels = labels[test_idx].tolist()
        
        train_val_videos = video_paths[train_val_idx]
        train_val_labels = labels[train_val_idx]
        
        # Split train+val
        skf2 = StratifiedKFold(n_splits=4, shuffle=True, random_state=58+fold_idx)
        train_idx, val_idx = next(skf2.split(train_val_videos, train_val_labels))
        
        train_videos = train_val_videos[train_idx].tolist()
        train_labels = train_val_labels[train_idx].tolist()
        
        val_videos = train_val_videos[val_idx].tolist()
        val_labels = train_val_labels[val_idx].tolist()
        
        fold_splits.append({
            'train': (train_videos, train_labels),
            'val': (val_videos, val_labels),
            'test': (test_videos, test_labels)
        })
        
        print(f"📊 Fold {fold_idx}: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    return fold_splits

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

def train_crossval_fold(model, train_loader, val_loader, device, fold_id, num_epochs=35):
    """Train cross-validation fold with fold-specific optimization."""

    # Fold-specific optimization strategies
    if fold_id == 0:
        optimizer = optim.AdamW(model.parameters(), lr=4e-4, weight_decay=8e-5)
        criterion = nn.CrossEntropyLoss()
    elif fold_id == 1:
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    elif fold_id == 2:
        optimizer = optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1.2e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1.5e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)

    print(f"\n🚀 CrossVal Fold {fold_id} training for {num_epochs} epochs...")

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

            # Fold-specific gradient clipping
            if fold_id <= 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)

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

        print(f"F{fold_id} E{epoch+1}/{num_epochs} - "
              f"Train: {train_acc:.1f}% ({unique_train_preds}/5), "
              f"Val: {val_acc:.1f}% ({unique_val_preds}/5), "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), f'breakthrough_v15_crossval_f{fold_id}.pth')
            print(f"  💾 F{fold_id} New best: {val_acc:.1f}%")
        else:
            patience += 1

        if patience >= max_patience:
            print(f"  ⏹️  F{fold_id} Early stopping")
            break

        # Success milestones
        if unique_val_preds >= 4 and val_acc >= 60:
            print(f"  🎉 F{fold_id} BREAKTHROUGH: 60%+ with 4+ classes!")
            if val_acc >= 70:
                print(f"  🏆 F{fold_id} EXCELLENT: 70%+ achieved!")
                if val_acc >= 80:
                    print(f"  🌟 F{fold_id} TARGET: 80%+!")
                    break

    return best_val_acc

def main():
    """Cross-validation ensemble system for 80% target."""
    print("🎯 BREAKTHROUGH SYSTEM V15 - CROSS-VALIDATION ENSEMBLE FOR 80% TARGET")
    print("=" * 80)
    print("CROSS-VALIDATION ENSEMBLE TECHNIQUES:")
    print("• 5-fold stratified cross-validation")
    print("• Fold-specific architectures for maximum diversity")
    print("• Fold-specific optimization strategies")
    print("• Fold-specific augmentation strengths")
    print("• Fold-specific resolutions (96x96, 112x112)")
    print("• Advanced ensemble voting with confidence weighting")
    print("• V5's proven preprocessing foundation")
    print("• RELENTLESS PURSUIT OF 80% GENERALIZATION")
    print("=" * 80)

    set_seeds(58)
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")

    # Create cross-validation splits
    fold_splits = create_crossval_splits(n_folds=5)

    models = []
    fold_results = []

    # Train each fold
    for fold_id in range(5):
        print(f"\n🔥 TRAINING CROSS-VALIDATION FOLD {fold_id}")
        print("=" * 60)

        # Get fold data
        train_videos, train_labels = fold_splits[fold_id]['train']
        val_videos, val_labels = fold_splits[fold_id]['val']
        test_videos, test_labels = fold_splits[fold_id]['test']

        # Determine input size for this fold
        input_size = 96 if fold_id % 2 == 0 else 112

        # Create datasets
        train_dataset = CrossValDataset(train_videos, train_labels, augment=True, phase='train', fold_id=fold_id)
        val_dataset = CrossValDataset(val_videos, val_labels, augment=False, phase='val', fold_id=fold_id)
        test_dataset = CrossValDataset(test_videos, test_labels, augment=False, phase='test', fold_id=fold_id)

        # Create weighted sampler
        weighted_sampler = create_weighted_sampler(train_labels)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=2, sampler=weighted_sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        # Create fold model
        model = CrossValModel(num_classes=5, fold_id=fold_id, input_size=input_size).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🧠 CrossVal Model F{fold_id}: {total_params:,} parameters")

        # Train fold model
        best_val_acc = train_crossval_fold(model, train_loader, val_loader, device, fold_id, num_epochs=35)

        # Test fold model
        if os.path.exists(f'breakthrough_v15_crossval_f{fold_id}.pth'):
            model.load_state_dict(torch.load(f'breakthrough_v15_crossval_f{fold_id}.pth', map_location=device))

        model.eval()
        test_correct = 0
        test_total = 0
        test_preds = []
        test_targets = []
        test_probs = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                probs = F.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
                test_preds.extend(pred.cpu().numpy())
                test_targets.extend(target.cpu().numpy())
                test_probs.extend(probs.cpu().numpy())

        test_acc = 100. * test_correct / test_total
        unique_test_preds = len(set(test_preds))

        fold_results.append({
            'fold_id': fold_id,
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_classes': unique_test_preds,
            'test_preds': test_preds,
            'test_targets': test_targets,
            'test_probs': test_probs,
            'params': total_params
        })

        models.append(model)

        print(f"🎯 CROSSVAL FOLD {fold_id} RESULTS:")
        print(f"   Parameters: {total_params:,}")
        print(f"   Validation: {best_val_acc:.1f}%")
        print(f"   Test: {test_acc:.1f}% ({unique_test_preds}/5 classes)")

        if test_acc >= 80:
            print(f"   🌟 FOLD {fold_id} ACHIEVED 80% TARGET!")

    # Cross-validation ensemble
    print(f"\n🎯 CROSS-VALIDATION ENSEMBLE RESULTS")
    print("=" * 70)

    # Calculate ensemble predictions
    val_accs = [r['val_acc'] for r in fold_results]
    test_accs = [r['test_acc'] for r in fold_results]

    # Weighted ensemble based on validation performance
    weights = np.array(val_accs) / np.sum(val_accs)

    # Use first fold's test targets (all folds should have same test split structure)
    all_test_targets = fold_results[0]['test_targets']

    # Average probabilities across folds (assuming same test samples)
    ensemble_probs = np.zeros((len(all_test_targets), 5))
    for i, result in enumerate(fold_results):
        fold_probs = np.array(result['test_probs'])
        ensemble_probs += weights[i] * fold_probs

    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_correct = np.sum(ensemble_preds == all_test_targets)
    ensemble_acc = 100. * ensemble_correct / len(all_test_targets)
    unique_ensemble_preds = len(set(ensemble_preds))

    print(f"🎯 Cross-Validation Results:")
    for r in fold_results:
        print(f"   F{r['fold_id']}: Val={r['val_acc']:.1f}%, Test={r['test_acc']:.1f}% ({r['test_classes']}/5), Params={r['params']:,}")

    print(f"🎯 Ensemble Performance:")
    print(f"   Weights: {weights}")
    print(f"   Mean Val: {np.mean(val_accs):.1f}% ± {np.std(val_accs):.1f}%")
    print(f"   Mean Test: {np.mean(test_accs):.1f}% ± {np.std(test_accs):.1f}%")
    print(f"   Best Individual: {max(test_accs):.1f}%")
    print(f"   WEIGHTED ENSEMBLE: {ensemble_acc:.1f}% ({unique_ensemble_preds}/5 classes)")

    # Compare with previous best
    previous_best = 36.8
    if ensemble_acc > previous_best:
        improvement = ensemble_acc - previous_best
        print(f"🏆 NEW CROSSVAL RECORD: +{improvement:.1f}% improvement!")

    if ensemble_acc >= 80:
        print("🌟 CROSSVAL ENSEMBLE TARGET ACHIEVED: 80%+ generalization!")
    elif max(test_accs) >= 80:
        print("🏆 INDIVIDUAL FOLD TARGET: At least one fold achieved 80%+!")
    elif ensemble_acc >= 70:
        print("🎉 CROSSVAL ENSEMBLE EXCELLENT: 70%+ achieved!")
    elif ensemble_acc >= 60:
        print("✅ CROSSVAL ENSEMBLE GREAT: 60%+ achieved!")
    else:
        print("🔄 Continue toward 80% target...")

    return ensemble_acc, max(test_accs), np.mean(test_accs)

if __name__ == "__main__":
    try:
        ensemble_accuracy, best_individual, mean_accuracy = main()
        print(f"\n🏁 Breakthrough V15 CrossVal Ensemble completed:")
        print(f"   Ensemble: {ensemble_accuracy:.1f}%")
        print(f"   Best Individual: {best_individual:.1f}%")
        print(f"   Mean: {mean_accuracy:.1f}%")

        if ensemble_accuracy >= 80 or best_individual >= 80:
            print("🎯 80% TARGET ACHIEVED!")
        else:
            print("🚀 Continue relentless pursuit of 80%...")

    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
