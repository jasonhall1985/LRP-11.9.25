#!/usr/bin/env python3
"""
Breakthrough System V7 Ensemble - Ultimate Ensemble for 80% Target
Multiple specialized models with ensemble voting for maximum accuracy
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

def set_seeds(seed=50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class EnsembleDataset(Dataset):
    """Ensemble dataset with V5 foundation + specialized augmentations."""
    def __init__(self, video_paths, labels, augment=False, phase='train', specialist_id=0):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        self.phase = phase
        self.specialist_id = specialist_id
        
        print(f"📊 Ensemble Dataset Specialist-{specialist_id} ({phase}): {len(video_paths)} videos, Augment: {augment}")
        label_counts = Counter(labels)
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        for i, name in enumerate(class_names):
            print(f"   {name}: {label_counts.get(i, 0)} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_v5_foundation(self, path):
        """V5's proven video loading."""
        cap = cv2.VideoCapture(path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            h, w = gray.shape
            crop_h = int(0.5 * h)
            crop_w_start = int(0.335 * w)
            crop_w_end = int(0.665 * w)
            
            cropped = gray[0:crop_h, crop_w_start:crop_w_end]
            resized = cv2.resize(cropped, (96, 96))
            frames.append(resized)
        
        cap.release()
        
        if len(frames) == 0:
            frames = [np.zeros((96, 96), dtype=np.uint8)]
        
        target_frames = 32
        if len(frames) >= target_frames:
            indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < target_frames:
                frames.extend(frames[:min(len(frames), target_frames - len(frames))])
        
        return np.array(frames[:target_frames])
    
    def apply_specialist_augmentation(self, frames):
        """Specialist augmentation based on model ID."""
        if not self.augment:
            return frames
        
        # Base V5 augmentations (all specialists)
        if random.random() < 0.5:
            frames = np.flip(frames, axis=2).copy()
        
        if random.random() < 0.3:
            brightness_factor = random.uniform(0.85, 1.15)
            frames = np.clip(frames * brightness_factor, 0, 255).astype(np.uint8)
        
        if random.random() < 0.2:
            contrast_factor = random.uniform(0.9, 1.1)
            frames = np.clip((frames - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
        
        # Specialist-specific augmentations
        if self.specialist_id == 0:
            # Specialist 0: Conservative (minimal additional augmentation)
            pass
        elif self.specialist_id == 1:
            # Specialist 1: Spatial focus
            if random.random() < 0.2:
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                h, w = frames.shape[1], frames.shape[2]
                translated_frames = []
                for frame in frames:
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    translated = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                    translated_frames.append(translated)
                frames = np.array(translated_frames)
        elif self.specialist_id == 2:
            # Specialist 2: Photometric focus
            if random.random() < 0.15:
                gamma = random.uniform(0.8, 1.2)
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
                gamma_frames = []
                for frame in frames:
                    gamma_corrected = cv2.LUT(frame, table)
                    gamma_frames.append(gamma_corrected)
                frames = np.array(gamma_frames)
        elif self.specialist_id == 3:
            # Specialist 3: Noise robustness
            if random.random() < 0.1:
                noise_std = random.uniform(1, 3)
                noise = np.random.normal(0, noise_std, frames.shape).astype(np.int16)
                frames = np.clip(frames.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        else:  # specialist_id == 4
            # Specialist 4: Rotation robustness
            if random.random() < 0.15:
                angle = random.uniform(-2, 2)
                h, w = frames.shape[1], frames.shape[2]
                center = (w // 2, h // 2)
                rotated_frames = []
                for frame in frames:
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                    rotated_frames.append(rotated)
                frames = np.array(rotated_frames)
        
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
        
        frames = self.load_video_v5_foundation(video_path)
        frames = self.apply_specialist_augmentation(frames)
        frames = self.apply_v5_preprocessing(frames)
        frames = torch.from_numpy(frames).float()
        
        return frames, label

class SpecialistModel(nn.Module):
    """Specialist model with different architectures for ensemble diversity."""
    def __init__(self, num_classes=5, specialist_id=0):
        super(SpecialistModel, self).__init__()
        self.specialist_id = specialist_id
        
        if specialist_id == 0:
            # Specialist 0: V5's proven architecture
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 7, stride=2, padding=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(3)
            )
            feature_dim = 128 * 9
            
        elif specialist_id == 1:
            # Specialist 1: Deeper network
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 7, stride=2, padding=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(2)
            )
            feature_dim = 256 * 4
            
        elif specialist_id == 2:
            # Specialist 2: Wider network
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(3)
            )
            feature_dim = 256 * 9
            
        elif specialist_id == 3:
            # Specialist 3: Compact efficient network
            self.features = nn.Sequential(
                nn.Conv2d(1, 48, 5, stride=2, padding=2),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(48, 96, 3, stride=2, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(96, 192, 3, stride=1, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(3)
            )
            feature_dim = 192 * 9
            
        else:  # specialist_id == 4
            # Specialist 4: Alternative architecture
            self.features = nn.Sequential(
                nn.Conv2d(1, 40, 6, stride=2, padding=2),
                nn.BatchNorm2d(40),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(40, 80, 4, stride=2, padding=1),
                nn.BatchNorm2d(80),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(80, 160, 3, stride=1, padding=1),
                nn.BatchNorm2d(160),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4)
            )
            feature_dim = 160 * 16
        
        # Specialist-specific temporal processing
        if specialist_id in [0, 1]:
            # Conservative temporal processing
            self.temporal = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.15)
            )
        else:
            # Aggressive temporal processing
            self.temporal = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.35),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True)
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
        
        x = x.view(B * T, 1, H, W)
        features = self.features(x)
        features = features.view(B * T, -1)
        features = features.view(B, T, -1)
        
        temporal_features = features.mean(dim=1)
        processed = self.temporal(temporal_features)
        output = self.classifier(processed)
        
        return output

def create_ensemble_splits(dataset_path="the_best_videos_so_far"):
    """Create splits optimized for ensemble training."""
    print("📊 Creating ensemble splits from FULL dataset...")

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

    # Stratified split for ensemble
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
    train_val_idx, test_idx = next(skf.split(video_paths, labels))

    test_videos = video_paths[test_idx].tolist()
    test_labels = labels[test_idx].tolist()

    train_val_videos = video_paths[train_val_idx]
    train_val_labels = labels[train_val_idx]

    # Split train+val
    skf2 = StratifiedKFold(n_splits=4, shuffle=True, random_state=50)
    train_idx, val_idx = next(skf2.split(train_val_videos, train_val_labels))

    train_videos = train_val_videos[train_idx].tolist()
    train_labels = train_val_labels[train_idx].tolist()

    val_videos = train_val_videos[val_idx].tolist()
    val_labels = train_val_labels[val_idx].tolist()

    print(f"📊 Ensemble splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")

    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def train_specialist_model(model, train_loader, val_loader, device, specialist_id, num_epochs=45):
    """Train specialist model with optimized settings."""

    # Specialist-specific optimization
    if specialist_id == 0:
        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    elif specialist_id == 1:
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=8e-5)
    elif specialist_id == 2:
        optimizer = optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1.2e-4)
    elif specialist_id == 3:
        optimizer = optim.AdamW(model.parameters(), lr=6e-4, weight_decay=6e-5)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=4.5e-4, weight_decay=9e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.65, patience=7)

    # Specialist-specific loss
    if specialist_id in [0, 1]:
        criterion = nn.CrossEntropyLoss()  # Conservative
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Aggressive

    print(f"\n🚀 Training Specialist-{specialist_id} for {num_epochs} epochs...")

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

            # Specialist-specific gradient clipping
            if specialist_id in [0, 1]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)

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

        print(f"S{specialist_id} E{epoch+1}/{num_epochs} - "
              f"Train: {train_acc:.1f}% ({unique_train_preds}/5), "
              f"Val: {val_acc:.1f}% ({unique_val_preds}/5), "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), f'breakthrough_v7_specialist_{specialist_id}.pth')
            print(f"  💾 S{specialist_id} New best: {val_acc:.1f}%")
        else:
            patience += 1

        if patience >= max_patience:
            print(f"  ⏹️  S{specialist_id} Early stopping")
            break

        # Success milestones
        if unique_val_preds >= 4 and val_acc >= 60:
            print(f"  🎉 S{specialist_id} BREAKTHROUGH: 60%+ with 4+ classes!")
            if val_acc >= 70:
                print(f"  🏆 S{specialist_id} EXCELLENT: 70%+ achieved!")
                if val_acc >= 80:
                    print(f"  🌟 S{specialist_id} TARGET: 80%+!")
                    break

    return best_val_acc

def main():
    """Ultimate ensemble system for 80% target."""
    print("🎯 BREAKTHROUGH SYSTEM V7 - ULTIMATE ENSEMBLE FOR 80% TARGET")
    print("=" * 80)
    print("ULTIMATE TECHNIQUES:")
    print("• 5 specialist models with diverse architectures")
    print("• Specialist-specific augmentation strategies")
    print("• Optimized hyperparameters per specialist")
    print("• Ensemble voting for maximum accuracy")
    print("• V5's proven preprocessing foundation")
    print("• Relentless pursuit of 80% generalization")
    print("=" * 80)

    set_seeds(50)
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")

    # Create ensemble splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_ensemble_splits()

    specialists = []
    specialist_results = []

    # Train each specialist
    for specialist_id in range(5):
        print(f"\n🔥 TRAINING SPECIALIST {specialist_id}")
        print("=" * 50)

        # Create datasets
        train_dataset = EnsembleDataset(train_videos, train_labels, augment=True, phase='train', specialist_id=specialist_id)
        val_dataset = EnsembleDataset(val_videos, val_labels, augment=False, phase='val', specialist_id=specialist_id)
        test_dataset = EnsembleDataset(test_videos, test_labels, augment=False, phase='test', specialist_id=specialist_id)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        # Create specialist model
        model = SpecialistModel(num_classes=5, specialist_id=specialist_id).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🧠 Specialist-{specialist_id}: {total_params:,} parameters")

        # Train specialist
        best_val_acc = train_specialist_model(model, train_loader, val_loader, device, specialist_id, num_epochs=45)

        # Test specialist
        if os.path.exists(f'breakthrough_v7_specialist_{specialist_id}.pth'):
            model.load_state_dict(torch.load(f'breakthrough_v7_specialist_{specialist_id}.pth', map_location=device))

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

        specialist_results.append({
            'specialist_id': specialist_id,
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_classes': unique_test_preds,
            'test_preds': test_preds,
            'test_targets': test_targets,
            'test_probs': test_probs
        })

        specialists.append(model)

        print(f"🎯 SPECIALIST {specialist_id} RESULTS:")
        print(f"   Validation: {best_val_acc:.1f}%")
        print(f"   Test: {test_acc:.1f}% ({unique_test_preds}/5 classes)")

        if test_acc >= 80:
            print(f"   🌟 SPECIALIST {specialist_id} ACHIEVED 80% TARGET!")

    # Ensemble predictions
    print(f"\n🎯 ENSEMBLE VOTING")
    print("=" * 50)

    # Collect all probabilities
    all_probs = np.array([r['test_probs'] for r in specialist_results])  # (5, n_samples, 5)
    ensemble_probs = np.mean(all_probs, axis=0)  # (n_samples, 5)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    test_targets = specialist_results[0]['test_targets']
    ensemble_correct = np.sum(ensemble_preds == test_targets)
    ensemble_acc = 100. * ensemble_correct / len(test_targets)
    unique_ensemble_preds = len(set(ensemble_preds))

    # Individual specialist summary
    val_accs = [r['val_acc'] for r in specialist_results]
    test_accs = [r['test_acc'] for r in specialist_results]

    print(f"🎯 BREAKTHROUGH SYSTEM V7 ENSEMBLE RESULTS")
    print("=" * 70)
    print(f"🎯 Individual Specialists:")
    for r in specialist_results:
        print(f"   S{r['specialist_id']}: Val={r['val_acc']:.1f}%, Test={r['test_acc']:.1f}% ({r['test_classes']}/5)")

    print(f"🎯 Ensemble Performance:")
    print(f"   Mean Validation: {np.mean(val_accs):.1f}% ± {np.std(val_accs):.1f}%")
    print(f"   Mean Test: {np.mean(test_accs):.1f}% ± {np.std(test_accs):.1f}%")
    print(f"   Best Individual: {max(test_accs):.1f}%")
    print(f"   ENSEMBLE TEST: {ensemble_acc:.1f}% ({unique_ensemble_preds}/5 classes)")

    if ensemble_acc >= 80:
        print("🌟 ENSEMBLE TARGET ACHIEVED: 80%+ generalization!")
    elif max(test_accs) >= 80:
        print("🏆 INDIVIDUAL TARGET: At least one specialist achieved 80%+!")
    elif ensemble_acc >= 70:
        print("🎉 ENSEMBLE EXCELLENT: 70%+ achieved!")
    elif ensemble_acc >= 60:
        print("✅ ENSEMBLE GREAT: 60%+ achieved!")
    else:
        print("🔄 Continue toward 80% target...")

    return ensemble_acc, max(test_accs)

if __name__ == "__main__":
    try:
        ensemble_accuracy, best_individual = main()
        print(f"\n🏁 Breakthrough V7 Ensemble completed:")
        print(f"   Ensemble: {ensemble_accuracy:.1f}%")
        print(f"   Best Individual: {best_individual:.1f}%")

        if ensemble_accuracy >= 80 or best_individual >= 80:
            print("🎯 80% TARGET ACHIEVED!")
        else:
            print("🚀 Continue relentless pursuit of 80%...")

    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
