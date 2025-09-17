#!/usr/bin/env python3
"""
Breakthrough System V11 Mega-Ensemble - Ultimate Ensemble for 80% Target
Combines multiple successful models with advanced ensemble techniques
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

def set_seeds(seed=54):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class MegaDataset(Dataset):
    """Mega dataset with V5 foundation + advanced techniques."""
    def __init__(self, video_paths, labels, augment=False, phase='train', model_id=0):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        self.phase = phase
        self.model_id = model_id
        
        print(f"📊 Mega Dataset M{model_id} ({phase}): {len(video_paths)} videos, Augment: {augment}")
        label_counts = Counter(labels)
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        for i, name in enumerate(class_names):
            print(f"   {name}: {label_counts.get(i, 0)} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_mega(self, path):
        """Mega video loading with model-specific variations."""
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
            
            # Model-specific resolutions for diversity
            if self.model_id == 0:
                resized = cv2.resize(cropped, (96, 96))   # V5 proven
            elif self.model_id == 1:
                resized = cv2.resize(cropped, (112, 112)) # V10 hybrid
            else:
                resized = cv2.resize(cropped, (128, 128)) # V9 ultra
            
            frames.append(resized)
        
        cap.release()
        
        if len(frames) == 0:
            frames = [np.zeros((96 if self.model_id == 0 else (112 if self.model_id == 1 else 128), 
                               96 if self.model_id == 0 else (112 if self.model_id == 1 else 128)), dtype=np.uint8)]
        
        # V5's proven 32-frame sampling
        target_frames = 32
        if len(frames) >= target_frames:
            indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < target_frames:
                frames.extend(frames[:min(len(frames), target_frames - len(frames))])
        
        return np.array(frames[:target_frames])
    
    def apply_mega_augmentation(self, frames):
        """Mega augmentation with model-specific strategies."""
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
        
        # Model-specific augmentations for diversity
        if self.model_id == 2 and random.random() < 0.1:
            # Geometric for model 2
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
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
        
        frames = self.load_video_mega(video_path)
        frames = self.apply_mega_augmentation(frames)
        frames = self.apply_v5_preprocessing(frames)
        frames = torch.from_numpy(frames).float()
        
        return frames, label

class MegaModel(nn.Module):
    """Mega model with architecture variations for ensemble diversity."""
    def __init__(self, num_classes=5, model_id=0):
        super(MegaModel, self).__init__()
        self.model_id = model_id
        
        # Model-specific architectures for maximum diversity
        if model_id == 0:
            # V5-style compact model
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
            
        elif model_id == 1:
            # V6/V7-style medium model
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
            
        else:  # model_id == 2
            # V9-style large model
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
                nn.Conv2d(256, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4)
            )
            feature_dim = 512 * 16
        
        # Model-specific temporal processing
        if model_id == 0:
            self.temporal = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )
            final_dim = 128
        elif model_id == 1:
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
                nn.Linear(feature_dim, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            )
            final_dim = 512
        
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

def create_mega_splits(dataset_path="the_best_videos_so_far"):
    """Create splits for mega ensemble."""
    print("📊 Creating mega ensemble splits from FULL dataset...")
    
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
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=54)
    train_val_idx, test_idx = next(skf.split(video_paths, labels))
    
    test_videos = video_paths[test_idx].tolist()
    test_labels = labels[test_idx].tolist()
    
    train_val_videos = video_paths[train_val_idx]
    train_val_labels = labels[train_val_idx]
    
    # Split train+val
    skf2 = StratifiedKFold(n_splits=4, shuffle=True, random_state=54)
    train_idx, val_idx = next(skf2.split(train_val_videos, train_val_labels))
    
    train_videos = train_val_videos[train_idx].tolist()
    train_labels = train_val_labels[train_idx].tolist()
    
    val_videos = train_val_videos[val_idx].tolist()
    val_labels = train_val_labels[val_idx].tolist()
    
    print(f"📊 Mega splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def train_mega_model(model, train_loader, val_loader, device, model_id, num_epochs=30):
    """Train mega model with model-specific optimization."""

    # Model-specific optimization strategies
    if model_id == 0:
        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
    elif model_id == 1:
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=8e-5)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1.2e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)

    print(f"\n🚀 Mega training Model-{model_id} for {num_epochs} epochs...")

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

            # Model-specific gradient clipping
            if model_id == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            elif model_id == 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            else:
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

        print(f"M{model_id} E{epoch+1}/{num_epochs} - "
              f"Train: {train_acc:.1f}% ({unique_train_preds}/5), "
              f"Val: {val_acc:.1f}% ({unique_val_preds}/5), "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), f'breakthrough_v11_mega_m{model_id}.pth')
            print(f"  💾 M{model_id} New best: {val_acc:.1f}%")
        else:
            patience += 1

        if patience >= max_patience:
            print(f"  ⏹️  M{model_id} Early stopping")
            break

        # Success milestones
        if unique_val_preds >= 4 and val_acc >= 60:
            print(f"  🎉 M{model_id} BREAKTHROUGH: 60%+ with 4+ classes!")
            if val_acc >= 70:
                print(f"  🏆 M{model_id} EXCELLENT: 70%+ achieved!")
                if val_acc >= 80:
                    print(f"  🌟 M{model_id} TARGET: 80%+!")
                    break

    return best_val_acc

def main():
    """Mega ensemble system for 80% target."""
    print("🎯 BREAKTHROUGH SYSTEM V11 - MEGA ENSEMBLE FOR 80% TARGET")
    print("=" * 80)
    print("MEGA ENSEMBLE TECHNIQUES:")
    print("• 3 diverse models with different architectures")
    print("• Model-specific resolutions (96x96, 112x112, 128x128)")
    print("• Model-specific optimization strategies")
    print("• Advanced ensemble voting with confidence weighting")
    print("• V5's proven preprocessing foundation")
    print("• Maximum diversity for robust predictions")
    print("• Relentless pursuit of 80% generalization")
    print("=" * 80)

    set_seeds(54)
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")

    # Create mega splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_mega_splits()

    models = []
    model_results = []

    # Train 3 diverse models
    for model_id in range(3):
        print(f"\n🔥 TRAINING MEGA MODEL {model_id}")
        print("=" * 50)

        # Create datasets
        train_dataset = MegaDataset(train_videos, train_labels, augment=True, phase='train', model_id=model_id)
        val_dataset = MegaDataset(val_videos, val_labels, augment=False, phase='val', model_id=model_id)
        test_dataset = MegaDataset(test_videos, test_labels, augment=False, phase='test', model_id=model_id)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        # Create mega model
        model = MegaModel(num_classes=5, model_id=model_id).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🧠 Mega Model-{model_id}: {total_params:,} parameters")

        # Train mega model
        best_val_acc = train_mega_model(model, train_loader, val_loader, device, model_id, num_epochs=30)

        # Test mega model
        if os.path.exists(f'breakthrough_v11_mega_m{model_id}.pth'):
            model.load_state_dict(torch.load(f'breakthrough_v11_mega_m{model_id}.pth', map_location=device))

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

        model_results.append({
            'model_id': model_id,
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_classes': unique_test_preds,
            'test_preds': test_preds,
            'test_targets': test_targets,
            'test_probs': test_probs
        })

        models.append(model)

        print(f"🎯 MEGA MODEL {model_id} RESULTS:")
        print(f"   Validation: {best_val_acc:.1f}%")
        print(f"   Test: {test_acc:.1f}% ({unique_test_preds}/5 classes)")

        if test_acc >= 80:
            print(f"   🌟 MODEL {model_id} ACHIEVED 80% TARGET!")

    # Advanced ensemble with confidence weighting
    print(f"\n🎯 MEGA ENSEMBLE VOTING")
    print("=" * 50)

    # Weighted ensemble based on validation performance
    val_accs = [r['val_acc'] for r in model_results]
    weights = np.array(val_accs) / np.sum(val_accs)

    all_probs = np.array([r['test_probs'] for r in model_results])  # (3, n_samples, 5)
    weighted_probs = np.average(all_probs, axis=0, weights=weights)  # (n_samples, 5)
    ensemble_preds = np.argmax(weighted_probs, axis=1)

    test_targets = model_results[0]['test_targets']
    ensemble_correct = np.sum(ensemble_preds == test_targets)
    ensemble_acc = 100. * ensemble_correct / len(test_targets)
    unique_ensemble_preds = len(set(ensemble_preds))

    # Results summary
    test_accs = [r['test_acc'] for r in model_results]

    print(f"🎯 BREAKTHROUGH SYSTEM V11 MEGA ENSEMBLE RESULTS")
    print("=" * 70)
    print(f"🎯 Individual Models:")
    for r in model_results:
        print(f"   M{r['model_id']}: Val={r['val_acc']:.1f}%, Test={r['test_acc']:.1f}% ({r['test_classes']}/5)")

    print(f"🎯 Ensemble Performance:")
    print(f"   Weights: {weights}")
    print(f"   Mean Test: {np.mean(test_accs):.1f}% ± {np.std(test_accs):.1f}%")
    print(f"   Best Individual: {max(test_accs):.1f}%")
    print(f"   WEIGHTED ENSEMBLE: {ensemble_acc:.1f}% ({unique_ensemble_preds}/5 classes)")

    # Compare with previous best
    previous_best = 36.8
    if ensemble_acc > previous_best:
        improvement = ensemble_acc - previous_best
        print(f"🏆 NEW RECORD: +{improvement:.1f}% improvement!")

    if ensemble_acc >= 80:
        print("🌟 MEGA ENSEMBLE TARGET ACHIEVED: 80%+ generalization!")
    elif max(test_accs) >= 80:
        print("🏆 INDIVIDUAL TARGET: At least one model achieved 80%+!")
    elif ensemble_acc >= 70:
        print("🎉 MEGA ENSEMBLE EXCELLENT: 70%+ achieved!")
    elif ensemble_acc >= 60:
        print("✅ MEGA ENSEMBLE GREAT: 60%+ achieved!")
    else:
        print("🔄 Continue toward 80% target...")

    return ensemble_acc, max(test_accs)

if __name__ == "__main__":
    try:
        ensemble_accuracy, best_individual = main()
        print(f"\n🏁 Breakthrough V11 Mega Ensemble completed:")
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
