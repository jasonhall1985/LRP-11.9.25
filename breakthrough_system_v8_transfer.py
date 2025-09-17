#!/usr/bin/env python3
"""
Breakthrough System V8 Transfer Learning - Pre-trained Features for 80% Target
Transfer learning from pre-trained models for maximum feature extraction
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
import torchvision.models as models
import torchvision.transforms as transforms

def set_seeds(seed=51):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class TransferDataset(Dataset):
    """Transfer learning dataset with V5 foundation + RGB conversion."""
    def __init__(self, video_paths, labels, augment=False, phase='train'):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        self.phase = phase
        
        print(f"📊 Transfer Dataset ({phase}): {len(video_paths)} videos, Augment: {augment}")
        label_counts = Counter(labels)
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        for i, name in enumerate(class_names):
            print(f"   {name}: {label_counts.get(i, 0)} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_for_transfer(self, path):
        """Load video optimized for transfer learning."""
        cap = cv2.VideoCapture(path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Keep as RGB for transfer learning
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # V5's proven ICU-style crop
            h, w = rgb.shape[:2]
            crop_h = int(0.5 * h)
            crop_w_start = int(0.335 * w)
            crop_w_end = int(0.665 * w)
            
            cropped = rgb[0:crop_h, crop_w_start:crop_w_end]
            # Resize to 224x224 for pre-trained models
            resized = cv2.resize(cropped, (224, 224))
            frames.append(resized)
        
        cap.release()
        
        if len(frames) == 0:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
        
        # V5's proven 32-frame sampling
        target_frames = 32
        if len(frames) >= target_frames:
            indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < target_frames:
                frames.extend(frames[:min(len(frames), target_frames - len(frames))])
        
        return np.array(frames[:target_frames])
    
    def apply_transfer_augmentation(self, frames):
        """Transfer learning compatible augmentation."""
        if not self.augment:
            return frames
        
        # V5's proven augmentations adapted for RGB
        if random.random() < 0.5:
            frames = np.flip(frames, axis=2).copy()  # Horizontal flip
        
        if random.random() < 0.3:
            brightness_factor = random.uniform(0.85, 1.15)
            frames = np.clip(frames * brightness_factor, 0, 255).astype(np.uint8)
        
        if random.random() < 0.2:
            contrast_factor = random.uniform(0.9, 1.1)
            frames = np.clip((frames - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
        
        # Additional transfer learning augmentations
        if random.random() < 0.15:
            # Color jittering
            hue_shift = random.uniform(-10, 10)
            hsv = cv2.cvtColor(frames[0], cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179)
            frames[0] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return frames
    
    def apply_transfer_preprocessing(self, frames):
        """Transfer learning preprocessing with ImageNet normalization."""
        # Convert to float and normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Apply V5's proven enhancement to each frame
        processed_frames = []
        for frame in frames:
            # Convert to grayscale for CLAHE, then back to RGB
            gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray).astype(np.float32) / 255.0
            
            # Apply enhancement to all channels
            enhancement_factor = np.mean(enhanced_gray) / (np.mean(frame) + 1e-8)
            enhanced_frame = np.clip(frame * enhancement_factor, 0, 1)
            
            processed_frames.append(enhanced_frame)
        
        frames = np.array(processed_frames)
        
        # ImageNet normalization for transfer learning
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        frames = (frames - mean) / std
        
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self.load_video_for_transfer(video_path)
        frames = self.apply_transfer_augmentation(frames)
        frames = self.apply_transfer_preprocessing(frames)
        
        # Convert to tensor (T, C, H, W)
        frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2)
        
        return frames, label

class TransferModel(nn.Module):
    """Transfer learning model with pre-trained backbone."""
    def __init__(self, num_classes=5, backbone='resnet18', freeze_backbone=False):
        super(TransferModel, self).__init__()
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone
        
        # Load pre-trained backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final layer
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=True)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:  # efficientnet_b0
            self.backbone = models.efficientnet_b0(pretrained=True)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Temporal aggregation
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Advanced temporal processing
        self.temporal_processor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_new_layers()
    
    def _initialize_new_layers(self):
        """Initialize only the new layers (not pre-trained backbone)."""
        for module in [self.temporal_processor, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # Process each frame through backbone
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)  # (B*T, feature_dim)
        features = features.view(B, T, -1)  # (B, T, feature_dim)
        
        # Temporal aggregation
        temporal_features = features.mean(dim=1)  # (B, feature_dim)
        
        # Process temporally
        processed = self.temporal_processor(temporal_features)
        
        # Classify
        output = self.classifier(processed)
        
        return output

def create_transfer_splits(dataset_path="the_best_videos_so_far"):
    """Create splits for transfer learning."""
    print("📊 Creating transfer learning splits from FULL dataset...")
    
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
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=51)
    train_val_idx, test_idx = next(skf.split(video_paths, labels))
    
    test_videos = video_paths[test_idx].tolist()
    test_labels = labels[test_idx].tolist()
    
    train_val_videos = video_paths[train_val_idx]
    train_val_labels = labels[train_val_idx]
    
    # Split train+val
    skf2 = StratifiedKFold(n_splits=4, shuffle=True, random_state=51)
    train_idx, val_idx = next(skf2.split(train_val_videos, train_val_labels))
    
    train_videos = train_val_videos[train_idx].tolist()
    train_labels = train_val_labels[train_idx].tolist()
    
    val_videos = train_val_videos[val_idx].tolist()
    val_labels = train_val_labels[val_idx].tolist()
    
    print(f"📊 Transfer splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def train_transfer_model(model, train_loader, val_loader, device, backbone_name, num_epochs=40):
    """Train transfer learning model with advanced techniques."""

    # Different learning rates for backbone and new layers
    if model.freeze_backbone:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        # Different learning rates for different parts
        backbone_params = list(model.backbone.parameters())
        new_params = list(model.temporal_processor.parameters()) + list(model.classifier.parameters())

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},  # Lower LR for pre-trained
            {'params': new_params, 'lr': 5e-4}       # Higher LR for new layers
        ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=6)

    # Advanced loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n🚀 Transfer training {backbone_name} for {num_epochs} epochs...")

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

        unique_train_preds = len(set(train_preds))
        unique_val_preds = len(set(val_preds))

        print(f"{backbone_name} E{epoch+1}/{num_epochs} - "
              f"Train: {train_acc:.1f}% ({unique_train_preds}/5), "
              f"Val: {val_acc:.1f}% ({unique_val_preds}/5)")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), f'breakthrough_v8_{backbone_name}.pth')
            print(f"  💾 {backbone_name} New best: {val_acc:.1f}%")
        else:
            patience += 1

        if patience >= max_patience:
            print(f"  ⏹️  {backbone_name} Early stopping")
            break

        # Success milestones
        if unique_val_preds >= 4 and val_acc >= 60:
            print(f"  🎉 {backbone_name} BREAKTHROUGH: 60%+ with 4+ classes!")
            if val_acc >= 70:
                print(f"  🏆 {backbone_name} EXCELLENT: 70%+ achieved!")
                if val_acc >= 80:
                    print(f"  🌟 {backbone_name} TARGET: 80%+!")
                    break

    return best_val_acc

def main():
    """Transfer learning system for 80% target."""
    print("🎯 BREAKTHROUGH SYSTEM V8 - TRANSFER LEARNING FOR 80% TARGET")
    print("=" * 80)
    print("TRANSFER LEARNING TECHNIQUES:")
    print("• Pre-trained ResNet18/34, MobileNetV2, EfficientNet-B0")
    print("• ImageNet features + lip-reading fine-tuning")
    print("• Different learning rates for backbone vs new layers")
    print("• RGB processing with V5's proven preprocessing")
    print("• Advanced temporal aggregation")
    print("• Label smoothing and gradient clipping")
    print("• Relentless pursuit of 80% generalization")
    print("=" * 80)

    set_seeds(51)
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")

    # Create transfer learning splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_transfer_splits()

    # Test multiple backbones
    backbones = ['resnet18', 'mobilenet_v2']  # Start with lighter models for CPU
    transfer_results = []

    for backbone_name in backbones:
        print(f"\n🔥 TRAINING TRANSFER MODEL: {backbone_name.upper()}")
        print("=" * 60)

        # Create datasets
        train_dataset = TransferDataset(train_videos, train_labels, augment=True, phase='train')
        val_dataset = TransferDataset(val_videos, val_labels, augment=False, phase='val')
        test_dataset = TransferDataset(test_videos, test_labels, augment=False, phase='test')

        # Create data loaders (smaller batch size for transfer learning)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        # Create transfer model
        model = TransferModel(num_classes=5, backbone=backbone_name, freeze_backbone=False).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🧠 {backbone_name}: {total_params:,} total, {trainable_params:,} trainable parameters")

        # Train transfer model
        best_val_acc = train_transfer_model(model, train_loader, val_loader, device, backbone_name, num_epochs=40)

        # Test transfer model
        if os.path.exists(f'breakthrough_v8_{backbone_name}.pth'):
            model.load_state_dict(torch.load(f'breakthrough_v8_{backbone_name}.pth', map_location=device))

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

        transfer_results.append({
            'backbone': backbone_name,
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_classes': unique_test_preds,
            'test_preds': test_preds,
            'test_targets': test_targets
        })

        print(f"🎯 {backbone_name.upper()} RESULTS:")
        print(f"   Validation: {best_val_acc:.1f}%")
        print(f"   Test: {test_acc:.1f}% ({unique_test_preds}/5 classes)")

        if test_acc >= 80:
            print(f"   🌟 {backbone_name.upper()} ACHIEVED 80% TARGET!")

    # Transfer learning summary
    val_accs = [r['val_acc'] for r in transfer_results]
    test_accs = [r['test_acc'] for r in transfer_results]

    print(f"\n🎯 BREAKTHROUGH SYSTEM V8 TRANSFER LEARNING RESULTS")
    print("=" * 70)
    print(f"🎯 Transfer Learning Models:")
    for r in transfer_results:
        print(f"   {r['backbone']}: Val={r['val_acc']:.1f}%, Test={r['test_acc']:.1f}% ({r['test_classes']}/5)")

    best_model = max(transfer_results, key=lambda x: x['test_acc'])
    print(f"🎯 Best Transfer Model: {best_model['backbone']} with {best_model['test_acc']:.1f}% test accuracy")

    if max(test_accs) >= 80:
        print("🌟 TRANSFER TARGET ACHIEVED: 80%+ generalization!")
    elif max(test_accs) >= 70:
        print("🏆 TRANSFER EXCELLENT: 70%+ achieved!")
    elif max(test_accs) >= 60:
        print("🎉 TRANSFER GREAT: 60%+ achieved!")
    else:
        print("🔄 Continue toward 80% target...")

    return max(test_accs), np.mean(test_accs)

if __name__ == "__main__":
    try:
        best_accuracy, mean_accuracy = main()
        print(f"\n🏁 Breakthrough V8 Transfer Learning completed:")
        print(f"   Best: {best_accuracy:.1f}%")
        print(f"   Mean: {mean_accuracy:.1f}%")

        if best_accuracy >= 80:
            print("🎯 80% TARGET ACHIEVED!")
        else:
            print("🚀 Continue relentless pursuit of 80%...")

    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
