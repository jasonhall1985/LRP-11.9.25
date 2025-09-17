#!/usr/bin/env python3
"""
Breakthrough System V2 - LSTM + Transformer Approach
Different architecture for 80% accuracy breakthrough
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
import math

def set_seeds(seed=43):  # Different seed for diversity
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class TransformerVideoDataset(Dataset):
    """Dataset optimized for transformer-based processing."""
    def __init__(self, video_paths, labels, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        
        print(f"📊 Transformer Dataset: {len(video_paths)} videos, Augment: {augment}")
        label_counts = Counter(labels)
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        for i, name in enumerate(class_names):
            print(f"   {name}: {label_counts.get(i, 0)} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_for_transformer(self, path):
        """Load video optimized for transformer processing."""
        cap = cv2.VideoCapture(path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale and resize to 128x128
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128))
            frames.append(resized)
        
        cap.release()
        
        if len(frames) == 0:
            frames = [np.zeros((128, 128), dtype=np.uint8)]
        
        # Use 48 frames for transformer processing
        target_frames = 48
        if len(frames) >= target_frames:
            # Multiple sampling strategies
            if self.augment and random.random() < 0.4:
                # Random sampling
                indices = sorted(random.sample(range(len(frames)), target_frames))
            else:
                # Uniform sampling
                indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            # Repeat frames to reach target
            while len(frames) < target_frames:
                frames.extend(frames[:min(len(frames), target_frames - len(frames))])
        
        return np.array(frames[:target_frames])
    
    def apply_transformer_augmentation(self, frames):
        """Apply augmentation optimized for transformer learning."""
        if not self.augment:
            return frames
        
        # Temporal augmentations
        if random.random() < 0.3:
            # Random frame dropping and duplication
            drop_indices = random.sample(range(len(frames)), random.randint(1, 5))
            for idx in sorted(drop_indices, reverse=True):
                if len(frames) > 32:  # Keep minimum frames
                    frames = np.delete(frames, idx, axis=0)
            
            # Pad back to target length
            while len(frames) < 48:
                frames = np.append(frames, [frames[-1]], axis=0)
        
        # Spatial augmentations
        if random.random() < 0.6:
            # Horizontal flip
            frames = np.flip(frames, axis=2).copy()
        
        if random.random() < 0.4:
            # Random brightness/contrast
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            frames = np.clip(frames * contrast + (brightness - 1) * 128, 0, 255).astype(np.uint8)
        
        if random.random() < 0.3:
            # Random crop and resize
            h, w = frames.shape[1], frames.shape[2]
            crop_size = random.randint(int(0.8 * min(h, w)), min(h, w))
            start_h = random.randint(0, h - crop_size)
            start_w = random.randint(0, w - crop_size)
            
            cropped_frames = []
            for frame in frames:
                cropped = frame[start_h:start_h+crop_size, start_w:start_w+crop_size]
                resized = cv2.resize(cropped, (128, 128))
                cropped_frames.append(resized)
            frames = np.array(cropped_frames)
        
        if random.random() < 0.2:
            # Add gaussian noise
            noise = np.random.normal(0, 10, frames.shape).astype(np.int16)
            frames = np.clip(frames.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load and augment video
        frames = self.load_video_for_transformer(video_path)
        frames = self.apply_transformer_augmentation(frames)
        
        # Normalize
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - 0.5) / 0.5  # [-1, 1] range
        
        # Convert to tensor (T, H, W)
        frames = torch.from_numpy(frames)
        
        return frames, label

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class LSTMTransformerModel(nn.Module):
    """Hybrid LSTM + Transformer model for lip reading."""
    def __init__(self, num_classes=5):
        super(LSTMTransformerModel, self).__init__()
        
        # CNN feature extractor for each frame
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        
        # Transformer for attention-based processing
        self.pos_encoding = PositionalEncoding(512, max_len=48)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
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
            elif isinstance(m, nn.LSTM):
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
        features = self.feature_extractor(x)  # (B*T, 256, 1, 1)
        features = features.view(B * T, -1)  # (B*T, 256)
        features = features.view(B, T, -1)  # (B, T, 256)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)  # (B, T, 512)
        
        # Transformer processing
        # Transpose for transformer: (T, B, 512)
        transformer_input = lstm_out.transpose(0, 1)
        transformer_input = self.pos_encoding(transformer_input)
        transformer_out = self.transformer(transformer_input)  # (T, B, 512)
        
        # Global temporal pooling
        transformer_out = transformer_out.transpose(0, 1)  # (B, T, 512)
        pooled = transformer_out.mean(dim=1)  # (B, 512)
        
        # Classification
        output = self.classifier(pooled)  # (B, num_classes)
        
        return output

def create_transformer_splits(dataset_path="the_best_videos_so_far"):
    """Create splits optimized for transformer training."""
    print("📊 Creating transformer splits from FULL dataset...")
    
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
    
    # Create balanced splits: 75% train, 12.5% val, 12.5% test
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []
    
    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
    
    for class_name, videos in class_videos.items():
        random.shuffle(videos)
        n_videos = len(videos)
        
        # 75% train, 12.5% val, 12.5% test
        n_train = max(1, int(0.75 * n_videos))
        n_val = max(1, int(0.125 * n_videos))
        
        train_videos.extend(videos[:n_train])
        train_labels.extend([class_to_idx[class_name]] * n_train)
        
        val_videos.extend(videos[n_train:n_train+n_val])
        val_labels.extend([class_to_idx[class_name]] * n_val)
        
        test_videos.extend(videos[n_train+n_val:])
        test_labels.extend([class_to_idx[class_name]] * (len(videos) - n_train - n_val))
    
    print(f"📊 Transformer splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def train_transformer_model(model, train_loader, val_loader, device, num_epochs=60):
    """Train transformer model with advanced techniques."""

    # Advanced optimizer with different learning rates
    backbone_params = []
    transformer_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if 'transformer' in name:
            transformer_params.append(param)
        elif 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': transformer_params, 'lr': 2e-4, 'weight_decay': 1e-5},
        {'params': classifier_params, 'lr': 5e-4, 'weight_decay': 1e-3}
    ])

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    # Label smoothing cross entropy
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

    print(f"\n🚀 Transformer training for {num_epochs} epochs...")

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

            # Gradient clipping for transformer stability
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
            torch.save(model.state_dict(), 'breakthrough_transformer_v2.pth')
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
    """Transformer breakthrough system main function."""
    print("🎯 BREAKTHROUGH SYSTEM V2 - LSTM + TRANSFORMER")
    print("=" * 80)
    print("TRANSFORMER TECHNIQUES:")
    print("• Hybrid CNN + LSTM + Transformer architecture")
    print("• Positional encoding for temporal understanding")
    print("• Multi-head attention mechanisms")
    print("• Bidirectional LSTM for sequence modeling")
    print("• 48-frame temporal processing")
    print("• Advanced transformer augmentation")
    print("• Differential learning rates")
    print("=" * 80)

    # Set seeds
    set_seeds(43)

    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")

    # Create transformer splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_transformer_splits()

    # Create datasets
    train_dataset = TransformerVideoDataset(train_videos, train_labels, augment=True)
    val_dataset = TransformerVideoDataset(val_videos, val_labels, augment=False)
    test_dataset = TransformerVideoDataset(test_videos, test_labels, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Create transformer model
    model = LSTMTransformerModel(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Transformer Model: {total_params:,} parameters")

    # Train
    best_val_acc = train_transformer_model(model, train_loader, val_loader, device, num_epochs=50)

    # Test
    print(f"\n🔍 Testing transformer model...")

    if os.path.exists('breakthrough_transformer_v2.pth'):
        model.load_state_dict(torch.load('breakthrough_transformer_v2.pth', map_location=device))
        print("📥 Loaded best transformer model")

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
        print(f"📊 Transformer Classification Report:\n{report}")

    print(f"\n🎯 BREAKTHROUGH SYSTEM V2 RESULTS")
    print("=" * 60)
    print(f"🎯 Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"🎯 Test Accuracy: {test_acc:.1f}%")
    print(f"🎯 Test Predictions: {sorted(set(test_preds))}")
    print(f"🎯 Test Targets: {sorted(set(test_targets))}")
    print(f"🎯 Unique Predictions: {unique_test_preds}/5 classes")
    print(f"🎯 Architecture: CNN + LSTM + Transformer")

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
        print("🔄 Continue to V3...")

    return test_acc, best_val_acc

if __name__ == "__main__":
    try:
        test_accuracy, val_accuracy = main()
        print(f"\n🏁 Breakthrough V2 completed:")
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
