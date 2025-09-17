#!/usr/bin/env python3
"""
Ultra Simple Training - Focus on Data Pipeline
Minimal complexity, maximum debugging visibility
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

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class UltraSimpleDataset(Dataset):
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels
        
        # Debug: Print dataset info
        print(f"📊 Dataset: {len(video_paths)} videos")
        label_counts = Counter(labels)
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        for i, name in enumerate(class_names):
            print(f"   {name}: {label_counts.get(i, 0)} videos")
        
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_simple(self, path):
        """Ultra simple video loading."""
        cap = cv2.VideoCapture(path)
        frames = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale and resize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))  # Even smaller for speed
            frames.append(resized)
            frame_count += 1
        
        cap.release()
        
        # Take exactly 16 frames (shorter sequence)
        if len(frames) >= 16:
            # Take evenly spaced frames
            indices = np.linspace(0, len(frames)-1, 16, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            # Repeat last frame
            while len(frames) < 16:
                frames.append(frames[-1] if frames else np.zeros((64, 64), dtype=np.uint8))
        
        return np.array(frames[:16])
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video
        frames = self.load_video_simple(video_path)
        
        # Normalize
        frames = frames.astype(np.float32) / 255.0
        
        # Add channel dimension: (C, T, H, W)
        frames = torch.from_numpy(frames).unsqueeze(0)
        
        return frames, label

class UltraSimpleModel(nn.Module):
    def __init__(self, num_classes=5):
        super(UltraSimpleModel, self).__init__()
        
        # Ultra simple 3D CNN
        self.conv1 = nn.Conv3d(1, 16, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(32, 64, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def analyze_dataset(dataset_path="corrected_balanced_dataset"):
    """Analyze the dataset structure."""
    print("🔍 ANALYZING DATASET")
    print("=" * 30)
    
    video_files = list(Path(dataset_path).glob("*.mp4"))
    print(f"Total videos: {len(video_files)}")
    
    # Analyze by class
    class_info = {}
    for video_file in video_files:
        parts = video_file.stem.split('_')
        if len(parts) >= 2:
            class_name = parts[0]
            speaker_id = parts[1]
            
            if class_name not in class_info:
                class_info[class_name] = {'videos': [], 'speakers': set()}
            
            class_info[class_name]['videos'].append(str(video_file))
            class_info[class_name]['speakers'].add(speaker_id)
    
    print("\n📊 Class Analysis:")
    for class_name, info in class_info.items():
        print(f"{class_name}: {len(info['videos'])} videos, {len(info['speakers'])} speakers")
        print(f"  Speakers: {sorted(info['speakers'])}")
    
    return class_info

def create_ultra_simple_splits(class_info):
    """Create ultra simple splits with clear separation."""
    print("\n📊 Creating ultra simple splits...")
    
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []
    
    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
    
    for class_name, info in class_info.items():
        videos = info['videos']
        random.shuffle(videos)
        
        # Simple split: 60% train, 20% val, 20% test
        n_videos = len(videos)
        n_train = max(1, int(0.6 * n_videos))
        n_val = max(1, int(0.2 * n_videos))
        
        train_videos.extend(videos[:n_train])
        train_labels.extend([class_to_idx[class_name]] * n_train)
        
        val_videos.extend(videos[n_train:n_train+n_val])
        val_labels.extend([class_to_idx[class_name]] * n_val)
        
        test_videos.extend(videos[n_train+n_val:])
        test_labels.extend([class_to_idx[class_name]] * (len(videos) - n_train - n_val))
    
    print(f"Splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def train_ultra_simple(model, train_loader, val_loader, device, num_epochs=5):
    """Ultra simple training with detailed logging."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)  # Higher learning rate
    
    print(f"\n🚀 Ultra simple training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            # Detailed batch logging
            batch_acc = 100. * pred.eq(target).sum().item() / target.size(0)
            print(f"  Batch {batch_idx+1}: Loss={loss.item():.4f}, Acc={batch_acc:.1f}%, "
                  f"Pred={pred.cpu().numpy()}, Target={target.cpu().numpy()}")
        
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
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Acc: {train_acc:.1f}%")
        print(f"  Val Acc: {val_acc:.1f}%")
        print(f"  Val Predictions: {val_preds}")
        print(f"  Val Targets: {val_targets}")
        
        # Check if model is learning different classes
        unique_preds = len(set(val_preds))
        print(f"  Unique predictions: {unique_preds}/5 classes")
        
        if unique_preds > 1:
            print("  ✅ Model is predicting multiple classes!")
        else:
            print("  ⚠️  Model stuck on one class")

def main():
    """Ultra simple main function."""
    print("🎯 ULTRA SIMPLE TRAINING")
    print("=" * 40)
    print("Minimal complexity, maximum visibility")
    print("64x64 images, 16 frames, tiny model")
    print("=" * 40)
    
    # Set seeds
    set_seeds(42)
    
    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")
    
    # Analyze dataset
    class_info = analyze_dataset()
    
    # Create splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_ultra_simple_splits(class_info)
    
    # Create datasets
    train_dataset = UltraSimpleDataset(train_videos, train_labels)
    val_dataset = UltraSimpleDataset(val_videos, val_labels)
    test_dataset = UltraSimpleDataset(test_videos, test_labels)
    
    # Create data loaders (batch size 1 for debugging)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Create ultra simple model
    model = UltraSimpleModel(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Ultra Simple Model: {total_params:,} parameters")
    
    # Train
    train_ultra_simple(model, train_loader, val_loader, device, num_epochs=3)
    
    # Quick test
    print(f"\n🔍 Quick test...")
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
    
    print(f"\n🎯 ULTRA SIMPLE RESULTS")
    print("=" * 30)
    print(f"🎯 Test Accuracy: {test_acc:.1f}%")
    print(f"🎯 Test Predictions: {test_preds}")
    print(f"🎯 Test Targets: {test_targets}")
    print(f"🎯 Unique Predictions: {unique_test_preds}/5 classes")
    
    if unique_test_preds > 1:
        print("✅ SUCCESS: Model predicts multiple classes!")
        print("🚀 Ready to scale up complexity")
    else:
        print("⚠️  ISSUE: Model stuck on one class")
        print("🔍 Need to investigate data pipeline")
    
    return test_acc

if __name__ == "__main__":
    try:
        final_accuracy = main()
        print(f"\n🏁 Ultra simple training completed: {final_accuracy:.1f}% accuracy")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
