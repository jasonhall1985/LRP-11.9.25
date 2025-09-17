#!/usr/bin/env python3
"""
Fresh Start Training - Simple and Fast
Get results quickly with a clean, working approach
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
from sklearn.metrics import accuracy_score, classification_report
import time

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class SimpleLipDataset(Dataset):
    def __init__(self, video_paths, labels, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        print(f"📊 Dataset: {len(video_paths)} videos, Augment: {augment}")
        
    def __len__(self):
        return len(self.video_paths)
    
    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize to 112x112
            resized = cv2.resize(gray, (112, 112))
            frames.append(resized)
        
        cap.release()
        
        # Sample 32 frames
        if len(frames) >= 32:
            indices = np.linspace(0, len(frames)-1, 32, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            # Pad with last frame
            while len(frames) < 32:
                frames.append(frames[-1] if frames else np.zeros((112, 112), dtype=np.uint8))
        
        return np.array(frames[:32])
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video
        frames = self.load_video(video_path)
        
        # Simple augmentation for training
        if self.augment and random.random() < 0.3:
            # Random brightness
            brightness = random.uniform(0.9, 1.1)
            frames = np.clip(frames * brightness, 0, 255).astype(np.uint8)
        
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor (C, T, H, W)
        frames = torch.from_numpy(frames).unsqueeze(0)
        
        return frames, label

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(Simple3DCNN, self).__init__()
        
        # Simple 3D CNN
        self.conv1 = nn.Conv3d(1, 32, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.pool1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.conv2 = nn.Conv3d(32, 64, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 128, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

def create_simple_splits(dataset_path="corrected_balanced_dataset"):
    """Create simple train/val/test splits."""
    print("📊 Creating simple data splits...")
    
    video_files = list(Path(dataset_path).glob("*.mp4"))
    print(f"Found {len(video_files)} videos")
    
    # Group by class
    class_videos = {'doctor': [], 'glasses': [], 'help': [], 'phone': [], 'pillow': []}
    
    for video_file in video_files:
        class_name = video_file.stem.split('_')[0]
        if class_name in class_videos:
            class_videos[class_name].append(str(video_file))
    
    # Create splits: 7 train, 2 val, 1 test per class
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []
    
    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
    
    for class_name, videos in class_videos.items():
        random.shuffle(videos)
        
        # 7 for training
        train_videos.extend(videos[:7])
        train_labels.extend([class_to_idx[class_name]] * 7)
        
        # 2 for validation
        if len(videos) > 7:
            val_videos.extend(videos[7:9])
            val_labels.extend([class_to_idx[class_name]] * min(2, len(videos) - 7))
        
        # 1 for testing
        if len(videos) > 9:
            test_videos.append(videos[9])
            test_labels.append(class_to_idx[class_name])
    
    print(f"📊 Splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    """Simple training loop."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    
    best_val_acc = 0.0
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
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
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Acc: {100.*train_correct/train_total:.1f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_simple_model.pth')
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%, "
              f"Best Val: {best_val_acc:.1f}%, LR: {scheduler.get_last_lr()[0]:.2e}")
    
    return best_val_acc

def test_model(model, test_loader, device):
    """Test the model."""
    print("\n🔍 Testing model...")
    
    # Load best model
    if os.path.exists('best_simple_model.pth'):
        model.load_state_dict(torch.load('best_simple_model.pth', map_location=device))
        print("📥 Loaded best model")
    
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    
    # Classification report
    class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
    if len(set(all_targets)) > 1:
        report = classification_report(all_targets, all_preds, target_names=class_names)
        print(f"📊 Classification Report:\n{report}")
    
    return test_acc

def main():
    """Main training function."""
    print("🎯 FRESH START TRAINING")
    print("=" * 40)
    print("Simple 3D CNN on lip-reading dataset")
    print("Target: Get working results quickly")
    print("=" * 40)
    
    # Set seeds
    set_seeds(42)
    
    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")
    
    # Create data splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_simple_splits()
    
    # Create datasets
    train_dataset = SimpleLipDataset(train_videos, train_labels, augment=True)
    val_dataset = SimpleLipDataset(val_videos, val_labels, augment=False)
    test_dataset = SimpleLipDataset(test_videos, test_labels, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Create model
    model = Simple3DCNN(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: {total_params:,} parameters")
    
    # Train model
    start_time = time.time()
    best_val_acc = train_model(model, train_loader, val_loader, device, num_epochs=8)
    training_time = (time.time() - start_time) / 60
    
    # Test model
    test_acc = test_model(model, test_loader, device)
    
    # Results
    print(f"\n🎯 FINAL RESULTS")
    print("=" * 30)
    print(f"🎯 Test Accuracy: {test_acc:.1f}%")
    print(f"🎯 Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"⏱️  Training Time: {training_time:.1f} minutes")
    
    if test_acc >= 50:
        print(f"✅ SUCCESS: Good baseline achieved!")
    elif test_acc >= 30:
        print(f"📈 PROGRESS: Reasonable results")
    else:
        print(f"⚠️  NEEDS WORK: Low accuracy")
    
    return test_acc

if __name__ == "__main__":
    try:
        final_accuracy = main()
        print(f"\n🏁 Training completed: {final_accuracy:.1f}% accuracy")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
