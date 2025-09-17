#!/usr/bin/env python3
"""
Lip Reading Training Pipeline - Using preprocessed MP4 videos
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

class LipReadingDataset(Dataset):
    """Dataset class for loading MP4 videos and converting to tensors."""
    
    def __init__(self, video_paths, labels, class_to_idx):
        self.video_paths = video_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video and convert to tensor
        frames = self.load_video_as_tensor(video_path)
        label_idx = self.class_to_idx[label]
        
        return frames, label_idx
    
    def load_video_as_tensor(self, video_path):
        """Load MP4 video and convert to tensor format."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        
        cap.release()
        
        # Convert to numpy array and normalize to [0, 1]
        frames = np.array(frames, dtype=np.float32) / 255.0
        
        # Ensure exactly 32 frames
        if len(frames) != 32:
            # Temporal sampling to get exactly 32 frames
            if len(frames) > 32:
                indices = np.linspace(0, len(frames)-1, 32, dtype=int)
                frames = frames[indices]
            else:
                # Repeat frames if not enough
                while len(frames) < 32:
                    frames = np.concatenate([frames, frames[:min(len(frames), 32-len(frames))]])
                frames = frames[:32]
        
        # Convert to tensor: (T, H, W) -> (1, T, H, W) for single channel
        frames = torch.from_numpy(frames).unsqueeze(0)  # Add channel dimension
        
        return frames

class LipReadingCNN3D(nn.Module):
    """3D CNN model for lip reading."""
    
    def __init__(self, num_classes=5):
        super(LipReadingCNN3D, self).__init__()
        
        # 3D Convolutional layers
        self.conv3d1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1)
        self.conv3d2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv3d3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        
        # Pooling layers
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Batch normalization
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after conv layers
        # Input: (1, 32, 96, 96)
        # After conv1 + pool: (32, 16, 48, 48)
        # After conv2 + pool: (64, 8, 24, 24)
        # After conv3 + pool: (128, 4, 12, 12)
        self.fc_input_size = 128 * 4 * 12 * 12
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, 1, 32, 96, 96)
        
        # 3D convolutions
        x = self.relu(self.bn1(self.conv3d1(x)))
        x = self.pool3d(x)
        
        x = self.relu(self.bn2(self.conv3d2(x)))
        x = self.pool3d(x)
        
        x = self.relu(self.bn3(self.conv3d3(x)))
        x = self.pool3d(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

def load_dataset(video_dir):
    """Load and organize videos by class."""
    video_dir = Path(video_dir)
    
    # Get all MP4 files
    video_files = list(video_dir.glob("*.mp4"))
    
    # Organize by class
    videos_by_class = defaultdict(list)
    
    for video_file in video_files:
        # Extract class from filename
        filename = video_file.name.lower()
        if 'doctor' in filename:
            class_name = 'doctor'
        elif 'glasses' in filename:
            class_name = 'glasses'
        elif 'help' in filename:
            class_name = 'help'
        elif 'phone' in filename:
            class_name = 'phone'
        elif 'pillow' in filename:
            class_name = 'pillow'
        else:
            continue  # Skip unknown classes
        
        videos_by_class[class_name].append(video_file)
    
    return videos_by_class

def create_stratified_splits(videos_by_class, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Create stratified train/val/test splits."""
    
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []
    
    print("📊 Creating stratified splits:")
    
    for class_name, videos in videos_by_class.items():
        n_videos = len(videos)
        n_train = int(n_videos * train_ratio)
        n_val = int(n_videos * val_ratio)
        n_test = n_videos - n_train - n_val  # Remaining videos
        
        print(f"   {class_name}: {n_videos} total → {n_train} train, {n_val} val, {n_test} test")
        
        # Shuffle videos for random split
        videos_shuffled = videos.copy()
        np.random.shuffle(videos_shuffled)
        
        # Split videos
        train_videos.extend(videos_shuffled[:n_train])
        train_labels.extend([class_name] * n_train)
        
        val_videos.extend(videos_shuffled[n_train:n_train+n_val])
        val_labels.extend([class_name] * n_val)
        
        test_videos.extend(videos_shuffled[n_train+n_val:])
        test_labels.extend([class_name] * n_test)
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def main():
    """Main training pipeline."""
    print("🚀 LIP READING TRAINING PIPELINE")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    video_dir = "data/training set 17.9.25/preview_videos_fixed"
    batch_size = 4  # Small batch size due to 3D convolutions
    learning_rate = 0.001
    num_epochs = 50
    target_accuracy = 80.0
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load dataset
    print(f"\n📁 Loading dataset from: {video_dir}")
    videos_by_class = load_dataset(video_dir)
    
    if not videos_by_class:
        print("❌ No videos found!")
        return
    
    # Print dataset statistics
    total_videos = sum(len(videos) for videos in videos_by_class.values())
    print(f"📊 Dataset loaded: {total_videos} videos across {len(videos_by_class)} classes")
    for class_name, videos in videos_by_class.items():
        print(f"   {class_name}: {len(videos)} videos")
    
    # Create class mapping
    classes = sorted(videos_by_class.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    print(f"\n🏷️  Class mapping: {class_to_idx}")
    
    # Create stratified splits
    print(f"\n📊 Creating stratified splits...")
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_stratified_splits(videos_by_class)
    
    print(f"   Training set: {len(train_videos)} videos")
    print(f"   Validation set: {len(val_videos)} videos") 
    print(f"   Test set: {len(test_videos)} videos")
    
    # Create datasets
    train_dataset = LipReadingDataset(train_videos, train_labels, class_to_idx)
    val_dataset = LipReadingDataset(val_videos, val_labels, class_to_idx)
    test_dataset = LipReadingDataset(test_videos, test_labels, class_to_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Initialize model
    print(f"\n🧠 Initializing 3D CNN model...")
    model = LipReadingCNN3D(num_classes=len(classes)).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training tracking
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_acc = 0.0
    best_model_path = "best_lip_reading_model.pth"
    
    print(f"\n🎯 Starting training (target: {target_accuracy}% validation accuracy)")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'class_to_idx': class_to_idx
            }, best_model_path)
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Check if target reached
        if val_acc >= target_accuracy:
            print(f"🎉 Target accuracy {target_accuracy}% reached!")
            break
        
        # Early stopping check
        if epoch > 10 and val_acc < max(val_accuracies[-10:]) - 5:
            print("⚠️  Early stopping triggered (validation accuracy declining)")
            break
    
    training_time = time.time() - start_time
    print(f"\n⏱️  Training completed in {training_time/60:.2f} minutes")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_acc,
        'training_time': training_time
    }
    
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"📊 Training history saved to training_history.json")
    print(f"💾 Best model saved to {best_model_path}")

if __name__ == "__main__":
    main()
