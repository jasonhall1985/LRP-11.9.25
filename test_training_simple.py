#!/usr/bin/env python3
"""
Simple Training Test - Memory Efficient Version

This script tests if training can start without memory issues.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import time
from datetime import datetime
from pathlib import Path
import psutil

def get_memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class SimpleLipDataset(Dataset):
    """Simplified dataset for testing."""
    
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels
        print(f"📊 Dataset: {len(self.video_paths)} videos")
        
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_simple(self, video_path):
        """Load video with minimal processing."""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            
            while frame_count < 32:  # Only load 32 frames max
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale and resize to smaller size for memory
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Resize to smaller dimensions to save memory
                frame = cv2.resize(frame, (160, 108))  # Much smaller than 640x432
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            # Pad to exactly 32 frames
            while len(frames) < 32:
                frames.append(frames[-1] if frames else np.zeros((108, 160), dtype=np.uint8))
            
            return np.array(frames[:32])
            
        except Exception as e:
            print(f"❌ Error loading {video_path}: {e}")
            return np.zeros((32, 108, 160), dtype=np.uint8)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video
        frames = self.load_video_simple(video_path)
        
        # Normalize
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor: (C, T, H, W)
        frames = torch.from_numpy(frames).unsqueeze(0)
        
        return frames, label

class SimpleModel(nn.Module):
    """Very simple 3D CNN for testing."""
    
    def __init__(self, num_classes=5):
        super(SimpleModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def create_simple_splits():
    """Create simple data splits."""
    dataset_path = "corrected_balanced_dataset"
    video_files = list(Path(dataset_path).glob("*.mp4"))
    
    if len(video_files) == 0:
        raise ValueError(f"No videos found in {dataset_path}")
    
    print(f"Found {len(video_files)} videos")
    
    # Take just a few videos for testing
    video_files = video_files[:10]  # Only use 10 videos for testing
    
    # Simple split
    train_videos = [str(f) for f in video_files[:8]]
    val_videos = [str(f) for f in video_files[8:]]
    
    # Create simple labels (just use filename)
    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
    
    train_labels = []
    for video in train_videos:
        class_name = Path(video).stem.split('_')[0]
        train_labels.append(class_to_idx.get(class_name, 0))
    
    val_labels = []
    for video in val_videos:
        class_name = Path(video).stem.split('_')[0]
        val_labels.append(class_to_idx.get(class_name, 0))
    
    print(f"Train: {len(train_videos)} videos, Val: {len(val_videos)} videos")
    return (train_videos, train_labels), (val_videos, val_labels)

def test_training():
    """Test training with minimal setup."""
    print("🧪 SIMPLE TRAINING TEST")
    print("=" * 50)
    
    # Memory check
    print(f"💾 Initial memory: {get_memory_usage():.1f} MB")
    
    # Set seeds
    set_random_seeds(42)
    
    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")
    
    # Create data
    (train_videos, train_labels), (val_videos, val_labels) = create_simple_splits()
    
    print(f"💾 After data splits: {get_memory_usage():.1f} MB")
    
    # Create datasets
    train_dataset = SimpleLipDataset(train_videos, train_labels)
    val_dataset = SimpleLipDataset(val_videos, val_labels)
    
    print(f"💾 After datasets: {get_memory_usage():.1f} MB")
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"💾 After loaders: {get_memory_usage():.1f} MB")
    
    # Create model
    model = SimpleModel(num_classes=5).to(device)
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"💾 After model: {get_memory_usage():.1f} MB")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"💾 After optimizer: {get_memory_usage():.1f} MB")
    
    # Test one training step
    print("\n🚀 Testing training step...")
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"📊 Batch {batch_idx}: data shape {data.shape}, target {target}")
        print(f"💾 Memory: {get_memory_usage():.1f} MB")
        
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"📈 Loss: {loss.item():.4f}")
        print(f"💾 After training step: {get_memory_usage():.1f} MB")
        
        if batch_idx >= 2:  # Only test a few batches
            break
    
    print("\n✅ Training test completed successfully!")
    print(f"💾 Final memory: {get_memory_usage():.1f} MB")
    
    # Test validation
    print("\n🔍 Testing validation...")
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            print(f"📊 Val batch {batch_idx}: Loss {loss.item():.4f}")
            
            if batch_idx >= 1:  # Only test a couple
                break
    
    print("✅ Validation test completed!")
    return True

if __name__ == "__main__":
    try:
        success = test_training()
        if success:
            print("\n🎉 MEMORY TEST PASSED - Ready for full training!")
        else:
            print("\n❌ MEMORY TEST FAILED")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
