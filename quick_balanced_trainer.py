#!/usr/bin/env python3
"""
Quick balanced 4-class model training to fix doctor bias.
Uses the existing 4-class data with aggressive class balancing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from collections import Counter
import time
import cv2
from torch.utils.data import Dataset

# Import existing model
from load_75_9_checkpoint import DoctorFocusedModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    def __init__(self, manifest_df):
        self.manifest = manifest_df.reset_index(drop=True)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(manifest_df['class'].unique()))}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        video_path = row['video_path']
        class_name = row['class']
        
        # Load video frames
        frames = self.load_video_frames(video_path)
        
        # Convert to tensor
        video_tensor = torch.FloatTensor(frames).unsqueeze(0)  # Add channel dimension
        
        # Get class index
        class_idx = self.class_to_idx[class_name]
        
        return video_tensor, class_idx
    
    def load_video_frames(self, video_path):
        """Load and preprocess video frames."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale and normalize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float32) / 255.0
            frames.append(gray)
        
        cap.release()
        
        # Sample 32 frames
        if len(frames) >= 32:
            indices = np.linspace(0, len(frames) - 1, 32, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            # Pad with last frame if too short
            while len(frames) < 32:
                frames.append(frames[-1] if frames else np.zeros((64, 96), dtype=np.float32))
        
        return np.array(frames[:32])

def create_balanced_sampler(dataset):
    """Create weighted sampler for balanced training."""
    # Count samples per class
    class_counts = Counter()
    for i in range(len(dataset)):
        _, class_idx = dataset[i]
        class_counts[class_idx] += 1
    
    # Calculate weights (inverse frequency)
    total_samples = len(dataset)
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (len(class_counts) * count)
    
    # Create sample weights
    sample_weights = []
    for i in range(len(dataset)):
        _, class_idx = dataset[i]
        sample_weights.append(class_weights[class_idx])
    
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def train_balanced_model():
    """Train a balanced 4-class model."""
    print("🚀 QUICK BALANCED 4-CLASS TRAINING")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Using device: {device}")
    
    # Load 4-class data
    train_df = pd.read_csv('backup_75.9_success_20250921_004410/4class_train_manifest.csv')
    val_df = pd.read_csv('backup_75.9_success_20250921_004410/4class_validation_manifest.csv')
    
    logger.info(f"📊 Training videos: {len(train_df)}")
    logger.info(f"📊 Validation videos: {len(val_df)}")
    
    # Check class distribution
    train_classes = train_df['class'].value_counts()
    logger.info("📊 Training class distribution:")
    for cls, count in train_classes.items():
        logger.info(f"   {cls}: {count} videos")
    
    # Create datasets
    train_dataset = VideoDataset(train_df)
    val_dataset = VideoDataset(val_df)
    
    logger.info(f"🎯 Classes: {list(train_dataset.class_to_idx.keys())}")
    
    # Create balanced sampler
    balanced_sampler = create_balanced_sampler(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        sampler=balanced_sampler,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False,
        num_workers=2
    )
    
    # Create model (DoctorFocusedModel is hardcoded for 4 classes)
    num_classes = len(train_dataset.class_to_idx)
    if num_classes != 4:
        raise ValueError(f"DoctorFocusedModel expects 4 classes, got {num_classes}")

    model = DoctorFocusedModel().to(device)
    
    logger.info(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss with class weights (anti-doctor bias)
    class_weights = torch.ones(num_classes)
    if 'doctor' in train_dataset.class_to_idx:
        doctor_idx = train_dataset.class_to_idx['doctor']
        class_weights[doctor_idx] = 0.5  # Reduce doctor weight
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    logger.info("🚀 Starting balanced training...")
    
    for epoch in range(50):  # Max 50 epochs
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        class_correct = Counter()
        class_total = Counter()
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
        
        val_acc = 100.0 * val_correct / val_total
        epoch_time = time.time() - start_time
        
        # Per-class results
        logger.info(f"Epoch {epoch+1}/{50} ({epoch_time:.1f}s)")
        logger.info(f"  Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")
        
        for class_idx, class_name in val_dataset.idx_to_class.items():
            if class_idx in class_total:
                class_acc = 100.0 * class_correct[class_idx] / class_total[class_idx]
                logger.info(f"  {class_name}: {class_acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': train_dataset.class_to_idx,
                'idx_to_class': train_dataset.idx_to_class,
                'val_accuracy': val_acc,
                'epoch': epoch + 1
            }, 'balanced_4class_model.pth')
            logger.info(f"✅ New best model saved: {val_acc:.1f}%")
        else:
            patience_counter += 1
        
        scheduler.step(val_loss)
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"🛑 Early stopping at epoch {epoch+1}")
            break
        
        # Quick success check
        if val_acc >= 60.0 and epoch >= 10:
            logger.info(f"🎯 Good accuracy achieved: {val_acc:.1f}%")
            break
    
    logger.info(f"🏆 Best validation accuracy: {best_val_acc:.1f}%")
    logger.info("✅ Balanced model training complete!")
    
    return best_val_acc

if __name__ == "__main__":
    train_balanced_model()
