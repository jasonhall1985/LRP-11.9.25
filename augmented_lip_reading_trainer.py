#!/usr/bin/env python3
"""
Augmented Lip Reading Training Pipeline - Using expanded dataset with lighting augmentations
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
import torch.nn.functional as F

class AugmentedLipReadingDataset(Dataset):
    """Dataset class for loading both original and augmented NPY files."""
    
    def __init__(self, npy_paths, labels, class_to_idx, augment=False):
        self.npy_paths = npy_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.augment = augment
        
    def __len__(self):
        return len(self.npy_paths)
    
    def __getitem__(self, idx):
        npy_path = self.npy_paths[idx]
        label = self.labels[idx]
        
        # Load preprocessed numpy array
        frames = np.load(npy_path)
        
        # Convert to tensor: (T, H, W) -> (1, T, H, W)
        frames = torch.from_numpy(frames).unsqueeze(0).float()
        
        # Apply additional augmentation if enabled (for training)
        if self.augment:
            frames = self.augment_tensor(frames)
        
        label_idx = self.class_to_idx[label]
        
        return frames, label_idx
    
    def augment_tensor(self, frames):
        """Apply minimal additional augmentations to tensor."""
        # Random horizontal flip (50% chance)
        if torch.rand(1) > 0.5:
            frames = torch.flip(frames, dims=[3])  # Flip width dimension
        
        return frames

class OptimizedLipReadingCNN(nn.Module):
    """Optimized CNN model for expanded dataset."""
    
    def __init__(self, num_classes=5):
        super(OptimizedLipReadingCNN, self).__init__()
        
        # 3D Convolutional layers with better architecture
        self.conv3d1 = nn.Conv3d(1, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.conv3d2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv3d3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        
        # Pooling
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Batch normalization
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        
        # Dropout with different rates
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.5)
        
        # Global average pooling to reduce parameters
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, 1, 32, 96, 96)
        
        # 3D convolutions
        x = self.relu(self.bn1(self.conv3d1(x)))
        x = self.pool3d(x)
        x = self.dropout1(x)
        
        x = self.relu(self.bn2(self.conv3d2(x)))
        x = self.pool3d(x)
        x = self.dropout2(x)
        
        x = self.relu(self.bn3(self.conv3d3(x)))
        x = self.pool3d(x)
        x = self.dropout3(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout1(x)
        
        x = self.fc3(x)
        
        return x

def load_augmented_dataset():
    """Load both original and augmented NPY files."""
    npy_dir = Path("data/training set 17.9.25")
    all_npy_files = list(npy_dir.glob("*.npy"))
    
    # Organize by class
    videos_by_class = defaultdict(list)
    
    for npy_file in all_npy_files:
        filename = npy_file.name.lower()
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
            continue
        
        videos_by_class[class_name].append(npy_file)
    
    return videos_by_class

def create_balanced_splits(videos_by_class, train_ratio=0.8, val_ratio=0.2):
    """Create balanced train/val splits."""
    
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    
    print("📊 Creating balanced splits:")
    
    for class_name, videos in videos_by_class.items():
        n_videos = len(videos)
        n_train = max(1, int(n_videos * train_ratio))
        n_val = n_videos - n_train
        
        print(f"   {class_name}: {n_videos} total → {n_train} train, {n_val} val")
        
        # Shuffle videos
        videos_shuffled = videos.copy()
        np.random.shuffle(videos_shuffled)
        
        # Split videos
        train_videos.extend(videos_shuffled[:n_train])
        train_labels.extend([class_name] * n_train)
        
        if n_val > 0:
            val_videos.extend(videos_shuffled[n_train:])
            val_labels.extend([class_name] * n_val)
    
    return (train_videos, train_labels), (val_videos, val_labels)

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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_preds, all_targets

def main():
    """Main training pipeline with augmented dataset."""
    print("🚀 AUGMENTED LIP READING TRAINING PIPELINE")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    batch_size = 4  # Slightly larger batch size with more data
    learning_rate = 0.0003  # Slightly lower learning rate
    num_epochs = 150
    target_accuracy = 60.0
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load augmented dataset
    print(f"\n📁 Loading augmented dataset...")
    videos_by_class = load_augmented_dataset()
    
    if not videos_by_class:
        print("❌ No videos found!")
        return
    
    # Print dataset statistics
    total_videos = sum(len(videos) for videos in videos_by_class.values())
    original_count = 0
    augmented_count = 0
    
    for class_name, videos in videos_by_class.items():
        class_original = len([v for v in videos if '_aug_' not in v.name])
        class_augmented = len([v for v in videos if '_aug_' in v.name])
        original_count += class_original
        augmented_count += class_augmented
        print(f"   {class_name}: {len(videos)} total ({class_original} original + {class_augmented} augmented)")
    
    print(f"📊 Dataset loaded: {total_videos} videos across {len(videos_by_class)} classes")
    print(f"   Original videos: {original_count}")
    print(f"   Augmented videos: {augmented_count}")
    print(f"   Dataset expansion: {(augmented_count/original_count)*100:.1f}%")
    
    # Create class mapping
    classes = sorted(videos_by_class.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    print(f"\n🏷️  Class mapping: {class_to_idx}")
    
    # Create balanced splits
    print(f"\n📊 Creating balanced splits...")
    (train_videos, train_labels), (val_videos, val_labels) = create_balanced_splits(videos_by_class)
    
    print(f"   Training set: {len(train_videos)} videos")
    print(f"   Validation set: {len(val_videos)} videos")
    
    # Create datasets
    train_dataset = AugmentedLipReadingDataset(train_videos, train_labels, class_to_idx, augment=True)
    val_dataset = AugmentedLipReadingDataset(val_videos, val_labels, class_to_idx, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Initialize optimized model
    print(f"\n🧠 Initializing optimized CNN model...")
    model = OptimizedLipReadingCNN(num_classes=len(classes)).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=15)
    
    # Training tracking
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_acc = 0.0
    best_model_path = "best_augmented_lip_reading_model.pth"
    patience_counter = 0
    max_patience = 30  # Increased patience for larger dataset
    
    print(f"\n🎯 Starting training with augmented dataset:")
    print(f"   Target: {target_accuracy}% validation accuracy")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Early stopping patience: {max_patience}")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Print results
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'class_to_idx': class_to_idx,
                'dataset_info': {
                    'total_videos': total_videos,
                    'original_videos': original_count,
                    'augmented_videos': augmented_count,
                    'expansion_percent': (augmented_count/original_count)*100
                }
            }, best_model_path)
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Check if target reached
        if val_acc >= target_accuracy:
            print(f"🎉 Target accuracy {target_accuracy}% reached!")
            break
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"⚠️  Early stopping triggered (no improvement for {max_patience} epochs)")
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
        'training_time': training_time,
        'dataset_info': {
            'total_videos': total_videos,
            'original_videos': original_count,
            'augmented_videos': augmented_count,
            'expansion_percent': (augmented_count/original_count)*100
        }
    }
    
    with open('augmented_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"📊 Training history saved to augmented_training_history.json")
    print(f"💾 Best model saved to {best_model_path}")
    
    # Final results summary
    print(f"\n🎯 FINAL RESULTS WITH AUGMENTED DATASET:")
    print(f"   Dataset expansion: {(augmented_count/original_count)*100:.1f}%")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Target ({target_accuracy}%) reached: {'✅' if best_val_acc >= target_accuracy else '❌'}")
    
    if best_val_acc >= target_accuracy:
        print(f"✅ SUCCESS: Augmented dataset achieved target performance!")
    else:
        improvement = best_val_acc - 29.41  # Previous best was 29.41%
        print(f"📈 IMPROVEMENT: +{improvement:.2f} percentage points from baseline")
        print(f"   Augmentation impact: {'Positive' if improvement > 0 else 'Neutral'}")

if __name__ == "__main__":
    main()
