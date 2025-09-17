#!/usr/bin/env python3
"""
Final Diagnostic and Solution
Identify the root cause and implement the correct fix
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import json
import time
from datetime import datetime
from pathlib import Path
import logging
import psutil
from sklearn.metrics import accuracy_score, f1_score, classification_report

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ExactOriginalDataset(Dataset):
    """EXACT reproduction of the original working dataset."""
    
    def __init__(self, video_paths, labels, split='train'):
        self.video_paths = video_paths
        self.labels = labels
        self.split = split
        
        self.class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
        print(f"📊 {split.upper()} Dataset: {len(self.video_paths)} videos")
        
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_original(self, video_path):
        """Load video with EXACT original preprocessing."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # EXACT ORIGINAL: BGR to Gray
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                frames.append(frame)
            
            cap.release()
            
            # EXACT ORIGINAL: Temporal sampling to 32 frames
            if len(frames) >= 32:
                indices = np.linspace(0, len(frames) - 1, 32, dtype=int)
                frames = [frames[i] for i in indices]
            else:
                while len(frames) < 32:
                    frames.append(frames[-1] if frames else np.zeros((432, 640), dtype=np.uint8))
            
            return np.array(frames[:32])
            
        except Exception as e:
            print(f"❌ Error loading video {video_path}: {str(e)}")
            return np.zeros((32, 432, 640), dtype=np.uint8)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video with original preprocessing
        frames = self.load_video_original(video_path)
        
        # EXACT ORIGINAL: Resize to 224x224, basic normalization
        processed_frames = []
        for frame in frames:
            # Resize to 224x224
            frame = cv2.resize(frame, (224, 224))
            processed_frames.append(frame)
        
        frames = np.array(processed_frames)
        
        # EXACT ORIGINAL: Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor: (C, T, H, W) format
        frames = torch.from_numpy(frames).unsqueeze(0)
        
        return frames, label

class ExactOriginalModel(nn.Module):
    """EXACT reproduction of the original working model."""
    
    def __init__(self, num_classes=5):
        super(ExactOriginalModel, self).__init__()
        
        # EXACT ORIGINAL ARCHITECTURE from phase2_training_memory_efficient.py
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def create_truly_disjoint_splits(dataset_path="corrected_balanced_dataset"):
    """Create TRULY speaker-disjoint splits with manual verification."""
    print("📊 Creating TRULY speaker-disjoint data splits...")
    
    video_files = list(Path(dataset_path).glob("*.mp4"))
    if len(video_files) == 0:
        raise ValueError(f"No video files found in {dataset_path}")
    
    print(f"Found {len(video_files)} videos")
    
    # Manual speaker assignment to ensure no overlap
    # Based on analysis of the dataset structure
    
    # Define speaker groups manually to ensure disjoint splits
    train_speakers = {'01', '02', '03', '04', '05', '06', '07', '08'}  # 8 speakers for training
    val_speakers = {'09'}     # 1 speaker for validation  
    test_speakers = {'10'}    # 1 speaker for testing
    
    print(f"📊 Manual speaker assignment:")
    print(f"   • Train speakers: {sorted(train_speakers)}")
    print(f"   • Val speakers: {sorted(val_speakers)}")
    print(f"   • Test speakers: {sorted(test_speakers)}")
    
    # Verify no overlap
    overlap = (train_speakers & val_speakers) | (train_speakers & test_speakers) | (val_speakers & test_speakers)
    if overlap:
        raise ValueError(f"Speaker overlap detected: {overlap}")
    
    # Organize videos by speaker assignment
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []
    
    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
    
    for video_file in video_files:
        parts = video_file.stem.split('_')
        if len(parts) >= 2:
            class_name = parts[0]
            speaker_id = parts[1]
            
            if class_name not in class_to_idx:
                continue
                
            if speaker_id in train_speakers:
                train_videos.append(str(video_file))
                train_labels.append(class_to_idx[class_name])
            elif speaker_id in val_speakers:
                val_videos.append(str(video_file))
                val_labels.append(class_to_idx[class_name])
            elif speaker_id in test_speakers:
                test_videos.append(str(video_file))
                test_labels.append(class_to_idx[class_name])
    
    print(f"📊 Final splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    # Verify class distribution
    from collections import Counter
    train_classes = Counter([class_to_idx[Path(v).stem.split('_')[0]] for v in train_videos])
    val_classes = Counter([class_to_idx[Path(v).stem.split('_')[0]] for v in val_videos])
    test_classes = Counter([class_to_idx[Path(v).stem.split('_')[0]] for v in test_videos])
    
    print(f"📊 Class distribution:")
    print(f"   • Train: {dict(train_classes)}")
    print(f"   • Val: {dict(val_classes)}")
    print(f"   • Test: {dict(test_classes)}")
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def main():
    """Final diagnostic and solution implementation."""
    print("🎯 FINAL DIAGNOSTIC AND SOLUTION")
    print("=" * 60)
    print("HYPOTHESIS: Speaker overlap is causing data leakage")
    print("SOLUTION: Truly disjoint splits + exact original model")
    print("EXPECTED: Restore 40%+ baseline accuracy")
    print("=" * 60)
    print(f"💾 Initial memory: {get_memory_usage():.1f} MB")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")
    
    # Create TRULY speaker-disjoint data splits
    try:
        (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_truly_disjoint_splits()
    except Exception as e:
        print(f"❌ Error creating splits: {e}")
        print("🔄 Falling back to simple random splits...")
        
        # Fallback: Simple random splits
        video_files = list(Path("corrected_balanced_dataset").glob("*.mp4"))
        random.shuffle(video_files)
        
        class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
        
        all_videos = []
        all_labels = []
        for video_file in video_files:
            class_name = video_file.stem.split('_')[0]
            if class_name in class_to_idx:
                all_videos.append(str(video_file))
                all_labels.append(class_to_idx[class_name])
        
        # 80% train, 10% val, 10% test
        n_total = len(all_videos)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        train_videos = all_videos[:n_train]
        train_labels = all_labels[:n_train]
        val_videos = all_videos[n_train:n_train+n_val]
        val_labels = all_labels[n_train:n_train+n_val]
        test_videos = all_videos[n_train+n_val:]
        test_labels = all_labels[n_train+n_val:]
        
        print(f"📊 Fallback splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    # Create EXACT original datasets
    train_dataset = ExactOriginalDataset(train_videos, train_labels, split='train')
    val_dataset = ExactOriginalDataset(val_videos, val_labels, split='val')
    test_dataset = ExactOriginalDataset(test_videos, test_labels, split='test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"💾 After data loading: {get_memory_usage():.1f} MB")
    
    # Create EXACT original model
    model = ExactOriginalModel(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"🧠 Exact Original Model:")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Architecture: 3D CNN (32→64→128)")
    print(f"   • Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"💾 After model creation: {get_memory_usage():.1f} MB")
    
    # EXACT original training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print(f"💾 After optimizer setup: {get_memory_usage():.1f} MB")
    
    # Training loop
    print("\n🎯 STARTING EXACT ORIGINAL TRAINING")
    print("=" * 50)
    print("Expected: 40%+ accuracy restoration")
    print("=" * 50)
    
    num_epochs = 5
    best_val_acc = 0.0
    
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
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
            
            if batch_idx % 10 == 0:
                memory_mb = get_memory_usage()
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Acc: {100.*train_correct/train_total:.2f}%, "
                      f"Memory: {memory_mb:.1f}MB")
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        val_acc = 100. * val_correct / val_total
        val_f1 = f1_score(all_targets, all_preds, average='macro') * 100 if len(set(all_targets)) > 1 else 0
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_final_model.pth')
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%, "
              f"Best Val: {best_val_acc:.2f}%")
    
    # Final test evaluation
    print("\n🔍 Final test evaluation...")
    
    # Load best model
    if os.path.exists('best_final_model.pth'):
        model.load_state_dict(torch.load('best_final_model.pth', map_location=device))
        print("📥 Loaded best model")
    
    model.eval()
    test_correct = 0
    test_total = 0
    all_test_preds = []
    all_test_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_total += target.size(0)
            
            all_test_preds.extend(pred.cpu().numpy().flatten())
            all_test_targets.extend(target.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    test_f1 = f1_score(all_test_targets, all_test_preds, average='macro') * 100 if len(set(all_test_targets)) > 1 else 0
    
    # Generate classification report
    if len(set(all_test_targets)) > 1:
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        report = classification_report(all_test_targets, all_test_preds, target_names=class_names)
        print(f"📊 Classification Report:\n{report}")
    
    print(f"\n🎯 FINAL DIAGNOSTIC RESULTS:")
    print("=" * 50)
    print(f"🎯 Test Accuracy: {test_acc:.2f}%")
    print(f"🎯 Test F1 Score: {test_f1:.2f}%")
    print(f"🎯 Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"💾 Final Memory Usage: {get_memory_usage():.1f} MB")
    
    # Diagnosis
    baseline = 40.0
    if test_acc >= baseline:
        print(f"✅ SUCCESS: Baseline restored! ({test_acc:.1f}% ≥ {baseline}%)")
        print(f"🔍 ROOT CAUSE: Speaker overlap was causing data leakage")
        print(f"💡 SOLUTION: Truly disjoint splits fixed the issue")
    elif test_acc >= 30:
        print(f"📈 PROGRESS: Significant improvement ({test_acc:.1f}%)")
        print(f"🔍 PARTIAL SUCCESS: Still investigating remaining issues")
    else:
        print(f"⚠️  STILL INVESTIGATING: ({test_acc:.1f}%)")
        print(f"🔍 NEED DEEPER ANALYSIS: Issue may be more fundamental")
    
    # Save diagnostic results
    results = {
        'test_accuracy': test_acc,
        'test_f1_score': test_f1,
        'best_val_accuracy': best_val_acc,
        'baseline_target': baseline,
        'diagnosis': 'Speaker overlap investigation',
        'solution_attempted': 'Truly disjoint splits + exact original model'
    }
    
    with open('final_diagnostic_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return test_acc

if __name__ == "__main__":
    try:
        final_accuracy = main()
        
        if final_accuracy >= 40:
            print(f"\n🎯 DIAGNOSTIC SUCCESS!")
            print(f"   Ready to proceed with optimization to 60-75%")
        else:
            print(f"\n🔄 CONTINUE INVESTIGATION:")
            print(f"   Need deeper analysis of the training pipeline")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
