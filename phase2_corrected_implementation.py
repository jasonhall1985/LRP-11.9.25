#!/usr/bin/env python3
"""
Phase 2: CORRECTED Implementation - Following User's Exact Specifications
Fixes: Keep original 290K backbone + add BiGRU head, 112x112 input, proper training recipe
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

class StandardizedLipDataset(Dataset):
    """Standardized dataset following user's exact preprocessing specifications."""
    
    def __init__(self, video_paths, labels, split='train', augment=True):
        self.video_paths = video_paths
        self.labels = labels
        self.split = split
        self.augment = augment and (split == 'train')
        
        self.class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
        print(f"📊 {split.upper()} Dataset: {len(self.video_paths)} videos")
        
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_standardized(self, video_path):
        """Load video with EXACT user specifications: 224x224 → 112x112 center crop."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # USER SPEC: 224x224 → 112x112 center crop
                frame = cv2.resize(frame, (224, 224))
                # Center crop to 112x112
                h, w = frame.shape
                start_h, start_w = (h - 112) // 2, (w - 112) // 2
                frame = frame[start_h:start_h + 112, start_w:start_w + 112]
                
                frames.append(frame)
            
            cap.release()
            
            # USER SPEC: Maintain 32 frames with uniform sampling
            if len(frames) >= 32:
                indices = np.linspace(0, len(frames) - 1, 32, dtype=int)
                frames = [frames[i] for i in indices]
            else:
                while len(frames) < 32:
                    frames.append(frames[-1] if frames else np.zeros((112, 112), dtype=np.uint8))
            
            return np.array(frames[:32])
            
        except Exception as e:
            print(f"❌ Error loading video {video_path}: {str(e)}")
            return np.zeros((32, 112, 112), dtype=np.uint8)
    
    def apply_strategic_augmentations(self, frames):
        """Apply USER SPECIFIED strategic training augmentations."""
        if not self.augment:
            return frames
        
        # USER SPEC: Temporal augmentations
        # ±2 frame jitter
        if random.random() < 0.3:
            jitter = random.randint(-2, 2)
            if jitter != 0:
                if jitter > 0:
                    frames = frames[jitter:]
                    frames = np.pad(frames, ((0, jitter), (0, 0), (0, 0)), mode='edge')
                else:
                    frames = frames[:jitter]
                    frames = np.pad(frames, ((-jitter, 0), (0, 0), (0, 0)), mode='edge')
        
        # 10% random frame dropout
        if random.random() < 0.1:
            dropout_frames = random.randint(1, 3)
            for _ in range(dropout_frames):
                idx = random.randint(0, len(frames) - 1)
                if idx > 0:
                    frames[idx] = frames[idx - 1]
        
        # USER SPEC: Spatial augmentations
        # ±3px translation
        if random.random() < 0.4:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx != 0 or dy != 0:
                frames = np.roll(frames, (dy, dx), axis=(1, 2))
        
        # USER SPEC: Appearance augmentations
        # ±10% brightness/contrast
        if random.random() < 0.3:
            brightness = random.uniform(0.9, 1.1)
            frames = np.clip(frames * brightness, 0, 255).astype(np.uint8)
        
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video with standardized preprocessing
        frames = self.load_video_standardized(video_path)
        
        # Apply strategic augmentations (training only)
        frames = self.apply_strategic_augmentations(frames)
        
        # USER SPEC: Per-video normalize to [0,1] with mean=0.5, std=0.5
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - 0.5) / 0.5  # Normalize to [-1, 1] then to mean=0.5, std=0.5
        frames = frames * 0.5 + 0.5  # Back to [0, 1] with proper normalization
        
        # Convert to tensor: (C, T, H, W) format
        frames = torch.from_numpy(frames).unsqueeze(0)
        
        return frames, label

class CorrectedLipModel(nn.Module):
    """USER SPECIFICATION: Keep original 290K backbone + add BiGRU head."""
    
    def __init__(self, num_classes=5):
        super(CorrectedLipModel, self).__init__()
        
        # USER SPEC: Keep current 3D CNN feature extractor (290K parameters)
        self.backbone = nn.Sequential(
            # First conv block - EXACT original architecture
            nn.Conv3d(1, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # Second conv block - EXACT original architecture
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Third conv block - EXACT original architecture
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        
        # USER SPEC: Enhanced Head with BiGRU layers
        self.enhanced_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            
            # Reshape for GRU: (batch, seq_len, features)
            # We'll use the temporal dimension from backbone
        )
        
        # USER SPEC: 2-layer Bidirectional GRU with hidden_size=256
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # USER SPEC: LayerNorm before final classifier
        self.layer_norm = nn.LayerNorm(512)  # 256 * 2 (bidirectional)
        
        # USER SPEC: Final linear layer: 512 → 5 classes
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Extract features with backbone
        features = self.backbone(x)  # (batch, 128, 1, 1, 1)
        
        # For GRU, we need sequence dimension
        # Since we have temporal pooling, we'll use the original temporal features
        # Let's modify to preserve some temporal information
        batch_size = x.size(0)
        
        # Get features before final pooling
        x_temp = x
        for i, layer in enumerate(self.backbone[:-1]):  # All except AdaptiveAvgPool3d
            x_temp = layer(x_temp)
        
        # x_temp shape: (batch, 128, T', H', W') where T' is reduced temporal dim
        # Global spatial pooling but keep temporal
        x_temp = F.adaptive_avg_pool3d(x_temp, (x_temp.size(2), 1, 1))  # (batch, 128, T', 1, 1)
        x_temp = x_temp.squeeze(-1).squeeze(-1)  # (batch, 128, T')
        x_temp = x_temp.transpose(1, 2)  # (batch, T', 128) for GRU
        
        # Apply BiGRU
        gru_out, _ = self.gru(x_temp)  # (batch, T', 512)
        
        # Take the last output
        gru_out = gru_out[:, -1, :]  # (batch, 512)
        
        # Apply LayerNorm
        gru_out = self.layer_norm(gru_out)
        
        # Final classification
        output = self.classifier(gru_out)
        
        return output

class LabelSmoothingCrossEntropy(nn.Module):
    """USER SPEC: Label smoothing: 0.1"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_prob = F.log_softmax(pred, dim=-1)
        loss = -(one_hot * log_prob).sum(dim=-1).mean()
        return loss

def create_speaker_disjoint_splits(dataset_path="corrected_balanced_dataset"):
    """USER SPEC: Ensure speaker-disjoint data splits."""
    print("📊 Creating speaker-disjoint data splits...")
    
    video_files = list(Path(dataset_path).glob("*.mp4"))
    if len(video_files) == 0:
        raise ValueError(f"No video files found in {dataset_path}")
    
    print(f"Found {len(video_files)} videos")
    
    # Group by speaker (assuming filename format: class_speaker_*.mp4)
    speaker_videos = {}
    class_speakers = {'doctor': set(), 'glasses': set(), 'help': set(), 'phone': set(), 'pillow': set()}
    
    for video_file in video_files:
        parts = video_file.stem.split('_')
        if len(parts) >= 2:
            class_name = parts[0]
            speaker_id = parts[1] if len(parts) > 1 else parts[0]
            
            if class_name in class_speakers:
                speaker_key = f"{class_name}_{speaker_id}"
                if speaker_key not in speaker_videos:
                    speaker_videos[speaker_key] = []
                speaker_videos[speaker_key].append(str(video_file))
                class_speakers[class_name].add(speaker_id)
    
    # Report speaker analysis
    for class_name, speakers in class_speakers.items():
        print(f"📊 {class_name}: {len(speakers)} speakers, {sum(1 for k in speaker_videos.keys() if k.startswith(class_name))} videos")
    
    # Create speaker-disjoint splits
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []
    
    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
    
    # For each class, split speakers
    for class_name in class_speakers.keys():
        class_speaker_videos = [(k, v) for k, v in speaker_videos.items() if k.startswith(class_name)]
        random.shuffle(class_speaker_videos)
        
        # Split speakers: 8 for train, 1 for val, 1 for test
        train_speakers = class_speaker_videos[:8]
        val_speakers = class_speaker_videos[8:9] if len(class_speaker_videos) > 8 else []
        test_speakers = class_speaker_videos[9:10] if len(class_speaker_videos) > 9 else []
        
        # Add videos from each speaker group
        for speaker_key, videos in train_speakers:
            train_videos.extend(videos)
            train_labels.extend([class_to_idx[class_name]] * len(videos))
        
        for speaker_key, videos in val_speakers:
            val_videos.extend(videos)
            val_labels.extend([class_to_idx[class_name]] * len(videos))
        
        for speaker_key, videos in test_speakers:
            test_videos.extend(videos)
            test_labels.extend([class_to_idx[class_name]] * len(videos))
    
    print(f"📊 Final splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    # Verify speaker disjoint
    print("🔍 Verifying speaker-disjoint splits...")
    train_speakers = set()
    val_speakers = set()
    test_speakers = set()
    
    for video in train_videos:
        speaker = Path(video).stem.split('_')[1] if '_' in Path(video).stem else Path(video).stem
        train_speakers.add(speaker)
    
    for video in val_videos:
        speaker = Path(video).stem.split('_')[1] if '_' in Path(video).stem else Path(video).stem
        val_speakers.add(speaker)
    
    for video in test_videos:
        speaker = Path(video).stem.split('_')[1] if '_' in Path(video).stem else Path(video).stem
        test_speakers.add(speaker)
    
    print(f"📊 Speaker analysis:")
    print(f"   • Train speakers: {len(train_speakers)}")
    print(f"   • Val speakers: {len(val_speakers)}")
    print(f"   • Test speakers: {len(test_speakers)}")
    
    # Check for overlap
    train_val_overlap = train_speakers & val_speakers
    train_test_overlap = train_speakers & test_speakers
    val_test_overlap = val_speakers & test_speakers
    
    if not train_val_overlap and not train_test_overlap and not val_test_overlap:
        print("✅ Splits are speaker-disjoint!")
    else:
        print(f"⚠️  Speaker overlap detected:")
        if train_val_overlap:
            print(f"   Train-Val: {train_val_overlap}")
        if train_test_overlap:
            print(f"   Train-Test: {train_test_overlap}")
        if val_test_overlap:
            print(f"   Val-Test: {val_test_overlap}")
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def main():
    """Main corrected implementation following user's exact specifications."""
    print("🎯 PHASE 2: CORRECTED IMPLEMENTATION")
    print("=" * 60)
    print("Following user's EXACT specifications:")
    print("• Keep original 290K backbone + add BiGRU head")
    print("• 224x224 → 112x112 center crop")
    print("• Strategic augmentations")
    print("• Proper training recipe with AdamW")
    print("=" * 60)
    print(f"💾 Initial memory: {get_memory_usage():.1f} MB")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")
    
    # Create speaker-disjoint data splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_speaker_disjoint_splits()
    
    # Create standardized datasets
    train_dataset = StandardizedLipDataset(train_videos, train_labels, split='train', augment=True)
    val_dataset = StandardizedLipDataset(val_videos, val_labels, split='val', augment=False)
    test_dataset = StandardizedLipDataset(test_videos, test_labels, split='test', augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"💾 After data loading: {get_memory_usage():.1f} MB")
    
    # Create corrected model
    model = CorrectedLipModel(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    
    print(f"🧠 Corrected Model:")
    print(f"   • Backbone parameters: {backbone_params:,} (target: ~290K)")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"💾 After model creation: {get_memory_usage():.1f} MB")
    
    # USER SPEC: Proper training recipe
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # USER SPEC: AdamW with different learning rates
    head_params = []
    backbone_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    # USER SPEC: Head layers: 6e-4, Backbone: 1e-4
    optimizer = optim.AdamW([
        {'params': head_params, 'lr': 6e-4, 'weight_decay': 1e-4},
        {'params': backbone_params, 'lr': 1e-4, 'weight_decay': 1e-4}
    ])
    
    print(f"💾 After optimizer setup: {get_memory_usage():.1f} MB")
    
    print(f"\n🎯 STARTING CORRECTED TRAINING")
    print("=" * 40)
    print("Expected: Significant improvement over 40% baseline")
    print("Target: 60-75% accuracy")
    print("=" * 40)
    
    # Simple training loop for now (will implement staged training next)
    model.train()
    
    for epoch in range(3):  # Quick test
        print(f"\n📊 Epoch {epoch + 1}")
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # USER SPEC: Gradient clipping: max_norm=1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                memory_mb = get_memory_usage()
                print(f"   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, "
                      f"Acc: {100.*correct/total:.2f}%, Memory: {memory_mb:.1f}MB")
        
        avg_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f"   📊 Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
    
    print(f"\n✅ CORRECTED IMPLEMENTATION TEST COMPLETED")
    print(f"💾 Final Memory Usage: {get_memory_usage():.1f} MB")
    print(f"🎯 Ready for full staged training implementation")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
