#!/usr/bin/env python3
"""
🎯 ADVANCED TRAINING COMPONENTS
==============================

Advanced components for comprehensive speaker-disjoint training:
- Enhanced 3D CNN-LSTM with temporal attention
- Conservative data augmentation
- Focal loss with class weights
- Comprehensive preprocessing pipeline
- Advanced validation strategies
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
import random
from pathlib import Path
import json
from collections import Counter

class EnhancedLightweightCNNLSTM(nn.Module):
    """Enhanced 3D CNN-LSTM with temporal attention pooling"""
    
    def __init__(self, num_classes=4, dropout=0.4, input_channels=1):
        super(EnhancedLightweightCNNLSTM, self).__init__()
        
        # 3D CNN Feature Extraction
        self.conv3d1 = nn.Conv3d(input_channels, 16, kernel_size=(3, 3, 3), padding=1)
        self.bn3d1 = nn.BatchNorm3d(16)
        self.pool3d1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv3d2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn3d2 = nn.BatchNorm3d(32)
        self.pool3d2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d3 = nn.Conv3d(32, 48, kernel_size=(3, 3, 3), padding=1)
        self.bn3d3 = nn.BatchNorm3d(48)
        self.pool3d3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Adaptive pooling for consistent feature size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 6))
        
        # LSTM for temporal modeling
        self.lstm_input_size = 48 * 4 * 6  # 1152
        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # Temporal attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_hidden_size,
            num_heads=8,
            dropout=dropout * 0.5,
            batch_first=True
        )
        
        # Classification head with regularization
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.lstm_hidden_size, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout * 0.75)
        self.fc_out = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 3D CNN feature extraction
        x = torch.relu(self.bn3d1(self.conv3d1(x)))
        x = self.pool3d1(x)
        
        x = torch.relu(self.bn3d2(self.conv3d2(x)))
        x = self.pool3d2(x)
        
        x = torch.relu(self.bn3d3(self.conv3d3(x)))
        x = self.pool3d3(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Reshape for LSTM: (batch, time, features)
        batch_size = x.size(0)
        timesteps = x.size(2)
        x = x.permute(0, 2, 1, 3, 4)  # (batch, time, channels, height, width)
        x = x.contiguous().view(batch_size, timesteps, -1)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)
        
        # Temporal attention pooling
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling over time dimension
        x = torch.mean(attn_out, dim=1)
        
        # Classification head
        x = self.dropout1(x)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc_out(x)
        
        return x
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ConservativeAugmentation:
    """Conservative data augmentation for lip-reading"""
    
    def __init__(self, 
                 brightness_range=0.15,
                 contrast_range=0.1,
                 temporal_speed_range=0.05,
                 spatial_translation=0.03,
                 rotation_degrees=2.0,
                 horizontal_flip_prob=0.5):
        
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.temporal_speed_range = temporal_speed_range
        self.spatial_translation = spatial_translation
        self.rotation_degrees = rotation_degrees
        self.horizontal_flip_prob = horizontal_flip_prob
    
    def __call__(self, video_tensor):
        """Apply conservative augmentation to video tensor"""
        # video_tensor shape: (T, H, W) or (C, T, H, W)
        
        if len(video_tensor.shape) == 3:
            T, H, W = video_tensor.shape
            C = 1
            video_tensor = video_tensor.unsqueeze(0)  # Add channel dimension
        else:
            C, T, H, W = video_tensor.shape
        
        # Photometric augmentations
        if random.random() < 0.7:  # 70% chance
            # Brightness adjustment
            brightness_factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
            video_tensor = torch.clamp(video_tensor * brightness_factor, 0, 1)
            
            # Contrast adjustment
            contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
            mean_val = video_tensor.mean()
            video_tensor = torch.clamp((video_tensor - mean_val) * contrast_factor + mean_val, 0, 1)
        
        # Spatial augmentations (minimal)
        if random.random() < 0.3:  # 30% chance for spatial
            # Small translation
            tx = int(W * random.uniform(-self.spatial_translation, self.spatial_translation))
            ty = int(H * random.uniform(-self.spatial_translation, self.spatial_translation))
            
            if tx != 0 or ty != 0:
                video_tensor = self._translate_video(video_tensor, tx, ty)
        
        # Horizontal flip (preserves lip-reading semantics)
        if random.random() < self.horizontal_flip_prob:
            video_tensor = torch.flip(video_tensor, dims=[-1])  # Flip width dimension
        
        # Temporal augmentations (very conservative)
        if random.random() < 0.2:  # 20% chance
            # Temporal jitter (±1 frame)
            if T > 2:
                jitter = random.randint(-1, 1)
                if jitter != 0:
                    video_tensor = self._temporal_jitter(video_tensor, jitter)
        
        return video_tensor.squeeze(0) if C == 1 else video_tensor
    
    def _translate_video(self, video_tensor, tx, ty):
        """Apply translation to video tensor"""
        C, T, H, W = video_tensor.shape
        translated = torch.zeros_like(video_tensor)
        
        # Calculate valid regions
        src_x_start = max(0, -tx)
        src_x_end = min(W, W - tx)
        src_y_start = max(0, -ty)
        src_y_end = min(H, H - ty)
        
        dst_x_start = max(0, tx)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_start = max(0, ty)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        translated[:, :, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            video_tensor[:, :, src_y_start:src_y_end, src_x_start:src_x_end]
        
        return translated
    
    def _temporal_jitter(self, video_tensor, jitter):
        """Apply temporal jitter to video tensor"""
        C, T, H, W = video_tensor.shape
        
        if jitter > 0:
            # Duplicate first frame
            first_frames = video_tensor[:, :jitter].clone()
            jittered = torch.cat([first_frames, video_tensor[:, :-jitter]], dim=1)
        elif jitter < 0:
            # Duplicate last frame
            last_frames = video_tensor[:, jitter:].clone()
            jittered = torch.cat([video_tensor[:, -jitter:], last_frames], dim=1)
        else:
            jittered = video_tensor
        
        return jittered

class StandardizedPreprocessor:
    """Standardized preprocessing pipeline for lip-reading videos"""
    
    def __init__(self, 
                 target_size=(64, 96),  # (H, W) - 96×64 ROI
                 target_frames=32,
                 grayscale=True,
                 normalize=True):
        
        self.target_size = target_size
        self.target_frames = target_frames
        self.grayscale = grayscale
        self.normalize = normalize
    
    def process_video(self, video_path):
        """Process video file to standardized format"""
        
        if video_path.endswith('.npy'):
            # Load preprocessed numpy array
            video_array = np.load(video_path)
            
            # Handle different input shapes
            if len(video_array.shape) == 4:  # (T, H, W, C)
                if self.grayscale and video_array.shape[-1] == 3:
                    # Convert RGB to grayscale
                    video_array = np.mean(video_array, axis=-1)
                elif video_array.shape[-1] == 1:
                    video_array = video_array.squeeze(-1)
            
            # Ensure shape is (T, H, W)
            if len(video_array.shape) != 3:
                raise ValueError(f"Unexpected video array shape: {video_array.shape}")
            
        else:
            # Load video file using OpenCV
            video_array = self._load_video_opencv(video_path)
        
        # Temporal processing
        video_array = self._process_temporal(video_array)
        
        # Spatial processing
        video_array = self._process_spatial(video_array)
        
        # Normalization
        if self.normalize:
            video_array = video_array.astype(np.float32) / 255.0
        
        return torch.from_numpy(video_array).float()
    
    def _load_video_opencv(self, video_path):
        """Load video using OpenCV"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from video: {video_path}")
        
        return np.array(frames)
    
    def _process_temporal(self, video_array):
        """Process temporal dimension to target frame count"""
        T = video_array.shape[0]
        
        if T == self.target_frames:
            return video_array
        elif T > self.target_frames:
            # Uniform sampling
            indices = np.linspace(0, T - 1, self.target_frames, dtype=int)
            return video_array[indices]
        else:
            # Padding by repeating last frame
            padding_needed = self.target_frames - T
            last_frame = video_array[-1:].repeat(padding_needed, axis=0)
            return np.concatenate([video_array, last_frame], axis=0)
    
    def _process_spatial(self, video_array):
        """Process spatial dimensions to target size"""
        T, H, W = video_array.shape[:3]
        target_h, target_w = self.target_size
        
        if (H, W) == (target_h, target_w):
            return video_array
        
        # Resize each frame
        resized_frames = []
        for i in range(T):
            frame = video_array[i]
            if len(frame.shape) == 2:  # Grayscale
                resized_frame = cv2.resize(frame, (target_w, target_h))
            else:  # Color
                resized_frame = cv2.resize(frame, (target_w, target_h))
            resized_frames.append(resized_frame)
        
        return np.array(resized_frames)

class ComprehensiveVideoDataset(Dataset):
    """Comprehensive dataset for speaker-disjoint training"""
    
    def __init__(self,
                 manifest_path,
                 preprocessor=None,
                 data_root=None,
                 augmentation=None,
                 synthetic_ratio=0.25):

        self.manifest_path = manifest_path
        self.data_root = data_root or ""
        self.preprocessor = preprocessor or StandardizedPreprocessor()
        self.augmentation = augmentation
        self.synthetic_ratio = synthetic_ratio
        
        # Load manifest
        import pandas as pd
        self.df = pd.read_csv(manifest_path)
        
        # Create class to index mapping
        self.classes = sorted(self.df['class_label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Store labels for external access
        self.labels = self.df['class_label'].tolist()

        print(f"📊 Dataset loaded: {len(self.df)} samples")
        print(f"📊 Classes: {self.classes}")

        # Print class distribution
        class_counts = self.df['class_label'].value_counts()
        for cls in self.classes:
            count = class_counts.get(cls, 0)
            print(f"   {cls}: {count} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and preprocess video
        try:
            # Construct full path
            video_path = os.path.join(self.data_root, row['video_path']) if self.data_root else row['video_path']
            video_tensor = self.preprocessor.process_video(video_path)

            # Apply augmentation if specified
            if self.augmentation is not None:
                # Apply augmentation based on synthetic ratio
                if random.random() < self.synthetic_ratio:
                    video_tensor = self.augmentation(video_tensor)

            # Get class label
            class_label = self.class_to_idx[row['class_label']]
            
            return video_tensor.unsqueeze(0), class_label  # Add channel dimension

        except Exception as e:
            print(f"⚠️  Error loading video {row['video_path']}: {e}")
            # Return a dummy tensor and label
            dummy_tensor = torch.zeros(1, 32, 64, 96)
            return dummy_tensor, 0

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross-entropy loss for better generalization."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)

        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)

        return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))

def create_weighted_sampler(labels):
    """Create weighted random sampler for class balancing."""
    # Count class frequencies
    class_counts = Counter(labels)
    total_samples = len(labels)

    # Calculate weights (inverse frequency)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # Create sample weights
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
