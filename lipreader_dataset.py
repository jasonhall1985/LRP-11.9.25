#!/usr/bin/env python3
"""
Lip Reading Video Dataset
========================

PyTorch Dataset for loading 96x96 mouth ROI videos with:
- Grayscale conversion
- CLAHE contrast enhancement
- Temporal sampling (T=24 frames)
- Conservative augmentations for lip reading

Features:
- Uniform temporal sampling with center cropping for >24 frames
- Loop padding for <24 frames
- CLAHE normalization for lighting standardization
- Conservative augmentations that preserve lip reading semantics

Author: Production Lip Reading System
Date: 2025-09-15
"""

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging
import random
from torchvision import transforms
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LipReaderDataset(Dataset):
    """
    Video dataset for lip reading with grayscale conversion and CLAHE enhancement.
    """
    
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        clip_len: int = 24,
        img_size: int = 96,
        resize_for_backbone: int = 112,
        clahe_enabled: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        augmentation_config: Optional[Dict[str, Any]] = None,
        is_training: bool = True,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the lip reading dataset.
        
        Args:
            manifest_df: DataFrame with video information
            clip_len: Number of frames per clip (T=24)
            img_size: Input video resolution (96x96)
            resize_for_backbone: Resize to this size for R(2+1)D backbone (112x112)
            clahe_enabled: Enable CLAHE contrast enhancement
            clahe_clip_limit: CLAHE clip limit
            clahe_tile_grid: CLAHE tile grid size
            augmentation_config: Augmentation configuration
            is_training: Whether this is training dataset
            class_names: List of class names
        """
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.clip_len = clip_len
        self.img_size = img_size
        self.resize_for_backbone = resize_for_backbone
        self.is_training = is_training
        
        # CLAHE setup
        self.clahe_enabled = clahe_enabled
        if clahe_enabled:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=clahe_tile_grid
            )
        
        # Class mapping
        if class_names is None:
            class_names = [
                "help", "doctor", "glasses", "phone", "pillow", 
                "i_need_to_move", "my_mouth_is_dry"
            ]
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.num_classes = len(class_names)
        
        # Augmentation setup
        self.augmentation_config = augmentation_config or {}
        self.setup_augmentations()
        
        # Enhanced normalization with Kinetics-400 statistics for 3-channel grayscale
        # Applied AFTER grayscale replication to 3 channels
        self.kinetics_normalize = transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989]
        )
        
        logger.info(f"Dataset initialized: {len(self.manifest_df)} videos, "
                   f"{self.num_classes} classes, training={is_training}")
        
    def setup_augmentations(self):
        """Setup conservative augmentations for lip reading."""
        config = self.augmentation_config
        
        # Only apply augmentations during training
        if not self.is_training or not config.get('enabled', True):
            self.augment_prob = 0.0
            return
            
        self.augment_prob = config.get('aug_prob', 0.5)
        
        # Temporal augmentations
        self.temporal_jitter = config.get('temporal_jitter', 2)
        
        # Spatial augmentations (conservative)
        self.brightness_range = config.get('brightness_range', 0.1)
        self.rotation_degrees = config.get('rotation_degrees', 2)
        self.translation_pixels = config.get('translation_pixels', 2)
        
        # Disabled augmentations (preserve lip reading semantics)
        self.horizontal_flip = config.get('horizontal_flip', False)  # Always False
        self.vertical_flip = config.get('vertical_flip', False)      # Always False
        
        logger.info(f"Augmentations setup: prob={self.augment_prob}, "
                   f"temporal_jitter={self.temporal_jitter}, "
                   f"brightness={self.brightness_range}")
        
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.manifest_df)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """
        Get a video clip and its label.
        
        Returns:
            video_tensor: (C, T, H, W) tensor
            label: Class index
            metadata: Additional information
        """
        try:
            # Get video info
            video_info = self.manifest_df.iloc[idx]
            video_path = video_info['path']
            class_name = video_info['class']
            
            # Load video
            frames = self.load_video(video_path)
            if frames is None or len(frames) == 0:
                # Return a dummy sample if video loading fails
                return self.get_dummy_sample(class_name)
                
            # Temporal sampling to T=24 frames
            frames = self.temporal_sampling(frames)
            
            # Convert to grayscale and apply CLAHE
            frames = self.preprocess_frames(frames)
            
            # Apply augmentations
            if self.is_training and random.random() < self.augment_prob:
                frames = self.apply_augmentations(frames)
                
            # Resize for backbone compatibility
            frames = self.resize_frames(frames)
            
            # Convert to tensor and normalize
            video_tensor = self.frames_to_tensor(frames)
            
            # Get label
            label = self.class_to_idx[class_name]
            
            # Metadata
            metadata = {
                'video_path': video_path,
                'class_name': class_name,
                'original_frames': len(frames),
                'gender': video_info.get('gender', 'unknown'),
                'age_band': video_info.get('age_band', 'unknown'),
                'ethnicity': video_info.get('ethnicity', 'unknown'),
                'source': video_info.get('source', 'unknown'),
                'processed_version': video_info.get('processed_version', 'unknown')
            }
            
            return video_tensor, label, metadata
            
        except Exception as e:
            logger.warning(f"Error loading video {idx}: {e}")
            # Return dummy sample on error
            try:
                class_name = self.manifest_df.iloc[idx]['class']
            except:
                class_name = 'unknown'
            return self.get_dummy_sample(class_name)
            
    def load_video(self, video_path: str) -> Optional[List[np.ndarray]]:
        """Load video frames using OpenCV."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return None
                
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                
            cap.release()
            
            if len(frames) == 0:
                logger.warning(f"No frames loaded from: {video_path}")
                return None
                
            return frames
            
        except Exception as e:
            logger.warning(f"Error loading video {video_path}: {e}")
            return None
            
    def temporal_sampling(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Sample T=24 frames from video.
        - Center cropping for >24 frames
        - Loop padding for <24 frames
        """
        num_frames = len(frames)
        
        if num_frames == self.clip_len:
            return frames
        elif num_frames > self.clip_len:
            # Center cropping for longer videos
            start_idx = (num_frames - self.clip_len) // 2
            return frames[start_idx:start_idx + self.clip_len]
        else:
            # Loop padding for shorter videos
            sampled_frames = []
            for i in range(self.clip_len):
                frame_idx = i % num_frames
                sampled_frames.append(frames[frame_idx])
            return sampled_frames
            
    def preprocess_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Enhanced frame preprocessing pipeline:
        1. Load original color videos
        2. Convert to grayscale using cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) consistently
        4. Replicate grayscale to 3 channels: np.stack([gray, gray, gray], axis=-1)
        5. Spatial resize: 96×96 → 112×112 using bilinear interpolation
        """
        processed_frames = []

        for frame in frames:
            # Step 1: Convert to grayscale on-the-fly
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame

            # Step 2: Apply CLAHE consistently across all splits if enabled
            if self.clahe_enabled:
                gray_frame = self.clahe.apply(gray_frame)

            # Step 3: Replicate grayscale to 3 channels for pretrained model compatibility
            # This creates a 3-channel image where all channels are identical
            rgb_frame = np.stack([gray_frame, gray_frame, gray_frame], axis=-1)

            # Step 4: Spatial resize from 96×96 → 112×112 using bilinear interpolation
            # This matches pretrained R(2+1)D model expectations
            if rgb_frame.shape[:2] != (self.resize_for_backbone, self.resize_for_backbone):
                rgb_frame = cv2.resize(
                    rgb_frame,
                    (self.resize_for_backbone, self.resize_for_backbone),
                    interpolation=cv2.INTER_LINEAR
                )

            processed_frames.append(rgb_frame)

        return processed_frames
        
    def apply_augmentations(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply conservative augmentations."""
        augmented_frames = []
        
        # Random brightness adjustment
        brightness_delta = random.uniform(-self.brightness_range, self.brightness_range)
        
        # Random rotation (small)
        rotation_angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        
        # Random translation (small)
        tx = random.randint(-self.translation_pixels, self.translation_pixels)
        ty = random.randint(-self.translation_pixels, self.translation_pixels)
        
        for frame in frames:
            h, w = frame.shape[:2]
            
            # Apply brightness
            frame_aug = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness_delta * 255)
            
            # Apply rotation and translation
            if abs(rotation_angle) > 0.1 or abs(tx) > 0 or abs(ty) > 0:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                M[0, 2] += tx
                M[1, 2] += ty
                frame_aug = cv2.warpAffine(frame_aug, M, (w, h))
                
            augmented_frames.append(frame_aug)
            
        return augmented_frames
        
    def resize_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Resize frames for backbone compatibility."""
        if self.resize_for_backbone == self.img_size:
            return frames
            
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (self.resize_for_backbone, self.resize_for_backbone))
            resized_frames.append(resized)
            
        return resized_frames
        
    def frames_to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Convert 3-channel replicated grayscale frames to tensor with Kinetics-400 normalization.

        Input: List of (H, W, 3) frames where all 3 channels are identical grayscale
        Returns: tensor (C, T, H, W) where C=3 for 3-channel grayscale
        """
        # Stack frames: (T, H, W, 3)
        video_array = np.stack(frames, axis=0)

        # Normalize to [0, 1]
        video_array = video_array.astype(np.float32) / 255.0

        # Convert to tensor and rearrange: (T, H, W, 3) -> (3, T, H, W)
        video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2)

        # Apply Kinetics-400 normalization frame-by-frame
        # The normalization expects (C, H, W) format, so we need to apply it to each frame
        normalized_frames = []
        for t in range(video_tensor.shape[1]):  # Iterate over time dimension
            frame = video_tensor[:, t, :, :]  # Shape: (3, H, W)
            normalized_frame = self.kinetics_normalize(frame)
            normalized_frames.append(normalized_frame)

        # Stack back to (3, T, H, W)
        video_tensor = torch.stack(normalized_frames, dim=1)

        return video_tensor
        
    def get_dummy_sample(self, class_name: str) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """Return a dummy sample for error cases."""
        # Create dummy video tensor (3, T, H, W) for 3-channel grayscale
        dummy_frames = torch.zeros(3, self.clip_len, self.resize_for_backbone, self.resize_for_backbone)

        # Apply Kinetics-400 normalization frame-by-frame
        normalized_frames = []
        for t in range(dummy_frames.shape[1]):  # Iterate over time dimension
            frame = dummy_frames[:, t, :, :]  # Shape: (3, H, W)
            normalized_frame = self.kinetics_normalize(frame)
            normalized_frames.append(normalized_frame)

        # Stack back to (3, T, H, W)
        dummy_frames = torch.stack(normalized_frames, dim=1)
        
        # Get label
        label = self.class_to_idx.get(class_name, 0)
        
        # Dummy metadata
        metadata = {
            'video_path': 'dummy',
            'class_name': class_name,
            'original_frames': self.clip_len,
            'gender': 'unknown',
            'age_band': 'unknown',
            'ethnicity': 'unknown',
            'source': 'dummy',
            'processed_version': 'dummy'
        }
        
        return dummy_frames, label, metadata
        
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balancing."""
        class_counts = self.manifest_df['class'].value_counts()
        
        # Ensure all classes are represented
        weights = []
        for class_name in self.class_names:
            count = class_counts.get(class_name, 1)  # Avoid division by zero
            weights.append(1.0 / count)
            
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum() * len(weights)
        
        return torch.FloatTensor(weights)
        
    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for WeightedRandomSampler."""
        class_counts = self.manifest_df['class'].value_counts()
        
        sample_weights = []
        for _, row in self.manifest_df.iterrows():
            class_name = row['class']
            count = class_counts[class_name]
            weight = 1.0 / np.sqrt(count)  # Inverse square root weighting
            sample_weights.append(weight)
            
        return torch.FloatTensor(sample_weights)


def collate_fn(batch: List[Tuple[torch.Tensor, int, Dict[str, Any]]]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """
    Custom collate function for video batches.
    
    Args:
        batch: List of (video_tensor, label, metadata) tuples
        
    Returns:
        videos: (B, C, T, H, W) tensor
        labels: (B,) tensor
        metadata: List of metadata dicts
    """
    videos, labels, metadata = zip(*batch)
    
    # Stack videos: (B, C, T, H, W)
    videos = torch.stack(videos, dim=0)
    
    # Stack labels: (B,)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return videos, labels, list(metadata)
