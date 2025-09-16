#!/usr/bin/env python3
"""
Video Transforms for Lip Reading
================================

Conservative augmentations specifically designed for lip reading:
- Preserves lip reading semantics
- NO horizontal flip (breaks directionality)
- Temporal jitter for temporal robustness
- CLAHE contrast enhancement
- Small spatial perturbations

Features:
- Temporal augmentations (jitter, dropout)
- Conservative spatial augmentations
- CLAHE contrast enhancement
- Grayscale-specific transforms
- Lip reading semantic preservation

Author: Production Lip Reading System
Date: 2025-09-15
"""

import cv2
import torch
import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Any
import torch.nn.functional as F
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class VideoTransforms:
    """
    Conservative video transforms for lip reading.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        is_training: bool = True
    ):
        """
        Initialize video transforms.
        
        Args:
            config: Transform configuration dictionary
            is_training: Whether this is for training (enables augmentations)
        """
        self.config = config
        self.is_training = is_training
        self.enabled = config.get('enabled', True) and is_training
        
        # Augmentation parameters
        self.aug_prob = config.get('aug_prob', 0.5)
        
        # Temporal augmentations
        self.temporal_jitter = config.get('temporal_jitter', 2)
        self.temporal_dropout = config.get('temporal_dropout', 0.0)
        
        # Spatial augmentations (conservative)
        self.brightness_range = config.get('brightness_range', 0.1)
        self.rotation_degrees = config.get('rotation_degrees', 2)
        self.translation_pixels = config.get('translation_pixels', 2)
        
        # Disabled augmentations (preserve lip reading semantics)
        self.horizontal_flip = config.get('horizontal_flip', False)  # Always False
        self.vertical_flip = config.get('vertical_flip', False)      # Always False
        self.elastic_transform = config.get('elastic_transform', False)  # Always False
        
        # CLAHE setup
        self.clahe_enabled = config.get('clahe_enabled', True)
        if self.clahe_enabled:
            self.clahe = cv2.createCLAHE(
                clipLimit=config.get('clahe_clip_limit', 2.0),
                tileGridSize=tuple(config.get('clahe_tile_grid', [8, 8]))
            )
            
        logger.info(f"VideoTransforms initialized: enabled={self.enabled}, "
                   f"training={is_training}, aug_prob={self.aug_prob}")
        
    def __call__(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply transforms to video frames.
        
        Args:
            frames: List of video frames (H, W) or (H, W, C)
            
        Returns:
            Transformed frames
        """
        if not self.enabled:
            return self.apply_clahe(frames) if self.clahe_enabled else frames
            
        # Apply augmentations with probability
        if random.random() < self.aug_prob:
            frames = self.apply_augmentations(frames)
        else:
            # Always apply CLAHE even without other augmentations
            frames = self.apply_clahe(frames) if self.clahe_enabled else frames
            
        return frames
        
    def apply_augmentations(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply all enabled augmentations."""
        # Temporal augmentations
        frames = self.apply_temporal_jitter(frames)
        frames = self.apply_temporal_dropout(frames)
        
        # Spatial augmentations
        frames = self.apply_spatial_augmentations(frames)
        
        # CLAHE enhancement
        if self.clahe_enabled:
            frames = self.apply_clahe(frames)
            
        return frames
        
    def apply_temporal_jitter(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply temporal jitter by slightly shifting frame selection.
        """
        if self.temporal_jitter <= 0 or len(frames) <= self.temporal_jitter * 2:
            return frames
            
        # Random jitter amount
        jitter = random.randint(-self.temporal_jitter, self.temporal_jitter)
        
        if jitter == 0:
            return frames
            
        # Apply jitter by shifting frame indices
        num_frames = len(frames)
        jittered_frames = []
        
        for i in range(num_frames):
            # Calculate jittered index with bounds checking
            jittered_idx = max(0, min(num_frames - 1, i + jitter))
            jittered_frames.append(frames[jittered_idx])
            
        return jittered_frames
        
    def apply_temporal_dropout(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply temporal dropout by randomly dropping frames.
        """
        if self.temporal_dropout <= 0.0:
            return frames
            
        dropout_frames = []
        for frame in frames:
            if random.random() > self.temporal_dropout:
                dropout_frames.append(frame)
            else:
                # Replace dropped frame with previous frame (or first frame if at start)
                if dropout_frames:
                    dropout_frames.append(dropout_frames[-1].copy())
                else:
                    dropout_frames.append(frame)
                    
        return dropout_frames
        
    def apply_spatial_augmentations(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply conservative spatial augmentations."""
        # Sample augmentation parameters once for all frames (consistency)
        brightness_delta = random.uniform(-self.brightness_range, self.brightness_range)
        rotation_angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        tx = random.randint(-self.translation_pixels, self.translation_pixels)
        ty = random.randint(-self.translation_pixels, self.translation_pixels)
        
        augmented_frames = []
        
        for frame in frames:
            augmented_frame = frame.copy()
            
            # Apply brightness adjustment
            if abs(brightness_delta) > 0.01:
                augmented_frame = cv2.convertScaleAbs(
                    augmented_frame, 
                    alpha=1.0, 
                    beta=brightness_delta * 255
                )
                
            # Apply rotation and translation
            if abs(rotation_angle) > 0.1 or abs(tx) > 0 or abs(ty) > 0:
                h, w = augmented_frame.shape[:2]
                center = (w // 2, h // 2)
                
                # Create transformation matrix
                M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                M[0, 2] += tx
                M[1, 2] += ty
                
                # Apply transformation
                augmented_frame = cv2.warpAffine(augmented_frame, M, (w, h))
                
            augmented_frames.append(augmented_frame)
            
        return augmented_frames
        
    def apply_clahe(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply CLAHE contrast enhancement."""
        if not self.clahe_enabled:
            return frames
            
        enhanced_frames = []
        
        for frame in frames:
            # Ensure frame is grayscale
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame
                
            # Apply CLAHE
            enhanced_frame = self.clahe.apply(gray_frame)
            enhanced_frames.append(enhanced_frame)
            
        return enhanced_frames


class VideoNormalization:
    """
    Video normalization utilities.
    """
    
    def __init__(
        self,
        mean: List[float] = [0.5],
        std: List[float] = [0.5],
        normalize_range: Tuple[float, float] = (0, 1)
    ):
        """
        Initialize video normalization.
        
        Args:
            mean: Normalization mean (for grayscale: [0.5])
            std: Normalization std (for grayscale: [0.5])
            normalize_range: Range to normalize pixel values to
        """
        self.mean = mean
        self.std = std
        self.normalize_range = normalize_range
        
        # Create torchvision normalizer
        self.normalizer = transforms.Normalize(mean=mean, std=std)
        
    def normalize_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Normalize frames and convert to tensor.
        
        Args:
            frames: List of frames (H, W) grayscale
            
        Returns:
            Normalized tensor (1, T, H, W)
        """
        # Stack frames: (T, H, W)
        video_array = np.stack(frames, axis=0)
        
        # Normalize to specified range
        min_val, max_val = self.normalize_range
        video_array = video_array.astype(np.float32) / 255.0
        video_array = video_array * (max_val - min_val) + min_val
        
        # Convert to tensor: (T, H, W) -> (1, T, H, W)
        video_tensor = torch.from_numpy(video_array).unsqueeze(0)
        
        # Apply normalization
        video_tensor = self.normalizer(video_tensor)
        
        return video_tensor


class VideoResize:
    """
    Video resizing utilities.
    """
    
    def __init__(
        self,
        input_size: int = 96,
        output_size: int = 112,
        interpolation: int = cv2.INTER_LINEAR
    ):
        """
        Initialize video resizer.
        
        Args:
            input_size: Input frame size (assumed square)
            output_size: Output frame size (assumed square)
            interpolation: OpenCV interpolation method
        """
        self.input_size = input_size
        self.output_size = output_size
        self.interpolation = interpolation
        
    def resize_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Resize frames to target size.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of resized frames
        """
        if self.input_size == self.output_size:
            return frames
            
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(
                frame, 
                (self.output_size, self.output_size),
                interpolation=self.interpolation
            )
            resized_frames.append(resized)
            
        return resized_frames


class VideoTemporalSampling:
    """
    Temporal sampling utilities for videos.
    """
    
    def __init__(
        self,
        clip_len: int = 24,
        sampling_mode: str = 'uniform',
        padding_mode: str = 'loop'
    ):
        """
        Initialize temporal sampler.
        
        Args:
            clip_len: Target number of frames
            sampling_mode: Sampling strategy ('uniform', 'random', 'center')
            padding_mode: Padding strategy for short videos ('loop', 'repeat', 'zero')
        """
        self.clip_len = clip_len
        self.sampling_mode = sampling_mode
        self.padding_mode = padding_mode
        
    def sample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Sample frames to target length.
        
        Args:
            frames: Input frames
            
        Returns:
            Sampled frames
        """
        num_frames = len(frames)
        
        if num_frames == self.clip_len:
            return frames
        elif num_frames > self.clip_len:
            return self._downsample_frames(frames)
        else:
            return self._upsample_frames(frames)
            
    def _downsample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Downsample frames when video is longer than target."""
        num_frames = len(frames)
        
        if self.sampling_mode == 'uniform':
            # Uniform sampling
            indices = np.linspace(0, num_frames - 1, self.clip_len, dtype=int)
        elif self.sampling_mode == 'center':
            # Center cropping
            start_idx = (num_frames - self.clip_len) // 2
            indices = list(range(start_idx, start_idx + self.clip_len))
        elif self.sampling_mode == 'random':
            # Random sampling (for training)
            start_idx = random.randint(0, num_frames - self.clip_len)
            indices = list(range(start_idx, start_idx + self.clip_len))
        else:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")
            
        return [frames[i] for i in indices]
        
    def _upsample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Upsample frames when video is shorter than target."""
        num_frames = len(frames)
        sampled_frames = []
        
        for i in range(self.clip_len):
            if self.padding_mode == 'loop':
                # Loop through frames
                frame_idx = i % num_frames
            elif self.padding_mode == 'repeat':
                # Repeat last frame
                frame_idx = min(i, num_frames - 1)
            elif self.padding_mode == 'zero':
                # Zero padding (create black frames)
                if i < num_frames:
                    frame_idx = i
                else:
                    # Create zero frame with same shape as first frame
                    zero_frame = np.zeros_like(frames[0])
                    sampled_frames.append(zero_frame)
                    continue
            else:
                raise ValueError(f"Unknown padding_mode: {self.padding_mode}")
                
            sampled_frames.append(frames[frame_idx])
            
        return sampled_frames


def create_video_transforms(
    config: Dict[str, Any],
    is_training: bool = True
) -> Dict[str, Any]:
    """
    Create video transform pipeline.
    
    Args:
        config: Transform configuration
        is_training: Whether for training
        
    Returns:
        Dictionary of transform components
    """
    transforms_dict = {
        'augmentation': VideoTransforms(config.get('augmentation', {}), is_training),
        'temporal_sampling': VideoTemporalSampling(
            clip_len=config.get('clip_len', 24),
            sampling_mode=config.get('temporal_sampling', 'uniform'),
            padding_mode=config.get('padding_mode', 'loop')
        ),
        'resize': VideoResize(
            input_size=config.get('img_size', 96),
            output_size=config.get('resize_for_backbone', 112)
        ),
        'normalization': VideoNormalization(
            mean=config.get('mean', [0.5]),
            std=config.get('std', [0.5]),
            normalize_range=config.get('normalize_range', [0, 1])
        )
    }
    
    return transforms_dict


if __name__ == "__main__":
    # Test transforms
    config = {
        'augmentation': {
            'enabled': True,
            'aug_prob': 0.5,
            'temporal_jitter': 2,
            'brightness_range': 0.1,
            'rotation_degrees': 2,
            'translation_pixels': 2,
            'clahe_enabled': True,
            'clahe_clip_limit': 2.0,
            'clahe_tile_grid': [8, 8]
        },
        'clip_len': 24,
        'img_size': 96,
        'resize_for_backbone': 112,
        'mean': [0.5],
        'std': [0.5]
    }
    
    # Create transforms
    transforms_dict = create_video_transforms(config, is_training=True)
    
    # Test with dummy frames
    dummy_frames = [np.random.randint(0, 255, (96, 96), dtype=np.uint8) for _ in range(30)]
    
    print(f"Input: {len(dummy_frames)} frames of shape {dummy_frames[0].shape}")
    
    # Apply transforms
    augmented = transforms_dict['augmentation'](dummy_frames)
    sampled = transforms_dict['temporal_sampling'].sample_frames(augmented)
    resized = transforms_dict['resize'].resize_frames(sampled)
    tensor = transforms_dict['normalization'].normalize_frames(resized)
    
    print(f"Output tensor shape: {tensor.shape}")
    print(f"Output tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")
