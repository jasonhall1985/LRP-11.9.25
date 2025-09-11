"""
PyTorch dataset classes for ICU lip reading data.
Handles loading preprocessed data with proper transforms and augmentations.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import logging
from pathlib import Path
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICULipReadingDataset(Dataset):
    """Dataset for ICU lip reading classification."""
    
    def __init__(self,
                 data_dir: str,
                 split: str = "train",
                 classes: List[str] = None,
                 transform: Optional[Callable] = None,
                 target_frames: int = 16,
                 load_in_memory: bool = False):
        """
        Initialize ICU lip reading dataset.
        
        Args:
            data_dir: Directory containing processed data
            split: Data split ("train", "val", "test")
            classes: List of class names
            transform: Optional transform function
            target_frames: Target number of frames per video
            load_in_memory: Whether to load all data in memory
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.classes = classes or ["doctor", "glasses", "phone", "pillow", "help"]
        self.transform = transform
        self.target_frames = target_frames
        self.load_in_memory = load_in_memory
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load file paths and labels
        self.samples = self._load_samples()
        
        # Load data in memory if requested
        self.data_cache = {}
        if load_in_memory:
            self._load_all_data()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        self._print_class_distribution()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load sample paths and labels."""
        samples = []
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory does not exist: {split_dir}")
        
        for class_name in self.classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory does not exist: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Get all .npy files in class directory
            for file_path in class_dir.glob("*.npy"):
                samples.append((str(file_path), class_idx))
        
        return samples
    
    def _load_all_data(self):
        """Load all data into memory."""
        logger.info(f"Loading all {len(self.samples)} samples into memory...")
        for i, (file_path, label) in enumerate(self.samples):
            try:
                data = np.load(file_path)
                self.data_cache[i] = data
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        logger.info(f"Loaded {len(self.data_cache)} samples into memory")
    
    def _print_class_distribution(self):
        """Print class distribution."""
        class_counts = {}
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        logger.info(f"Class distribution for {self.split}:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (data, label)
        """
        file_path, label = self.samples[idx]
        
        # Load data
        if self.load_in_memory and idx in self.data_cache:
            data = self.data_cache[idx]
        else:
            try:
                data = np.load(file_path)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                # Return zeros as fallback
                data = np.zeros((self.target_frames, 96, 96, 1), dtype=np.float32)
        
        # Ensure correct shape and type
        data = data.astype(np.float32)
        
        # Handle different input shapes
        if data.ndim == 4:  # (T, H, W, C)
            pass
        elif data.ndim == 3:  # (T, H, W) - add channel dimension
            data = np.expand_dims(data, axis=-1)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
        
        # Standardize sequence length
        if data.shape[0] != self.target_frames:
            data = self._standardize_sequence_length(data)
        
        # Apply transforms
        if self.transform:
            data = self.transform(data)
        
        # Convert to tensor with proper dtype
        data_tensor = torch.from_numpy(data).float()

        return data_tensor, label
    
    def _standardize_sequence_length(self, data: np.ndarray) -> np.ndarray:
        """Standardize sequence to target length."""
        current_length = data.shape[0]
        
        if current_length == self.target_frames:
            return data
        elif current_length < self.target_frames:
            # Pad with last frame
            padding_needed = self.target_frames - current_length
            last_frame = data[-1:].repeat(padding_needed, axis=0)
            return np.concatenate([data, last_frame], axis=0)
        else:
            # Center crop
            start_idx = (current_length - self.target_frames) // 2
            return data[start_idx:start_idx + self.target_frames]
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total_samples = len(self.samples)
        num_classes = len(self.classes)
        
        weights = []
        for i in range(num_classes):
            if i in class_counts:
                weight = total_samples / (num_classes * class_counts[i])
            else:
                weight = 1.0
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


class DataAugmentation:
    """Data augmentation for lip reading videos."""
    
    def __init__(self,
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 noise_std: float = 0.01,
                 temporal_jitter: bool = True,
                 apply_prob: float = 0.5):
        """
        Initialize data augmentation.
        
        Args:
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            noise_std: Standard deviation for Gaussian noise
            temporal_jitter: Whether to apply temporal jittering
            apply_prob: Probability of applying each augmentation
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.temporal_jitter = temporal_jitter
        self.apply_prob = apply_prob
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to data.
        
        Args:
            data: Input data of shape (T, H, W, C)
            
        Returns:
            Augmented data
        """
        data = data.copy()
        
        # Brightness adjustment
        if random.random() < self.apply_prob:
            brightness_factor = random.uniform(*self.brightness_range)
            data = np.clip(data * brightness_factor, 0, 1)
        
        # Contrast adjustment
        if random.random() < self.apply_prob:
            contrast_factor = random.uniform(*self.contrast_range)
            mean = data.mean()
            data = np.clip((data - mean) * contrast_factor + mean, 0, 1)
        
        # Gaussian noise
        if random.random() < self.apply_prob:
            noise = np.random.normal(0, self.noise_std, data.shape)
            data = np.clip(data + noise, 0, 1)
        
        # Temporal jittering (slight frame reordering)
        if self.temporal_jitter and random.random() < self.apply_prob:
            data = self._apply_temporal_jitter(data)
        
        return data
    
    def _apply_temporal_jitter(self, data: np.ndarray, max_shift: int = 1) -> np.ndarray:
        """Apply slight temporal jittering."""
        T = data.shape[0]
        if T <= 2 * max_shift:
            return data
        
        # Create slightly jittered indices
        indices = list(range(T))
        for i in range(max_shift, T - max_shift):
            if random.random() < 0.3:  # 30% chance to jitter each frame
                shift = random.randint(-max_shift, max_shift)
                new_idx = max(0, min(T - 1, i + shift))
                indices[i] = new_idx
        
        return data[indices]


def create_data_loaders(data_dir: str,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       target_frames: int = 16,
                       augment_train: bool = True,
                       load_in_memory: bool = False) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing processed data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        target_frames: Target number of frames per video
        augment_train: Whether to apply augmentation to training data
        load_in_memory: Whether to load all data in memory
        
    Returns:
        Dictionary of data loaders
    """
    classes = ["doctor", "glasses", "phone", "pillow", "help"]
    
    # Create augmentation transform for training
    train_transform = DataAugmentation() if augment_train else None
    
    # Create datasets
    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = ICULipReadingDataset(
            data_dir=data_dir,
            split=split,
            classes=classes,
            transform=train_transform if split == "train" else None,
            target_frames=target_frames,
            load_in_memory=load_in_memory
        )
    
    # Create data loaders
    data_loaders = {}
    for split, dataset in datasets.items():
        shuffle = (split == "train")
        data_loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train")  # Drop last incomplete batch for training
        )
    
    return data_loaders


def test_dataset():
    """Test the dataset implementation."""
    print("Testing ICU Dataset...")
    
    # Test with simple processed data directory
    data_dir = "simple_processed_data"
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Skipping test.")
        return
    
    # Create dataset
    dataset = ICULipReadingDataset(
        data_dir=data_dir,
        split="train",
        target_frames=16
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test loading a sample
        data, label = dataset[0]
        print(f"Sample shape: {data.shape}")
        print(f"Sample label: {label} ({dataset.idx_to_class[label]})")
        
        # Test class weights
        weights = dataset.get_class_weights()
        print(f"Class weights: {weights}")
        
        # Test data loader
        data_loaders = create_data_loaders(
            data_dir=data_dir,
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            target_frames=16
        )
        
        for split, loader in data_loaders.items():
            print(f"{split} loader: {len(loader)} batches")
            if len(loader) > 0:
                batch_data, batch_labels = next(iter(loader))
                print(f"  Batch shape: {batch_data.shape}")
                print(f"  Batch labels: {batch_labels}")
        
        print("✅ Dataset test passed!")
    else:
        print("⚠️ Dataset is empty")


if __name__ == "__main__":
    test_dataset()
