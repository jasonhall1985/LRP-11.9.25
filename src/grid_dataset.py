"""
GRID corpus dataset implementation for LipNet pretraining.
Handles character-level sequence learning with CTC loss.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import logging
import random
from pathlib import Path
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GRIDVocabulary:
    """Vocabulary for GRID corpus character-level recognition."""
    
    def __init__(self):
        """Initialize vocabulary with characters and special tokens."""
        # GRID corpus uses specific command words and letters
        self.chars = list(string.ascii_lowercase) + [' ']  # a-z + space
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # Add blank token for CTC (index 0 is reserved for blank)
        self.blank_idx = len(self.chars)
        self.vocab_size = len(self.chars) + 1  # +1 for blank
        
        logger.info(f"GRID vocabulary size: {self.vocab_size} (including blank)")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to character indices."""
        text = text.lower().strip()
        return [self.char_to_idx.get(char, self.char_to_idx[' ']) for char in text]
    
    def decode(self, indices: List[int]) -> str:
        """Decode character indices to text."""
        chars = [self.idx_to_char.get(idx, '') for idx in indices if idx != self.blank_idx]
        return ''.join(chars)


class MockGRIDDataset(Dataset):
    """
    Mock GRID dataset for demonstration purposes.
    In production, this would load actual GRID corpus data.
    """
    
    def __init__(self, 
                 split: str = "train",
                 num_samples: int = 1000,
                 sequence_length: int = 16,
                 vocab: Optional[GRIDVocabulary] = None):
        """
        Initialize mock GRID dataset.
        
        Args:
            split: Dataset split ("train", "val", "test")
            num_samples: Number of synthetic samples to generate
            sequence_length: Video sequence length
            vocab: Vocabulary object
        """
        self.split = split
        self.sequence_length = sequence_length
        self.vocab = vocab or GRIDVocabulary()
        
        # GRID corpus typical command structure: "bin [color] [preposition] [letter] [digit] [adverb]"
        self.colors = ["blue", "green", "red", "white"]
        self.prepositions = ["at", "by", "in", "with"]
        self.letters = list(string.ascii_lowercase[:10])  # a-j
        self.digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        self.adverbs = ["again", "now", "please", "soon"]
        
        # Generate synthetic samples
        self.samples = self._generate_samples(num_samples)
        
        logger.info(f"Generated {len(self.samples)} mock GRID samples for {split}")
    
    def _generate_samples(self, num_samples: int) -> List[Tuple[np.ndarray, str]]:
        """Generate synthetic GRID-like samples."""
        samples = []
        
        for _ in range(num_samples):
            # Generate GRID-like command
            color = random.choice(self.colors)
            prep = random.choice(self.prepositions)
            letter = random.choice(self.letters)
            digit = random.choice(self.digits)
            adverb = random.choice(self.adverbs)
            
            # Create command text
            command = f"bin {color} {prep} {letter} {digit} {adverb}"
            
            # Generate synthetic video data (random for demo)
            video_data = np.random.randn(self.sequence_length, 96, 96, 1).astype(np.float32)
            video_data = np.clip(video_data * 0.1 + 0.5, 0, 1)  # Normalize to [0,1]
            
            samples.append((video_data, command))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            video: Video tensor of shape (T, H, W, C)
            target: Target character indices
            target_length: Length of target sequence
        """
        video_data, text = self.samples[idx]
        
        # Encode text to character indices
        target_indices = self.vocab.encode(text)
        
        # Convert to tensors
        video_tensor = torch.from_numpy(video_data).float()
        target_tensor = torch.tensor(target_indices, dtype=torch.long)
        target_length = torch.tensor(len(target_indices), dtype=torch.long)
        
        return video_tensor, target_tensor, target_length


def ctc_collate_fn(batch):
    """
    Collate function for CTC training.
    Handles variable-length sequences.
    """
    videos, targets, target_lengths = zip(*batch)
    
    # Stack videos (all same length)
    videos = torch.stack(videos, dim=0)
    
    # Concatenate targets and create lengths tensor
    targets_cat = torch.cat(targets, dim=0)
    target_lengths = torch.stack(target_lengths, dim=0)
    
    # Input lengths (all sequences have same length after encoder)
    batch_size = len(batch)
    # Assuming encoder reduces sequence length by factor of 4 (due to temporal pooling)
    input_length = videos.shape[1] // 4
    input_lengths = torch.full((batch_size,), input_length, dtype=torch.long)
    
    return videos, targets_cat, input_lengths, target_lengths


def create_grid_data_loaders(batch_size: int = 16,
                           num_workers: int = 4,
                           train_samples: int = 1000,
                           val_samples: int = 200,
                           test_samples: int = 200) -> Dict[str, DataLoader]:
    """
    Create GRID data loaders for CTC training.
    
    Args:
        batch_size: Batch size
        num_workers: Number of worker processes
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        
    Returns:
        Dictionary of data loaders
    """
    vocab = GRIDVocabulary()
    
    # Create datasets
    datasets = {
        'train': MockGRIDDataset('train', train_samples, vocab=vocab),
        'val': MockGRIDDataset('val', val_samples, vocab=vocab),
        'test': MockGRIDDataset('test', test_samples, vocab=vocab)
    }
    
    # Create data loaders
    data_loaders = {}
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        data_loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=ctc_collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == 'train')
        )
    
    return data_loaders, vocab


class RealGRIDDataset(Dataset):
    """
    Real GRID dataset implementation.
    This would be used if actual GRID corpus data is available.
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = "train",
                 vocab: Optional[GRIDVocabulary] = None):
        """
        Initialize real GRID dataset.
        
        Args:
            data_dir: Directory containing GRID corpus data
            split: Dataset split
            vocab: Vocabulary object
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.vocab = vocab or GRIDVocabulary()
        
        # Load file paths and transcriptions
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} GRID samples for {split}")
    
    def _load_samples(self) -> List[Tuple[str, str]]:
        """Load GRID samples from directory structure."""
        samples = []
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            logger.warning(f"GRID split directory does not exist: {split_dir}")
            return samples
        
        # GRID corpus structure: speaker/video_file.mp4 and corresponding .txt
        for video_file in split_dir.glob("**/*.mp4"):
            txt_file = video_file.with_suffix('.txt')
            if txt_file.exists():
                with open(txt_file, 'r') as f:
                    transcription = f.read().strip()
                samples.append((str(video_file), transcription))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample from the real GRID dataset."""
        video_path, text = self.samples[idx]
        
        # Load and preprocess video (would use actual video loading here)
        # For now, return mock data
        video_data = np.random.randn(16, 96, 96, 1).astype(np.float32)
        video_data = np.clip(video_data * 0.1 + 0.5, 0, 1)
        
        # Encode text
        target_indices = self.vocab.encode(text)
        
        # Convert to tensors
        video_tensor = torch.from_numpy(video_data).float()
        target_tensor = torch.tensor(target_indices, dtype=torch.long)
        target_length = torch.tensor(len(target_indices), dtype=torch.long)
        
        return video_tensor, target_tensor, target_length


def test_grid_dataset():
    """Test GRID dataset implementation."""
    print("Testing GRID Dataset...")
    
    # Test vocabulary
    vocab = GRIDVocabulary()
    test_text = "bin blue at a zero now"
    encoded = vocab.encode(test_text)
    decoded = vocab.decode(encoded)
    
    print(f"Original: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    print(f"Vocabulary size: {vocab.vocab_size}")
    
    # Test dataset
    dataset = MockGRIDDataset('train', num_samples=10)
    print(f"Dataset size: {len(dataset)}")
    
    # Test sample
    video, target, target_length = dataset[0]
    print(f"Video shape: {video.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Target length: {target_length}")
    print(f"Target text: '{vocab.decode(target.tolist())}'")
    
    # Test data loader
    data_loaders, vocab = create_grid_data_loaders(
        batch_size=4,
        num_workers=0,
        train_samples=20,
        val_samples=10,
        test_samples=10
    )
    
    # Test batch
    train_loader = data_loaders['train']
    videos, targets, input_lengths, target_lengths = next(iter(train_loader))
    
    print(f"Batch videos shape: {videos.shape}")
    print(f"Batch targets shape: {targets.shape}")
    print(f"Input lengths: {input_lengths}")
    print(f"Target lengths: {target_lengths}")
    
    print("✅ GRID dataset test passed!")


if __name__ == "__main__":
    test_grid_dataset()
