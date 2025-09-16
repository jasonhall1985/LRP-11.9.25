#!/usr/bin/env python3
"""
Debug index mismatch issues in the training pipeline.
"""

import sys
import os
sys.path.append('.')

import pandas as pd
import torch
from torch.utils.data import DataLoader
import logging
from lipreader_dataset import LipReaderDataset

# Set up logging to capture warnings
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def test_index_boundaries():
    """Test if DataLoader is accessing indices outside the dataset bounds."""
    print("=== TESTING INDEX BOUNDARIES ===")
    
    # Load manifest and create splits
    df = pd.read_csv("clean_balanced_manifest.csv")
    print(f"Full dataset size: {len(df)}")
    
    # Create demographic splits
    val_mask = df['age_band'] == '40-64'
    val_df = df[val_mask].copy()
    
    test_mask = (df['gender'] == 'female') & (df['age_band'] == '18-39')
    test_df = df[test_mask].copy()
    
    train_mask = ~(val_mask | test_mask)
    train_df = df[train_mask].copy()
    
    print(f"Split sizes: TRAIN={len(train_df)}, VAL={len(val_df)}, TEST={len(test_df)}")
    
    # Create train dataset
    augmentation_config = {
        'enabled': True,
        'aug_prob': 0.5,
        'temporal_jitter': 2,
        'brightness_factor': 0.1,
        'contrast_range': (0.9, 1.1),
        'horizontal_flip_prob': 0.5
    }
    
    train_dataset = LipReaderDataset(
        manifest_df=train_df,
        clip_len=24,
        img_size=96,
        resize_for_backbone=112,
        clahe_enabled=True,
        augmentation_config=augmentation_config,
        is_training=True
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Train dataset manifest shape: {train_dataset.manifest_df.shape}")
    print(f"Train dataset index range: 0 to {len(train_dataset)-1}")
    
    # Test boundary indices
    boundary_indices = [0, 1, len(train_dataset)-2, len(train_dataset)-1]
    out_of_bounds_indices = [len(train_dataset), len(train_dataset)+1, 1500, 2000]
    
    print(f"\nTesting boundary indices:")
    for idx in boundary_indices:
        try:
            video_tensor, label, metadata = train_dataset[idx]
            if metadata['video_path'] == 'dummy':
                print(f"  Index {idx}: DUMMY (should not happen for valid indices)")
            else:
                print(f"  Index {idx}: SUCCESS - {metadata['class_name']}")
        except Exception as e:
            print(f"  Index {idx}: ERROR - {e}")
    
    print(f"\nTesting out-of-bounds indices:")
    for idx in out_of_bounds_indices:
        try:
            video_tensor, label, metadata = train_dataset[idx]
            if metadata['video_path'] == 'dummy':
                print(f"  Index {idx}: DUMMY (expected for out-of-bounds)")
            else:
                print(f"  Index {idx}: UNEXPECTED SUCCESS - {metadata['class_name']}")
        except Exception as e:
            print(f"  Index {idx}: ERROR (expected) - {e}")

def test_dataloader_indices():
    """Test what indices the DataLoader is actually requesting."""
    print("\n=== TESTING DATALOADER INDICES ===")
    
    # Create a custom dataset that logs index access
    class LoggingDataset(LipReaderDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.accessed_indices = []
            self.dummy_count = 0
            
        def __getitem__(self, idx):
            self.accessed_indices.append(idx)
            result = super().__getitem__(idx)
            
            # Check if it's a dummy sample
            if result[2]['video_path'] == 'dummy':
                self.dummy_count += 1
                print(f"    DUMMY returned for index {idx} (dataset size: {len(self)})")
            
            return result
    
    # Load data and create splits
    df = pd.read_csv("clean_balanced_manifest.csv")
    
    val_mask = df['age_band'] == '40-64'
    test_mask = (df['gender'] == 'female') & (df['age_band'] == '18-39')
    train_mask = ~(val_mask | test_mask)
    train_df = df[train_mask].copy()
    
    # Create logging dataset
    augmentation_config = {
        'enabled': True,
        'aug_prob': 0.5,
        'temporal_jitter': 2,
        'brightness_factor': 0.1,
        'contrast_range': (0.9, 1.1),
        'horizontal_flip_prob': 0.5
    }
    
    train_dataset = LoggingDataset(
        manifest_df=train_df,
        clip_len=24,
        img_size=96,
        resize_for_backbone=112,
        clahe_enabled=True,
        augmentation_config=augmentation_config,
        is_training=True
    )
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    
    print(f"DataLoader created with dataset size: {len(train_dataset)}")
    
    # Process one batch
    print(f"Processing one batch...")
    for batch_idx, (videos, labels, metadata_list) in enumerate(train_loader):
        print(f"Batch {batch_idx}: processed {len(videos)} samples")
        break
    
    print(f"\nIndex access analysis:")
    print(f"  Indices accessed: {sorted(train_dataset.accessed_indices)}")
    print(f"  Min index: {min(train_dataset.accessed_indices)}")
    print(f"  Max index: {max(train_dataset.accessed_indices)}")
    print(f"  Dataset size: {len(train_dataset)}")
    print(f"  Out-of-bounds accesses: {sum(1 for idx in train_dataset.accessed_indices if idx >= len(train_dataset))}")
    print(f"  Dummy samples returned: {train_dataset.dummy_count}")

if __name__ == "__main__":
    test_index_boundaries()
    test_dataloader_indices()
