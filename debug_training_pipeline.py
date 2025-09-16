#!/usr/bin/env python3
"""
Debug the exact training pipeline to identify video loading failures.
"""

import sys
import os
sys.path.append('.')

import pandas as pd
import torch
from torch.utils.data import DataLoader
import logging
import traceback
from lipreader_dataset import LipReaderDataset

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def replicate_training_setup():
    """Replicate the exact training setup to identify issues."""
    print("=== REPLICATING TRAINING SETUP ===")
    
    # Load manifest
    manifest_path = "clean_balanced_manifest.csv"
    df = pd.read_csv(manifest_path)
    print(f"Total videos in manifest: {len(df)}")
    
    # Create demographic splits (same as training)
    def create_demographic_splits(df, seed=42):
        """Create the same demographic splits as training."""
        import numpy as np
        np.random.seed(seed)
        
        # VAL: age_band=40-64 (both genders)
        val_mask = df['age_band'] == '40-64'
        val_df = df[val_mask].copy()
        
        # TEST: gender=female AND age_band=18-39
        test_mask = (df['gender'] == 'female') & (df['age_band'] == '18-39')
        test_df = df[test_mask].copy()
        
        # TRAIN: all remaining
        train_mask = ~(val_mask | test_mask)
        train_df = df[train_mask].copy()
        
        return train_df, val_df, test_df
    
    train_df, val_df, test_df = create_demographic_splits(df)
    print(f"Split sizes: TRAIN={len(train_df)}, VAL={len(val_df)}, TEST={len(test_df)}")
    
    # Create datasets with same config as training
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
    
    # Test specific indices that were failing
    failing_indices = [336, 379, 1251, 550, 521, 235, 464, 734, 1350, 302]
    print(f"\nTesting specific failing indices from training logs...")
    
    for idx in failing_indices:
        if idx < len(train_dataset):
            print(f"\nTesting index {idx}:")
            try:
                video_tensor, label, metadata = train_dataset[idx]
                if metadata['video_path'] == 'dummy':
                    print(f"  FAILED: Returned dummy sample")
                    # Get the actual video info
                    video_info = train_df.iloc[idx]
                    print(f"  Video path: {video_info['path']}")
                    print(f"  Class: {video_info['class']}")
                    print(f"  File exists: {os.path.exists(video_info['path'])}")
                    
                    # Try manual loading
                    try:
                        frames = train_dataset.load_video(video_info['path'])
                        if frames is None:
                            print(f"  Manual load: FAILED")
                        else:
                            print(f"  Manual load: SUCCESS ({len(frames)} frames)")
                    except Exception as e:
                        print(f"  Manual load: ERROR - {e}")
                        traceback.print_exc()
                else:
                    print(f"  SUCCESS: {metadata['class_name']} - shape {video_tensor.shape}")
            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()
        else:
            print(f"Index {idx}: OUT OF RANGE")

def test_dataloader_with_failing_indices():
    """Test DataLoader with the exact same setup as training."""
    print("\n=== TESTING DATALOADER ===")
    
    # Load manifest and create splits
    df = pd.read_csv("clean_balanced_manifest.csv")
    
    def create_demographic_splits(df, seed=42):
        import numpy as np
        np.random.seed(seed)
        
        val_mask = df['age_band'] == '40-64'
        val_df = df[val_mask].copy()
        
        test_mask = (df['gender'] == 'female') & (df['age_band'] == '18-39')
        test_df = df[test_mask].copy()
        
        train_mask = ~(val_mask | test_mask)
        train_df = df[train_mask].copy()
        
        return train_df, val_df, test_df
    
    train_df, val_df, test_df = create_demographic_splits(df)
    
    # Create dataset
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
    
    # Create DataLoader with same settings as training
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Same as training config
        pin_memory=False,
        drop_last=True
    )
    
    print(f"DataLoader created with {len(train_dataset)} samples")
    
    # Test first few batches
    dummy_count = 0
    total_samples = 0
    
    for batch_idx, (videos, labels, metadata_list) in enumerate(train_loader):
        batch_dummy_count = 0
        for i, metadata in enumerate(metadata_list):
            if metadata['video_path'] == 'dummy':
                batch_dummy_count += 1
                dummy_count += 1
            total_samples += 1
        
        print(f"Batch {batch_idx}: {videos.shape}, dummy samples: {batch_dummy_count}/{len(videos)}")
        
        if batch_idx >= 5:  # Test first 5 batches
            break
    
    print(f"\nDataLoader Results:")
    print(f"  Total samples processed: {total_samples}")
    print(f"  Dummy samples: {dummy_count}")
    print(f"  Dummy rate: {dummy_count/total_samples*100:.1f}%")

if __name__ == "__main__":
    replicate_training_setup()
    test_dataloader_with_failing_indices()
