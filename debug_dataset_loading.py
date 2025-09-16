#!/usr/bin/env python3
"""
Debug dataset loading to identify why videos are failing during training.
"""

import sys
import os
sys.path.append('.')

from lipreader_dataset import LipReaderDataset
from transforms_video import VideoTransforms
import pandas as pd
import torch
from torch.utils.data import DataLoader
import logging

# Set up logging to see warnings
logging.basicConfig(level=logging.WARNING)

def test_dataset_loading():
    """Test the actual dataset loading pipeline."""
    print("=== DATASET LOADING DEBUG ===")
    
    # Load manifest
    manifest_path = "clean_balanced_manifest.csv"
    df = pd.read_csv(manifest_path)
    print(f"Total videos in manifest: {len(df)}")
    
    # Create transforms
    transform_config = {
        'enabled': True,
        'aug_prob': 0.5,
        'temporal_jitter': 2,
        'brightness_factor': 0.1,
        'contrast_range': (0.9, 1.1),
        'horizontal_flip_prob': 0.5
    }
    transforms = VideoTransforms(config=transform_config, is_training=True)
    
    # Create dataset
    dataset = LipReaderDataset(
        manifest_df=df,
        clip_len=24,
        img_size=96,
        resize_for_backbone=112,
        augmentation_config=transform_config,
        is_training=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Class to index mapping: {dataset.class_to_idx}")
    
    # Test first 50 samples
    print("\nTesting first 50 samples...")
    failed_indices = []
    successful_indices = []
    
    for i in range(min(50, len(dataset))):
        try:
            video_tensor, label, metadata = dataset[i]
            
            # Check if it's a dummy sample
            if metadata['video_path'] == 'dummy':
                failed_indices.append(i)
                print(f"  Sample {i}: DUMMY (failed to load)")
            else:
                successful_indices.append(i)
                print(f"  Sample {i}: SUCCESS - {metadata['class_name']} - shape {video_tensor.shape}")
                
        except Exception as e:
            failed_indices.append(i)
            print(f"  Sample {i}: ERROR - {e}")
    
    print(f"\nResults for first 50 samples:")
    print(f"  Successful: {len(successful_indices)}")
    print(f"  Failed: {len(failed_indices)}")
    print(f"  Success rate: {len(successful_indices)/50*100:.1f}%")
    
    if failed_indices:
        print(f"\nFailed indices: {failed_indices}")
        
        # Analyze failed samples
        print("\nAnalyzing failed samples...")
        for idx in failed_indices[:5]:  # Check first 5 failures
            video_info = df.iloc[idx]
            print(f"  Index {idx}:")
            print(f"    Path: {video_info['path']}")
            print(f"    Class: {video_info['class']}")
            print(f"    Exists: {os.path.exists(video_info['path'])}")
            
            # Try to load manually
            try:
                frames = dataset.load_video(video_info['path'])
                if frames is None:
                    print(f"    Manual load: FAILED (returned None)")
                else:
                    print(f"    Manual load: SUCCESS ({len(frames)} frames)")
            except Exception as e:
                print(f"    Manual load: ERROR - {e}")

def test_dataloader():
    """Test with DataLoader to see batch loading."""
    print("\n=== DATALOADER TEST ===")
    
    # Load manifest
    manifest_path = "clean_balanced_manifest.csv"
    df = pd.read_csv(manifest_path)
    
    # Create transforms
    transform_config = {
        'enabled': False,
        'aug_prob': 0.0
    }
    transforms = VideoTransforms(config=transform_config, is_training=False)
    
    # Create dataset
    dataset = LipReaderDataset(
        manifest_df=df.head(100),  # Test with first 100 samples
        clip_len=24,
        img_size=96,
        resize_for_backbone=112,
        augmentation_config=transform_config,
        is_training=False
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,  # Single-threaded for debugging
        pin_memory=False
    )
    
    print(f"Testing DataLoader with {len(dataset)} samples...")
    
    batch_count = 0
    total_dummy_samples = 0
    
    for batch_idx, (videos, labels, metadata_list) in enumerate(dataloader):
        batch_count += 1
        
        # Count dummy samples in this batch
        dummy_count = sum(1 for meta in metadata_list if meta['video_path'] == 'dummy')
        total_dummy_samples += dummy_count
        
        print(f"  Batch {batch_idx}: {videos.shape}, dummy samples: {dummy_count}/{len(videos)}")
        
        if batch_idx >= 5:  # Test first 5 batches
            break
    
    print(f"\nDataLoader Results:")
    print(f"  Batches processed: {batch_count}")
    print(f"  Total dummy samples: {total_dummy_samples}")
    print(f"  Dummy rate: {total_dummy_samples/(batch_count*16)*100:.1f}%")

if __name__ == "__main__":
    test_dataset_loading()
    test_dataloader()
