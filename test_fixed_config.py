#!/usr/bin/env python3
"""
Test the fixed configuration and training pipeline.
"""

import sys
import os
sys.path.append('.')

import yaml
import pandas as pd
from train import LipReadingTrainer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_fixed_configuration():
    """Test the fixed configuration."""
    print("=== TESTING FIXED CONFIGURATION ===")
    
    # Load config
    with open('config_staged.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Config classes: {config['classes']['names']}")
    print(f"Config clip_len: {config['data']['clip_len']}")
    print(f"Config img_size: {config['data']['img_size']}")
    print(f"Config early_stop_bonus: {config['targets']['early_stop_bonus']}")
    
    # Load manifest and check classes
    df = pd.read_csv('clean_balanced_manifest.csv')
    actual_classes = sorted(df['class'].unique())
    config_classes = sorted(config['classes']['names'])
    
    print(f"\nActual classes in data: {actual_classes}")
    print(f"Config classes: {config_classes}")
    print(f"Classes match: {actual_classes == config_classes}")
    
    if actual_classes != config_classes:
        print("❌ CLASS MISMATCH DETECTED!")
        missing_in_config = set(actual_classes) - set(config_classes)
        extra_in_config = set(config_classes) - set(actual_classes)
        if missing_in_config:
            print(f"  Missing in config: {missing_in_config}")
        if extra_in_config:
            print(f"  Extra in config: {extra_in_config}")
        return False
    else:
        print("✅ Classes match perfectly!")
    
    return True

def test_trainer_initialization():
    """Test trainer initialization with fixed config."""
    print("\n=== TESTING TRAINER INITIALIZATION ===")
    
    try:
        # Load config
        with open('config_staged.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Initialize trainer
        trainer = LipReadingTrainer(
            config=config,
            output_dir='experiments/test_fixed'
        )
        
        print("✅ Trainer initialized successfully")
        
        # Test data loading
        print("Testing data loading...")
        train_loader, val_loader, test_loader = trainer.load_data('clean_balanced_manifest.csv')
        
        print(f"✅ Data loaded successfully:")
        print(f"  TRAIN: {len(train_loader.dataset)} samples")
        print(f"  VAL: {len(val_loader.dataset)} samples") 
        print(f"  TEST: {len(test_loader.dataset)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_batch():
    """Test processing a single batch."""
    print("\n=== TESTING SINGLE BATCH PROCESSING ===")
    
    try:
        # Load config
        with open('config_staged.yaml', 'r') as f:
            config = yaml.safe_load(f)

        trainer = LipReadingTrainer(
            config=config,
            output_dir='experiments/test_fixed'
        )
        
        train_loader, val_loader, test_loader = trainer.load_data('clean_balanced_manifest.csv')
        
        # Test train batch
        print("Processing train batch...")
        train_batch = next(iter(train_loader))
        videos, labels, metadata_list = train_batch
        
        print(f"  Batch shape: {videos.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Metadata count: {len(metadata_list)}")
        
        # Check for dummy samples
        dummy_count = sum(1 for meta in metadata_list if meta.get('video_path') == 'dummy')
        print(f"  Dummy samples: {dummy_count}/{len(metadata_list)} ({dummy_count/len(metadata_list)*100:.1f}%)")
        
        if dummy_count == 0:
            print("✅ No dummy samples detected")
        else:
            print(f"⚠️  {dummy_count} dummy samples detected")
        
        # Test val batch
        print("Processing val batch...")
        val_batch = next(iter(val_loader))
        videos, labels, metadata_list = val_batch
        
        dummy_count = sum(1 for meta in metadata_list if meta.get('video_path') == 'dummy')
        print(f"  Val dummy samples: {dummy_count}/{len(metadata_list)} ({dummy_count/len(metadata_list)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = True
    
    success &= test_fixed_configuration()
    success &= test_trainer_initialization()
    success &= test_single_batch()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - Configuration is fixed!")
    else:
        print("\n❌ SOME TESTS FAILED - Issues remain")
    
    print("\nReady to restart training with fixed configuration.")
