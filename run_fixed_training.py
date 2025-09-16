#!/usr/bin/env python3
"""
Run the fixed training with proper configuration.
"""

import sys
import os
sys.path.append('.')

import yaml
from train import LipReadingTrainer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Run the fixed training."""
    print("🚀 Starting Fixed Staged Training")
    print("=" * 60)
    
    # Load config
    with open('config_staged.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Config loaded:")
    print(f"  Classes: {config['classes']['names']}")
    print(f"  Clip length: {config['data']['clip_len']}")
    print(f"  Image size: {config['data']['img_size']}")
    print(f"  Staged training: {config['training']['staged_training']}")
    
    # Initialize trainer
    trainer = LipReadingTrainer(config=config, output_dir='experiments/run_003_fixed')
    
    # Load data
    print("\n📊 Loading data...")
    train_loader, val_loader, test_loader = trainer.load_data('clean_balanced_manifest.csv')
    
    print(f"✅ Data loaded:")
    print(f"  TRAIN: {len(train_loader.dataset)} samples")
    print(f"  VAL: {len(val_loader.dataset)} samples")
    print(f"  TEST: {len(test_loader.dataset)} samples")
    
    # Setup model and training components
    print("\n🔧 Setting up model and training components...")
    trainer.setup_model_and_training(train_loader)

    # Start training
    print("\n🎯 Starting training...")
    print("Target: >80% generalization accuracy")
    print("=" * 60)

    trainer.train(train_loader, val_loader, test_loader)
    
    print("\n🎉 Training completed!")

if __name__ == "__main__":
    main()
