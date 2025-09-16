#!/usr/bin/env python3
"""
Resume training script with stabilization improvements
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the resume training command."""
    
    # Check if we have the required files
    manifest_path = "clean_balanced_manifest.csv"
    checkpoint_dir = "experiments/run_001"
    
    if not Path(manifest_path).exists():
        print(f"Error: Manifest file {manifest_path} not found")
        sys.exit(1)
    
    if not Path(checkpoint_dir).exists():
        print(f"Error: Checkpoint directory {checkpoint_dir} not found")
        sys.exit(1)
    
    # Build the command
    cmd = [
        "python", "train.py",
        "--manifest", manifest_path,
        "--resume", checkpoint_dir,  # Will auto-find latest checkpoint
        "--unfreeze_layers", "layer3,layer4",
        "--head_lr", "2e-4",
        "--backbone_lr", "2e-5",
        "--weight_decay", "0.01",
        "--dropout", "0.3",
        "--label_smoothing", "0.05",
        "--clip_len", "32",
        "--ema_beta", "0.999",
        "--early_stop_patience", "6",
        "--balance", "weighted_sampler",
        "--aug", "temporal10,affine2,bc0.1",
        "--save_dir", "experiments/run_001"
    ]
    
    print("Starting training with stabilization improvements...")
    print("Command:", " ".join(cmd))
    print()
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
