#!/usr/bin/env python3
"""
Lighting Augmentation Pipeline - Expand dataset by 25% using only lighting variations
"""

import numpy as np
import cv2
import torch
import subprocess
import tempfile
import os
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm
import json

class LightingAugmenter:
    """Apply lighting-only augmentations to video frames."""
    
    def __init__(self):
        self.brightness_range = (-0.15, 0.15)  # ±15% variation
        self.contrast_range = (0.9, 1.1)      # 0.9-1.1x multiplier
        self.gamma_range = (0.95, 1.05)       # 0.95-1.05x range
    
    def apply_brightness_adjustment(self, frames, factor):
        """Apply brightness adjustment to frames."""
        # Convert to [0, 1] range for processing
        frames_01 = (frames + 1) / 2.0
        
        # Apply brightness adjustment
        adjusted = np.clip(frames_01 + factor, 0, 1)
        
        # Convert back to [-1, 1] range
        return (adjusted * 2.0) - 1.0
    
    def apply_contrast_adjustment(self, frames, factor):
        """Apply contrast adjustment to frames."""
        # Convert to [0, 1] range for processing
        frames_01 = (frames + 1) / 2.0
        
        # Apply contrast adjustment around 0.5 (middle gray)
        adjusted = np.clip((frames_01 - 0.5) * factor + 0.5, 0, 1)
        
        # Convert back to [-1, 1] range
        return (adjusted * 2.0) - 1.0
    
    def apply_gamma_correction(self, frames, gamma):
        """Apply gamma correction to frames."""
        # Convert to [0, 1] range for processing
        frames_01 = (frames + 1) / 2.0
        
        # Apply gamma correction
        adjusted = np.power(frames_01, 1.0 / gamma)
        adjusted = np.clip(adjusted, 0, 1)
        
        # Convert back to [-1, 1] range
        return (adjusted * 2.0) - 1.0
    
    def augment_video(self, frames, augmentation_type):
        """Apply specific lighting augmentation to video frames."""
        if augmentation_type == 'brightness_increase':
            factor = random.uniform(0.05, self.brightness_range[1])
            return self.apply_brightness_adjustment(frames, factor), f"brightness_increase_{factor:.3f}"
        
        elif augmentation_type == 'brightness_decrease':
            factor = random.uniform(self.brightness_range[0], -0.05)
            return self.apply_brightness_adjustment(frames, factor), f"brightness_decrease_{abs(factor):.3f}"
        
        elif augmentation_type == 'contrast_increase':
            factor = random.uniform(1.02, self.contrast_range[1])
            return self.apply_contrast_adjustment(frames, factor), f"contrast_increase_{factor:.3f}"
        
        elif augmentation_type == 'contrast_decrease':
            factor = random.uniform(self.contrast_range[0], 0.98)
            return self.apply_contrast_adjustment(frames, factor), f"contrast_decrease_{factor:.3f}"
        
        elif augmentation_type == 'gamma_increase':
            gamma = random.uniform(1.01, self.gamma_range[1])
            return self.apply_gamma_correction(frames, gamma), f"gamma_increase_{gamma:.3f}"
        
        elif augmentation_type == 'gamma_decrease':
            gamma = random.uniform(self.gamma_range[0], 0.99)
            return self.apply_gamma_correction(frames, gamma), f"gamma_decrease_{gamma:.3f}"
        
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")

def load_original_dataset():
    """Load the original processed dataset."""
    npy_dir = Path("data/training set 17.9.25")
    npy_files = list(npy_dir.glob("*.npy"))
    
    # Organize by class
    videos_by_class = defaultdict(list)
    
    for npy_file in npy_files:
        filename = npy_file.name.lower()
        if 'doctor' in filename:
            class_name = 'doctor'
        elif 'glasses' in filename:
            class_name = 'glasses'
        elif 'help' in filename:
            class_name = 'help'
        elif 'phone' in filename:
            class_name = 'phone'
        elif 'pillow' in filename:
            class_name = 'pillow'
        else:
            continue
        
        videos_by_class[class_name].append(npy_file)
    
    return videos_by_class

def calculate_augmentation_targets(videos_by_class, target_increase=0.25):
    """Calculate how many augmented videos to create per class."""
    total_videos = sum(len(videos) for videos in videos_by_class.values())
    target_total = int(total_videos * (1 + target_increase))
    target_augmented = target_total - total_videos
    
    print(f"📊 AUGMENTATION TARGETS:")
    print(f"   Current total: {total_videos} videos")
    print(f"   Target total: {target_total} videos")
    print(f"   Augmented needed: {target_augmented} videos")
    
    # Distribute augmented videos proportionally by class
    augmentation_targets = {}
    for class_name, videos in videos_by_class.items():
        class_proportion = len(videos) / total_videos
        class_target = max(1, int(target_augmented * class_proportion))
        augmentation_targets[class_name] = class_target
        print(f"   {class_name}: +{class_target} augmented videos")
    
    return augmentation_targets

def npy_to_mp4_ffmpeg(npy_path, output_path):
    """Convert numpy array to proper grayscale MP4 using FFmpeg."""
    try:
        # Load the numpy array
        frames = np.load(npy_path)
        
        # Convert from [-1, 1] back to [0, 255]
        frames_uint8 = ((frames + 1) * 127.5).astype(np.uint8)
        frames_uint8 = np.clip(frames_uint8, 0, 255)
        
        # Create temporary raw video file
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
            temp_raw_path = temp_file.name
            temp_file.write(frames_uint8.tobytes())
        
        # Use FFmpeg to convert raw grayscale to proper MP4
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '96x96',  # size
            '-pix_fmt', 'gray',  # grayscale pixel format
            '-r', '8',  # frame rate
            '-i', temp_raw_path,  # input
            '-c:v', 'libx264',  # H.264 codec
            '-pix_fmt', 'yuv420p',  # compatible pixel format
            '-vf', 'format=gray,format=yuv420p',  # ensure grayscale
            '-loglevel', 'quiet',  # suppress FFmpeg output
            str(output_path)  # output
        ]
        
        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temp file
        os.unlink(temp_raw_path)
        
        return result.returncode == 0
            
    except Exception as e:
        return False

def verify_augmented_quality(frames, original_filename):
    """Verify augmented video meets quality standards."""
    # Check shape
    if frames.shape != (32, 96, 96):
        return False, f"Wrong shape: {frames.shape}"
    
    # Check range
    if not (-1.1 <= frames.min() <= -0.8 and 0.8 <= frames.max() <= 1.1):
        return False, f"Wrong range: [{frames.min():.3f}, {frames.max():.3f}]"
    
    # Check for extreme values (should be < 10%)
    extreme_low = np.sum(frames < -0.9) / frames.size * 100
    extreme_high = np.sum(frames > 0.9) / frames.size * 100
    total_extreme = extreme_low + extreme_high
    
    if total_extreme > 10.0:
        return False, f"Too many extreme values: {total_extreme:.2f}%"
    
    # Check for NaN or Inf
    if np.isnan(frames).any() or np.isinf(frames).any():
        return False, "Contains NaN or Inf values"
    
    return True, "Quality check passed"

def create_augmented_dataset():
    """Create augmented dataset with lighting variations only."""
    print("🔧 LIGHTING AUGMENTATION PIPELINE")
    print("=" * 50)
    
    # Load original dataset
    print("\n📁 Loading original dataset...")
    videos_by_class = load_original_dataset()
    
    if not videos_by_class:
        print("❌ No original videos found!")
        return
    
    # Calculate augmentation targets
    augmentation_targets = calculate_augmentation_targets(videos_by_class)
    
    # Create output directories
    augmented_dir = Path("data/training set 17.9.25")  # Same directory as originals
    preview_dir = Path("data/training set 17.9.25/preview_videos_fixed")
    
    # Initialize augmenter
    augmenter = LightingAugmenter()
    
    # Define augmentation types
    augmentation_types = [
        'brightness_increase', 'brightness_decrease',
        'contrast_increase', 'contrast_decrease',
        'gamma_increase', 'gamma_decrease'
    ]
    
    print(f"\n🎨 CREATING AUGMENTED VIDEOS:")
    print(f"   Augmentation types: {len(augmentation_types)}")
    print(f"   Brightness range: ±10-15%")
    print(f"   Contrast range: 0.9-1.1x")
    print(f"   Gamma range: 0.95-1.05x")
    
    successful_augmentations = 0
    failed_augmentations = 0
    augmentation_log = []
    
    # Process each class
    for class_name, original_videos in videos_by_class.items():
        target_count = augmentation_targets[class_name]
        print(f"\n📹 Processing {class_name} class:")
        print(f"   Original videos: {len(original_videos)}")
        print(f"   Target augmented: {target_count}")
        
        # Randomly select videos to augment (with replacement if needed)
        videos_to_augment = []
        for _ in range(target_count):
            selected_video = random.choice(original_videos)
            selected_aug_type = random.choice(augmentation_types)
            videos_to_augment.append((selected_video, selected_aug_type))
        
        # Create augmented versions
        for i, (original_path, aug_type) in enumerate(tqdm(videos_to_augment, desc=f"Augmenting {class_name}")):
            try:
                # Load original video
                original_frames = np.load(original_path)
                
                # Apply augmentation
                augmented_frames, aug_description = augmenter.augment_video(original_frames, aug_type)
                
                # Verify quality
                is_valid, quality_msg = verify_augmented_quality(augmented_frames, original_path.name)
                
                if not is_valid:
                    print(f"❌ Quality check failed for {original_path.name}: {quality_msg}")
                    failed_augmentations += 1
                    continue
                
                # Generate augmented filename
                base_name = original_path.stem.replace('_gentle_v5', '')
                aug_filename = f"{base_name}_aug_{aug_description}.npy"
                aug_path = augmented_dir / aug_filename
                
                # Save augmented NPY file
                np.save(aug_path, augmented_frames)
                
                # Create MP4 preview
                mp4_filename = f"{base_name}_aug_{aug_description}.mp4"
                mp4_path = preview_dir / mp4_filename
                
                mp4_success = npy_to_mp4_ffmpeg(aug_path, mp4_path)
                
                # Log augmentation
                augmentation_log.append({
                    'original_file': original_path.name,
                    'augmented_file': aug_filename,
                    'class': class_name,
                    'augmentation_type': aug_type,
                    'augmentation_description': aug_description,
                    'quality_check': quality_msg,
                    'mp4_created': mp4_success,
                    'shape': list(augmented_frames.shape),
                    'range': [float(augmented_frames.min()), float(augmented_frames.max())],
                    'extreme_values_percent': float(np.sum((augmented_frames < -0.9) | (augmented_frames > 0.9)) / augmented_frames.size * 100)
                })
                
                successful_augmentations += 1
                
            except Exception as e:
                print(f"❌ Error augmenting {original_path.name}: {e}")
                failed_augmentations += 1
    
    # Save augmentation log
    log_path = "augmentation_log.json"
    with open(log_path, 'w') as f:
        json.dump(augmentation_log, f, indent=2)
    
    print(f"\n✅ AUGMENTATION COMPLETE:")
    print(f"   ✅ Successful augmentations: {successful_augmentations}")
    print(f"   ❌ Failed augmentations: {failed_augmentations}")
    print(f"   📊 Success rate: {successful_augmentations/(successful_augmentations+failed_augmentations)*100:.1f}%")
    print(f"   📁 Augmentation log saved: {log_path}")
    
    # Verify final dataset size
    final_npy_files = list(augmented_dir.glob("*.npy"))
    final_mp4_files = list(preview_dir.glob("*.mp4"))
    
    print(f"\n📊 FINAL DATASET STATISTICS:")
    print(f"   Total NPY files: {len(final_npy_files)}")
    print(f"   Total MP4 previews: {len(final_mp4_files)}")
    
    # Count by class
    final_by_class = defaultdict(int)
    for npy_file in final_npy_files:
        filename = npy_file.name.lower()
        if 'doctor' in filename:
            final_by_class['doctor'] += 1
        elif 'glasses' in filename:
            final_by_class['glasses'] += 1
        elif 'help' in filename:
            final_by_class['help'] += 1
        elif 'phone' in filename:
            final_by_class['phone'] += 1
        elif 'pillow' in filename:
            final_by_class['pillow'] += 1
    
    print(f"\n🏷️  FINAL CLASS DISTRIBUTION:")
    total_final = 0
    for class_name, count in sorted(final_by_class.items()):
        original_count = len(videos_by_class[class_name])
        augmented_count = count - original_count
        print(f"   {class_name}: {count} total ({original_count} original + {augmented_count} augmented)")
        total_final += count
    
    original_total = sum(len(videos) for videos in videos_by_class.values())
    increase_percent = ((total_final - original_total) / original_total) * 100
    print(f"   Total: {total_final} videos ({increase_percent:.1f}% increase)")
    
    return successful_augmentations > 0

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    success = create_augmented_dataset()
    
    if success:
        print(f"\n🎉 AUGMENTATION PIPELINE COMPLETE!")
        print(f"   Ready for training with expanded dataset")
        print(f"   Use improved_lip_reading_trainer.py for training")
    else:
        print(f"\n❌ AUGMENTATION PIPELINE FAILED!")
        print(f"   Check error messages above")
