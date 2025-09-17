#!/usr/bin/env python3
"""
Enhanced Lighting Augmentation Pipeline - Reach exactly 25% increase with robust quality handling
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

class RobustLightingAugmenter:
    """Apply lighting-only augmentations with robust quality preservation."""
    
    def __init__(self):
        # More conservative ranges to ensure quality
        self.brightness_range = (-0.10, 0.10)  # ±10% variation (more conservative)
        self.contrast_range = (0.92, 1.08)     # 0.92-1.08x multiplier (more conservative)
        self.gamma_range = (0.96, 1.04)        # 0.96-1.04x range (more conservative)
    
    def apply_brightness_adjustment(self, frames, factor):
        """Apply brightness adjustment with range preservation."""
        # Convert to [0, 1] range for processing
        frames_01 = (frames + 1) / 2.0
        
        # Apply brightness adjustment
        adjusted = frames_01 + factor
        
        # Ensure we stay within valid range
        adjusted = np.clip(adjusted, 0.05, 0.95)  # Leave margin for contrast/gamma
        
        # Convert back to [-1, 1] range
        return (adjusted * 2.0) - 1.0
    
    def apply_contrast_adjustment(self, frames, factor):
        """Apply contrast adjustment with range preservation."""
        # Convert to [0, 1] range for processing
        frames_01 = (frames + 1) / 2.0
        
        # Apply contrast adjustment around 0.5 (middle gray)
        adjusted = (frames_01 - 0.5) * factor + 0.5
        
        # Ensure we stay within valid range
        adjusted = np.clip(adjusted, 0.05, 0.95)  # Leave margin
        
        # Convert back to [-1, 1] range
        return (adjusted * 2.0) - 1.0
    
    def apply_gamma_correction(self, frames, gamma):
        """Apply gamma correction with range preservation."""
        # Convert to [0, 1] range for processing
        frames_01 = (frames + 1) / 2.0
        
        # Ensure we have positive values for gamma
        frames_01 = np.clip(frames_01, 0.01, 0.99)
        
        # Apply gamma correction
        adjusted = np.power(frames_01, 1.0 / gamma)
        
        # Ensure we stay within valid range
        adjusted = np.clip(adjusted, 0.05, 0.95)  # Leave margin
        
        # Convert back to [-1, 1] range
        return (adjusted * 2.0) - 1.0
    
    def augment_video_safe(self, frames, augmentation_type, max_attempts=5):
        """Apply augmentation with multiple attempts to ensure quality."""
        for attempt in range(max_attempts):
            try:
                if augmentation_type == 'brightness_increase':
                    factor = random.uniform(0.03, self.brightness_range[1])
                    augmented = self.apply_brightness_adjustment(frames, factor)
                    desc = f"brightness_increase_{factor:.3f}"
                
                elif augmentation_type == 'brightness_decrease':
                    factor = random.uniform(self.brightness_range[0], -0.03)
                    augmented = self.apply_brightness_adjustment(frames, factor)
                    desc = f"brightness_decrease_{abs(factor):.3f}"
                
                elif augmentation_type == 'contrast_increase':
                    factor = random.uniform(1.01, self.contrast_range[1])
                    augmented = self.apply_contrast_adjustment(frames, factor)
                    desc = f"contrast_increase_{factor:.3f}"
                
                elif augmentation_type == 'contrast_decrease':
                    factor = random.uniform(self.contrast_range[0], 0.99)
                    augmented = self.apply_contrast_adjustment(frames, factor)
                    desc = f"contrast_decrease_{factor:.3f}"
                
                elif augmentation_type == 'gamma_increase':
                    gamma = random.uniform(1.005, self.gamma_range[1])
                    augmented = self.apply_gamma_correction(frames, gamma)
                    desc = f"gamma_increase_{gamma:.3f}"
                
                elif augmentation_type == 'gamma_decrease':
                    gamma = random.uniform(self.gamma_range[0], 0.995)
                    augmented = self.apply_gamma_correction(frames, gamma)
                    desc = f"gamma_decrease_{gamma:.3f}"
                
                else:
                    raise ValueError(f"Unknown augmentation type: {augmentation_type}")
                
                # Quick quality check
                if (-1.05 <= augmented.min() <= -0.7 and 0.7 <= augmented.max() <= 1.05):
                    return augmented, desc, True
                
            except Exception as e:
                continue
        
        # If all attempts failed, return original with minimal modification
        minimal_factor = random.uniform(-0.02, 0.02)
        safe_augmented = self.apply_brightness_adjustment(frames, minimal_factor)
        return safe_augmented, f"minimal_brightness_{abs(minimal_factor):.3f}", False

def load_original_dataset():
    """Load the original processed dataset."""
    npy_dir = Path("data/training set 17.9.25")
    npy_files = list(npy_dir.glob("*.npy"))
    
    # Filter out already augmented files
    original_files = [f for f in npy_files if '_aug_' not in f.name]
    
    # Organize by class
    videos_by_class = defaultdict(list)
    
    for npy_file in original_files:
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

def calculate_precise_augmentation_targets(videos_by_class, target_increase=0.25):
    """Calculate precise augmentation targets to reach exactly 25% increase."""
    original_total = sum(len(videos) for videos in videos_by_class.values())
    target_total = int(original_total * (1 + target_increase))  # 66 * 1.25 = 82.5 -> 82
    target_augmented = target_total - original_total  # 82 - 66 = 16
    
    print(f"📊 PRECISE AUGMENTATION TARGETS:")
    print(f"   Original total: {original_total} videos")
    print(f"   Target total: {target_total} videos")
    print(f"   Augmented needed: {target_augmented} videos")
    
    # Distribute augmented videos to balance classes better
    augmentation_targets = {}
    remaining_augmented = target_augmented
    
    # Sort classes by size (smallest first) to balance better
    sorted_classes = sorted(videos_by_class.items(), key=lambda x: len(x[1]))
    
    for i, (class_name, videos) in enumerate(sorted_classes):
        if i == len(sorted_classes) - 1:  # Last class gets remaining
            class_target = remaining_augmented
        else:
            # Give smaller classes proportionally more augmentations
            base_target = max(1, target_augmented // len(videos_by_class))
            class_target = min(base_target, remaining_augmented)
        
        augmentation_targets[class_name] = class_target
        remaining_augmented -= class_target
        print(f"   {class_name}: +{class_target} augmented videos")
    
    return augmentation_targets

def npy_to_mp4_ffmpeg(npy_path, output_path):
    """Convert numpy array to proper grayscale MP4 using FFmpeg."""
    try:
        frames = np.load(npy_path)
        frames_uint8 = ((frames + 1) * 127.5).astype(np.uint8)
        frames_uint8 = np.clip(frames_uint8, 0, 255)
        
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
            temp_raw_path = temp_file.name
            temp_file.write(frames_uint8.tobytes())
        
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', '96x96', '-pix_fmt', 'gray', '-r', '8',
            '-i', temp_raw_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-vf', 'format=gray,format=yuv420p', '-loglevel', 'quiet',
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        os.unlink(temp_raw_path)
        return result.returncode == 0
    except:
        return False

def verify_augmented_quality(frames, original_filename):
    """Verify augmented video meets quality standards."""
    if frames.shape != (32, 96, 96):
        return False, f"Wrong shape: {frames.shape}"
    
    if not (-1.1 <= frames.min() <= -0.7 and 0.7 <= frames.max() <= 1.1):
        return False, f"Wrong range: [{frames.min():.3f}, {frames.max():.3f}]"
    
    extreme_low = np.sum(frames < -0.9) / frames.size * 100
    extreme_high = np.sum(frames > 0.9) / frames.size * 100
    total_extreme = extreme_low + extreme_high
    
    if total_extreme > 15.0:  # More lenient threshold
        return False, f"Too many extreme values: {total_extreme:.2f}%"
    
    if np.isnan(frames).any() or np.isinf(frames).any():
        return False, "Contains NaN or Inf values"
    
    return True, "Quality check passed"

def create_enhanced_augmented_dataset():
    """Create augmented dataset with enhanced robustness."""
    print("🔧 ENHANCED LIGHTING AUGMENTATION PIPELINE")
    print("=" * 60)
    
    # Load original dataset (excluding already augmented)
    print("\n📁 Loading original dataset...")
    videos_by_class = load_original_dataset()
    
    if not videos_by_class:
        print("❌ No original videos found!")
        return False
    
    # Calculate precise augmentation targets
    augmentation_targets = calculate_precise_augmentation_targets(videos_by_class)
    
    # Create output directories
    augmented_dir = Path("data/training set 17.9.25")
    preview_dir = Path("data/training set 17.9.25/preview_videos_fixed")
    
    # Initialize robust augmenter
    augmenter = RobustLightingAugmenter()
    
    # Define augmentation types
    augmentation_types = [
        'brightness_increase', 'brightness_decrease',
        'contrast_increase', 'contrast_decrease',
        'gamma_increase', 'gamma_decrease'
    ]
    
    print(f"\n🎨 CREATING ENHANCED AUGMENTED VIDEOS:")
    print(f"   Conservative brightness range: ±10%")
    print(f"   Conservative contrast range: 0.92-1.08x")
    print(f"   Conservative gamma range: 0.96-1.04x")
    print(f"   Multiple attempts per augmentation for quality")
    
    successful_augmentations = 0
    failed_augmentations = 0
    augmentation_log = []
    
    # Process each class
    for class_name, original_videos in videos_by_class.items():
        target_count = augmentation_targets[class_name]
        print(f"\n📹 Processing {class_name} class:")
        print(f"   Original videos: {len(original_videos)}")
        print(f"   Target augmented: {target_count}")
        
        # Create augmented versions with persistence
        created_count = 0
        max_total_attempts = target_count * 10  # Allow many attempts
        
        for attempt in range(max_total_attempts):
            if created_count >= target_count:
                break
            
            # Randomly select video and augmentation type
            selected_video = random.choice(original_videos)
            selected_aug_type = random.choice(augmentation_types)
            
            try:
                # Load original video
                original_frames = np.load(selected_video)
                
                # Apply robust augmentation
                augmented_frames, aug_description, success = augmenter.augment_video_safe(
                    original_frames, selected_aug_type
                )
                
                # Verify quality
                is_valid, quality_msg = verify_augmented_quality(augmented_frames, selected_video.name)
                
                if not is_valid:
                    failed_augmentations += 1
                    continue
                
                # Generate unique augmented filename
                base_name = selected_video.stem.replace('_gentle_v5', '')
                aug_filename = f"{base_name}_aug_{aug_description}_{created_count:02d}.npy"
                aug_path = augmented_dir / aug_filename
                
                # Check if file already exists (avoid duplicates)
                if aug_path.exists():
                    continue
                
                # Save augmented NPY file
                np.save(aug_path, augmented_frames)
                
                # Create MP4 preview
                mp4_filename = f"{base_name}_aug_{aug_description}_{created_count:02d}.mp4"
                mp4_path = preview_dir / mp4_filename
                mp4_success = npy_to_mp4_ffmpeg(aug_path, mp4_path)
                
                # Log augmentation
                augmentation_log.append({
                    'original_file': selected_video.name,
                    'augmented_file': aug_filename,
                    'class': class_name,
                    'augmentation_type': selected_aug_type,
                    'augmentation_description': aug_description,
                    'quality_check': quality_msg,
                    'mp4_created': mp4_success,
                    'robust_success': success,
                    'shape': list(augmented_frames.shape),
                    'range': [float(augmented_frames.min()), float(augmented_frames.max())],
                    'extreme_values_percent': float(np.sum((augmented_frames < -0.9) | (augmented_frames > 0.9)) / augmented_frames.size * 100)
                })
                
                successful_augmentations += 1
                created_count += 1
                
            except Exception as e:
                failed_augmentations += 1
                continue
        
        print(f"   ✅ Created: {created_count}/{target_count} augmented videos")
    
    # Save augmentation log
    log_path = "enhanced_augmentation_log.json"
    with open(log_path, 'w') as f:
        json.dump(augmentation_log, f, indent=2)
    
    print(f"\n✅ ENHANCED AUGMENTATION COMPLETE:")
    print(f"   ✅ Successful augmentations: {successful_augmentations}")
    print(f"   ❌ Failed attempts: {failed_augmentations}")
    print(f"   📊 Success rate: {successful_augmentations/(successful_augmentations+failed_augmentations)*100:.1f}%")
    print(f"   📁 Augmentation log saved: {log_path}")
    
    # Verify final dataset size
    all_npy_files = list(augmented_dir.glob("*.npy"))
    original_files = [f for f in all_npy_files if '_aug_' not in f.name]
    augmented_files = [f for f in all_npy_files if '_aug_' in f.name]
    
    print(f"\n📊 FINAL DATASET STATISTICS:")
    print(f"   Original NPY files: {len(original_files)}")
    print(f"   Augmented NPY files: {len(augmented_files)}")
    print(f"   Total NPY files: {len(all_npy_files)}")
    
    # Calculate actual increase
    if len(original_files) > 0:
        actual_increase = (len(augmented_files) / len(original_files)) * 100
        print(f"   Actual increase: {actual_increase:.1f}%")
        
        target_reached = actual_increase >= 24.0  # Allow small margin
        print(f"   Target (25%) reached: {'✅' if target_reached else '❌'}")
        
        return target_reached
    
    return False

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    success = create_enhanced_augmented_dataset()
    
    if success:
        print(f"\n🎉 ENHANCED AUGMENTATION PIPELINE COMPLETE!")
        print(f"   Dataset successfully expanded by ~25%")
        print(f"   Ready for training with expanded dataset")
    else:
        print(f"\n⚠️  AUGMENTATION PIPELINE COMPLETED WITH ISSUES")
        print(f"   May not have reached full 25% target")
        print(f"   Check logs for details")
