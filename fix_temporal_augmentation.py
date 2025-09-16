#!/usr/bin/env python3
"""
Fix temporal augmentation issues by removing speed-based augmentations
and replacing them with temporal-preserving alternatives.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import shutil
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalAugmentationFixer:
    """
    Fix temporal augmentation issues by replacing problematic augmentations.
    """
    
    def __init__(self, balanced_manifest_path: str):
        """Initialize the fixer."""
        self.manifest_path = balanced_manifest_path
        self.manifest_df = pd.read_csv(balanced_manifest_path)
        
        # Get augmented videos
        self.augmented_df = self.manifest_df[
            self.manifest_df['source'].str.contains('augmented', na=False)
        ].copy()
        
        logger.info(f"Found {len(self.augmented_df)} augmented videos")
        
        # Safe augmentation types (preserve temporal characteristics)
        self.safe_augmentation_types = [
            'horizontal_flip',
            'brightness_increase',
            'brightness_decrease', 
            'contrast_increase',
            'contrast_decrease',
            'gaussian_noise',
            'slight_rotation'
        ]
        
    def identify_problematic_videos(self) -> List[Dict]:
        """Identify videos with speed-based augmentations."""
        problematic_videos = []
        
        for _, row in self.augmented_df.iterrows():
            filename = Path(row['path']).name
            if 'speed_' in filename:
                problematic_videos.append(row.to_dict())
                
        logger.info(f"Found {len(problematic_videos)} videos with speed augmentation")
        return problematic_videos
        
    def apply_gaussian_noise(self, video_path: str, output_path: str, noise_level: float = 5.0) -> bool:
        """Apply Gaussian noise augmentation (temporal-preserving)."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer with SAME fps
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level, frame.shape).astype(np.int16)
                noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                # Write frame
                out.write(noisy_frame)
                
            cap.release()
            out.release()
            return True
            
        except Exception as e:
            logger.error(f"Error applying Gaussian noise to {video_path}: {e}")
            return False
            
    def apply_slight_rotation(self, video_path: str, output_path: str, angle: float = 2.0) -> bool:
        """Apply slight rotation augmentation (temporal-preserving)."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer with SAME fps
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
            
            # Rotation matrix
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                # Apply rotation
                rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))
                
                # Write frame
                out.write(rotated_frame)
                
            cap.release()
            out.release()
            return True
            
        except Exception as e:
            logger.error(f"Error applying rotation to {video_path}: {e}")
            return False
            
    def replace_problematic_augmentation(self, original_path: str, output_path: str, replacement_type: str) -> bool:
        """Replace a problematic augmentation with a safe one."""
        if replacement_type == 'gaussian_noise':
            return self.apply_gaussian_noise(original_path, output_path)
        elif replacement_type == 'slight_rotation':
            return self.apply_slight_rotation(original_path, output_path)
        elif replacement_type == 'horizontal_flip':
            return self.apply_horizontal_flip(original_path, output_path)
        elif replacement_type == 'brightness_increase':
            return self.apply_brightness_adjustment(original_path, output_path, 25)
        elif replacement_type == 'brightness_decrease':
            return self.apply_brightness_adjustment(original_path, output_path, -25)
        elif replacement_type == 'contrast_increase':
            return self.apply_contrast_adjustment(original_path, output_path, 1.1)
        elif replacement_type == 'contrast_decrease':
            return self.apply_contrast_adjustment(original_path, output_path, 0.9)
        else:
            logger.error(f"Unknown replacement type: {replacement_type}")
            return False
            
    def apply_horizontal_flip(self, video_path: str, output_path: str) -> bool:
        """Apply horizontal flip augmentation."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer with SAME fps
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                # Apply horizontal flip
                flipped_frame = cv2.flip(frame, 1)
                
                # Write frame
                out.write(flipped_frame)
                
            cap.release()
            out.release()
            return True
            
        except Exception as e:
            logger.error(f"Error applying horizontal flip to {video_path}: {e}")
            return False
            
    def apply_brightness_adjustment(self, video_path: str, output_path: str, factor: float) -> bool:
        """Apply brightness adjustment augmentation."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer with SAME fps
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                # Apply brightness adjustment
                adjusted_frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=factor)
                
                # Write frame
                out.write(adjusted_frame)
                
            cap.release()
            out.release()
            return True
            
        except Exception as e:
            logger.error(f"Error applying brightness adjustment to {video_path}: {e}")
            return False
            
    def apply_contrast_adjustment(self, video_path: str, output_path: str, factor: float) -> bool:
        """Apply contrast adjustment augmentation."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer with SAME fps
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                # Apply contrast adjustment
                adjusted_frame = cv2.convertScaleAbs(frame, alpha=factor, beta=0)
                
                # Write frame
                out.write(adjusted_frame)
                
            cap.release()
            out.release()
            return True
            
        except Exception as e:
            logger.error(f"Error applying contrast adjustment to {video_path}: {e}")
            return False
            
    def find_original_video(self, augmented_path: str) -> str:
        """Find the original video for an augmented video."""
        augmented_filename = Path(augmented_path).name
        
        # Extract the base filename (remove augmentation suffix)
        parts = augmented_filename.split('_aug_')
        if len(parts) >= 2:
            base_name = parts[0] + '.mp4'
            
            # Find in original manifest
            original_manifest = pd.read_csv("comprehensive_manifest.csv")
            for _, row in original_manifest.iterrows():
                original_filename = Path(row['path']).name
                if original_filename == base_name:
                    return row['path']
                    
        return None
        
    def fix_temporal_issues(self) -> str:
        """Fix all temporal augmentation issues."""
        logger.info("🔧 Fixing temporal augmentation issues...")
        
        # Identify problematic videos
        problematic_videos = self.identify_problematic_videos()
        
        if not problematic_videos:
            logger.info("✅ No temporal issues found!")
            return self.manifest_path
            
        # Create backup directory
        backup_dir = Path("./augmented_videos_backup")
        backup_dir.mkdir(exist_ok=True)
        
        fixed_entries = []
        replacement_types = ['gaussian_noise', 'slight_rotation']
        
        for i, video_info in enumerate(problematic_videos):
            old_path = video_info['path']
            old_filename = Path(old_path).name
            
            logger.info(f"Fixing {i+1}/{len(problematic_videos)}: {old_filename}")
            
            # Move problematic video to backup
            backup_path = backup_dir / old_filename
            if Path(old_path).exists():
                shutil.move(old_path, backup_path)
                logger.info(f"  Moved to backup: {backup_path}")
            
            # Find original video
            original_path = self.find_original_video(old_path)
            if not original_path:
                logger.error(f"  Could not find original for {old_filename}")
                continue
                
            # Choose replacement augmentation
            replacement_type = replacement_types[i % len(replacement_types)]
            
            # Generate new filename
            base_name = old_filename.split('_aug_')[0]
            new_filename = f"{base_name}_aug_{replacement_type}_{i:03d}.mp4"
            new_path = Path("./augmented_videos") / new_filename
            
            # Apply replacement augmentation
            success = self.replace_problematic_augmentation(
                original_path, str(new_path), replacement_type
            )
            
            if success:
                # Update manifest entry
                video_info['path'] = str(new_path)
                video_info['processed_version'] = f"cropped_aug_{replacement_type}"
                fixed_entries.append(video_info)
                logger.info(f"  ✅ Replaced with {replacement_type}: {new_filename}")
            else:
                logger.error(f"  ❌ Failed to create replacement for {old_filename}")
                
        # Update manifest
        if fixed_entries:
            # Remove problematic entries from manifest
            updated_manifest_df = self.manifest_df[
                ~self.manifest_df['path'].isin([v['path'] for v in problematic_videos])
            ].copy()
            
            # Add fixed entries
            fixed_df = pd.DataFrame(fixed_entries)
            updated_manifest_df = pd.concat([updated_manifest_df, fixed_df], ignore_index=True)
            
            # Save updated manifest
            output_path = "fixed_balanced_comprehensive_manifest.csv"
            updated_manifest_df.to_csv(output_path, index=False)
            
            logger.info(f"✅ Fixed {len(fixed_entries)} problematic videos")
            logger.info(f"📄 Updated manifest saved: {output_path}")
            
            return output_path
        else:
            logger.error("❌ No videos were successfully fixed")
            return self.manifest_path

def main():
    """Main function to fix temporal augmentation issues."""
    logger.info("🚀 Starting Temporal Augmentation Fix")
    
    # Initialize fixer
    fixer = TemporalAugmentationFixer("balanced_comprehensive_manifest.csv")
    
    # Fix temporal issues
    updated_manifest_path = fixer.fix_temporal_issues()
    
    logger.info("✅ Temporal augmentation fix completed!")
    logger.info(f"📊 Updated manifest: {updated_manifest_path}")
    
    return updated_manifest_path

if __name__ == "__main__":
    main()
