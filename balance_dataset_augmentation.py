#!/usr/bin/env python3
"""
Dataset class balancing through video augmentation.
Applies minimal augmentations to preserve lip-reading quality while balancing classes.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import random
from typing import Dict, List, Tuple
import shutil
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetBalancer:
    """
    Balance dataset classes through minimal video augmentation.
    """
    
    def __init__(self, manifest_path: str, output_dir: str = "./augmented_videos"):
        """
        Initialize the dataset balancer.
        
        Args:
            manifest_path: Path to the comprehensive manifest CSV
            output_dir: Directory to store augmented videos
        """
        self.manifest_path = manifest_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load manifest
        self.manifest_df = pd.read_csv(manifest_path)
        logger.info(f"Loaded manifest with {len(self.manifest_df)} videos")
        
        # Augmentation types
        self.augmentation_types = [
            'horizontal_flip',
            'brightness_increase',
            'brightness_decrease', 
            'contrast_increase',
            'contrast_decrease',
            'speed_increase',
            'speed_decrease'
        ]
        
        # Track augmented videos
        self.augmented_videos = []
        
    def get_class_counts(self) -> Dict[str, int]:
        """Get current class counts from manifest."""
        return self.manifest_df['class'].value_counts().to_dict()
        
    def calculate_augmentation_needs(self) -> Dict[str, int]:
        """Calculate how many videos each class needs for balancing."""
        class_counts = self.get_class_counts()
        target_count = max(class_counts.values())  # Use largest class as target
        
        augmentation_needs = {}
        for class_name, current_count in class_counts.items():
            needed = target_count - current_count
            augmentation_needs[class_name] = max(0, needed)
            
        logger.info(f"Target count per class: {target_count}")
        logger.info(f"Augmentation needs: {augmentation_needs}")
        
        return augmentation_needs, target_count
        
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
            
            # Create video writer
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
            
            # Create video writer
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
            
            # Create video writer
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
            
    def apply_speed_adjustment(self, video_path: str, output_path: str, factor: float) -> bool:
        """Apply speed adjustment augmentation."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
                
            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            new_fps = original_fps * factor
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer with adjusted FPS
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height), isColor=False)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                # Write frame (speed change is handled by FPS adjustment)
                out.write(frame)
                
            cap.release()
            out.release()
            return True
            
        except Exception as e:
            logger.error(f"Error applying speed adjustment to {video_path}: {e}")
            return False
            
    def apply_augmentation(self, video_path: str, augmentation_type: str, output_path: str) -> bool:
        """Apply specified augmentation to video."""
        if augmentation_type == 'horizontal_flip':
            return self.apply_horizontal_flip(video_path, output_path)
        elif augmentation_type == 'brightness_increase':
            return self.apply_brightness_adjustment(video_path, output_path, 25)  # +25 brightness
        elif augmentation_type == 'brightness_decrease':
            return self.apply_brightness_adjustment(video_path, output_path, -25)  # -25 brightness
        elif augmentation_type == 'contrast_increase':
            return self.apply_contrast_adjustment(video_path, output_path, 1.1)  # 1.1x contrast
        elif augmentation_type == 'contrast_decrease':
            return self.apply_contrast_adjustment(video_path, output_path, 0.9)  # 0.9x contrast
        elif augmentation_type == 'speed_increase':
            return self.apply_speed_adjustment(video_path, output_path, 1.05)  # 1.05x speed
        elif augmentation_type == 'speed_decrease':
            return self.apply_speed_adjustment(video_path, output_path, 0.95)  # 0.95x speed
        else:
            logger.error(f"Unknown augmentation type: {augmentation_type}")
            return False

    def generate_augmented_filename(self, original_path: str, augmentation_type: str, index: int) -> str:
        """Generate filename for augmented video."""
        original_path = Path(original_path)
        stem = original_path.stem
        suffix = original_path.suffix

        # Add augmentation info to filename
        augmented_name = f"{stem}_aug_{augmentation_type}_{index:03d}{suffix}"
        return str(self.output_dir / augmented_name)

    def balance_class(self, class_name: str, needed_count: int) -> List[Dict]:
        """Balance a specific class by creating augmented videos."""
        logger.info(f"Balancing class '{class_name}': need {needed_count} additional videos")

        # Get videos for this class
        class_videos = self.manifest_df[self.manifest_df['class'] == class_name].copy()

        if len(class_videos) == 0:
            logger.warning(f"No videos found for class {class_name}")
            return []

        augmented_entries = []
        created_count = 0

        # Create augmented videos
        for i in range(needed_count):
            # Select source video (cycle through available videos)
            source_idx = i % len(class_videos)
            source_video = class_videos.iloc[source_idx]

            # Select augmentation type (cycle through types)
            aug_type = self.augmentation_types[i % len(self.augmentation_types)]

            # Generate output path
            output_path = self.generate_augmented_filename(
                source_video['path'], aug_type, i
            )

            # Apply augmentation
            success = self.apply_augmentation(
                source_video['path'], aug_type, output_path
            )

            if success:
                # Create manifest entry for augmented video
                augmented_entry = source_video.to_dict()
                augmented_entry['path'] = output_path
                augmented_entry['source'] = f"{source_video['source']}_augmented"
                augmented_entry['processed_version'] = f"cropped_aug_{aug_type}"

                augmented_entries.append(augmented_entry)
                created_count += 1

                if created_count % 10 == 0:
                    logger.info(f"Created {created_count}/{needed_count} augmented videos for {class_name}")
            else:
                logger.warning(f"Failed to create augmented video: {output_path}")

        logger.info(f"Successfully created {created_count} augmented videos for class '{class_name}'")
        return augmented_entries

    def balance_dataset(self) -> str:
        """Balance the entire dataset and create updated manifest."""
        logger.info("Starting dataset balancing...")

        # Calculate augmentation needs
        augmentation_needs, target_count = self.calculate_augmentation_needs()

        # Track all augmented videos
        all_augmented_entries = []

        # Balance each class that needs augmentation
        for class_name, needed_count in augmentation_needs.items():
            if needed_count > 0:
                augmented_entries = self.balance_class(class_name, needed_count)
                all_augmented_entries.extend(augmented_entries)
                self.augmented_videos.extend(augmented_entries)

        # Create updated manifest
        updated_manifest_df = pd.concat([
            self.manifest_df,
            pd.DataFrame(all_augmented_entries)
        ], ignore_index=True)

        # Save updated manifest
        output_manifest_path = "balanced_comprehensive_manifest.csv"
        updated_manifest_df.to_csv(output_manifest_path, index=False)

        # Log final statistics
        final_class_counts = updated_manifest_df['class'].value_counts().to_dict()
        logger.info("=== FINAL DATASET STATISTICS ===")
        logger.info(f"Total videos: {len(updated_manifest_df)}")
        logger.info(f"Class distribution: {final_class_counts}")
        logger.info(f"Augmented videos created: {len(all_augmented_entries)}")
        logger.info(f"Updated manifest saved: {output_manifest_path}")

        return output_manifest_path

    def get_augmented_samples(self, num_samples: int = 20) -> List[Dict]:
        """Get random samples from augmented videos only."""
        if not self.augmented_videos:
            logger.warning("No augmented videos available for sampling")
            return []

        # Ensure we have samples from all augmented classes
        augmented_df = pd.DataFrame(self.augmented_videos)
        samples = []

        # Get samples from each augmented class
        augmented_classes = augmented_df['class'].unique()
        samples_per_class = max(1, num_samples // len(augmented_classes))

        for class_name in augmented_classes:
            class_augmented = augmented_df[augmented_df['class'] == class_name]
            class_samples = class_augmented.sample(
                n=min(samples_per_class, len(class_augmented))
            ).to_dict('records')
            samples.extend(class_samples)

        # If we need more samples, randomly select from remaining
        if len(samples) < num_samples:
            remaining_needed = num_samples - len(samples)
            remaining_videos = [v for v in self.augmented_videos if v not in samples]
            if remaining_videos:
                additional_samples = random.sample(
                    remaining_videos,
                    min(remaining_needed, len(remaining_videos))
                )
                samples.extend(additional_samples)

        return samples[:num_samples]

def main():
    """Main function to balance dataset."""
    logger.info("🚀 Starting Dataset Class Balancing")

    # Initialize balancer
    balancer = DatasetBalancer("comprehensive_manifest.csv")

    # Balance dataset
    updated_manifest_path = balancer.balance_dataset()

    logger.info("✅ Dataset balancing completed successfully!")
    logger.info(f"📊 Updated manifest: {updated_manifest_path}")

    return updated_manifest_path, balancer

if __name__ == "__main__":
    main()
