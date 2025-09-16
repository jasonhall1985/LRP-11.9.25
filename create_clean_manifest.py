#!/usr/bin/env python3
"""
Create a clean manifest that only includes videos that actually exist
and removes any references to problematic speed augmentations.
"""

import pandas as pd
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_clean_manifest():
    """Create a clean manifest with only existing videos."""
    logger.info("🧹 Creating clean manifest...")
    
    # Load the fixed manifest
    df = pd.read_csv("fixed_balanced_comprehensive_manifest.csv")
    logger.info(f"Loaded manifest with {len(df)} entries")
    
    # Filter out videos that don't exist
    existing_videos = []
    missing_videos = []
    
    for _, row in df.iterrows():
        video_path = row['path']
        if os.path.exists(video_path):
            existing_videos.append(row.to_dict())
        else:
            missing_videos.append(video_path)
            
    logger.info(f"Found {len(existing_videos)} existing videos")
    logger.info(f"Found {len(missing_videos)} missing videos")
    
    # Create clean dataframe
    clean_df = pd.DataFrame(existing_videos)
    
    # Remove any remaining speed augmentations
    speed_mask = clean_df['processed_version'].str.contains('speed', na=False)
    speed_videos = clean_df[speed_mask]
    
    if len(speed_videos) > 0:
        logger.warning(f"Removing {len(speed_videos)} remaining speed-augmented videos")
        clean_df = clean_df[~speed_mask]
    
    # Check class distribution
    class_counts = clean_df['class'].value_counts().sort_index()
    logger.info("Class distribution after cleaning:")
    for class_name, count in class_counts.items():
        logger.info(f"  {class_name}: {count} videos")
        
    # Check augmentation types
    augmented_df = clean_df[clean_df['source'].str.contains('augmented', na=False)]
    logger.info(f"Total augmented videos: {len(augmented_df)}")
    
    aug_types = {}
    for _, row in augmented_df.iterrows():
        processed_version = row['processed_version']
        if 'aug_' in processed_version:
            aug_type = processed_version.split('aug_')[1]
            aug_types[aug_type] = aug_types.get(aug_type, 0) + 1
            
    logger.info("Augmentation type distribution:")
    for aug_type, count in sorted(aug_types.items()):
        logger.info(f"  {aug_type}: {count} videos")
        
    # Save clean manifest
    output_path = "clean_balanced_manifest.csv"
    clean_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Clean manifest saved: {output_path}")
    logger.info(f"📊 Total videos in clean manifest: {len(clean_df)}")
    
    return output_path

if __name__ == "__main__":
    create_clean_manifest()
