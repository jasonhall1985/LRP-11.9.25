#!/usr/bin/env python3
"""
Check video durations to ensure temporal length preservation during augmentation.
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_video_properties(video_path: str) -> Dict:
    """Get comprehensive video properties including duration."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate duration
        duration_seconds = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration_seconds': duration_seconds,
            'duration_ms': duration_seconds * 1000
        }
        
    except Exception as e:
        logger.error(f"Error getting properties for {video_path}: {e}")
        return None

def find_original_video(augmented_path: str, original_manifest_df: pd.DataFrame) -> str:
    """Find the original video that corresponds to an augmented video."""
    augmented_filename = Path(augmented_path).name
    
    # Extract the base filename (remove augmentation suffix)
    # Format: original_name_aug_type_number.mp4
    parts = augmented_filename.split('_aug_')
    if len(parts) >= 2:
        base_name = parts[0] + '.mp4'  # Reconstruct original filename
        
        # Find matching original video
        for _, row in original_manifest_df.iterrows():
            original_filename = Path(row['path']).name
            if original_filename == base_name:
                return row['path']
                
    return None

def check_temporal_preservation():
    """Check if augmented videos preserve temporal characteristics."""
    logger.info("🔍 Checking temporal preservation in augmented videos...")
    
    # Load manifests
    original_manifest = pd.read_csv("comprehensive_manifest.csv")
    balanced_manifest = pd.read_csv("clean_balanced_manifest.csv")
    
    # Get augmented videos
    augmented_videos = balanced_manifest[
        balanced_manifest['source'].str.contains('augmented', na=False)
    ].copy()
    
    logger.info(f"Found {len(augmented_videos)} augmented videos to check")
    
    # Sample videos for detailed checking
    sample_size = min(20, len(augmented_videos))
    sampled_videos = augmented_videos.sample(n=sample_size)
    
    temporal_issues = []
    duration_comparisons = []
    
    logger.info(f"Checking {sample_size} sampled augmented videos...")
    
    for idx, row in sampled_videos.iterrows():
        augmented_path = row['path']
        augmented_filename = Path(augmented_path).name
        
        logger.info(f"Checking: {augmented_filename}")
        
        # Get augmented video properties
        aug_props = get_video_properties(augmented_path)
        if not aug_props:
            temporal_issues.append(f"Could not read augmented video: {augmented_filename}")
            continue
            
        # Find corresponding original video
        original_path = find_original_video(augmented_path, original_manifest)
        if not original_path:
            temporal_issues.append(f"Could not find original for: {augmented_filename}")
            continue
            
        # Get original video properties
        orig_props = get_video_properties(original_path)
        if not orig_props:
            temporal_issues.append(f"Could not read original video: {Path(original_path).name}")
            continue
            
        # Extract augmentation type
        aug_type = "unknown"
        if "_aug_" in augmented_filename:
            parts = augmented_filename.split("_aug_")
            if len(parts) > 1:
                aug_type = parts[1].split("_")[0]
        
        # Compare durations
        duration_diff = abs(aug_props['duration_seconds'] - orig_props['duration_seconds'])
        duration_diff_percent = (duration_diff / orig_props['duration_seconds']) * 100 if orig_props['duration_seconds'] > 0 else 0
        
        comparison = {
            'augmented_file': augmented_filename,
            'original_file': Path(original_path).name,
            'augmentation_type': aug_type,
            'original_duration': orig_props['duration_seconds'],
            'augmented_duration': aug_props['duration_seconds'],
            'duration_diff_seconds': duration_diff,
            'duration_diff_percent': duration_diff_percent,
            'original_fps': orig_props['fps'],
            'augmented_fps': aug_props['fps'],
            'original_frames': orig_props['frame_count'],
            'augmented_frames': aug_props['frame_count'],
            'fps_changed': abs(orig_props['fps'] - aug_props['fps']) > 0.1,
            'frame_count_changed': orig_props['frame_count'] != aug_props['frame_count']
        }
        
        duration_comparisons.append(comparison)
        
        # Check for significant temporal changes
        if duration_diff_percent > 5:  # More than 5% difference
            temporal_issues.append(
                f"{augmented_filename} ({aug_type}): Duration changed by {duration_diff_percent:.1f}% "
                f"({orig_props['duration_seconds']:.2f}s → {aug_props['duration_seconds']:.2f}s)"
            )
            
        if comparison['fps_changed']:
            temporal_issues.append(
                f"{augmented_filename} ({aug_type}): FPS changed "
                f"({orig_props['fps']:.2f} → {aug_props['fps']:.2f})"
            )
            
        if comparison['frame_count_changed']:
            temporal_issues.append(
                f"{augmented_filename} ({aug_type}): Frame count changed "
                f"({orig_props['frame_count']} → {aug_props['frame_count']})"
            )
    
    # Generate report
    logger.info("=" * 60)
    logger.info("🎬 TEMPORAL PRESERVATION ANALYSIS RESULTS")
    logger.info("=" * 60)
    
    if not temporal_issues:
        logger.info("✅ NO TEMPORAL ISSUES FOUND!")
        logger.info("All augmented videos preserve their original temporal characteristics.")
    else:
        logger.warning(f"⚠️  FOUND {len(temporal_issues)} TEMPORAL ISSUES:")
        for issue in temporal_issues:
            logger.warning(f"  - {issue}")
    
    # Summary statistics
    if duration_comparisons:
        duration_diffs = [comp['duration_diff_percent'] for comp in duration_comparisons]
        avg_duration_diff = np.mean(duration_diffs)
        max_duration_diff = np.max(duration_diffs)
        
        logger.info(f"\n📊 DURATION STATISTICS:")
        logger.info(f"  Average duration difference: {avg_duration_diff:.2f}%")
        logger.info(f"  Maximum duration difference: {max_duration_diff:.2f}%")
        logger.info(f"  Videos with FPS changes: {sum(1 for c in duration_comparisons if c['fps_changed'])}")
        logger.info(f"  Videos with frame count changes: {sum(1 for c in duration_comparisons if c['frame_count_changed'])}")
        
        # Group by augmentation type
        aug_type_stats = {}
        for comp in duration_comparisons:
            aug_type = comp['augmentation_type']
            if aug_type not in aug_type_stats:
                aug_type_stats[aug_type] = []
            aug_type_stats[aug_type].append(comp['duration_diff_percent'])
            
        logger.info(f"\n📈 BY AUGMENTATION TYPE:")
        for aug_type, diffs in aug_type_stats.items():
            avg_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            logger.info(f"  {aug_type}: avg {avg_diff:.2f}%, max {max_diff:.2f}%")
    
    # Save detailed report
    if duration_comparisons:
        df_report = pd.DataFrame(duration_comparisons)
        df_report.to_csv("temporal_preservation_report.csv", index=False)
        logger.info(f"\n📄 Detailed report saved: temporal_preservation_report.csv")
    
    return len(temporal_issues) == 0, temporal_issues, duration_comparisons

def check_specific_augmentation_types():
    """Check specific augmentation types that might affect temporal characteristics."""
    logger.info("\n🔍 Checking specific augmentation types for temporal effects...")

    balanced_manifest = pd.read_csv("clean_balanced_manifest.csv")
    augmented_videos = balanced_manifest[
        balanced_manifest['source'].str.contains('augmented', na=False)
    ].copy()
    
    # Group by augmentation type
    aug_type_counts = {}
    for _, row in augmented_videos.iterrows():
        filename = Path(row['path']).name
        if "_aug_" in filename:
            parts = filename.split("_aug_")
            if len(parts) > 1:
                aug_type = parts[1].split("_")[0]
                aug_type_counts[aug_type] = aug_type_counts.get(aug_type, 0) + 1
    
    logger.info("Augmentation type distribution:")
    for aug_type, count in sorted(aug_type_counts.items()):
        logger.info(f"  {aug_type}: {count} videos")
        
        # Check if this type should affect temporal characteristics
        if aug_type in ['speed_increase', 'speed_decrease']:
            logger.warning(f"  ⚠️  {aug_type} SHOULD affect temporal characteristics!")
        else:
            logger.info(f"  ✅ {aug_type} should NOT affect temporal characteristics")

if __name__ == "__main__":
    # Check augmentation type distribution first
    check_specific_augmentation_types()
    
    # Then check temporal preservation
    no_issues, issues, comparisons = check_temporal_preservation()
    
    if no_issues:
        logger.info("\n🎉 SUCCESS: All augmented videos preserve temporal characteristics!")
    else:
        logger.error(f"\n❌ ISSUES FOUND: {len(issues)} temporal preservation problems detected")
        logger.error("Please review the issues above and consider re-running augmentation")
