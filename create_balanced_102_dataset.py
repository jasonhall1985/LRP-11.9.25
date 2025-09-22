#!/usr/bin/env python3
"""
Balanced 102-Video-Per-Class Dataset Creation
Create perfectly balanced dataset with exactly 102 videos per class
from available videos in data/the_best_videos_so_far/
"""

import os
import glob
import pandas as pd
import numpy as np
import random
from datetime import datetime
from collections import defaultdict
import re

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def extract_video_metadata(filename):
    """Extract class and demographic information from video filename"""
    filename = os.path.basename(filename)
    
    # Skip augmented videos
    if 'augmented' in filename:
        return None
    
    # Pattern: class__useruser01__age__gender__ethnicity__timestamp_topmid_96x64_processed.mp4
    if '__' in filename and '_processed' in filename:
        parts = filename.split('__')
        if len(parts) >= 5:
            class_name = parts[0]
            age_group = parts[2]
            gender = parts[3]
            ethnicity = parts[4].split('_')[0]  # Remove timestamp part
            
            return {
                'filename': filename,
                'class': class_name,
                'age_group': age_group,
                'gender': gender,
                'ethnicity': ethnicity,
                'demographic_group': f"{age_group}_{gender}_{ethnicity}",
                'original_or_augmented': 'original'
            }
    
    # Legacy pattern: class number_processed.mp4
    elif '_processed' in filename:
        base_name = filename.replace('_processed.mp4', '').replace('_processed copy.mp4', '')
        match = re.match(r'^([a-zA-Z_]+)', base_name)
        if match:
            class_name = match.group(1)
            return {
                'filename': filename,
                'class': class_name,
                'age_group': 'unknown',
                'gender': 'unknown',
                'ethnicity': 'unknown',
                'demographic_group': 'unknown',
                'original_or_augmented': 'original'
            }
    
    return None

def analyze_available_videos():
    """Analyze all available videos in the main directory"""
    print("🔍 ANALYZING AVAILABLE VIDEOS")
    print("=" * 50)
    
    video_dir = "data/the_best_videos_so_far"
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    print(f"Found {len(video_files)} total video files")
    
    # Extract metadata for all videos
    video_metadata = []
    target_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    
    for video_path in video_files:
        metadata = extract_video_metadata(video_path)
        if metadata and metadata['class'] in target_classes:
            metadata['video_path'] = video_path
            video_metadata.append(metadata)
    
    df = pd.DataFrame(video_metadata)
    
    print(f"\n📊 ORIGINAL VIDEOS BY CLASS:")
    class_counts = df['class'].value_counts().sort_index()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} videos")
    
    print(f"\n📊 DEMOGRAPHIC DISTRIBUTION:")
    demo_counts = df['demographic_group'].value_counts()
    for demo, count in demo_counts.items():
        print(f"  {demo}: {count} videos")
    
    return df

def stratified_selection(df, target_per_class=102):
    """Select exactly target_per_class videos per class using stratified sampling"""
    print(f"\n🎯 STRATIFIED SELECTION (Target: {target_per_class} per class)")
    print("=" * 60)
    
    selected_videos = []
    excluded_videos = []
    
    target_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    
    for class_name in target_classes:
        print(f"\n📋 Processing {class_name}:")
        
        class_videos = df[df['class'] == class_name].copy()
        available_count = len(class_videos)
        
        print(f"  Available: {available_count} videos")
        print(f"  Target: {target_per_class} videos")
        
        if available_count < target_per_class:
            print(f"  ⚠️  WARNING: Only {available_count} available, need {target_per_class}")
            selected = class_videos
            print(f"  ✅ Selected all {len(selected)} available videos")
        elif available_count == target_per_class:
            selected = class_videos
            print(f"  ✅ Perfect match - selected all {len(selected)} videos")
        else:
            # Stratified sampling to preserve demographic diversity
            print(f"  📉 Need to select {target_per_class} from {available_count}")
            
            # Group by demographic for proportional sampling
            demo_groups = class_videos.groupby('demographic_group')
            print(f"  Demographics found: {len(demo_groups)} groups")
            
            selected_parts = []
            for demo_name, demo_group in demo_groups:
                demo_count = len(demo_group)
                # Calculate proportional selection
                proportion = demo_count / available_count
                target_from_demo = max(1, round(proportion * target_per_class))
                
                # Don't exceed available in this demographic
                actual_from_demo = min(target_from_demo, demo_count)
                
                demo_selected = demo_group.sample(n=actual_from_demo, random_state=42)
                selected_parts.append(demo_selected)
                
                print(f"    {demo_name}: {actual_from_demo}/{demo_count} selected")
            
            # Combine all demographic selections
            selected = pd.concat(selected_parts, ignore_index=True)
            
            # If we have too many, randomly remove excess
            if len(selected) > target_per_class:
                selected = selected.sample(n=target_per_class, random_state=42)
            
            # If we have too few, randomly add more
            elif len(selected) < target_per_class:
                remaining = class_videos[~class_videos.index.isin(selected.index)]
                needed = target_per_class - len(selected)
                additional = remaining.sample(n=min(needed, len(remaining)), random_state=42)
                selected = pd.concat([selected, additional], ignore_index=True)
            
            print(f"  ✅ Final selection: {len(selected)} videos")
            
            # Track excluded videos
            excluded = class_videos[~class_videos['filename'].isin(selected['filename'])]
            excluded_videos.extend(excluded.to_dict('records'))
            print(f"  📝 Excluded: {len(excluded)} videos")
        
        selected_videos.extend(selected.to_dict('records'))
    
    selected_df = pd.DataFrame(selected_videos)
    
    print(f"\n📊 FINAL SELECTION SUMMARY:")
    final_counts = selected_df['class'].value_counts().sort_index()
    for class_name, count in final_counts.items():
        print(f"  {class_name}: {count} videos")
    
    print(f"  Total selected: {len(selected_df)} videos")
    print(f"  Total excluded: {len(excluded_videos)} videos")
    
    return selected_df, excluded_videos

def create_train_val_split(selected_df, train_ratio=0.8):
    """Create stratified train/validation split"""
    print(f"\n🔄 CREATING TRAIN/VALIDATION SPLIT ({train_ratio:.0%}/{1-train_ratio:.0%})")
    print("=" * 60)
    
    train_videos = []
    val_videos = []
    
    target_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    
    for class_name in target_classes:
        class_videos = selected_df[selected_df['class'] == class_name].copy()
        class_count = len(class_videos)
        
        # Calculate split sizes
        train_size = int(class_count * train_ratio)
        val_size = class_count - train_size
        
        # Shuffle and split
        class_videos = class_videos.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_class = class_videos[:train_size]
        val_class = class_videos[train_size:]
        
        train_videos.extend(train_class.to_dict('records'))
        val_videos.extend(val_class.to_dict('records'))
        
        print(f"  {class_name}: {len(train_class)} train, {len(val_class)} val")
    
    train_df = pd.DataFrame(train_videos)
    val_df = pd.DataFrame(val_videos)
    
    print(f"\n📊 SPLIT SUMMARY:")
    print(f"  Training: {len(train_df)} videos ({len(train_df)/(len(train_df)+len(val_df))*100:.1f}%)")
    print(f"  Validation: {len(val_df)} videos ({len(val_df)/(len(train_df)+len(val_df))*100:.1f}%)")
    
    return train_df, val_df

def save_dataset_manifests(train_df, val_df, excluded_videos):
    """Save dataset manifests and exclusion log"""
    print(f"\n💾 SAVING DATASET MANIFESTS")
    print("=" * 40)
    
    # Create output directory
    output_dir = "balanced_102_training_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training manifest
    train_manifest_path = os.path.join(output_dir, "balanced_408_train_manifest.csv")
    train_df.to_csv(train_manifest_path, index=False)
    print(f"  ✅ Training manifest: {train_manifest_path}")
    
    # Save validation manifest
    val_manifest_path = os.path.join(output_dir, "balanced_408_validation_manifest.csv")
    val_df.to_csv(val_manifest_path, index=False)
    print(f"  ✅ Validation manifest: {val_manifest_path}")
    
    # Save exclusion log
    exclusion_path = os.path.join(output_dir, "balanced_102_dataset_exclusions.txt")
    with open(exclusion_path, 'w') as f:
        f.write(f"# Balanced 102-Per-Class Dataset Exclusions\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total excluded videos: {len(excluded_videos)}\n\n")
        
        # Group exclusions by class
        exclusions_by_class = defaultdict(list)
        for video in excluded_videos:
            exclusions_by_class[video['class']].append(video)
        
        for class_name in sorted(exclusions_by_class.keys()):
            class_excluded = exclusions_by_class[class_name]
            f.write(f"## {class_name.upper()} ({len(class_excluded)} excluded)\n")
            for video in class_excluded:
                f.write(f"  - {video['filename']}\n")
            f.write(f"\n")
    
    print(f"  ✅ Exclusion log: {exclusion_path}")
    
    return train_manifest_path, val_manifest_path, exclusion_path

def main():
    """Execute balanced 102-per-class dataset creation"""
    print("🎯 BALANCED 102-VIDEO-PER-CLASS DATASET CREATION")
    print("=" * 70)
    
    # Step 1: Analyze available videos
    df = analyze_available_videos()
    
    # Step 2: Stratified selection
    selected_df, excluded_videos = stratified_selection(df, target_per_class=102)
    
    # Step 3: Train/validation split
    train_df, val_df = create_train_val_split(selected_df, train_ratio=0.8)
    
    # Step 4: Save manifests
    train_path, val_path, exclusion_path = save_dataset_manifests(train_df, val_df, excluded_videos)
    
    print("=" * 70)
    print("📊 DATASET CREATION RESULTS:")
    print(f"  ✅ Balanced dataset: 102 videos per class")
    print(f"  ✅ Training set: {len(train_df)} videos")
    print(f"  ✅ Validation set: {len(val_df)} videos")
    print(f"  ✅ Total dataset: {len(train_df) + len(val_df)} videos")
    print(f"  📝 Excluded videos: {len(excluded_videos)}")
    print(f"  📁 Output directory: balanced_102_training_results/")
    
    # Verify perfect balance
    train_counts = train_df['class'].value_counts().sort_index()
    val_counts = val_df['class'].value_counts().sort_index()
    
    print(f"\n🎯 FINAL CLASS VERIFICATION:")
    all_balanced = True
    for class_name in ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']:
        train_count = train_counts.get(class_name, 0)
        val_count = val_counts.get(class_name, 0)
        total_count = train_count + val_count
        
        if total_count != 102:
            all_balanced = False
        
        print(f"  {class_name}: {total_count} total ({train_count} train + {val_count} val)")
    
    if all_balanced:
        print("✅ PERFECT BALANCE ACHIEVED: All classes have exactly 102 videos!")
        print("🎯 Ready for model training with balanced 408-video dataset")
        return True
    else:
        print("⚠️  BALANCE NOT ACHIEVED: Manual adjustment required")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
