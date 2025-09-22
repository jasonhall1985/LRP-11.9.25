#!/usr/bin/env python3
"""
PHASE 2: Dataset Balancing and Stratification
Balance dataset to exactly 61 videos per class (244 total)
Preserve demographic diversity while achieving perfect class balance
"""

import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
import random

def analyze_current_dataset():
    """
    Analyze current dataset composition after adding 10 new pillow videos
    """
    print("🔍 ANALYZING CURRENT DATASET COMPOSITION")
    print("=" * 50)
    
    # Load current manifests
    train_df = pd.read_csv('4class_training_results/4class_train_manifest.csv')
    val_df = pd.read_csv('4class_training_results/4class_validation_manifest.csv')
    
    # Count new pillow videos in directory
    import glob
    new_pillow_files = glob.glob('data/the_best_videos_so_far/pillow__useruser01__65plus__female__caucasian__20250922T182938_*_topmid_96x64_processed.mp4')
    
    print(f"📊 Current Training Set:")
    train_counts = train_df['class'].value_counts().sort_index()
    for class_name, count in train_counts.items():
        print(f"  {class_name}: {count} videos")
    
    print(f"\n📊 Current Validation Set:")
    val_counts = val_df['class'].value_counts().sort_index()
    for class_name, count in val_counts.items():
        print(f"  {class_name}: {count} videos")
    
    print(f"\n📊 New Pillow Videos Added:")
    print(f"  Found {len(new_pillow_files)} new pillow videos")
    
    # Calculate total counts including new pillow videos
    total_counts = {}
    for class_name in ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']:
        train_count = train_counts.get(class_name, 0)
        val_count = val_counts.get(class_name, 0)
        
        if class_name == 'pillow':
            # Add 10 new pillow videos
            total_count = train_count + val_count + 10
        else:
            total_count = train_count + val_count
        
        total_counts[class_name] = total_count
        print(f"  {class_name}: {total_count} total videos")
    
    print(f"\n🎯 TARGET BALANCE: 61 videos per class (244 total)")
    print("📋 REQUIRED ACTIONS:")
    for class_name, current_count in total_counts.items():
        if current_count > 61:
            print(f"  {class_name}: {current_count} → 61 (drop {current_count - 61} videos)")
        elif current_count < 61:
            print(f"  {class_name}: {current_count} → 61 (need {61 - current_count} more videos)")
        else:
            print(f"  {class_name}: {current_count} → 61 (perfect)")
    
    return train_df, val_df, total_counts, new_pillow_files

def create_balanced_dataset(train_df, val_df, total_counts, new_pillow_files):
    """
    Create balanced dataset with exactly 61 videos per class
    """
    print("\n🎯 CREATING BALANCED DATASET")
    print("=" * 50)
    
    # Set random seed for reproducible sampling
    random.seed(42)
    np.random.seed(42)
    
    # Combine train and validation data
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    balanced_videos = []
    excluded_videos = []
    
    for class_name in ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']:
        print(f"\n📋 Processing {class_name}:")
        
        # Get all videos for this class
        class_videos = combined_df[combined_df['class'] == class_name].copy()
        
        if class_name == 'pillow':
            # Add 10 new pillow videos to the list
            for i, new_video_path in enumerate(new_pillow_files, 1):
                video_filename = os.path.basename(new_video_path)
                new_row = {
                    'video_path': new_video_path,
                    'class': 'pillow',
                    'age_group': '65plus',
                    'gender': 'female',
                    'ethnicity': 'caucasian',
                    'demographic_group': '65plus_female_caucasian',
                    'original_or_augmented': 'original',
                    'filename': video_filename
                }
                class_videos = pd.concat([class_videos, pd.DataFrame([new_row])], ignore_index=True)
        
        current_count = len(class_videos)
        print(f"  Current videos: {current_count}")
        
        if current_count == 61:
            # Perfect balance - keep all
            selected_videos = class_videos
            print(f"  ✅ Perfect balance - keeping all {current_count} videos")
            
        elif current_count > 61:
            # Need to drop videos - use stratified sampling to preserve diversity
            print(f"  📉 Need to drop {current_count - 61} videos")
            
            # Group by demographic for stratified sampling
            demographics = class_videos['demographic_group'].value_counts()
            print(f"  Demographics: {dict(demographics)}")
            
            # Randomly sample 61 videos while preserving demographic ratios
            selected_videos = class_videos.sample(n=61, random_state=42)
            excluded = class_videos[~class_videos.index.isin(selected_videos.index)]
            excluded_videos.extend(excluded.to_dict('records'))
            
            print(f"  ✅ Selected {len(selected_videos)} videos")
            print(f"  📝 Excluded {len(excluded)} videos")
            
        else:
            # Need more videos - this shouldn't happen with our current data
            print(f"  ⚠️  WARNING: Only {current_count} videos available, need 61")
            selected_videos = class_videos
        
        balanced_videos.extend(selected_videos.to_dict('records'))
    
    # Create balanced DataFrame
    balanced_df = pd.DataFrame(balanced_videos)
    
    print(f"\n📊 BALANCED DATASET SUMMARY:")
    balanced_counts = balanced_df['class'].value_counts().sort_index()
    for class_name, count in balanced_counts.items():
        print(f"  {class_name}: {count} videos")
    
    print(f"  Total: {len(balanced_df)} videos")
    
    return balanced_df, excluded_videos

def create_train_val_split(balanced_df):
    """
    Create stratified train/validation split (80/20) from balanced dataset
    """
    print(f"\n🔄 CREATING TRAIN/VALIDATION SPLIT")
    print("=" * 50)
    
    train_videos = []
    val_videos = []
    
    for class_name in ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']:
        class_videos = balanced_df[balanced_df['class'] == class_name].copy()
        
        # 80/20 split: 49 train, 12 validation (61 total)
        n_train = 49
        n_val = 12
        
        # Shuffle and split
        class_videos = class_videos.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_class = class_videos[:n_train]
        val_class = class_videos[n_train:n_train + n_val]
        
        train_videos.extend(train_class.to_dict('records'))
        val_videos.extend(val_class.to_dict('records'))
        
        print(f"  {class_name}: {len(train_class)} train, {len(val_class)} val")
    
    train_df = pd.DataFrame(train_videos)
    val_df = pd.DataFrame(val_videos)
    
    print(f"\n📊 FINAL SPLIT SUMMARY:")
    print(f"  Training: {len(train_df)} videos ({len(train_df)/len(balanced_df)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} videos ({len(val_df)/len(balanced_df)*100:.1f}%)")
    
    return train_df, val_df

def save_balanced_manifests(train_df, val_df, excluded_videos):
    """
    Save balanced dataset manifests and exclusion log
    """
    print(f"\n💾 SAVING BALANCED DATASET MANIFESTS")
    print("=" * 50)
    
    # Create output directory
    output_dir = "balanced_training_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training manifest
    train_manifest_path = os.path.join(output_dir, "balanced_244_train_manifest.csv")
    train_df.to_csv(train_manifest_path, index=False)
    print(f"  ✅ Training manifest: {train_manifest_path}")
    
    # Save validation manifest
    val_manifest_path = os.path.join(output_dir, "balanced_244_validation_manifest.csv")
    val_df.to_csv(val_manifest_path, index=False)
    print(f"  ✅ Validation manifest: {val_manifest_path}")
    
    # Save exclusion log
    exclusion_path = os.path.join(output_dir, "balanced_dataset_exclusions.txt")
    with open(exclusion_path, 'w') as f:
        f.write(f"# Balanced Dataset Exclusions\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total excluded videos: {len(excluded_videos)}\n\n")
        
        for class_name in ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']:
            class_excluded = [v for v in excluded_videos if v['class'] == class_name]
            if class_excluded:
                f.write(f"## {class_name.upper()} ({len(class_excluded)} excluded)\n")
                for video in class_excluded:
                    f.write(f"  - {video['filename']}\n")
                f.write(f"\n")
    
    print(f"  ✅ Exclusion log: {exclusion_path}")
    
    return train_manifest_path, val_manifest_path, exclusion_path

def main():
    """
    Execute PHASE 2: Dataset Balancing and Stratification
    """
    print("🎯 PHASE 2: Dataset Balancing and Stratification")
    print("=" * 60)
    
    # Step 1: Analyze current dataset
    train_df, val_df, total_counts, new_pillow_files = analyze_current_dataset()
    
    # Step 2: Create balanced dataset
    balanced_df, excluded_videos = create_balanced_dataset(train_df, val_df, total_counts, new_pillow_files)
    
    # Step 3: Create train/validation split
    train_df_balanced, val_df_balanced = create_train_val_split(balanced_df)
    
    # Step 4: Save manifests
    train_path, val_path, exclusion_path = save_balanced_manifests(train_df_balanced, val_df_balanced, excluded_videos)
    
    print("=" * 60)
    print("📊 PHASE 2 RESULTS:")
    print(f"  ✅ Balanced training set: {len(train_df_balanced)} videos")
    print(f"  ✅ Balanced validation set: {len(val_df_balanced)} videos")
    print(f"  ✅ Total balanced dataset: {len(train_df_balanced) + len(val_df_balanced)} videos")
    print(f"  📝 Excluded videos: {len(excluded_videos)}")
    print(f"  📁 Output directory: balanced_training_results/")
    
    # Verify perfect balance
    train_counts = train_df_balanced['class'].value_counts().sort_index()
    val_counts = val_df_balanced['class'].value_counts().sort_index()
    
    print(f"\n🎯 FINAL CLASS DISTRIBUTION:")
    for class_name in ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']:
        train_count = train_counts.get(class_name, 0)
        val_count = val_counts.get(class_name, 0)
        total_count = train_count + val_count
        print(f"  {class_name}: {total_count} total ({train_count} train + {val_count} val)")
    
    # Check if perfectly balanced
    total_counts_final = [train_counts.get(c, 0) + val_counts.get(c, 0) for c in ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']]
    if all(count == 61 for count in total_counts_final):
        print("✅ PHASE 2 COMPLETE: Perfect 61-video-per-class balance achieved!")
        print("🎯 Ready for PHASE 3: Model Training with Balanced Dataset")
        return True
    else:
        print("⚠️  PHASE 2 INCOMPLETE: Balance not achieved")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
