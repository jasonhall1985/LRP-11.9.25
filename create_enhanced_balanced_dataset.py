#!/usr/bin/env python3
"""
Create Enhanced Balanced Dataset Using Highest Available Class Count
Maximize dataset size while maintaining perfect balance and demographic diversity
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import json
import random

def analyze_current_video_pool():
    """Analyze current video pool after integration of new diverse videos"""
    print("🔍 ANALYZING CURRENT VIDEO POOL AFTER INTEGRATION")
    print("=" * 70)
    
    data_dir = "data/the_best_videos_so_far"
    video_files = glob.glob(os.path.join(data_dir, "*.mp4"))
    
    print(f"📁 Found {len(video_files)} total videos in pool")
    
    # Analyze videos by class and demographics
    video_analysis = []
    class_counts = defaultdict(int)
    demographic_counts = defaultdict(int)
    
    target_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        
        # Skip augmented videos
        if 'augmented' in filename:
            continue
        
        # Extract class from filename
        class_name = None
        for target_class in target_classes:
            if filename.startswith(target_class + '__') or filename.startswith(target_class + '_'):
                class_name = target_class
                break
        
        if not class_name:
            continue
        
        # Extract demographic info from processed filename
        # Format: class__useruser01__age__gender__ethnicity__timestamp_uniqueid_topmid_96x64_processed.mp4
        parts = filename.split('__')
        if len(parts) >= 5:
            age_group = parts[2] if parts[2] != 'unknown' else 'unknown'
            gender = parts[3] if parts[3] != 'unknown' else 'unknown'
            ethnicity = parts[4].split('_')[0] if parts[4] != 'unknown' else 'unknown'
        else:
            # Fallback for older naming conventions
            age_group = 'unknown'
            gender = 'unknown'
            ethnicity = 'unknown'
            
            # Try to extract from filename patterns
            filename_lower = filename.lower()
            if '18to39' in filename_lower:
                age_group = '18to39'
            elif '40to64' in filename_lower:
                age_group = '40to64'
            elif '65plus' in filename_lower:
                age_group = '65plus'
            
            if 'female' in filename_lower:
                gender = 'female'
            elif 'male' in filename_lower:
                gender = 'male'
            
            if 'caucasian' in filename_lower:
                ethnicity = 'caucasian'
            elif 'asian' in filename_lower:
                ethnicity = 'asian'
            elif 'aboriginal' in filename_lower:
                ethnicity = 'aboriginal'
        
        demographic_group = f"{age_group}_{gender}_{ethnicity}"
        
        video_analysis.append({
            'filename': filename,
            'class': class_name,
            'age_group': age_group,
            'gender': gender,
            'ethnicity': ethnicity,
            'demographic_group': demographic_group
        })
        
        class_counts[class_name] += 1
        demographic_counts[demographic_group] += 1
    
    print(f"\n📊 CURRENT VIDEO POOL BY CLASS:")
    total_videos = 0
    for class_name in sorted(target_classes):
        count = class_counts[class_name]
        total_videos += count
        print(f"  {class_name}: {count} videos")
    
    print(f"  Total original videos: {total_videos}")
    
    print(f"\n🌍 DEMOGRAPHIC DIVERSITY:")
    print(f"  Unique demographic groups: {len(demographic_counts)}")
    for demo, count in sorted(demographic_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_videos) * 100 if total_videos > 0 else 0
        print(f"  {demo}: {count} videos ({percentage:.1f}%)")
    
    return video_analysis, dict(class_counts), dict(demographic_counts)

def create_maximum_balanced_dataset(video_analysis, class_counts):
    """Create balanced dataset using the highest available class count"""
    print(f"\n🎯 CREATING MAXIMUM BALANCED DATASET")
    print("=" * 50)
    
    # Find maximum class count for balancing
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    
    print(f"📊 CLASS COUNT ANALYSIS:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count} videos")
    
    print(f"\n🎯 BALANCING STRATEGY:")
    print(f"  Maximum available: {max_count} videos")
    print(f"  Minimum available: {min_count} videos")
    print(f"  Balance target: {max_count} videos per class")
    print(f"  Total balanced dataset: {max_count * 4} videos")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(video_analysis)
    
    # Create balanced dataset by sampling up to max_count per class
    balanced_videos = []
    
    for class_name in sorted(class_counts.keys()):
        class_videos = df[df['class'] == class_name].copy()
        available_count = len(class_videos)
        
        if available_count >= max_count:
            # Stratified sampling to preserve demographic diversity
            selected_videos = stratified_sample_by_demographics(class_videos, max_count)
        else:
            # Use all available videos
            selected_videos = class_videos
        
        balanced_videos.extend(selected_videos.to_dict('records'))
        print(f"  {class_name}: Selected {len(selected_videos)} videos")
    
    return balanced_videos, max_count

def stratified_sample_by_demographics(class_videos, target_count):
    """Stratified sampling to preserve demographic diversity"""
    # Group by demographic groups
    demo_groups = class_videos.groupby('demographic_group')
    
    selected_videos = []
    remaining_target = target_count
    
    # Calculate proportional sampling
    for demo_name, demo_group in demo_groups:
        demo_count = len(demo_group)
        proportion = demo_count / len(class_videos)
        target_from_demo = max(1, round(proportion * target_count))
        
        # Don't exceed available videos in this demographic
        actual_from_demo = min(target_from_demo, demo_count, remaining_target)
        
        if actual_from_demo > 0:
            sampled = demo_group.sample(n=actual_from_demo, random_state=42)
            selected_videos.append(sampled)
            remaining_target -= actual_from_demo
        
        if remaining_target <= 0:
            break
    
    # If we still need more videos, sample randomly from remaining
    if remaining_target > 0 and len(selected_videos) > 0:
        combined_selected = pd.concat(selected_videos)
        remaining_videos = class_videos[~class_videos.index.isin(combined_selected.index)]
        
        if len(remaining_videos) > 0:
            additional = remaining_videos.sample(n=min(remaining_target, len(remaining_videos)), random_state=42)
            selected_videos.append(additional)
    
    return pd.concat(selected_videos) if selected_videos else pd.DataFrame()

def create_train_validation_split(balanced_videos, balance_target):
    """Create 80/20 train/validation split with stratified sampling"""
    print(f"\n📊 CREATING 80/20 TRAIN/VALIDATION SPLIT")
    print("=" * 50)
    
    df = pd.DataFrame(balanced_videos)
    
    train_videos = []
    val_videos = []
    
    # Split each class separately to maintain balance
    for class_name in df['class'].unique():
        class_videos = df[df['class'] == class_name].copy()
        
        # Calculate split sizes
        total_class_videos = len(class_videos)
        train_size = int(total_class_videos * 0.8)
        val_size = total_class_videos - train_size
        
        # Stratified split by demographic groups
        demo_groups = class_videos.groupby('demographic_group')
        
        class_train = []
        class_val = []
        
        for demo_name, demo_group in demo_groups:
            demo_count = len(demo_group)
            demo_train_size = max(1, int(demo_count * 0.8))
            demo_val_size = demo_count - demo_train_size
            
            # Shuffle and split
            shuffled = demo_group.sample(frac=1, random_state=42)
            demo_train = shuffled.iloc[:demo_train_size]
            demo_val = shuffled.iloc[demo_train_size:]
            
            class_train.append(demo_train)
            if len(demo_val) > 0:
                class_val.append(demo_val)
        
        # Combine demographic groups for this class
        if class_train:
            class_train_combined = pd.concat(class_train)
            train_videos.extend(class_train_combined.to_dict('records'))
        
        if class_val:
            class_val_combined = pd.concat(class_val)
            val_videos.extend(class_val_combined.to_dict('records'))
        
        print(f"  {class_name}: {len(class_train_combined) if class_train else 0} train, {len(class_val_combined) if class_val else 0} val")
    
    print(f"\n📊 SPLIT SUMMARY:")
    print(f"  Training videos: {len(train_videos)}")
    print(f"  Validation videos: {len(val_videos)}")
    print(f"  Total: {len(train_videos) + len(val_videos)}")
    
    return train_videos, val_videos

def save_manifests(train_videos, val_videos, balance_target):
    """Save training and validation manifests"""
    print(f"\n💾 SAVING ENHANCED BALANCED DATASET MANIFESTS")
    print("=" * 50)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = "enhanced_balanced_training_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create training manifest
    train_df = pd.DataFrame(train_videos)
    train_manifest_path = os.path.join(output_dir, f"enhanced_balanced_{balance_target * 4}_train_manifest.csv")
    train_df.to_csv(train_manifest_path, index=False)
    
    # Create validation manifest
    val_df = pd.DataFrame(val_videos)
    val_manifest_path = os.path.join(output_dir, f"enhanced_balanced_{balance_target * 4}_validation_manifest.csv")
    val_df.to_csv(val_manifest_path, index=False)
    
    print(f"✅ Training manifest: {train_manifest_path}")
    print(f"✅ Validation manifest: {val_manifest_path}")
    
    # Generate summary statistics
    print(f"\n📊 TRAINING SET ANALYSIS:")
    train_class_counts = train_df['class'].value_counts().sort_index()
    for class_name, count in train_class_counts.items():
        print(f"  {class_name}: {count} videos")
    
    print(f"\n📊 VALIDATION SET ANALYSIS:")
    val_class_counts = val_df['class'].value_counts().sort_index()
    for class_name, count in val_class_counts.items():
        print(f"  {class_name}: videos")
    
    print(f"\n🌍 VALIDATION SET DEMOGRAPHIC DIVERSITY:")
    val_demo_counts = val_df['demographic_group'].value_counts()
    print(f"  Unique demographic groups: {len(val_demo_counts)}")
    for demo, count in val_demo_counts.items():
        print(f"  {demo}: {count} videos")
    
    return train_manifest_path, val_manifest_path, output_dir

def compare_with_previous_dataset(balance_target):
    """Compare with previous 85-per-class dataset"""
    print(f"\n📈 COMPARISON WITH PREVIOUS 85-PER-CLASS DATASET")
    print("=" * 60)
    
    previous_size = 85 * 4  # 340 videos
    current_size = balance_target * 4
    improvement = current_size - previous_size
    improvement_pct = (improvement / previous_size) * 100 if previous_size > 0 else 0
    
    print(f"📊 DATASET SIZE COMPARISON:")
    print(f"  Previous dataset: {previous_size} videos (85 per class)")
    print(f"  Enhanced dataset: {current_size} videos ({balance_target} per class)")
    print(f"  Improvement: +{improvement} videos ({improvement_pct:+.1f}%)")
    
    print(f"\n🎯 TRAINING DATA COMPARISON:")
    previous_train = int(previous_size * 0.8)  # ~272 videos
    current_train = int(current_size * 0.8)
    train_improvement = current_train - previous_train
    train_improvement_pct = (train_improvement / previous_train) * 100 if previous_train > 0 else 0
    
    print(f"  Previous training: {previous_train} videos")
    print(f"  Enhanced training: {current_train} videos")
    print(f"  Training improvement: +{train_improvement} videos ({train_improvement_pct:+.1f}%)")
    
    return {
        'previous_size': previous_size,
        'current_size': current_size,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'previous_train': previous_train,
        'current_train': current_train,
        'train_improvement': train_improvement,
        'train_improvement_pct': train_improvement_pct
    }

def main():
    """Execute enhanced balanced dataset creation"""
    print("🎯 ENHANCED BALANCED DATASET CREATION")
    print("=" * 70)
    print("Objective: Create largest possible balanced dataset with demographic diversity")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Analyze current video pool
    video_analysis, class_counts, demographic_counts = analyze_current_video_pool()
    
    # Create maximum balanced dataset
    balanced_videos, balance_target = create_maximum_balanced_dataset(video_analysis, class_counts)
    
    # Create train/validation split
    train_videos, val_videos = create_train_validation_split(balanced_videos, balance_target)
    
    # Save manifests
    train_manifest_path, val_manifest_path, output_dir = save_manifests(train_videos, val_videos, balance_target)
    
    # Compare with previous dataset
    comparison = compare_with_previous_dataset(balance_target)
    
    # Save comprehensive results
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'balance_target': balance_target,
        'total_videos': balance_target * 4,
        'train_videos': len(train_videos),
        'val_videos': len(val_videos),
        'class_counts': class_counts,
        'demographic_diversity': len(demographic_counts),
        'comparison': comparison,
        'manifests': {
            'train': train_manifest_path,
            'validation': val_manifest_path
        }
    }
    
    results_path = os.path.join(output_dir, f"enhanced_balanced_dataset_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved: {results_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 ENHANCED BALANCED DATASET CREATION COMPLETE")
    print(f"🎯 Balance achieved: {balance_target} videos per class")
    print(f"📊 Total dataset: {balance_target * 4} videos")
    print(f"📈 Improvement: {comparison['improvement_pct']:+.1f}% vs. previous 85-per-class")
    print(f"🌍 Demographic groups: {len(demographic_counts)}")
    print("🚀 Ready for Phase 3: Lightweight Model Training")
    
    return True, results_path, balance_target

if __name__ == "__main__":
    success, results_path, balance_target = main()
    exit(0 if success else 1)
