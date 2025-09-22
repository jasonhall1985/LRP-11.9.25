#!/usr/bin/env python3
"""
Gap Analysis for 85-Video-Per-Class Target
Analyze how many additional original videos are needed to reach 85 videos per class
without using any augmented videos
"""

import os
import glob
import pandas as pd
import numpy as np
import re
from collections import defaultdict

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

def analyze_current_original_videos():
    """Analyze current original video counts"""
    print("🔍 ANALYZING CURRENT ORIGINAL VIDEOS")
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
    
    print(f"\n📊 CURRENT ORIGINAL VIDEOS BY CLASS:")
    class_counts = df['class'].value_counts().sort_index()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} videos")
    
    print(f"\nTotal original videos: {len(df)}")
    
    return df, class_counts

def calculate_gap_to_85(class_counts, target_per_class=85):
    """Calculate gap to reach 85 videos per class"""
    print(f"\n🎯 GAP ANALYSIS FOR {target_per_class}-VIDEO-PER-CLASS TARGET")
    print("=" * 60)
    
    target_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    
    total_current = 0
    total_needed = 0
    gap_analysis = {}
    
    print("📋 PER-CLASS ANALYSIS:")
    for class_name in target_classes:
        current = class_counts.get(class_name, 0)
        gap = max(0, target_per_class - current)
        
        total_current += current
        total_needed += gap
        
        gap_analysis[class_name] = {
            'current': current,
            'target': target_per_class,
            'gap': gap,
            'status': '✅ Complete' if gap == 0 else f'❌ Need {gap} more'
        }
        
        print(f"\n  {class_name.upper()}:")
        print(f"    Current:  {current:3d} videos")
        print(f"    Target:   {target_per_class:3d} videos")
        print(f"    Gap:      {gap:3d} videos")
        print(f"    Status:   {gap_analysis[class_name]['status']}")
    
    # Overall summary
    total_target = target_per_class * len(target_classes)
    completion_percentage = (total_current / total_target) * 100
    
    print(f"\n📊 OVERALL SUMMARY:")
    print(f"  Current total:     {total_current:3d} videos")
    print(f"  Target total:      {total_target:3d} videos")
    print(f"  Additional needed: {total_needed:3d} videos")
    print(f"  Completion:        {completion_percentage:.1f}%")
    
    return gap_analysis, total_needed

def analyze_collection_requirements(gap_analysis):
    """Analyze video collection requirements"""
    print(f"\n📋 VIDEO COLLECTION REQUIREMENTS")
    print("=" * 50)
    
    classes_needing_videos = []
    for class_name, analysis in gap_analysis.items():
        if analysis['gap'] > 0:
            classes_needing_videos.append((class_name, analysis['gap']))
    
    if not classes_needing_videos:
        print("✅ NO ADDITIONAL VIDEOS NEEDED!")
        print("   All classes already have 85+ original videos")
        return
    
    print("📝 CLASSES REQUIRING ADDITIONAL VIDEOS:")
    for class_name, gap in classes_needing_videos:
        print(f"  • {class_name}: {gap} additional videos needed")
    
    # Sort by priority (largest gap first)
    classes_needing_videos.sort(key=lambda x: x[1], reverse=True)
    priority_class, max_gap = classes_needing_videos[0]
    
    print(f"\n🎯 COLLECTION STRATEGY:")
    print(f"  Priority class: {priority_class} (needs {max_gap} videos)")
    print(f"  Total videos to collect: {sum(gap for _, gap in classes_needing_videos)}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("  1. Focus on collecting videos for classes with largest gaps")
    print("  2. Maintain demographic diversity during collection")
    print("  3. Ensure consistent video quality and preprocessing")
    print("  4. Consider recording new videos if existing sources are insufficient")
    
    # Alternative strategies
    print(f"\n🔄 ALTERNATIVE STRATEGIES:")
    
    # Find maximum feasible balanced size
    min_current = min(analysis['current'] for analysis in gap_analysis.values())
    print(f"  • Current maximum balanced size: {min_current} videos per class")
    print(f"    (Total: {min_current * 4} videos)")
    
    # Find intermediate targets
    current_counts = [analysis['current'] for analysis in gap_analysis.values()]
    current_counts.sort()
    
    for intermediate_target in [60, 70, 75, 80]:
        if intermediate_target > min_current and intermediate_target < 85:
            needed_for_target = sum(max(0, intermediate_target - analysis['current']) 
                                  for analysis in gap_analysis.values())
            if needed_for_target > 0:
                print(f"  • {intermediate_target} videos per class: Need {needed_for_target} additional videos")

def main():
    """Execute gap analysis for 85-video-per-class target"""
    print("📊 GAP ANALYSIS: 85 VIDEOS PER CLASS (ORIGINAL VIDEOS ONLY)")
    print("=" * 70)
    print("Objective: Determine how many additional original videos are needed")
    print("Constraint: No augmented videos allowed")
    
    # Step 1: Analyze current original videos
    df, class_counts = analyze_current_original_videos()
    
    # Step 2: Calculate gap to 85 per class
    gap_analysis, total_needed = calculate_gap_to_85(class_counts, target_per_class=85)
    
    # Step 3: Analyze collection requirements
    analyze_collection_requirements(gap_analysis)
    
    print("=" * 70)
    if total_needed == 0:
        print("🎉 READY FOR 85-PER-CLASS TRAINING!")
        print("   All classes have sufficient original videos")
    else:
        print(f"📋 COLLECTION REQUIRED: {total_needed} additional original videos needed")
        print("   Consider alternative balanced sizes or video collection strategy")
    
    return gap_analysis, total_needed

if __name__ == "__main__":
    gap_analysis, total_needed = main()
    
    print(f"\n📄 Analysis complete.")
    if total_needed > 0:
        print(f"💡 Consider training with current maximum balanced size or collecting {total_needed} additional videos")
    else:
        print("✅ Ready to proceed with 85-per-class balanced dataset creation!")
