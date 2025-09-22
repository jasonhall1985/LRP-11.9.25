#!/usr/bin/env python3
"""
Comprehensive Video Inventory Analysis
Analyze video inventory across both source directories for 4-class lip-reading training
Focus on original videos only, exclude augmented videos
"""

import os
import glob
import pandas as pd
import numpy as np
import re
from collections import defaultdict

def extract_class_from_processed_filename(filename):
    """Extract class from processed video filename in the_best_videos_so_far"""
    filename = os.path.basename(filename)
    
    # Skip augmented videos
    if 'augmented' in filename.lower():
        return None, 'augmented'
    
    # Pattern: class__useruser01__age__gender__ethnicity__timestamp_topmid_96x64_processed.mp4
    if '__' in filename and '_processed' in filename:
        parts = filename.split('__')
        if len(parts) >= 1:
            class_name = parts[0]
            return class_name, 'processed_original'
    
    # Legacy pattern: class number_processed.mp4
    elif '_processed' in filename:
        base_name = filename.replace('_processed.mp4', '').replace('_processed copy.mp4', '')
        match = re.match(r'^([a-zA-Z_]+)', base_name)
        if match:
            class_name = match.group(1)
            return class_name, 'processed_original'
    
    return None, 'unknown'

def normalize_class_name(raw_class_name):
    """Normalize class name variations to standard format"""
    if not raw_class_name:
        return None

    # Convert to lowercase and replace spaces with underscores
    normalized = raw_class_name.lower().replace(' ', '_')

    # Handle specific variations
    class_mappings = {
        'my_mouth_is_dry': 'my_mouth_is_dry',
        'i_need_to_move': 'i_need_to_move',
        'doctor': 'doctor',
        'pillow': 'pillow',
        # Handle variations
        'my mouth is dry': 'my_mouth_is_dry',
        'i need to move': 'i_need_to_move',
        'myneed_to_move': 'i_need_to_move',  # Potential typo
        'my_need_to_move': 'i_need_to_move',  # Potential variation
    }

    return class_mappings.get(normalized, normalized)

def extract_class_from_unprocessed_filename(filename):
    """Extract class from unprocessed video filename in extra videos directory with normalization"""
    filename = os.path.basename(filename)

    # Pattern examples from extra videos:
    # "pillow_female_65plus_causasian_video 1.mp4"
    # "doctor_male_18to39_caucasian_video 2.mp4"
    # "my mouth is dry_female_18-39_asian_video  (2) 4.mp4"
    # "I need to move_female_18-29_caucasian_video 2 copy 2.mp4"

    # Try to extract class name from various patterns
    potential_class = None

    # Pattern 1: Standard underscore format (pillow_, doctor_)
    if filename.startswith('pillow_'):
        potential_class = 'pillow'
    elif filename.startswith('doctor_'):
        potential_class = 'doctor'
    elif filename.startswith('i_need_to_move_'):
        potential_class = 'i_need_to_move'
    elif filename.startswith('my_mouth_is_dry_'):
        potential_class = 'my_mouth_is_dry'

    # Pattern 2: Space-separated format (need to find first underscore or space pattern)
    elif 'my mouth is dry' in filename.lower():
        potential_class = 'my mouth is dry'
    elif 'i need to move' in filename.lower():
        potential_class = 'i need to move'

    # Pattern 3: Mixed case variations
    elif filename.lower().startswith('my mouth is dry'):
        potential_class = 'my mouth is dry'
    elif filename.lower().startswith('i need to move'):
        potential_class = 'i need to move'

    # Pattern 4: Generic extraction from beginning
    else:
        # Try to extract from beginning until first underscore or space followed by demographic info
        parts = filename.split('_')
        if len(parts) > 1:
            # Check if first part might be a class name
            first_part = parts[0].lower()
            if first_part in ['doctor', 'pillow']:
                potential_class = first_part
            else:
                # Try extracting longer phrases for multi-word classes
                # Look for patterns like "my mouth is dry_" or "i need to move_"
                for i in range(2, min(5, len(parts))):  # Check up to 4 words
                    phrase = '_'.join(parts[:i]).lower()
                    if 'mouth' in phrase and 'dry' in phrase:
                        potential_class = phrase
                        break
                    elif 'need' in phrase and 'move' in phrase:
                        potential_class = phrase
                        break

    # Normalize the extracted class name
    if potential_class:
        normalized_class = normalize_class_name(potential_class)
        if normalized_class in ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']:
            return normalized_class, 'unprocessed_original'

    return None, 'unknown'

def analyze_processed_videos_directory():
    """Analyze processed videos in data/the_best_videos_so_far/"""
    print("📁 ANALYZING: data/the_best_videos_so_far/ (Processed Videos)")
    print("=" * 60)
    
    directory = "data/the_best_videos_so_far"
    
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return {}
    
    # Find all MP4 files
    video_files = glob.glob(os.path.join(directory, "*.mp4"))
    print(f"Total MP4 files found: {len(video_files)}")
    
    # Analyze each video
    target_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    class_counts = defaultdict(int)
    augmented_counts = defaultdict(int)
    video_details = []
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        class_name, video_type = extract_class_from_processed_filename(filename)
        
        if class_name in target_classes:
            if video_type == 'processed_original':
                class_counts[class_name] += 1
                video_details.append({
                    'filename': filename,
                    'class': class_name,
                    'type': 'processed_original',
                    'directory': 'the_best_videos_so_far'
                })
            elif video_type == 'augmented':
                augmented_counts[class_name] += 1
    
    # Display results
    print(f"\n📊 PROCESSED ORIGINAL VIDEOS BY CLASS:")
    total_processed = 0
    for class_name in sorted(target_classes):
        count = class_counts[class_name]
        total_processed += count
        print(f"  {class_name}: {count} videos")
    
    print(f"\n📊 AUGMENTED VIDEOS (EXCLUDED FROM COUNT):")
    total_augmented = 0
    for class_name in sorted(target_classes):
        count = augmented_counts[class_name]
        total_augmented += count
        print(f"  {class_name}: {count} videos")
    
    print(f"\nSummary:")
    print(f"  Total processed original videos: {total_processed}")
    print(f"  Total augmented videos (excluded): {total_augmented}")
    
    return dict(class_counts), video_details

def analyze_unprocessed_videos_directory():
    """Analyze unprocessed videos in data/extra videos 22.9.25/"""
    print(f"\n📁 ANALYZING: data/extra videos 22.9.25/ (Unprocessed Videos)")
    print("=" * 60)
    
    directory = "data/extra videos 22.9.25"
    
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return {}
    
    # Find all MP4 files
    video_files = glob.glob(os.path.join(directory, "*.mp4"))
    print(f"Total MP4 files found: {len(video_files)}")
    
    # Analyze each video
    target_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    class_counts = defaultdict(int)
    video_details = []
    
    print(f"\n📋 UNPROCESSED VIDEO FILES (with class name normalization):")
    unknown_files = []

    for video_path in video_files:
        filename = os.path.basename(video_path)
        class_name, video_type = extract_class_from_unprocessed_filename(filename)

        status_indicator = "✅" if class_name in target_classes else "❌"
        print(f"  {status_indicator} {filename}")
        print(f"      -> Detected class: {class_name if class_name else 'UNKNOWN'}")

        if class_name in target_classes:
            class_counts[class_name] += 1
            video_details.append({
                'filename': filename,
                'class': class_name,
                'type': 'unprocessed_original',
                'directory': 'extra videos 22.9.25'
            })
        else:
            unknown_files.append(filename)

    if unknown_files:
        print(f"\n⚠️  UNRECOGNIZED FILES ({len(unknown_files)} files):")
        for filename in unknown_files[:10]:  # Show first 10
            print(f"    • {filename}")
        if len(unknown_files) > 10:
            print(f"    ... and {len(unknown_files) - 10} more")
    
    # Display results
    print(f"\n📊 UNPROCESSED ORIGINAL VIDEOS BY CLASS:")
    total_unprocessed = 0
    for class_name in sorted(target_classes):
        count = class_counts[class_name]
        total_unprocessed += count
        print(f"  {class_name}: {count} videos")
    
    print(f"\nSummary:")
    print(f"  Total unprocessed original videos: {total_unprocessed}")
    
    return dict(class_counts), video_details

def calculate_combined_inventory(processed_counts, unprocessed_counts):
    """Calculate combined inventory and gap analysis"""
    print(f"\n🔄 COMBINED INVENTORY ANALYSIS")
    print("=" * 50)
    
    target_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    combined_analysis = {}
    
    print(f"📊 COMBINED TOTALS (Processed + Unprocessed):")
    total_available = 0
    
    for class_name in sorted(target_classes):
        processed = processed_counts.get(class_name, 0)
        unprocessed = unprocessed_counts.get(class_name, 0)
        total = processed + unprocessed
        
        combined_analysis[class_name] = {
            'processed': processed,
            'unprocessed': unprocessed,
            'total_available': total
        }
        
        total_available += total
        
        print(f"  {class_name}:")
        print(f"    Processed:   {processed:3d} videos")
        print(f"    Unprocessed: {unprocessed:3d} videos")
        print(f"    Total:       {total:3d} videos")
    
    print(f"\nOverall totals:")
    print(f"  Total available videos: {total_available}")
    
    return combined_analysis

def gap_analysis_for_85_per_class(combined_analysis, target_per_class=85):
    """Perform gap analysis for 85-per-class target"""
    print(f"\n🎯 GAP ANALYSIS FOR {target_per_class}-PER-CLASS TARGET")
    print("=" * 60)
    
    target_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    
    total_available = 0
    total_gaps = 0
    classes_with_gaps = []
    classes_complete = []
    
    print(f"📋 PER-CLASS GAP ANALYSIS:")
    
    for class_name in sorted(target_classes):
        analysis = combined_analysis[class_name]
        available = analysis['total_available']
        gap = max(0, target_per_class - available)
        
        total_available += available
        total_gaps += gap
        
        status = "✅ Complete" if gap == 0 else f"❌ Need {gap} more"
        
        if gap > 0:
            classes_with_gaps.append((class_name, gap, analysis))
        else:
            classes_complete.append(class_name)
        
        print(f"\n  {class_name.upper()}:")
        print(f"    Available: {available:3d} videos")
        print(f"    Target:    {target_per_class:3d} videos")
        print(f"    Gap:       {gap:3d} videos")
        print(f"    Status:    {status}")
        
        if analysis['unprocessed'] > 0:
            print(f"    Note:      {analysis['unprocessed']} videos need preprocessing")
    
    # Summary
    total_target = target_per_class * len(target_classes)
    completion_percentage = (total_available / total_target) * 100
    
    print(f"\n📊 OVERALL SUMMARY:")
    print(f"  Total available:   {total_available:3d} videos")
    print(f"  Total target:      {total_target:3d} videos")
    print(f"  Total gaps:        {total_gaps:3d} videos")
    print(f"  Completion:        {completion_percentage:.1f}%")
    
    # Processing requirements
    total_unprocessed = sum(analysis['unprocessed'] for analysis in combined_analysis.values())
    print(f"  Videos needing preprocessing: {total_unprocessed}")
    
    return {
        'classes_with_gaps': classes_with_gaps,
        'classes_complete': classes_complete,
        'total_gaps': total_gaps,
        'completion_percentage': completion_percentage,
        'total_unprocessed': total_unprocessed
    }

def generate_action_plan(gap_results, combined_analysis):
    """Generate action plan based on gap analysis"""
    print(f"\n🚀 ACTION PLAN")
    print("=" * 30)
    
    if gap_results['total_gaps'] == 0:
        print("✅ READY FOR 85-PER-CLASS TRAINING!")
        print("   All classes have sufficient videos")
        
        if gap_results['total_unprocessed'] > 0:
            print(f"\n📋 PREPROCESSING REQUIRED:")
            print(f"   Process {gap_results['total_unprocessed']} unprocessed videos")
            
            for class_name, analysis in combined_analysis.items():
                if analysis['unprocessed'] > 0:
                    print(f"   • {class_name}: {analysis['unprocessed']} videos to process")
    else:
        print(f"📋 COLLECTION REQUIRED: {gap_results['total_gaps']} additional videos needed")
        
        print(f"\n🎯 PRIORITY COLLECTION ORDER:")
        sorted_gaps = sorted(gap_results['classes_with_gaps'], key=lambda x: x[1], reverse=True)
        
        for class_name, gap, analysis in sorted_gaps:
            print(f"   {gap:2d} videos needed for {class_name}")
        
        if gap_results['total_unprocessed'] > 0:
            print(f"\n📋 PREPROCESSING ALSO REQUIRED:")
            print(f"   Process {gap_results['total_unprocessed']} existing unprocessed videos first")
    
    # Alternative balanced sizes
    print(f"\n🔄 ALTERNATIVE BALANCED DATASET SIZES:")
    
    available_counts = [analysis['total_available'] for analysis in combined_analysis.values()]
    min_available = min(available_counts)
    
    print(f"   • Maximum balanced size: {min_available} videos per class ({min_available * 4} total)")
    
    for target in [60, 70, 75, 80]:
        if target > min_available and target < 85:
            needed = sum(max(0, target - analysis['total_available']) 
                        for analysis in combined_analysis.values())
            print(f"   • {target} per class: Need {needed} additional videos ({target * 4} total)")

def main():
    """Execute comprehensive video inventory analysis"""
    print("📊 COMPREHENSIVE VIDEO INVENTORY ANALYSIS")
    print("=" * 70)
    print("Objective: Determine updated dataset composition for 4-class lip-reading")
    print("Scope: Original videos only (exclude augmented)")
    print("Target: 85 videos per class (340 total)")
    
    # Step 1: Analyze processed videos
    processed_counts, processed_details = analyze_processed_videos_directory()
    
    # Step 2: Analyze unprocessed videos
    unprocessed_counts, unprocessed_details = analyze_unprocessed_videos_directory()
    
    # Step 3: Calculate combined inventory
    combined_analysis = calculate_combined_inventory(processed_counts, unprocessed_counts)
    
    # Step 4: Gap analysis for 85-per-class
    gap_results = gap_analysis_for_85_per_class(combined_analysis, target_per_class=85)
    
    # Step 5: Generate action plan
    generate_action_plan(gap_results, combined_analysis)
    
    print("=" * 70)
    print("📄 INVENTORY ANALYSIS COMPLETE")
    
    if gap_results['total_gaps'] == 0:
        print("🎉 SUFFICIENT VIDEOS AVAILABLE FOR 85-PER-CLASS TRAINING!")
    else:
        print(f"📋 COLLECTION NEEDED: {gap_results['total_gaps']} additional videos required")
    
    if gap_results['total_unprocessed'] > 0:
        print(f"⚙️  PREPROCESSING NEEDED: {gap_results['total_unprocessed']} videos to process")
    
    return combined_analysis, gap_results

if __name__ == "__main__":
    combined_analysis, gap_results = main()
