#!/usr/bin/env python3
"""
Dataset Analysis for 85-Video-Per-Class Expansion
Analyze current dataset composition from both source directories to determine
what's needed to reach 85 videos per class (340 total videos)
"""

import os
import glob
import re
from collections import defaultdict, Counter

def extract_class_from_filename(filename):
    """Extract class name from video filename"""
    # Handle different naming patterns
    filename = os.path.basename(filename)
    
    # Pattern 1: class__useruser01__... (standard format)
    if '__' in filename:
        class_name = filename.split('__')[0]
        return class_name
    
    # Pattern 2: class number_processed.mp4 (legacy format)
    if '_processed' in filename:
        # Remove _processed and file extension
        base_name = filename.replace('_processed.mp4', '').replace('_processed copy.mp4', '')
        # Extract class name (everything before the number)
        match = re.match(r'^([a-zA-Z_]+)', base_name)
        if match:
            return match.group(1)
    
    return 'unknown'

def analyze_directory(directory_path, directory_name):
    """Analyze videos in a directory and return class counts"""
    print(f"\n📁 ANALYZING: {directory_name}")
    print(f"   Path: {directory_path}")
    
    if not os.path.exists(directory_path):
        print(f"   ❌ Directory not found!")
        return {}
    
    # Find all video files
    video_extensions = ['*.mp4', '*.mov', '*.avi']
    all_videos = []
    
    for ext in video_extensions:
        videos = glob.glob(os.path.join(directory_path, ext))
        all_videos.extend(videos)
        
        # Also check subdirectories (like augmented_videos)
        subdirs = glob.glob(os.path.join(directory_path, '*', ext))
        all_videos.extend(subdirs)
    
    print(f"   Total video files found: {len(all_videos)}")
    
    # Count by class
    class_counts = defaultdict(int)
    class_files = defaultdict(list)
    
    for video_path in all_videos:
        filename = os.path.basename(video_path)
        class_name = extract_class_from_filename(filename)
        
        # Map legacy class names to current 4-class system
        class_mapping = {
            'doctor': 'doctor',
            'i_need_to_move': 'i_need_to_move', 
            'my_mouth_is_dry': 'my_mouth_is_dry',
            'pillow': 'pillow',
            'phone': 'phone',  # Not in 4-class system
            'glasses': 'glasses',  # Not in 4-class system
            'help': 'help',  # Not in 4-class system
            'unknown': 'unknown'
        }
        
        mapped_class = class_mapping.get(class_name, class_name)
        class_counts[mapped_class] += 1
        class_files[mapped_class].append(filename)
    
    # Display results
    print(f"   Class distribution:")
    for class_name in sorted(class_counts.keys()):
        count = class_counts[class_name]
        print(f"     {class_name}: {count} videos")
    
    return dict(class_counts)

def analyze_extra_videos_directory():
    """Analyze the extra videos directory for unprocessed videos"""
    print(f"\n📁 ANALYZING: Extra Videos Directory")
    extra_dir = "data/extra videos 22.9.25"
    
    if not os.path.exists(extra_dir):
        print(f"   ❌ Directory not found: {extra_dir}")
        return {}
    
    # Find all video files
    video_files = glob.glob(os.path.join(extra_dir, "*.mp4"))
    print(f"   Total video files found: {len(video_files)}")
    
    # Count by class (extract from filename)
    class_counts = defaultdict(int)
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        
        # Extract class from filename patterns like "pillow_female_65plus_causasian_video 1.mp4"
        if filename.startswith('pillow_'):
            class_counts['pillow'] += 1
        elif filename.startswith('doctor_'):
            class_counts['doctor'] += 1
        elif filename.startswith('i_need_to_move_'):
            class_counts['i_need_to_move'] += 1
        elif filename.startswith('my_mouth_is_dry_'):
            class_counts['my_mouth_is_dry'] += 1
        else:
            # Try to extract class from beginning of filename
            class_name = filename.split('_')[0]
            class_counts[class_name] += 1
    
    print(f"   Class distribution:")
    for class_name in sorted(class_counts.keys()):
        count = class_counts[class_name]
        print(f"     {class_name}: {count} videos")
    
    return dict(class_counts)

def calculate_expansion_requirements(current_counts, target_per_class=85):
    """Calculate what's needed to reach target videos per class"""
    print(f"\n🎯 EXPANSION ANALYSIS (Target: {target_per_class} per class)")
    print("=" * 60)
    
    # Focus on 4-class system
    target_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    
    total_current = 0
    total_needed = 0
    expansion_plan = {}
    
    for class_name in target_classes:
        current = current_counts.get(class_name, 0)
        needed = max(0, target_per_class - current)
        
        total_current += current
        total_needed += needed
        
        expansion_plan[class_name] = {
            'current': current,
            'target': target_per_class,
            'needed': needed,
            'status': '✅ Complete' if needed == 0 else f'📈 Need {needed} more'
        }
        
        print(f"{class_name}:")
        print(f"  Current: {current} videos")
        print(f"  Target:  {target_per_class} videos")
        print(f"  Gap:     {needed} videos")
        print(f"  Status:  {expansion_plan[class_name]['status']}")
        print()
    
    # Summary
    total_target = target_per_class * len(target_classes)
    print(f"📊 SUMMARY:")
    print(f"  Current total: {total_current} videos")
    print(f"  Target total:  {total_target} videos")
    print(f"  Additional needed: {total_needed} videos")
    print(f"  Progress: {total_current}/{total_target} ({total_current/total_target*100:.1f}%)")
    
    return expansion_plan

def main():
    """Execute comprehensive dataset analysis"""
    print("🔍 DATASET ANALYSIS FOR 85-VIDEO-PER-CLASS EXPANSION")
    print("=" * 70)
    
    # Analyze main training directory
    main_counts = analyze_directory("data/the_best_videos_so_far", "Main Training Directory")
    
    # Analyze extra videos directory
    extra_counts = analyze_extra_videos_directory()
    
    # Combine counts (note: we already processed 10 pillow videos from extra)
    print(f"\n🔄 COMBINING COUNTS")
    print("=" * 30)
    
    combined_counts = defaultdict(int)
    
    # Add main directory counts
    for class_name, count in main_counts.items():
        combined_counts[class_name] += count
    
    # Add extra directory counts (but subtract the 10 pillow videos we already processed)
    for class_name, count in extra_counts.items():
        if class_name == 'pillow':
            # We already processed 10 pillow videos from extra directory
            remaining = max(0, count - 10)
            combined_counts[f'{class_name}_unprocessed'] = remaining
            print(f"   Note: {count} pillow videos in extra directory")
            print(f"         10 already processed in Phase 1")
            print(f"         {remaining} remaining unprocessed")
        else:
            combined_counts[f'{class_name}_unprocessed'] = count
    
    print(f"\n📊 COMBINED DATASET COMPOSITION:")
    print("   (Including processed videos from main directory)")
    
    # Focus on 4-class system for final counts
    final_counts = {}
    target_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    
    for class_name in target_classes:
        processed_count = main_counts.get(class_name, 0)
        unprocessed_count = combined_counts.get(f'{class_name}_unprocessed', 0)
        total_available = processed_count + unprocessed_count
        
        final_counts[class_name] = processed_count  # Only count processed videos
        
        print(f"   {class_name}:")
        print(f"     Processed: {processed_count}")
        print(f"     Unprocessed: {unprocessed_count}")
        print(f"     Total available: {total_available}")
    
    # Calculate expansion requirements
    expansion_plan = calculate_expansion_requirements(final_counts, target_per_class=85)
    
    # Action plan
    print(f"\n🚀 ACTION PLAN:")
    print("=" * 20)
    
    classes_needing_expansion = [cls for cls, plan in expansion_plan.items() if plan['needed'] > 0]
    
    if not classes_needing_expansion:
        print("✅ All classes already have 85+ videos!")
        print("   Ready to proceed with balanced 85-per-class dataset creation.")
    else:
        print("📋 Classes requiring additional videos:")
        for class_name in classes_needing_expansion:
            plan = expansion_plan[class_name]
            print(f"   • {class_name}: Need {plan['needed']} more videos")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("   1. Process any available unprocessed videos from extra directory")
        print("   2. Collect additional videos for classes with gaps")
        print("   3. Consider data augmentation for classes with small gaps")
        print("   4. Alternatively, reduce target to a feasible number (e.g., 61 per class)")
    
    return expansion_plan

if __name__ == "__main__":
    expansion_plan = main()
    
    print(f"\n📄 Analysis complete. Use this information to plan dataset expansion strategy.")
