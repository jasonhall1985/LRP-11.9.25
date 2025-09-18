#!/usr/bin/env python3
"""
Verify Balanced Dataset
=======================
Verify that the brightness augmentation successfully balanced the dataset
by counting original + augmented videos per class.
"""

import os
from pathlib import Path
from collections import defaultdict

def extract_class_from_filename(filename):
    """Extract class name from filename."""
    filename_lower = filename.lower()
    
    if filename.startswith('doctor'):
        return 'doctor'
    elif filename.startswith('glasses'):
        return 'glasses'
    elif filename.startswith('help'):
        return 'help'
    elif filename.startswith('phone'):
        return 'phone'
    elif filename.startswith('pillow'):
        return 'pillow'
    elif filename.startswith('i_need_to_move'):
        return 'i_need_to_move'
    elif filename.startswith('my_mouth_is_dry'):
        return 'my_mouth_is_dry'
    else:
        # Try structured filename
        parts = filename.split('__')
        if len(parts) > 0:
            return parts[0]
        else:
            return 'unknown'

def main():
    original_dir = Path("data/the_best_videos_so_far")
    augmented_dir = Path("data/the_best_videos_so_far/augmented_videos")
    
    print("🔍 VERIFYING BALANCED DATASET")
    print("=" * 60)
    print(f"📁 Original Directory: {original_dir}")
    print(f"📁 Augmented Directory: {augmented_dir}")
    print()
    
    # Count original videos
    original_counts = defaultdict(int)
    if original_dir.exists():
        for video_file in original_dir.iterdir():
            if video_file.is_file() and video_file.suffix.lower() == '.mp4':
                class_name = extract_class_from_filename(video_file.name)
                original_counts[class_name] += 1
    
    # Count augmented videos
    augmented_counts = defaultdict(int)
    if augmented_dir.exists():
        for video_file in augmented_dir.iterdir():
            if video_file.is_file() and video_file.suffix.lower() == '.mp4':
                class_name = extract_class_from_filename(video_file.name)
                augmented_counts[class_name] += 1
    
    # Calculate totals
    all_classes = set(original_counts.keys()) | set(augmented_counts.keys())
    total_counts = {}
    
    for class_name in all_classes:
        original = original_counts.get(class_name, 0)
        augmented = augmented_counts.get(class_name, 0)
        total = original + augmented
        total_counts[class_name] = {
            'original': original,
            'augmented': augmented,
            'total': total
        }
    
    # Sort classes by name
    sorted_classes = sorted(all_classes)
    
    print("📊 DATASET BALANCE VERIFICATION:")
    print("-" * 60)
    print(f"{'CLASS':<20} | {'ORIG':<4} | {'AUG':<4} | {'TOTAL':<5} | {'STATUS'}")
    print("-" * 60)
    
    target_count = 102
    balanced_classes = 0
    total_videos = 0
    
    for class_name in sorted_classes:
        counts = total_counts[class_name]
        original = counts['original']
        augmented = counts['augmented']
        total = counts['total']
        
        # Determine status
        if total == target_count:
            status = "✅ BALANCED"
            balanced_classes += 1
        elif total > target_count:
            status = f"⬆️ OVER (+{total - target_count})"
        else:
            status = f"⬇️ UNDER (-{target_count - total})"
        
        print(f"{class_name:<20} | {original:>4} | {augmented:>4} | {total:>5} | {status}")
        total_videos += total
    
    print("-" * 60)
    print(f"{'TOTAL':<20} | {sum(c['original'] for c in total_counts.values()):>4} | {sum(c['augmented'] for c in total_counts.values()):>4} | {total_videos:>5} |")
    
    # Summary statistics
    print(f"\n📈 BALANCE SUMMARY:")
    print("-" * 60)
    print(f"Target per class: {target_count} videos")
    print(f"Classes balanced: {balanced_classes}/{len(sorted_classes)}")
    print(f"Total videos: {total_videos}")
    print(f"Average per class: {total_videos/len(sorted_classes):.1f}")
    
    if balanced_classes == len(sorted_classes):
        print("🎯 ✅ DATASET PERFECTLY BALANCED!")
    elif balanced_classes >= len(sorted_classes) * 0.8:
        print("🎯 ⚠️ Dataset mostly balanced")
    else:
        print("🎯 ❌ Dataset needs more balancing")
    
    # Show augmented video examples
    print(f"\n📋 AUGMENTED VIDEO EXAMPLES:")
    print("-" * 60)
    
    if augmented_dir.exists():
        augmented_files = list(augmented_dir.glob("*.mp4"))
        if augmented_files:
            print(f"Total augmented videos: {len(augmented_files)}")
            print("Sample augmented videos:")
            for i, video_file in enumerate(sorted(augmented_files)[:5]):
                print(f"  • {video_file.name}")
            if len(augmented_files) > 5:
                print(f"  ... and {len(augmented_files) - 5} more")
        else:
            print("No augmented videos found")
    else:
        print("Augmented directory not found")
    
    print(f"\n🎯 VERIFICATION COMPLETE")

if __name__ == "__main__":
    main()
