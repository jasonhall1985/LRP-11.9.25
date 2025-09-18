#!/usr/bin/env python3
"""
Analyze Best Videos Dataset
===========================
Count videos per class in the data/the_best_videos_so_far folder
"""

import os
from pathlib import Path
from collections import defaultdict
import re

def extract_class_from_filename(filename):
    """Extract class name from filename."""
    filename_lower = filename.lower()
    
    # Handle different filename patterns
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
        # Try to extract from structured filename
        parts = filename.split('__')
        if len(parts) > 0:
            return parts[0]
        else:
            return 'unknown'

def main():
    dataset_dir = Path("data/the_best_videos_so_far")
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    print("📊 ANALYZING BEST VIDEOS DATASET")
    print("=" * 60)
    print(f"📁 Directory: {dataset_dir}")
    print()
    
    # Count videos by class
    class_counts = defaultdict(list)
    total_videos = 0
    
    # Get all video files
    video_files = [f for f in dataset_dir.iterdir() if f.is_file() and f.suffix.lower() == '.mp4']
    
    for video_file in video_files:
        class_name = extract_class_from_filename(video_file.name)
        class_counts[class_name].append(video_file.name)
        total_videos += 1
    
    # Sort classes by name
    sorted_classes = sorted(class_counts.keys())
    
    print("📈 VIDEO COUNT BY CLASS:")
    print("-" * 60)
    
    for class_name in sorted_classes:
        count = len(class_counts[class_name])
        print(f"{class_name:<20} | {count:>4} videos")
    
    print("-" * 60)
    print(f"{'TOTAL':<20} | {total_videos:>4} videos")
    print()
    
    # Show detailed breakdown for each class
    print("📋 DETAILED CLASS BREAKDOWN:")
    print("=" * 60)
    
    for class_name in sorted_classes:
        videos = class_counts[class_name]
        print(f"\n🎬 {class_name.upper()} ({len(videos)} videos):")
        print("-" * 40)
        
        # Group by filename pattern
        numbered_videos = []
        structured_videos = []
        
        for video in videos:
            if re.match(r'^[a-z_]+ \d+_processed', video):
                numbered_videos.append(video)
            else:
                structured_videos.append(video)
        
        if numbered_videos:
            print(f"  📝 Numbered format: {len(numbered_videos)} videos")
            # Show first few examples
            for i, video in enumerate(sorted(numbered_videos)[:3]):
                print(f"     • {video}")
            if len(numbered_videos) > 3:
                print(f"     ... and {len(numbered_videos) - 3} more")
        
        if structured_videos:
            print(f"  📝 Structured format: {len(structured_videos)} videos")
            # Show first few examples
            for i, video in enumerate(sorted(structured_videos)[:3]):
                print(f"     • {video}")
            if len(structured_videos) > 3:
                print(f"     ... and {len(structured_videos) - 3} more")
    
    # Summary statistics
    print("\n📊 DATASET STATISTICS:")
    print("=" * 60)
    
    if class_counts:
        min_count = min(len(videos) for videos in class_counts.values())
        max_count = max(len(videos) for videos in class_counts.values())
        avg_count = total_videos / len(class_counts)
        
        print(f"Number of classes: {len(class_counts)}")
        print(f"Total videos: {total_videos}")
        print(f"Average per class: {avg_count:.1f}")
        print(f"Min per class: {min_count}")
        print(f"Max per class: {max_count}")
        
        # Check balance
        if max_count - min_count <= 5:
            print("✅ Dataset is well balanced")
        elif max_count - min_count <= 20:
            print("⚠️  Dataset has moderate imbalance")
        else:
            print("❌ Dataset has significant imbalance")
    
    print("\n🎯 ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()
