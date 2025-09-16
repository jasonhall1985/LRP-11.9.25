#!/usr/bin/env python3
"""
Create Corrected Balanced Dataset

This script creates a corrected balanced dataset using only videos with exactly 32 frames
and proper preprocessing standards.
"""

import os
import glob
import shutil
import random
import csv
import cv2
from pathlib import Path
from collections import defaultdict

def validate_video_frame_count(video_path):
    """Check if video has exactly 32 frames."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 0
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        return frame_count == 32, frame_count
    except:
        return False, 0

def create_corrected_balanced_dataset():
    """Create a corrected balanced dataset with only valid 32-frame videos."""
    print("🔧 CREATING CORRECTED BALANCED DATASET")
    print("=" * 80)
    
    # Paths
    source_path = "grayscale_validation_output/processed_videos"
    output_path = "corrected_balanced_dataset"
    manifest_path = "corrected_balanced_manifest.csv"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get all processed videos and validate them
    video_files = glob.glob(os.path.join(source_path, "*_processed.mp4"))
    
    print(f"📊 Found {len(video_files)} processed videos")
    print("🔍 Validating frame counts...")
    
    # Organize valid videos by class
    valid_videos = defaultdict(list)
    invalid_videos = []
    
    for video_file in video_files:
        video_name = Path(video_file).stem.replace("_processed", "")
        class_name = video_name.split()[0].lower()
        
        is_valid, frame_count = validate_video_frame_count(video_file)
        
        if is_valid and class_name in ["doctor", "glasses", "help", "phone", "pillow"]:
            valid_videos[class_name].append({
                'original_name': video_name,
                'file_path': video_file,
                'frame_count': frame_count
            })
        else:
            invalid_videos.append({
                'name': video_name,
                'class': class_name,
                'frame_count': frame_count,
                'reason': 'Invalid frame count' if not is_valid else 'Unknown class'
            })
    
    # Report validation results
    print(f"\n📋 VALIDATION RESULTS:")
    print("-" * 50)
    
    total_valid = sum(len(videos) for videos in valid_videos.values())
    print(f"   • Valid videos (32 frames): {total_valid}")
    print(f"   • Invalid videos: {len(invalid_videos)}")
    
    print(f"\n📊 VALID VIDEOS BY CLASS:")
    print("-" * 50)
    
    for class_name in ["doctor", "glasses", "help", "phone", "pillow"]:
        count = len(valid_videos[class_name])
        print(f"   • {class_name:<12}: {count:>2} videos")
    
    if invalid_videos:
        print(f"\n❌ INVALID VIDEOS:")
        print("-" * 50)
        for video in invalid_videos[:10]:  # Show first 10
            print(f"   • {video['name']} ({video['class']}): {video['frame_count']} frames")
        if len(invalid_videos) > 10:
            print(f"   ... and {len(invalid_videos) - 10} more")
    
    # Determine target size (minimum class count to ensure we can balance)
    class_counts = {class_name: len(videos) for class_name, videos in valid_videos.items()}
    min_count = min(class_counts.values())
    
    print(f"\n🎯 BALANCING STRATEGY:")
    print("-" * 50)
    print(f"   • Target size per class: {min_count} videos")
    print(f"   • Strategy: Use all available videos from smallest class")
    print(f"   • For larger classes: randomly sample to match target")
    
    # Create balanced dataset
    manifest_entries = []
    
    print(f"\n🔄 CREATING BALANCED DATASET:")
    print("-" * 50)
    
    for class_name in ["doctor", "glasses", "help", "phone", "pillow"]:
        available_videos = valid_videos[class_name]
        
        # If we have more videos than target, randomly sample
        if len(available_videos) > min_count:
            selected_videos = random.sample(available_videos, min_count)
            print(f"   • {class_name:<12}: sampled {min_count} from {len(available_videos)} available")
        else:
            selected_videos = available_videos
            print(f"   • {class_name:<12}: using all {len(selected_videos)} available")
        
        # Copy selected videos to balanced dataset
        for i, video_info in enumerate(selected_videos):
            src_path = video_info['file_path']
            dst_name = f"{class_name}_{i+1:02d}.mp4"
            dst_path = os.path.join(output_path, dst_name)
            
            shutil.copy2(src_path, dst_path)
            
            # Verify the copied video
            is_valid, frame_count = validate_video_frame_count(dst_path)
            
            manifest_entries.append({
                'class': class_name,
                'filename': dst_name,
                'original_video': video_info['original_name'],
                'frame_count': frame_count,
                'valid': is_valid
            })
    
    # Save manifest
    with open(manifest_path, 'w', newline='') as csvfile:
        fieldnames = ['class', 'filename', 'original_video', 'frame_count', 'valid']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in manifest_entries:
            writer.writerow(entry)
    
    print(f"\n📄 Manifest saved: {manifest_path}")
    
    # Final verification
    print(f"\n✅ FINAL VERIFICATION:")
    print("-" * 50)
    
    final_class_counts = defaultdict(int)
    total_final_videos = 0
    all_valid = True
    
    for entry in manifest_entries:
        final_class_counts[entry['class']] += 1
        total_final_videos += 1
        if not entry['valid']:
            all_valid = False
    
    for class_name in ["doctor", "glasses", "help", "phone", "pillow"]:
        count = final_class_counts[class_name]
        is_balanced = count == min_count
        status = "✅" if is_balanced else "❌"
        print(f"   • {class_name:<12}: {count:>2} videos {status}")
    
    print(f"\n📊 CORRECTED DATASET SUMMARY:")
    print("=" * 50)
    print(f"   • Total videos: {total_final_videos}")
    print(f"   • Videos per class: {min_count}")
    print(f"   • All videos valid: {'✅' if all_valid else '❌'}")
    print(f"   • Perfect balance: {'✅' if len(set(final_class_counts.values())) == 1 else '❌'}")
    print(f"   • Dataset location: {output_path}")
    
    return total_final_videos, min_count, all_valid

def main():
    """Main execution function."""
    # Set random seed for reproducible sampling
    random.seed(42)
    
    total_videos, videos_per_class, all_valid = create_corrected_balanced_dataset()
    
    print(f"\n🎉 CORRECTED BALANCED DATASET CREATED!")
    print("=" * 80)
    
    if all_valid and videos_per_class >= 15:  # Minimum viable dataset size
        print("✅ DATASET IS READY FOR TRAINING!")
        print(f"   • {total_videos} high-quality videos")
        print(f"   • {videos_per_class} videos per class")
        print("   • All videos have exactly 32 frames")
        print("   • Perfect class balance")
        print()
        print("🚀 NEXT STEPS:")
        print("   1. Confirm dataset quality")
        print("   2. Proceed with training configuration")
        print("   3. Implement train/validation/test splits")
    else:
        print("⚠️  DATASET MAY NEED MORE VIDEOS")
        print(f"   • Current size: {videos_per_class} videos per class")
        print("   • Recommended minimum: 15 videos per class")
        print("   • Consider processing more source videos")
    
    return True

if __name__ == "__main__":
    main()
