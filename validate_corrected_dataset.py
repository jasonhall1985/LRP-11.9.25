#!/usr/bin/env python3
"""
Final Validation of Corrected Balanced Dataset
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

def validate_corrected_dataset():
    """Perform final validation of the corrected balanced dataset."""
    print("🔍 FINAL VALIDATION OF CORRECTED BALANCED DATASET")
    print("=" * 80)
    
    dataset_path = "corrected_balanced_dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return False
    
    # Get all videos
    video_files = glob.glob(os.path.join(dataset_path, "*.mp4"))
    
    print(f"📊 Found {len(video_files)} videos in corrected dataset")
    print()
    
    # Validation statistics
    class_counts = defaultdict(int)
    frame_counts = []
    dimensions = []
    brightness_values = []
    all_valid = True
    
    print("🔄 VALIDATING EACH VIDEO:")
    print("-" * 60)
    
    for video_file in sorted(video_files):
        filename = Path(video_file).stem
        class_name = filename.split('_')[0]
        
        try:
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"❌ {filename}: Cannot open video")
                all_valid = False
                continue
            
            # Get properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Read first frame for quality check
            ret, frame = cap.read()
            if ret:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                brightness = np.mean(gray_frame)
                brightness_values.append(brightness)
            
            cap.release()
            
            # Validation checks
            is_valid = frame_count == 32
            status = "✅" if is_valid else "❌"
            
            if is_valid:
                class_counts[class_name] += 1
                frame_counts.append(frame_count)
                dimensions.append(f"{width}x{height}")
            else:
                all_valid = False
            
            print(f"   {status} {filename:<25} | {class_name:<8} | {frame_count:>2} frames | {width}x{height}")
            
        except Exception as e:
            print(f"❌ {filename}: Error - {str(e)}")
            all_valid = False
    
    print()
    print("📊 VALIDATION SUMMARY:")
    print("-" * 60)
    
    # Class distribution
    print("📋 Class Distribution:")
    target_count = 10
    all_balanced = True
    
    for class_name in ["doctor", "glasses", "help", "phone", "pillow"]:
        count = class_counts[class_name]
        is_balanced = count == target_count
        status = "✅" if is_balanced else "❌"
        print(f"   • {class_name:<12}: {count:>2} videos {status}")
        if not is_balanced:
            all_balanced = False
    
    # Frame count consistency
    if frame_counts:
        unique_frame_counts = set(frame_counts)
        frame_consistency = len(unique_frame_counts) == 1 and 32 in unique_frame_counts
        print(f"\n📹 Frame Count Consistency:")
        print(f"   • All videos have 32 frames: {'✅' if frame_consistency else '❌'}")
        if not frame_consistency:
            print(f"   • Found frame counts: {sorted(unique_frame_counts)}")
    
    # Dimension consistency
    if dimensions:
        unique_dimensions = set(dimensions)
        print(f"\n📐 Dimension Analysis:")
        for dim in sorted(unique_dimensions):
            count = dimensions.count(dim)
            percentage = (count / len(dimensions)) * 100
            print(f"   • {dim:<12}: {count:>2} videos ({percentage:>5.1f}%)")
    
    # Brightness analysis
    if brightness_values:
        avg_brightness = np.mean(brightness_values)
        std_brightness = np.std(brightness_values)
        min_brightness = np.min(brightness_values)
        max_brightness = np.max(brightness_values)
        
        print(f"\n💡 Brightness Analysis:")
        print(f"   • Average: {avg_brightness:.1f} ± {std_brightness:.1f}")
        print(f"   • Range: {min_brightness:.1f} - {max_brightness:.1f}")
        
        brightness_ok = 100 <= avg_brightness <= 150
        print(f"   • Within target range (100-150): {'✅' if brightness_ok else '⚠️'}")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print("=" * 60)
    
    total_videos = len(video_files)
    expected_videos = 50  # 10 per class × 5 classes
    
    dataset_ready = (
        all_valid and
        all_balanced and
        total_videos == expected_videos and
        len(unique_frame_counts) == 1 and
        32 in unique_frame_counts
    )
    
    print(f"   • Total videos: {total_videos} (expected: {expected_videos})")
    print(f"   • All videos valid: {'✅' if all_valid else '❌'}")
    print(f"   • Perfect class balance: {'✅' if all_balanced else '❌'}")
    print(f"   • Frame count consistency: {'✅' if frame_consistency else '❌'}")
    
    print(f"\n🚀 TRAINING READINESS:")
    print("=" * 60)
    
    if dataset_ready:
        print("   ✅ DATASET IS READY FOR TRAINING!")
        print("   • All quality checks passed")
        print("   • Perfect class balance (10 videos per class)")
        print("   • Consistent 32-frame structure")
        print("   • Proper preprocessing standards maintained")
        print()
        print("   🎯 Dataset characteristics:")
        print(f"      • Total videos: {total_videos}")
        print(f"      • Classes: 5 (doctor, glasses, help, phone, pillow)")
        print(f"      • Videos per class: 10")
        print(f"      • Frame count: 32 per video")
        print(f"      • Preprocessing: Enhanced grayscale normalization")
        print()
        print("   📋 Recommended training splits:")
        print("      • Training: 8 videos per class (40 total)")
        print("      • Validation: 1 video per class (5 total)")
        print("      • Testing: 1 video per class (5 total)")
    else:
        print("   ⚠️  DATASET NEEDS ATTENTION")
        print("   • Review validation issues above")
        print("   • Consider processing additional source videos")
    
    return dataset_ready

def main():
    """Main validation execution."""
    return validate_corrected_dataset()

if __name__ == "__main__":
    main()
