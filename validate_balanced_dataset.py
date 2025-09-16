#!/usr/bin/env python3
"""
Comprehensive Validation Report for Balanced Dataset

This script performs thorough validation of the balanced dataset to ensure
all preprocessing standards are maintained and the dataset is ready for training.
"""

import os
import glob
import csv
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

def validate_video_quality(video_path):
    """Comprehensive video quality validation."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Cannot open video", {}
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Read a few frames to check quality
        frames_checked = 0
        brightness_values = []
        
        for i in range(min(5, frame_count)):  # Check first 5 frames
            ret, frame = cap.read()
            if ret:
                frames_checked += 1
                # Check if grayscale (should have same values across channels)
                if len(frame.shape) == 3:
                    is_grayscale = np.allclose(frame[:,:,0], frame[:,:,1]) and np.allclose(frame[:,:,1], frame[:,:,2])
                else:
                    is_grayscale = True
                
                # Calculate brightness
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                brightness = np.mean(gray_frame)
                brightness_values.append(brightness)
        
        cap.release()
        
        # Validation checks
        issues = []
        if frame_count != 32:
            issues.append(f"Frame count: {frame_count} (expected 32)")
        
        if width > 640 or height > 432:
            issues.append(f"Dimensions too large: {width}x{height}")
        
        if frames_checked == 0:
            issues.append("No frames could be read")
        
        avg_brightness = np.mean(brightness_values) if brightness_values else 0
        if avg_brightness < 50 or avg_brightness > 200:
            issues.append(f"Unusual brightness: {avg_brightness:.1f}")
        
        quality_info = {
            'frame_count': frame_count,
            'dimensions': f"{width}x{height}",
            'fps': fps,
            'avg_brightness': avg_brightness,
            'frames_readable': frames_checked
        }
        
        is_valid = len(issues) == 0
        status_msg = "Valid" if is_valid else f"Issues: {'; '.join(issues)}"
        
        return is_valid, status_msg, quality_info
        
    except Exception as e:
        return False, f"Error: {str(e)}", {}

def generate_validation_report():
    """Generate comprehensive validation report for balanced dataset."""
    print("🔍 COMPREHENSIVE BALANCED DATASET VALIDATION")
    print("=" * 80)
    
    # Check if balanced dataset exists
    balanced_path = "balanced_training_dataset"
    manifest_path = "balanced_dataset_manifest.csv"
    
    if not os.path.exists(balanced_path):
        print(f"❌ Balanced dataset not found at: {balanced_path}")
        return False
    
    if not os.path.exists(manifest_path):
        print(f"❌ Manifest file not found at: {manifest_path}")
        return False
    
    # Load manifest
    manifest_data = []
    with open(manifest_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        manifest_data = list(reader)
    
    print(f"📄 Loaded manifest with {len(manifest_data)} entries")
    print(f"📁 Validating videos in: {balanced_path}")
    print()
    
    # Validation statistics
    validation_stats = {
        'total_videos': 0,
        'valid_videos': 0,
        'invalid_videos': 0,
        'class_counts': defaultdict(int),
        'original_count': 0,
        'duplicate_count': 0,
        'quality_issues': []
    }
    
    dimension_stats = defaultdict(int)
    brightness_values = []
    
    print("🔄 VALIDATING INDIVIDUAL VIDEOS:")
    print("-" * 60)
    
    # Validate each video
    for entry in manifest_data:
        video_path = os.path.join(balanced_path, entry['filename'])
        class_name = entry['class']
        is_duplicate = entry['is_duplicate'].lower() == 'true'
        
        validation_stats['total_videos'] += 1
        validation_stats['class_counts'][class_name] += 1
        
        if is_duplicate:
            validation_stats['duplicate_count'] += 1
        else:
            validation_stats['original_count'] += 1
        
        # Validate video quality
        is_valid, status_msg, quality_info = validate_video_quality(video_path)
        
        if is_valid:
            validation_stats['valid_videos'] += 1
            if quality_info:
                dimension_stats[quality_info['dimensions']] += 1
                brightness_values.append(quality_info['avg_brightness'])
        else:
            validation_stats['invalid_videos'] += 1
            validation_stats['quality_issues'].append({
                'filename': entry['filename'],
                'class': class_name,
                'issue': status_msg
            })
        
        # Print validation result
        status_icon = "✅" if is_valid else "❌"
        duplicate_marker = " [DUP]" if is_duplicate else ""
        print(f"   {status_icon} {entry['filename']:<35} | {class_name:<8} | {status_msg}{duplicate_marker}")
    
    print()
    print("📊 VALIDATION SUMMARY:")
    print("-" * 60)
    print(f"   • Total videos validated: {validation_stats['total_videos']}")
    print(f"   • Valid videos: {validation_stats['valid_videos']} ✅")
    print(f"   • Invalid videos: {validation_stats['invalid_videos']} {'❌' if validation_stats['invalid_videos'] > 0 else '✅'}")
    print(f"   • Original videos: {validation_stats['original_count']}")
    print(f"   • Duplicate videos: {validation_stats['duplicate_count']}")
    
    print(f"\n📋 CLASS DISTRIBUTION VERIFICATION:")
    print("-" * 60)
    all_classes_balanced = True
    target_count = 20  # Expected count per class
    
    for class_name in ['doctor', 'glasses', 'help', 'phone', 'pillow']:
        count = validation_stats['class_counts'][class_name]
        is_balanced = count == target_count
        status = "✅" if is_balanced else "❌"
        print(f"   • {class_name:<12}: {count:>2} videos {status}")
        if not is_balanced:
            all_classes_balanced = False
    
    print(f"\n🎯 PREPROCESSING STANDARDS VERIFICATION:")
    print("-" * 60)
    
    # Dimension analysis
    print("   📐 Video Dimensions:")
    for dimensions, count in sorted(dimension_stats.items()):
        percentage = (count / validation_stats['valid_videos']) * 100 if validation_stats['valid_videos'] > 0 else 0
        print(f"      • {dimensions:<12}: {count:>2} videos ({percentage:>5.1f}%)")
    
    # Brightness analysis
    if brightness_values:
        avg_brightness = np.mean(brightness_values)
        std_brightness = np.std(brightness_values)
        min_brightness = np.min(brightness_values)
        max_brightness = np.max(brightness_values)
        
        print(f"   💡 Brightness Analysis:")
        print(f"      • Average: {avg_brightness:.1f} ± {std_brightness:.1f}")
        print(f"      • Range: {min_brightness:.1f} - {max_brightness:.1f}")
        print(f"      • Target range: 100-150 (enhanced normalization)")
        
        brightness_ok = 100 <= avg_brightness <= 150
        brightness_status = "✅" if brightness_ok else "⚠️"
        print(f"      • Status: {brightness_status}")
    
    # Quality issues report
    if validation_stats['quality_issues']:
        print(f"\n❌ QUALITY ISSUES FOUND:")
        print("-" * 60)
        for issue in validation_stats['quality_issues']:
            print(f"   • {issue['filename']} ({issue['class']}): {issue['issue']}")
    
    # Overall assessment
    print(f"\n🎯 OVERALL DATASET ASSESSMENT:")
    print("=" * 60)
    
    success_rate = (validation_stats['valid_videos'] / validation_stats['total_videos']) * 100 if validation_stats['total_videos'] > 0 else 0
    
    print(f"   • Validation success rate: {success_rate:.1f}%")
    print(f"   • Class balance status: {'✅ PERFECT' if all_classes_balanced else '❌ IMBALANCED'}")
    print(f"   • Quality issues: {len(validation_stats['quality_issues'])}")
    
    # Final recommendation
    is_ready_for_training = (
        success_rate >= 95.0 and
        all_classes_balanced and
        len(validation_stats['quality_issues']) == 0
    )
    
    print(f"\n🚀 TRAINING READINESS:")
    print("=" * 60)
    if is_ready_for_training:
        print("   ✅ DATASET IS READY FOR TRAINING!")
        print("   • All quality checks passed")
        print("   • Perfect class balance achieved")
        print("   • Preprocessing standards maintained")
        print()
        print("   🎯 Recommended next steps:")
        print("      1. Proceed with training configuration")
        print("      2. Implement train/validation/test splits")
        print("      3. Configure R2Plus1D model architecture")
        print("      4. Begin training with monitoring")
    else:
        print("   ⚠️  DATASET NEEDS ATTENTION BEFORE TRAINING")
        print("   • Review quality issues above")
        print("   • Fix any imbalances or preprocessing problems")
        print("   • Re-run validation before proceeding")
    
    return is_ready_for_training

def main():
    """Main validation execution."""
    return generate_validation_report()

if __name__ == "__main__":
    main()
