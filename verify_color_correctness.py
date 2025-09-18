#!/usr/bin/env python3
"""
Verify Color Correctness
========================
Test augmented videos to ensure proper color rendering (no green artifacts).
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_video_colors(video_path):
    """Analyze color properties of a video."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    # Read first frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Analyze color channels (BGR format)
    b_channel = frame[:, :, 0]  # Blue channel
    g_channel = frame[:, :, 1]  # Green channel  
    r_channel = frame[:, :, 2]  # Red channel
    
    # Calculate channel statistics
    b_mean = np.mean(b_channel)
    g_mean = np.mean(g_channel)
    r_mean = np.mean(r_channel)
    
    # Check for green dominance (indicator of color corruption)
    green_dominance = g_mean > (b_mean + r_mean) / 2 * 1.2
    
    return {
        'b_mean': b_mean,
        'g_mean': g_mean,
        'r_mean': r_mean,
        'green_dominance': green_dominance,
        'frame_shape': frame.shape
    }

def main():
    augmented_dir = Path("data/the_best_videos_so_far/augmented_videos")
    
    print("🎨 VERIFYING COLOR CORRECTNESS")
    print("=" * 60)
    print(f"📁 Augmented Directory: {augmented_dir}")
    print()
    
    # Test sample videos from different classes
    sample_videos = []
    classes_tested = set()
    
    for video_file in augmented_dir.glob("*.mp4"):
        # Extract class name
        class_name = video_file.name.split('_')[0] if '_' in video_file.name else video_file.name.split(' ')[0]
        
        if class_name not in classes_tested and len(sample_videos) < 7:
            sample_videos.append(video_file)
            classes_tested.add(class_name)
    
    print(f"🧪 Testing {len(sample_videos)} sample videos:")
    print("-" * 60)
    
    all_good = True
    
    for i, video_path in enumerate(sample_videos, 1):
        print(f"\n{i}. {video_path.name}")
        print("   " + "-" * 50)
        
        color_info = analyze_video_colors(video_path)
        
        if color_info:
            b_mean = color_info['b_mean']
            g_mean = color_info['g_mean']
            r_mean = color_info['r_mean']
            green_dom = color_info['green_dominance']
            
            print(f"   Blue Channel Mean:  {b_mean:6.1f}")
            print(f"   Green Channel Mean: {g_mean:6.1f}")
            print(f"   Red Channel Mean:   {r_mean:6.1f}")
            print(f"   Frame Shape: {color_info['frame_shape']}")
            
            if green_dom:
                print("   🚨 WARNING: Green dominance detected!")
                all_good = False
            else:
                print("   ✅ Color balance looks good")
        else:
            print("   ❌ Could not analyze video")
            all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("🎯 ✅ COLOR VERIFICATION PASSED")
        print("   All tested videos show proper color balance")
        print("   No green coloration artifacts detected")
    else:
        print("🎯 ❌ COLOR VERIFICATION FAILED")
        print("   Some videos may have color issues")
    
    print(f"\n📁 All {len(list(augmented_dir.glob('*.mp4')))} augmented videos ready for inspection")

if __name__ == "__main__":
    main()
