#!/usr/bin/env python3
"""
Simple Crop Test - Show the bigger crop area that includes all lips
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def load_original_crop(video_path):
    """Original V5 crop (50% height, 33% width)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Original small crop
        crop_h = int(0.50 * h)
        crop_w_start = int(0.335 * w)
        crop_w_end = int(0.665 * w)
        
        cropped = gray[0:crop_h, crop_w_start:crop_w_end]
        resized = cv2.resize(cropped, (96, 96))
        frames.append(resized)
    
    cap.release()
    
    # Get first 8 frames
    return np.array(frames[:8])

def load_bigger_crop(video_path):
    """New bigger crop (80% height, 60% width)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Bigger crop - zoom out more
        crop_h = int(0.80 * h)  # 80% height instead of 50%
        crop_w_start = int(0.20 * w)  # 60% width instead of 33%
        crop_w_end = int(0.80 * w)
        
        cropped = gray[0:crop_h, crop_w_start:crop_w_end]
        resized = cv2.resize(cropped, (96, 96))
        frames.append(resized)
    
    cap.release()
    
    # Get first 8 frames
    return np.array(frames[:8])

def show_crop_comparison(original_frames, bigger_frames):
    """Show side-by-side comparison."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Simple Crop Comparison: Original (50%H, 33%W) vs Bigger (80%H, 60%W)', fontsize=14)
    
    for i in range(4):
        # Original crop (top row)
        axes[0, i].imshow(original_frames[i], cmap='gray')
        axes[0, i].set_title(f'Original Crop\nFrame {i}')
        axes[0, i].axis('off')
        
        # Bigger crop (bottom row)
        axes[1, i].imshow(bigger_frames[i], cmap='gray')
        axes[1, i].set_title(f'Bigger Crop\nFrame {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_crop_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Crop comparison saved as 'simple_crop_comparison.png'")

def main():
    """Test the simple bigger crop."""
    print("🔧 SIMPLE CROP TEST")
    print("=" * 40)
    
    video_path = "/Users/client/Desktop/LRP classifier 11.9.25/data/TRAINING SET 2.9.25/doctor 4.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"📹 Testing on: {video_path}")
    
    # Load with different crops
    print("\n🔄 Loading with original crop...")
    original_frames = load_original_crop(video_path)
    print(f"✅ Original crop: {original_frames.shape}")
    
    print("\n🔄 Loading with bigger crop...")
    bigger_frames = load_bigger_crop(video_path)
    print(f"✅ Bigger crop: {bigger_frames.shape}")
    
    # Show comparison
    print("\n📊 Creating comparison...")
    show_crop_comparison(original_frames, bigger_frames)
    
    print("\n✅ SIMPLE CROP TEST COMPLETE")
    print("\n📏 CROP SIZES:")
    print("   Original: 50% height × 33% width")
    print("   Bigger:   80% height × 60% width")
    print("\n🎯 RESULT:")
    print("   ✅ Much bigger crop area")
    print("   ✅ More room for lips to fit")
    print("   ✅ No fancy positioning - just zoomed out")

if __name__ == "__main__":
    main()
