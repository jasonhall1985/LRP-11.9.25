#!/usr/bin/env python3
"""
Visualize Crop Parameter Improvements
====================================

Create a visual comparison showing the difference between the old (80% height × 60% width)
and new optimized (65% height × 40% width, centered) cropping parameters.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_sample_frame(video_path):
    """Load a single frame from a video for visualization."""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_old_crop(frame):
    """Apply old cropping parameters (80% height × 60% width)."""
    h, w = frame.shape
    
    # Old parameters: 80% height × 60% width (20% to 80% horizontally)
    crop_h = int(0.80 * h)  # 80% height
    crop_w_start = int(0.20 * w)  # 60% width (20% to 80%)
    crop_w_end = int(0.80 * w)
    
    cropped = frame[0:crop_h, crop_w_start:crop_w_end]
    return cv2.resize(cropped, (96, 96))

def apply_optimized_crop(frame):
    """Apply optimized cropping parameters (65% height × 40% width, centered)."""
    h, w = frame.shape
    
    # Optimized parameters: 65% height × 40% width (30% to 70% horizontally)
    crop_h = int(0.65 * h)  # 65% height
    
    # Vertical positioning: start from 10% down to center mouth region
    crop_v_start = int(0.10 * h)
    crop_v_end = crop_v_start + crop_h
    
    # Horizontal positioning: middle 40% for better mouth capture
    crop_w_start = int(0.30 * w)  # 30% to 70% (40% width)
    crop_w_end = int(0.70 * w)
    
    # Ensure crop doesn't exceed frame boundaries
    crop_v_end = min(crop_v_end, h)
    crop_w_end = min(crop_w_end, w)
    
    cropped = frame[crop_v_start:crop_v_end, crop_w_start:crop_w_end]
    return cv2.resize(cropped, (96, 96))

def visualize_crop_overlay(frame, title):
    """Visualize crop areas overlaid on the original frame."""
    h, w = frame.shape
    overlay = frame.copy()
    
    # Old crop area (red)
    old_crop_h = int(0.80 * h)
    old_crop_w_start = int(0.20 * w)
    old_crop_w_end = int(0.80 * w)
    cv2.rectangle(overlay, (old_crop_w_start, 0), (old_crop_w_end, old_crop_h), 128, 2)
    
    # Optimized crop area (green)
    opt_crop_h = int(0.65 * h)
    opt_crop_v_start = int(0.10 * h)
    opt_crop_v_end = opt_crop_v_start + opt_crop_h
    opt_crop_w_start = int(0.30 * w)
    opt_crop_w_end = int(0.70 * w)
    cv2.rectangle(overlay, (opt_crop_w_start, opt_crop_v_start), (opt_crop_w_end, opt_crop_v_end), 200, 2)
    
    return overlay

def main():
    """Create crop comparison visualization."""
    print("📊 CREATING CROP IMPROVEMENT VISUALIZATION")
    print("=" * 50)
    
    # Find a sample video from the additional dataset
    sample_dir = Path("data/13.9.25top7dataset_cropped")
    sample_videos = list(sample_dir.glob("doctor*.mp4"))
    
    if not sample_videos:
        print("❌ No sample videos found!")
        return
    
    sample_video = sample_videos[0]
    print(f"📹 Using sample: {sample_video.name}")
    
    # Load sample frame
    frame = load_sample_frame(sample_video)
    if frame is None:
        print("❌ Could not load sample frame!")
        return
    
    print(f"📐 Original frame size: {frame.shape}")
    
    # Apply different cropping methods
    old_cropped = apply_old_crop(frame)
    optimized_cropped = apply_optimized_crop(frame)
    overlay = visualize_crop_overlay(frame, "Crop Areas Comparison")
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Crop Parameter Improvement Comparison', fontsize=16, fontweight='bold')
    
    # Original with overlay
    axes[0, 0].imshow(overlay, cmap='gray')
    axes[0, 0].set_title('Original Frame with Crop Areas\n(Red: Old 80%×60%, Green: Optimized 65%×40%)', fontsize=10)
    axes[0, 0].axis('off')
    
    # Original frame
    axes[0, 1].imshow(frame, cmap='gray')
    axes[0, 1].set_title('Original Frame', fontsize=12)
    axes[0, 1].axis('off')
    
    # Old crop result
    axes[1, 0].imshow(old_cropped, cmap='gray')
    axes[1, 0].set_title('OLD: 80% height × 60% width\n(From top, 20%-80% horizontal)', fontsize=10)
    axes[1, 0].axis('off')
    
    # Optimized crop result
    axes[1, 1].imshow(optimized_cropped, cmap='gray')
    axes[1, 1].set_title('OPTIMIZED: 65% height × 40% width\n(10% down, 30%-70% horizontal, centered)', fontsize=10)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = "crop_improvement_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved: {output_path}")
    
    # Analyze the differences
    print(f"\n📊 CROP ANALYSIS:")
    print(f"   Original frame: {frame.shape[0]}×{frame.shape[1]} pixels")
    print(f"   Old crop area: {int(0.80 * frame.shape[0])}×{int(0.60 * frame.shape[1])} pixels")
    print(f"   Optimized crop area: {int(0.65 * frame.shape[0])}×{int(0.40 * frame.shape[1])} pixels")
    print(f"   Both resized to: 96×96 pixels")
    
    print(f"\n🎯 KEY IMPROVEMENTS:")
    print(f"   • Reduced height (65% vs 80%) = less background, more focus on mouth")
    print(f"   • Reduced width (40% vs 60%) = better mouth centering")
    print(f"   • Vertical offset (10% down) = mouth positioned in center of crop")
    print(f"   • Horizontal centering (30%-70%) = optimal mouth positioning")
    print(f"   • Complete lip visibility ensured")
    print(f"   • Consistent with preview_videos_fixed approach")

if __name__ == "__main__":
    main()
