#!/usr/bin/env python3
"""
Optimized Crop Test - Validate improved cropping parameters for complete lip capture
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def load_frames_original_crop(video_path, target_frames=32):
    """Load video with original V5 crop (top 50%, middle 33%)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Original V5 crop
        h, w = gray.shape
        crop_h = int(0.50 * h)  # Top 50%
        crop_w_start = int(0.335 * w)  # Middle 33%
        crop_w_end = int(0.665 * w)
        
        cropped = gray[0:crop_h, crop_w_start:crop_w_end]
        resized = cv2.resize(cropped, (96, 96))
        frames.append(resized)
    
    cap.release()
    
    if len(frames) >= target_frames:
        indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    
    return np.array(frames[:target_frames])

def load_frames_expanded_crop(video_path, target_frames=32):
    """Load video with previous expanded crop (top 55%, middle 36.3%)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Previous expanded crop
        h, w = gray.shape
        crop_h = int(0.55 * h)  # Top 55%
        crop_w_start = int(0.3185 * w)  # Middle 36.3%
        crop_w_end = int(0.6815 * w)
        
        cropped = gray[0:crop_h, crop_w_start:crop_w_end]
        resized = cv2.resize(cropped, (96, 96))
        frames.append(resized)
    
    cap.release()
    
    if len(frames) >= target_frames:
        indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    
    return np.array(frames[:target_frames])

def load_frames_optimized_crop(video_path, target_frames=32):
    """Load video with optimized crop (65% height, centered mouth positioning)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optimized crop for complete lip capture
        h, w = gray.shape
        
        # Increased height to 65% to capture complete lip area
        crop_h = int(0.65 * h)
        
        # Vertical positioning: start from 10% down to center mouth region
        crop_v_start = int(0.10 * h)
        crop_v_end = crop_v_start + crop_h
        
        # Horizontal positioning: middle 40% for better mouth capture
        crop_w_start = int(0.30 * w)
        crop_w_end = int(0.70 * w)
        
        # Ensure crop doesn't exceed frame boundaries
        crop_v_end = min(crop_v_end, h)
        crop_w_end = min(crop_w_end, w)
        
        cropped = gray[crop_v_start:crop_v_end, crop_w_start:crop_w_end]
        resized = cv2.resize(cropped, (96, 96))
        frames.append(resized)
    
    cap.release()
    
    if len(frames) >= target_frames:
        indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    
    return np.array(frames[:target_frames])

def visualize_crop_comparison(original_frames, expanded_frames, optimized_frames):
    """Create comprehensive crop comparison visualization."""
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle('Crop Optimization Comparison: Original → Expanded → Optimized', fontsize=16)
    
    # Select frames to display
    frame_indices = [0, 4, 8, 12, 16, 20]
    
    for i, frame_idx in enumerate(frame_indices):
        # Original crop (top row)
        axes[0, i].imshow(original_frames[frame_idx], cmap='gray')
        axes[0, i].set_title(f'Original (50%H, 33%W)\nFrame {frame_idx}', fontsize=10)
        axes[0, i].axis('off')
        
        # Expanded crop (middle row)
        axes[1, i].imshow(expanded_frames[frame_idx], cmap='gray')
        axes[1, i].set_title(f'Expanded (55%H, 36.3%W)\nFrame {frame_idx}', fontsize=10)
        axes[1, i].axis('off')
        
        # Optimized crop (bottom row)
        axes[2, i].imshow(optimized_frames[frame_idx], cmap='gray')
        axes[2, i].set_title(f'Optimized (65%H, 40%W)\nCentered Mouth', fontsize=10)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('optimized_crop_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Crop comparison saved as 'optimized_crop_comparison.png'")

def analyze_lip_visibility(frames, crop_name):
    """Analyze lip visibility and positioning in cropped frames."""
    print(f"\n📊 {crop_name} ANALYSIS:")
    print(f"   Shape: {frames.shape}")
    print(f"   Min pixel value: {frames.min()}")
    print(f"   Max pixel value: {frames.max()}")
    print(f"   Mean brightness: {frames.mean():.2f}")
    
    # Analyze bottom region (where lips should be visible)
    bottom_region = frames[:, -20:, :]  # Bottom 20 pixels
    bottom_brightness = bottom_region.mean()
    
    # Analyze center region (optimal lip position)
    center_region = frames[:, 35:60, :]  # Center region
    center_brightness = center_region.mean()
    
    print(f"   Bottom region brightness: {bottom_brightness:.2f}")
    print(f"   Center region brightness: {center_brightness:.2f}")
    
    # Check for potential lip cutoff (very dark bottom edge)
    bottom_edge = frames[:, -5:, :].mean()
    if bottom_edge < 50:  # Very dark bottom edge suggests cutoff
        print(f"   ⚠️  WARNING: Potential lip cutoff detected (bottom edge: {bottom_edge:.1f})")
    else:
        print(f"   ✅ Good lip visibility (bottom edge: {bottom_edge:.1f})")

def main():
    """Test optimized cropping parameters."""
    print("🔧 OPTIMIZED CROP PARAMETER TEST")
    print("=" * 60)
    
    # Test video path
    video_path = "/Users/client/Desktop/LRP classifier 11.9.25/data/TRAINING SET 2.9.25/doctor 1.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Test video not found: {video_path}")
        return
    
    print(f"📹 Testing on: {video_path}")
    
    # Load frames with different crop parameters
    print("\n🔄 Loading frames with different crop parameters...")
    
    original_frames = load_frames_original_crop(video_path, target_frames=32)
    print(f"✅ Original crop: {original_frames.shape}")
    
    expanded_frames = load_frames_expanded_crop(video_path, target_frames=32)
    print(f"✅ Expanded crop: {expanded_frames.shape}")
    
    optimized_frames = load_frames_optimized_crop(video_path, target_frames=32)
    print(f"✅ Optimized crop: {optimized_frames.shape}")
    
    # Analyze lip visibility for each crop
    analyze_lip_visibility(original_frames, "ORIGINAL CROP (50% height, 33% width)")
    analyze_lip_visibility(expanded_frames, "EXPANDED CROP (55% height, 36.3% width)")
    analyze_lip_visibility(optimized_frames, "OPTIMIZED CROP (65% height, 40% width, centered)")
    
    # Create comprehensive visualization
    print("\n📊 Creating crop comparison visualization...")
    visualize_crop_comparison(original_frames, expanded_frames, optimized_frames)
    
    print("\n✅ OPTIMIZED CROP TEST COMPLETE")
    print("📁 Results saved as:")
    print("   - optimized_crop_comparison.png")
    
    print("\n🎯 CROP PARAMETER SUMMARY:")
    print("   Original V5:  50% height, 33% width, top-aligned")
    print("   Expanded:     55% height, 36.3% width, top-aligned")
    print("   Optimized:    65% height, 40% width, 10% offset (centered mouth)")
    
    print("\n🚀 EXPECTED IMPROVEMENTS:")
    print("   ✅ Complete lip area captured (including bottom lip)")
    print("   ✅ Mouth positioned in upper-center of crop")
    print("   ✅ Adequate padding around all lip edges")
    print("   ✅ No lip cutoff at bottom of frame")
    print("   ✅ Better mouth region visibility for lip-reading")

if __name__ == "__main__":
    main()
