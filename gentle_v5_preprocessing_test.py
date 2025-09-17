#!/usr/bin/env python3
"""
Gentle V5 Preprocessing Pipeline - Improved version with expanded crop and gentler processing
Test implementation on doctor 1.mp4 to validate lip detail preservation
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def apply_original_v5_preprocessing(frames):
    """Original V5 preprocessing for comparison."""
    frames = frames.astype(np.float32) / 255.0
    
    processed_frames = []
    for frame in frames:
        frame_uint8 = (frame * 255).astype(np.uint8)
        
        # Original V5 CLAHE (harsh)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(frame_uint8).astype(np.float32) / 255.0
        
        # Original V5 percentile normalization (harsh)
        p2, p98 = np.percentile(enhanced, [2, 98])
        if p98 > p2:
            enhanced = np.clip((enhanced - p2) / (p98 - p2), 0, 1)
        
        # Original V5 gamma correction (strong)
        gamma = 1.2
        enhanced = np.power(enhanced, 1.0 / gamma)
        
        # Original V5 brightness standardization
        target_brightness = 0.5
        current_brightness = np.mean(enhanced)
        if current_brightness > 0:
            brightness_factor = target_brightness / current_brightness
            enhanced = np.clip(enhanced * brightness_factor, 0, 1)
        
        processed_frames.append(enhanced)
    
    frames = np.array(processed_frames)
    frames = (frames - 0.5) / 0.5  # Normalize to [-1, 1]
    
    return frames

def apply_gentle_v5_preprocessing(frames):
    """
    Gentle V5 preprocessing with improved parameters for lip detail preservation.

    Key improvements:
    - Gentler CLAHE: clipLimit 2.0→1.2, tileGridSize (8,8)→(10,10)
    - Softer percentile normalization: (p2,p98)→(p3,p97)
    - Reduced gamma correction: 1.2→1.1
    - Same brightness standardization and normalization
    """
    frames = frames.astype(np.float32) / 255.0

    processed_frames = []
    for frame in frames:
        frame_uint8 = (frame * 255).astype(np.uint8)

        # GENTLE CLAHE enhancement (less aggressive than original)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(10, 10))
        enhanced = clahe.apply(frame_uint8).astype(np.float32) / 255.0

        # GENTLE percentile normalization (less aggressive clipping)
        p3, p97 = np.percentile(enhanced, [3, 97])
        if p97 > p3:
            enhanced = np.clip((enhanced - p3) / (p97 - p3), 0, 1)

        # GENTLE gamma correction (subtle adjustment)
        gamma = 1.1
        enhanced = np.power(enhanced, 1.0 / gamma)

        # Same brightness standardization as V5
        target_brightness = 0.5
        current_brightness = np.mean(enhanced)
        if current_brightness > 0:
            brightness_factor = target_brightness / current_brightness
            enhanced = np.clip(enhanced * brightness_factor, 0, 1)

        processed_frames.append(enhanced)

    frames = np.array(processed_frames)
    frames = (frames - 0.5) / 0.5  # Normalize to [-1, 1]

    return frames

def load_and_crop_video(video_path, target_frames=32, crop_type='original'):
    """
    Load video and apply cropping.
    
    crop_type:
    - 'original': V5's original ICU crop (top 50%, middle 33%)
    - 'expanded': New expanded crop (top 55%, middle 36.3%)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale using ITU-R BT.709 weights (V5 standard)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply cropping based on type
        h, w = gray.shape
        
        if crop_type == 'original':
            # Original V5 ICU-style crop
            crop_h = int(0.5 * h)  # Top 50%
            crop_w_start = int(0.335 * w)  # Middle 33%
            crop_w_end = int(0.665 * w)
        else:  # expanded
            # New expanded crop
            crop_h = int(0.55 * h)  # Top 55% (10% more height)
            crop_w_start = int(0.3185 * w)  # Middle 36.3% (centered)
            crop_w_end = int(0.6815 * w)
        
        cropped = gray[0:crop_h, crop_w_start:crop_w_end]
        
        # Resize to standard size (96x96 for consistency)
        resized = cv2.resize(cropped, (96, 96))
        frames.append(resized)
    
    cap.release()
    
    if len(frames) == 0:
        print("❌ No frames loaded from video!")
        return None
    
    print(f"📹 Loaded {len(frames)} frames from video")
    
    # V5's proven 32-frame temporal sampling
    if len(frames) >= target_frames:
        indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    else:
        # Repeat frames if not enough
        while len(frames) < target_frames:
            frames.extend(frames[:min(len(frames), target_frames - len(frames))])
    
    return np.array(frames[:target_frames])

def visualize_preprocessing_comparison(original_frames, original_v5_frames, gentle_v5_frames, frame_indices=[0, 8, 16, 24]):
    """Visualize comparison between original, V5, and gentle V5 preprocessing."""
    
    fig, axes = plt.subplots(len(frame_indices), 3, figsize=(15, 4*len(frame_indices)))
    fig.suptitle('Preprocessing Comparison: Original vs V5 vs Gentle V5', fontsize=16)
    
    for i, frame_idx in enumerate(frame_indices):
        # Original frame
        axes[i, 0].imshow(original_frames[frame_idx], cmap='gray', vmin=0, vmax=255)
        axes[i, 0].set_title(f'Original Frame {frame_idx}')
        axes[i, 0].axis('off')
        
        # Original V5 processed (convert from [-1,1] to [0,1] for display)
        v5_display = (original_v5_frames[frame_idx] + 1) / 2
        axes[i, 1].imshow(v5_display, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Original V5 Frame {frame_idx}')
        axes[i, 1].axis('off')
        
        # Gentle V5 processed (convert from [-1,1] to [0,1] for display)
        gentle_display = (gentle_v5_frames[frame_idx] + 1) / 2
        axes[i, 2].imshow(gentle_display, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Gentle V5 Frame {frame_idx}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show for non-interactive

def analyze_preprocessing_stats(frames, name):
    """Analyze preprocessing statistics."""
    print(f"\n📊 {name} Statistics:")
    print(f"   Shape: {frames.shape}")
    print(f"   Min: {frames.min():.3f}")
    print(f"   Max: {frames.max():.3f}")
    print(f"   Mean: {frames.mean():.3f}")
    print(f"   Std: {frames.std():.3f}")
    
    # Check for harsh artifacts (too many extreme values)
    extreme_low = np.sum(frames < -0.9) / frames.size * 100
    extreme_high = np.sum(frames > 0.9) / frames.size * 100
    print(f"   Extreme low values (<-0.9): {extreme_low:.2f}%")
    print(f"   Extreme high values (>0.9): {extreme_high:.2f}%")

def test_crop_comparison(video_path):
    """Test original vs expanded crop to show lip preservation."""
    print("\n🔍 TESTING CROP COMPARISON")
    
    # Load with original crop
    original_crop_frames = load_and_crop_video(video_path, target_frames=4, crop_type='original')
    
    # Load with expanded crop
    expanded_crop_frames = load_and_crop_video(video_path, target_frames=4, crop_type='expanded')
    
    if original_crop_frames is None or expanded_crop_frames is None:
        print("❌ Failed to load frames for crop comparison")
        return
    
    # Visualize crop comparison
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Crop Comparison: Original (50% height, 33% width) vs Expanded (55% height, 36.3% width)', fontsize=14)
    
    for i in range(4):
        # Original crop
        axes[0, i].imshow(original_crop_frames[i], cmap='gray')
        axes[0, i].set_title(f'Original Crop - Frame {i*8}')
        axes[0, i].axis('off')
        
        # Expanded crop
        axes[1, i].imshow(expanded_crop_frames[i], cmap='gray')
        axes[1, i].set_title(f'Expanded Crop - Frame {i*8}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('crop_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show for non-interactive

    print("✅ Crop comparison saved as 'crop_comparison.png'")

def main():
    """Main test function for gentle V5 preprocessing."""
    print("🔧 GENTLE V5 PREPROCESSING TEST")
    print("=" * 50)
    
    # Test video path
    video_path = "/Users/client/Desktop/LRP classifier 11.9.25/data/TRAINING SET 2.9.25/doctor 1.mp4"
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        print("Please check the path and try again.")
        return
    
    print(f"📹 Testing on: {video_path}")
    
    # Test crop comparison first
    test_crop_comparison(video_path)
    
    # Load frames with expanded crop for preprocessing test
    print("\n🔄 Loading frames with expanded crop...")
    original_frames = load_and_crop_video(video_path, target_frames=32, crop_type='expanded')
    
    if original_frames is None:
        print("❌ Failed to load video frames")
        return
    
    print(f"✅ Loaded {len(original_frames)} frames")
    
    # Apply original V5 preprocessing
    print("\n🔄 Applying original V5 preprocessing...")
    original_v5_frames = apply_original_v5_preprocessing(original_frames.copy())
    analyze_preprocessing_stats(original_v5_frames, "Original V5")
    
    # Apply gentle V5 preprocessing
    print("\n🔄 Applying gentle V5 preprocessing...")
    gentle_v5_frames = apply_gentle_v5_preprocessing(original_frames.copy())
    analyze_preprocessing_stats(gentle_v5_frames, "Gentle V5")
    
    # Visualize comparison
    print("\n📊 Creating visualization...")
    visualize_preprocessing_comparison(
        original_frames, 
        original_v5_frames, 
        gentle_v5_frames,
        frame_indices=[0, 8, 16, 24]
    )
    
    print("\n✅ GENTLE V5 PREPROCESSING TEST COMPLETE")
    print("📁 Results saved as:")
    print("   - preprocessing_comparison.png")
    print("   - crop_comparison.png")
    
    # Validation summary
    print("\n🎯 VALIDATION SUMMARY:")
    print("✅ Gentle CLAHE: clipLimit 2.0→1.2, tileGridSize (8,8)→(10,10)")
    print("✅ Softer percentile: (p2,p98)→(p3,p97)")
    print("✅ Reduced gamma: 1.2→1.1")
    print("✅ Expanded crop: 50%→55% height, 33%→36.3% width")
    print("✅ Preserved brightness standardization and [-1,1] normalization")
    
    # Check for improvements
    original_extreme = np.sum((original_v5_frames < -0.9) | (original_v5_frames > 0.9)) / original_v5_frames.size * 100
    gentle_extreme = np.sum((gentle_v5_frames < -0.9) | (gentle_v5_frames > 0.9)) / gentle_v5_frames.size * 100
    
    print(f"\n📈 IMPROVEMENT METRICS:")
    print(f"   Extreme values reduced: {original_extreme:.2f}% → {gentle_extreme:.2f}%")
    print(f"   Improvement: {original_extreme - gentle_extreme:.2f}% fewer harsh artifacts")
    
    if gentle_extreme < original_extreme:
        print("🎉 SUCCESS: Gentle preprocessing reduces harsh artifacts!")
    else:
        print("⚠️  WARNING: May need further parameter adjustment")

if __name__ == "__main__":
    main()
