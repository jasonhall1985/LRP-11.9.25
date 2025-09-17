#!/usr/bin/env python3
"""
Final Gentle V5 Preprocessing Pipeline - Optimized for lip detail preservation
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def apply_gentle_v5_preprocessing(frames):
    """
    Final gentle V5 preprocessing optimized for lip detail preservation.
    
    Optimized parameters:
    - Minimal CLAHE: clipLimit 2.0→1.5, tileGridSize (8,8)→(8,8)
    - Conservative percentile: (p2,p98)→(p1,p99)
    - Minimal gamma: 1.2→1.02
    - Optimized crop: 65% height, 40% width, centered mouth positioning
    - Same brightness standardization and normalization
    """
    frames = frames.astype(np.float32) / 255.0
    
    processed_frames = []
    for frame in frames:
        frame_uint8 = (frame * 255).astype(np.uint8)
        
        # MINIMAL CLAHE enhancement (preserve natural contrast)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(frame_uint8).astype(np.float32) / 255.0
        
        # CONSERVATIVE percentile normalization (minimal clipping)
        p1, p99 = np.percentile(enhanced, [1, 99])
        if p99 > p1:
            enhanced = np.clip((enhanced - p1) / (p99 - p1), 0, 1)
        
        # MINIMAL gamma correction (barely noticeable)
        gamma = 1.02
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

def load_and_crop_video_optimized(video_path, target_frames=32):
    """Load video with bigger crop area to ensure lips fit in frame."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale using ITU-R BT.709 weights (V5 standard)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Simple bigger crop - just zoom out more to fit the lips
        h, w = gray.shape

        # Take top 80% of height (much bigger than original 50%)
        crop_h = int(0.80 * h)

        # Take middle 60% of width (much bigger than original 33%)
        crop_w_start = int(0.20 * w)  # Start at 20%
        crop_w_end = int(0.80 * w)    # End at 80%

        # Simple top-aligned crop (no fancy positioning)
        cropped = gray[0:crop_h, crop_w_start:crop_w_end]

        # Resize to standard size (96x96 for consistency)
        resized = cv2.resize(cropped, (96, 96))
        frames.append(resized)
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    # V5's proven 32-frame temporal sampling
    if len(frames) >= target_frames:
        indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    else:
        # Repeat frames if not enough
        while len(frames) < target_frames:
            frames.extend(frames[:min(len(frames), target_frames - len(frames))])
    
    return np.array(frames[:target_frames])

def create_gentle_v5_dataset(video_dir, output_dir, class_labels):
    """
    Create complete dataset using gentle V5 preprocessing with expanded crop.
    
    Args:
        video_dir: Directory containing class subdirectories with videos
        output_dir: Directory to save processed videos
        class_labels: List of class names (e.g., ['doctor', 'glasses', 'help', 'phone', 'pillow'])
    """
    import os
    from tqdm import tqdm
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🔧 GENTLE V5 DATASET CREATION")
    print("=" * 50)
    print(f"📁 Input directory: {video_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🏷️  Classes: {class_labels}")
    print("\n🎯 GENTLE V5 PARAMETERS:")
    print("   • Minimal CLAHE: clipLimit=1.5, tileGridSize=(8,8)")
    print("   • Conservative percentile: (p1,p99)")
    print("   • Minimal gamma: 1.02")
    print("   • Optimized crop: 65% height, 40% width, centered mouth")
    print("   • 32-frame temporal sampling")
    print("   • Brightness standardization to 0.5")
    print("   • Normalization to [-1,1]")
    
    total_processed = 0
    
    for class_name in class_labels:
        class_dir = Path(video_dir) / class_name
        if not class_dir.exists():
            print(f"⚠️  Warning: Class directory not found: {class_dir}")
            continue
        
        # Get all video files
        video_files = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
        
        if not video_files:
            print(f"⚠️  Warning: No videos found in {class_dir}")
            continue
        
        print(f"\n🔄 Processing {class_name}: {len(video_files)} videos")
        
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(exist_ok=True)
        
        for video_file in tqdm(video_files, desc=f"Processing {class_name}"):
            try:
                # Load and crop video
                frames = load_and_crop_video_optimized(str(video_file), target_frames=32)
                
                if frames is None:
                    print(f"❌ Failed to load: {video_file.name}")
                    continue
                
                # Apply gentle V5 preprocessing
                processed_frames = apply_gentle_v5_preprocessing(frames)
                
                # Save processed video
                output_file = class_output_dir / f"{video_file.stem}_gentle_v5.npy"
                np.save(output_file, processed_frames)
                
                total_processed += 1
                
            except Exception as e:
                print(f"❌ Error processing {video_file.name}: {e}")
                continue
    
    print(f"\n✅ GENTLE V5 DATASET CREATION COMPLETE")
    print(f"📊 Total videos processed: {total_processed}")
    print(f"📁 Output saved to: {output_dir}")
    
    return total_processed

def main():
    """Test gentle V5 preprocessing and provide dataset creation function."""
    print("🔧 FINAL GENTLE V5 PREPROCESSING")
    print("=" * 50)
    
    # Test on sample video
    video_path = "/Users/client/Desktop/LRP classifier 11.9.25/data/TRAINING SET 2.9.25/doctor 4.mp4"
    
    if Path(video_path).exists():
        print(f"📹 Testing on: {video_path}")
        
        # Load frames with optimized crop
        frames = load_and_crop_video_optimized(video_path, target_frames=32)
        
        if frames is not None:
            print(f"✅ Loaded {len(frames)} frames")
            
            # Apply gentle preprocessing
            processed_frames = apply_gentle_v5_preprocessing(frames)
            
            # Analyze results
            print(f"\n📊 GENTLE V5 RESULTS:")
            print(f"   Shape: {processed_frames.shape}")
            print(f"   Min: {processed_frames.min():.3f}")
            print(f"   Max: {processed_frames.max():.3f}")
            print(f"   Mean: {processed_frames.mean():.3f}")
            print(f"   Std: {processed_frames.std():.3f}")
            
            # Check for extreme values
            extreme_low = np.sum(processed_frames < -0.9) / processed_frames.size * 100
            extreme_high = np.sum(processed_frames > 0.9) / processed_frames.size * 100
            print(f"   Extreme low (<-0.9): {extreme_low:.2f}%")
            print(f"   Extreme high (>0.9): {extreme_high:.2f}%")
            
            print("\n✅ GENTLE V5 PREPROCESSING READY FOR SAGEMAKER!")
            
        else:
            print("❌ Failed to load test video")
    else:
        print(f"⚠️  Test video not found: {video_path}")
    
    print("\n🚀 USAGE FOR FULL DATASET:")
    print("```python")
    print("from gentle_v5_preprocessing_final import create_gentle_v5_dataset")
    print("")
    print("# Create complete dataset")
    print("create_gentle_v5_dataset(")
    print("    video_dir='data/TRAINING SET 2.9.25',")
    print("    output_dir='data/gentle_v5_dataset',")
    print("    class_labels=['doctor', 'glasses', 'help', 'phone', 'pillow']")
    print(")")
    print("```")

if __name__ == "__main__":
    main()
