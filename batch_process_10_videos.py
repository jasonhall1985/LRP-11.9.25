#!/usr/bin/env python3
"""
Batch Process 10 Videos - Apply gentle V5 preprocessing with bigger crop
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def apply_gentle_v5_preprocessing(frames):
    """
    Gentle V5 preprocessing with bigger crop for lip detail preservation.
    """
    frames = frames.astype(np.float32) / 255.0
    
    processed_frames = []
    for frame in frames:
        frame_uint8 = (frame * 255).astype(np.uint8)
        
        # GENTLE CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(frame_uint8).astype(np.float32) / 255.0
        
        # CONSERVATIVE percentile normalization
        p1, p99 = np.percentile(enhanced, [1, 99])
        if p99 > p1:
            enhanced = np.clip((enhanced - p1) / (p99 - p1), 0, 1)
        
        # MINIMAL gamma correction
        gamma = 1.02
        enhanced = np.power(enhanced, 1.0 / gamma)
        
        # Brightness standardization
        target_brightness = 0.5
        current_brightness = np.mean(enhanced)
        if current_brightness > 0:
            brightness_factor = target_brightness / current_brightness
            enhanced = np.clip(enhanced * brightness_factor, 0, 1)
        
        processed_frames.append(enhanced)
    
    frames = np.array(processed_frames)
    frames = (frames - 0.5) / 0.5  # Normalize to [-1, 1]
    
    return frames

def load_and_crop_video_bigger(video_path, target_frames=32):
    """Load video with bigger crop area (80% height, 60% width)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Bigger crop - zoom out more to fit lips
        h, w = gray.shape
        crop_h = int(0.80 * h)  # 80% height
        crop_w_start = int(0.20 * w)  # 60% width (20% to 80%)
        crop_w_end = int(0.80 * w)
        
        cropped = gray[0:crop_h, crop_w_start:crop_w_end]
        resized = cv2.resize(cropped, (96, 96))
        frames.append(resized)
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    # Ensure exactly 32 frames using temporal sampling
    if len(frames) >= target_frames:
        indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    else:
        # Repeat frames if not enough
        while len(frames) < target_frames:
            frames.extend(frames[:min(len(frames), target_frames - len(frames))])
    
    return np.array(frames[:target_frames])

def process_video(input_path, output_path):
    """Process a single video with gentle V5 preprocessing."""
    try:
        # Load and crop video
        frames = load_and_crop_video_bigger(str(input_path), target_frames=32)
        
        if frames is None:
            print(f"❌ Failed to load: {input_path.name}")
            return False
        
        # Apply gentle V5 preprocessing
        processed_frames = apply_gentle_v5_preprocessing(frames)
        
        # Verify output quality
        if processed_frames.shape != (32, 96, 96):
            print(f"❌ Wrong shape {processed_frames.shape}: {input_path.name}")
            return False
        
        if not (-1.1 <= processed_frames.min() <= -0.9 and 0.9 <= processed_frames.max() <= 1.1):
            print(f"❌ Wrong range [{processed_frames.min():.3f}, {processed_frames.max():.3f}]: {input_path.name}")
            return False
        
        # Save processed video
        np.save(output_path, processed_frames)
        
        # Print stats
        extreme_low = np.sum(processed_frames < -0.9) / processed_frames.size * 100
        extreme_high = np.sum(processed_frames > 0.9) / processed_frames.size * 100
        print(f"✅ {input_path.name} -> {output_path.name}")
        print(f"   Shape: {processed_frames.shape}, Range: [{processed_frames.min():.3f}, {processed_frames.max():.3f}]")
        print(f"   Mean: {processed_frames.mean():.3f}, Extreme: {extreme_low+extreme_high:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {input_path.name}: {e}")
        return False

def main():
    """Process 10 videos from training set."""
    print("🔧 BATCH PROCESS 10 VIDEOS")
    print("=" * 50)
    
    # Input and output directories
    input_dir = Path("data/TRAINING SET 2.9.25")
    output_dir = Path("data/training set 17.9.25")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Get all video files
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(list(input_dir.glob(ext)))
    
    if not video_files:
        print("❌ No video files found!")
        return
    
    print(f"📹 Found {len(video_files)} videos")
    
    # Process first 10 videos
    videos_to_process = video_files[:10]
    
    print(f"\n🔄 Processing first 10 videos:")
    for i, video_file in enumerate(videos_to_process, 1):
        print(f"   {i}. {video_file.name}")
    
    print(f"\n🚀 PROCESSING WITH GENTLE V5:")
    print("   • Bigger crop: 80% height × 60% width")
    print("   • Gentle CLAHE: clipLimit=1.5")
    print("   • Conservative percentiles: p1, p99")
    print("   • Minimal gamma: 1.02")
    print("   • 32 frames exactly")
    print("   • Consistent lighting & grayscale")
    
    successful = 0
    failed = 0
    
    for i, video_file in enumerate(videos_to_process, 1):
        print(f"\n📹 Processing {i}/10: {video_file.name}")
        
        # Create output filename
        output_filename = f"{video_file.stem}_gentle_v5.npy"
        output_path = output_dir / output_filename
        
        # Process video
        if process_video(video_file, output_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\n✅ BATCH PROCESSING COMPLETE")
    print(f"📊 Results:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📁 Output directory: {output_dir}")
    
    if successful > 0:
        print(f"\n🎯 All processed videos have:")
        print(f"   • Exactly 32 frames")
        print(f"   • Shape: (32, 96, 96)")
        print(f"   • Range: [-1, 1]")
        print(f"   • Consistent gentle preprocessing")
        print(f"   • Bigger crop with centered lips")

if __name__ == "__main__":
    main()
