#!/usr/bin/env python3
"""
Process one more video to get 10 total
"""

import cv2
import numpy as np
from pathlib import Path

def apply_gentle_v5_preprocessing(frames):
    """Gentle V5 preprocessing."""
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

def main():
    """Process one more video."""
    print("🔧 PROCESSING ONE MORE VIDEO")
    print("=" * 40)
    
    # Try phone 1.mp4
    input_path = Path("data/TRAINING SET 2.9.25/phone 1.mp4")
    output_dir = Path("data/training set 17.9.25")
    output_path = output_dir / "phone 1_gentle_v5.npy"
    
    if not input_path.exists():
        print(f"❌ Video not found: {input_path}")
        return
    
    print(f"📹 Processing: {input_path.name}")
    
    try:
        # Load and crop video
        frames = load_and_crop_video_bigger(str(input_path), target_frames=32)
        
        if frames is None:
            print(f"❌ Failed to load: {input_path.name}")
            return
        
        # Apply gentle V5 preprocessing
        processed_frames = apply_gentle_v5_preprocessing(frames)
        
        # Save processed video
        np.save(output_path, processed_frames)
        
        # Print stats
        extreme_low = np.sum(processed_frames < -0.9) / processed_frames.size * 100
        extreme_high = np.sum(processed_frames > 0.9) / processed_frames.size * 100
        print(f"✅ {input_path.name} -> {output_path.name}")
        print(f"   Shape: {processed_frames.shape}")
        print(f"   Range: [{processed_frames.min():.3f}, {processed_frames.max():.3f}]")
        print(f"   Mean: {processed_frames.mean():.3f}")
        print(f"   Extreme values: {extreme_low+extreme_high:.2f}%")
        
        print(f"\n🎯 NOW YOU HAVE 10 PROCESSED VIDEOS!")
        
    except Exception as e:
        print(f"❌ Error processing {input_path.name}: {e}")

if __name__ == "__main__":
    main()
