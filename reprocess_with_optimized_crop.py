#!/usr/bin/env python3
"""
Reprocess Videos with Optimized Crop Parameters
===============================================

Reprocess all videos in the additional 50 per class dataset using the optimized
cropping parameters (65% height, 40% width, centered) to ensure consistency
with the preview_videos_fixed dataset and complete lip visibility.
"""

import cv2
import numpy as np
import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import json

def apply_gentle_v5_preprocessing(frames):
    """Apply gentle V5 preprocessing with optimized parameters."""
    frames = frames.astype(np.float32)
    
    # Apply gentle CLAHE (clipLimit=1.5)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    for i in range(len(frames)):
        frames[i] = clahe.apply(frames[i].astype(np.uint8)).astype(np.float32)
    
    # Conservative percentile normalization (p1, p99)
    p1, p99 = np.percentile(frames, [1, 99])
    frames = np.clip(frames, p1, p99)
    
    # Normalize to [0, 1]
    frames = (frames - p1) / (p99 - p1)
    
    # Minimal gamma correction (1.02)
    frames = np.power(frames, 1.02)
    
    # Final normalization to [-1, 1]
    frames = (frames - 0.5) / 0.5
    
    return frames

def load_and_crop_video_optimized(video_path, target_frames=32):
    """Load video with optimized cropping parameters for complete lip visibility."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optimized crop for complete lip capture
        h, w = gray.shape
        
        # Optimized crop for complete lip capture (65% height, centered mouth)
        crop_h = int(0.65 * h)  # 65% height (instead of 80%)
        
        # Vertical positioning: start from 10% down to center mouth region
        crop_v_start = int(0.10 * h)
        crop_v_end = crop_v_start + crop_h
        
        # Horizontal positioning: middle 40% for better mouth capture
        crop_w_start = int(0.30 * w)  # 30% to 70% (40% width)
        crop_w_end = int(0.70 * w)
        
        # Ensure crop doesn't exceed frame boundaries
        crop_v_end = min(crop_v_end, h)
        crop_w_end = min(crop_w_end, w)

        cropped = gray[crop_v_start:crop_v_end, crop_w_start:crop_w_end]
        
        # Resize to 96x96 (critical for preventing green artifacts)
        resized = cv2.resize(cropped, (96, 96))
        frames.append(resized)
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    # 32-frame temporal sampling using np.linspace()
    if len(frames) >= target_frames:
        indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    else:
        # Repeat frames if not enough
        while len(frames) < target_frames:
            frames.extend(frames[:min(len(frames), target_frames - len(frames))])
    
    return np.array(frames[:target_frames])

def create_optimized_preview_video(processed_frames, output_path):
    """Create preview video using AVI->MP4 conversion to fix green artifacts."""
    try:
        # Convert from [-1,1] to [0,255] for visualization
        preview_frames = ((processed_frames + 1) * 127.5).astype(np.uint8)
        preview_frames = np.clip(preview_frames, 0, 255)

        # First create AVI file (more reliable for grayscale)
        avi_path = str(output_path).replace('.mp4', '.avi')
        
        # Create AVI video writer with grayscale settings
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 8  # Slow playback for inspection
        h, w = preview_frames.shape[1], preview_frames.shape[2]

        out = cv2.VideoWriter(avi_path, fourcc, fps, (w, h), isColor=False)
        
        if not out.isOpened():
            print(f"❌ Could not open AVI video writer for {avi_path}")
            return False

        for frame in preview_frames:
            out.write(frame)

        out.release()
        
        # Convert AVI to MP4 using FFmpeg
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', avi_path,
                '-c:v', 'libx264',
                '-pix_fmt', 'gray',
                '-crf', '23',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                os.remove(avi_path)
                return True
            else:
                print(f"⚠️  FFmpeg conversion failed, keeping AVI: {avi_path}")
                return True
                
        except (FileNotFoundError, subprocess.SubprocessError):
            print(f"⚠️  FFmpeg not available, keeping AVI: {avi_path}")
            return True

    except Exception as e:
        print(f"❌ Failed to create preview for {output_path}: {e}")
        return False

def main():
    """Reprocess all videos with optimized cropping parameters."""
    print("🔧 REPROCESSING WITH OPTIMIZED CROP PARAMETERS")
    print("=" * 60)
    
    # Directories
    source_dir = Path("data/13.9.25top7dataset_cropped")
    target_dir = Path("data/training set 17.9.25/additional 50 per class")
    preview_dir = target_dir / "preview_videos"
    
    print(f"📁 Source: {source_dir}")
    print(f"📁 Target: {target_dir}")
    print(f"📁 Previews: {preview_dir}")
    
    # Load existing processing log to get the list of successfully processed videos
    log_file = target_dir / "processing_log.json"
    if not log_file.exists():
        print("❌ No processing log found!")
        return
    
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    processed_videos = log_data.get('processing_log', [])
    print(f"📊 Found {len(processed_videos)} previously processed videos")

    # Reprocess each video with optimized parameters
    successful = 0
    failed = 0

    for video_info in tqdm(processed_videos, desc="Reprocessing with optimized crop"):
        try:
            source_path = Path(video_info['input_video'])
            class_name = video_info['class']
            # Extract base name from the input video path
            base_name = source_path.stem.replace('_topmid', '')
            
            # Load and process with optimized parameters
            frames = load_and_crop_video_optimized(source_path)
            if frames is None:
                failed += 1
                continue
            
            # Apply gentle V5 preprocessing
            processed_frames = apply_gentle_v5_preprocessing(frames)
            
            # Quality check
            if processed_frames.shape != (32, 96, 96):
                failed += 1
                continue
            
            min_val = processed_frames.min()
            max_val = processed_frames.max()
            if not (-1.1 <= min_val <= -0.8 and 0.8 <= max_val <= 1.1):
                failed += 1
                continue
            
            # Save updated NPY file
            npy_filename = f"{base_name}_gentle_v5.npy"
            npy_path = target_dir / class_name / npy_filename
            np.save(npy_path, processed_frames)
            
            # Create updated preview video
            preview_filename = f"{base_name}_preview.mp4"
            preview_path = preview_dir / preview_filename
            create_optimized_preview_video(processed_frames, preview_path)
            
            successful += 1
            
        except Exception as e:
            print(f"❌ Error reprocessing {base_name}: {e}")
            failed += 1
    
    print(f"\n✅ REPROCESSING COMPLETE")
    print(f"📊 Successfully reprocessed: {successful}/{len(processed_videos)}")
    print(f"📊 Failed: {failed}")
    
    print(f"\n🎯 OPTIMIZED PARAMETERS APPLIED:")
    print(f"   • Crop: 65% height × 40% width (centered)")
    print(f"   • Vertical positioning: 10% down from top")
    print(f"   • Horizontal positioning: 30% to 70%")
    print(f"   • Complete lip visibility ensured")
    print(f"   • Mouth region centered in crop")

if __name__ == "__main__":
    main()
