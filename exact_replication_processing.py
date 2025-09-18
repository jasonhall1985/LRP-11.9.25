#!/usr/bin/env python3
"""
EXACT REPLICATION PROCESSING
============================

Use the EXACT same functions from process_full_dataset_gentle_v5.py that created
the successful preview_videos_fixed dataset. No modifications, no optimizations,
just exact replication of the working code.
"""

import cv2
import numpy as np
import os
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
import json

def apply_gentle_v5_preprocessing(frames):
    """
    EXACT COPY from process_full_dataset_gentle_v5.py
    Gentle V5 preprocessing with validated parameters.
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
    """
    EXACT COPY from process_full_dataset_gentle_v5.py
    Load video with bigger crop area (80% height, 60% width).
    """
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

def npy_to_mp4_ffmpeg(npy_path, output_path):
    """
    EXACT COPY from process_full_dataset_gentle_v5.py
    Convert numpy array to proper grayscale MP4 using FFmpeg.
    """
    try:
        # Load the numpy array
        frames = np.load(npy_path)
        
        # Convert from [-1, 1] back to [0, 255]
        frames_uint8 = ((frames + 1) * 127.5).astype(np.uint8)
        frames_uint8 = np.clip(frames_uint8, 0, 255)
        
        # Create temporary raw video file
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
            temp_raw_path = temp_file.name
            temp_file.write(frames_uint8.tobytes())
        
        # Use FFmpeg to convert raw grayscale to proper MP4
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '96x96',  # size
            '-pix_fmt', 'gray',  # grayscale pixel format
            '-r', '8',  # frame rate
            '-i', temp_raw_path,  # input
            '-c:v', 'libx264',  # H.264 codec
            '-pix_fmt', 'yuv420p',  # compatible pixel format
            '-vf', 'format=gray,format=yuv420p',  # ensure grayscale
            '-loglevel', 'quiet',  # suppress FFmpeg output
            str(output_path)  # output
        ]
        
        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temp file
        os.unlink(temp_raw_path)
        
        return result.returncode == 0
            
    except Exception as e:
        return False

def main():
    """Reprocess all videos using EXACT same functions as preview_videos_fixed."""
    print("🔧 EXACT REPLICATION PROCESSING")
    print("=" * 60)
    print("Using EXACT same functions from process_full_dataset_gentle_v5.py")
    print("that created the successful preview_videos_fixed dataset")
    
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
    
    print(f"\n🎯 EXACT REPLICATION PARAMETERS:")
    print(f"   • Crop: 80% height × 60% width (EXACT from preview_videos_fixed)")
    print(f"   • From top: gray[0:crop_h, crop_w_start:crop_w_end]")
    print(f"   • CLAHE: clipLimit=1.5, tileGridSize=(8,8)")
    print(f"   • Percentiles: p1, p99")
    print(f"   • Gamma: 1.02")
    print(f"   • FFmpeg conversion with exact same parameters")
    
    # Reprocess each video with EXACT same functions
    successful = 0
    failed = 0
    
    for video_info in tqdm(processed_videos, desc="EXACT replication processing"):
        try:
            source_path = Path(video_info['input_video'])
            class_name = video_info['class']
            base_name = source_path.stem.replace('_topmid', '')
            
            # Use EXACT same function as preview_videos_fixed
            frames = load_and_crop_video_bigger(str(source_path), target_frames=32)
            if frames is None:
                failed += 1
                continue
            
            # Use EXACT same preprocessing function as preview_videos_fixed
            processed_frames = apply_gentle_v5_preprocessing(frames)
            
            # EXACT same quality checks as preview_videos_fixed
            if processed_frames.shape != (32, 96, 96):
                failed += 1
                continue
            
            if not (-1.1 <= processed_frames.min() <= -0.8 and 0.8 <= processed_frames.max() <= 1.1):
                failed += 1
                continue
            
            # Save updated NPY file
            npy_filename = f"{base_name}_gentle_v5.npy"
            npy_path = target_dir / class_name / npy_filename
            np.save(npy_path, processed_frames)
            
            # Use EXACT same video conversion function as preview_videos_fixed
            preview_filename = f"{base_name}_preview.mp4"
            preview_path = preview_dir / preview_filename
            
            if npy_to_mp4_ffmpeg(npy_path, preview_path):
                successful += 1
            else:
                failed += 1
            
        except Exception as e:
            print(f"❌ Error processing {base_name}: {e}")
            failed += 1
    
    print(f"\n✅ EXACT REPLICATION COMPLETE")
    print(f"📊 Successfully processed: {successful}/{len(processed_videos)}")
    print(f"📊 Failed: {failed}")
    
    print(f"\n🎯 THESE SHOULD NOW MATCH preview_videos_fixed EXACTLY:")
    print(f"   • Same cropping: 80% height × 60% width from top")
    print(f"   • Same preprocessing: gentle V5 with exact parameters")
    print(f"   • Same video conversion: FFmpeg with exact settings")
    print(f"   • Lips should be positioned identically to reference")

if __name__ == "__main__":
    main()
