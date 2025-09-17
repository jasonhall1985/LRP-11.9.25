#!/usr/bin/env python3
"""
Process Full Dataset - Apply gentle V5 preprocessing to all videos with balanced classes
"""

import cv2
import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def apply_gentle_v5_preprocessing(frames):
    """
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

def get_class_from_filename(filename):
    """Extract class name from filename."""
    filename_lower = filename.lower()
    if 'doctor' in filename_lower:
        return 'doctor'
    elif 'glasses' in filename_lower:
        return 'glasses'
    elif 'help' in filename_lower:
        return 'help'
    elif 'phone' in filename_lower:
        return 'phone'
    elif 'pillow' in filename_lower:
        return 'pillow'
    else:
        return 'unknown'

def balance_dataset(video_files_by_class, target_per_class=None):
    """Balance dataset to have equal samples per class."""
    if not target_per_class:
        # Use the minimum class count
        target_per_class = min(len(files) for files in video_files_by_class.values())
    
    balanced_files = {}
    for class_name, files in video_files_by_class.items():
        if len(files) >= target_per_class:
            # Take first N files
            balanced_files[class_name] = files[:target_per_class]
        else:
            # Use all available files
            balanced_files[class_name] = files
            print(f"⚠️  Warning: {class_name} has only {len(files)} videos (target: {target_per_class})")
    
    return balanced_files

def npy_to_mp4_ffmpeg(npy_path, output_path):
    """Convert numpy array to proper grayscale MP4 using FFmpeg."""
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
    """Process complete dataset with gentle V5 preprocessing."""
    print("🔧 PROCESS FULL DATASET - GENTLE V5 PREPROCESSING")
    print("=" * 70)
    
    # Directories
    input_dir = Path("data/TRAINING SET 2.9.25")
    output_dir = Path("data/training set 17.9.25")
    preview_dir = Path("data/training set 17.9.25/preview_videos_fixed")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"📁 Preview: {preview_dir}")
    
    # Get all video files and organize by class
    video_extensions = ['*.mp4', '*.avi', '*.mov']
    all_video_files = []
    for ext in video_extensions:
        all_video_files.extend(list(input_dir.glob(ext)))
    
    if not all_video_files:
        print("❌ No video files found!")
        return
    
    # Organize by class
    video_files_by_class = defaultdict(list)
    for video_file in all_video_files:
        class_name = get_class_from_filename(video_file.name)
        if class_name != 'unknown':
            video_files_by_class[class_name].append(video_file)
    
    print(f"\n📊 DATASET ANALYSIS:")
    total_videos = 0
    for class_name, files in video_files_by_class.items():
        print(f"   {class_name}: {len(files)} videos")
        total_videos += len(files)
    print(f"   Total: {total_videos} videos")
    
    # Balance dataset
    print(f"\n⚖️  BALANCING DATASET:")
    balanced_files = balance_dataset(video_files_by_class)
    
    balanced_total = 0
    for class_name, files in balanced_files.items():
        print(f"   {class_name}: {len(files)} videos (balanced)")
        balanced_total += len(files)
    print(f"   Balanced total: {balanced_total} videos")
    
    # Process all videos
    print(f"\n🚀 PROCESSING WITH GENTLE V5:")
    print("   • Bigger crop: 80% height × 60% width")
    print("   • Gentle CLAHE: clipLimit=1.5")
    print("   • Conservative percentiles: p1, p99")
    print("   • Minimal gamma: 1.02")
    print("   • 32 frames exactly")
    print("   • Consistent lighting & grayscale")
    
    successful_npy = 0
    successful_mp4 = 0
    failed = 0
    
    # Process each class
    for class_name, video_files in balanced_files.items():
        print(f"\n📹 Processing {class_name} class ({len(video_files)} videos):")
        
        for video_file in tqdm(video_files, desc=f"Processing {class_name}"):
            try:
                # Load and crop video
                frames = load_and_crop_video_bigger(str(video_file), target_frames=32)
                
                if frames is None:
                    failed += 1
                    continue
                
                # Apply gentle V5 preprocessing
                processed_frames = apply_gentle_v5_preprocessing(frames)
                
                # Verify quality
                if processed_frames.shape != (32, 96, 96):
                    failed += 1
                    continue
                
                if not (-1.1 <= processed_frames.min() <= -0.8 and 0.8 <= processed_frames.max() <= 1.1):
                    failed += 1
                    continue
                
                # Save NPY file
                npy_filename = f"{video_file.stem}_gentle_v5.npy"
                npy_path = output_dir / npy_filename
                np.save(npy_path, processed_frames)
                successful_npy += 1
                
                # Convert to MP4 preview
                mp4_filename = f"{video_file.stem}_preview.mp4"
                mp4_path = preview_dir / mp4_filename
                
                if npy_to_mp4_ffmpeg(npy_path, mp4_path):
                    successful_mp4 += 1
                
            except Exception as e:
                failed += 1
    
    print(f"\n✅ FULL DATASET PROCESSING COMPLETE")
    print(f"📊 FINAL RESULTS:")
    print(f"   ✅ NPY files created: {successful_npy}")
    print(f"   ✅ MP4 previews created: {successful_mp4}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📁 NPY files: {output_dir}")
    print(f"   📁 Preview videos: {preview_dir}")

if __name__ == "__main__":
    main()
