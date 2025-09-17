#!/usr/bin/env python3
"""
Fix Video Conversion - Use FFmpeg approach for proper grayscale MP4
"""

import numpy as np
import cv2
import subprocess
import tempfile
import os
from pathlib import Path

def npy_to_mp4_fixed(npy_path, output_path):
    """Convert numpy array to proper grayscale MP4 using FFmpeg."""
    try:
        # Load the numpy array
        frames = np.load(npy_path)
        print(f"📹 Loaded {npy_path.name}: {frames.shape}")
        
        # Convert from [-1, 1] back to [0, 255]
        frames_uint8 = ((frames + 1) * 127.5).astype(np.uint8)
        frames_uint8 = np.clip(frames_uint8, 0, 255)
        
        # Create temporary raw video file
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
            temp_raw_path = temp_file.name
            frames_uint8.tobytes()
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
            str(output_path)  # output
        ]
        
        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temp file
        os.unlink(temp_raw_path)
        
        if result.returncode == 0:
            print(f"✅ Saved video: {output_path.name}")
            print(f"   Frames: {len(frames)}, Size: 96x96, FPS: 8")
            return True
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error converting {npy_path.name}: {e}")
        return False

def npy_to_mp4_opencv_fixed(npy_path, output_path):
    """Fallback method using OpenCV with proper grayscale handling."""
    try:
        # Load the numpy array
        frames = np.load(npy_path)
        print(f"📹 Loaded {npy_path.name}: {frames.shape}")
        
        # Convert from [-1, 1] back to [0, 255]
        frames_uint8 = ((frames + 1) * 127.5).astype(np.uint8)
        frames_uint8 = np.clip(frames_uint8, 0, 255)
        
        # Create video writer with specific settings for grayscale
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 8.0, (96, 96), isColor=False)
        
        if not out.isOpened():
            print(f"❌ Could not open video writer for {output_path}")
            return False
        
        # Write frames
        for frame in frames_uint8:
            out.write(frame)
        
        out.release()
        
        print(f"✅ Saved video: {output_path.name}")
        print(f"   Frames: {len(frames)}, Size: 96x96, FPS: 8")
        return True
        
    except Exception as e:
        print(f"❌ Error converting {npy_path.name}: {e}")
        return False

def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def main():
    """Convert NPY files to proper grayscale MP4."""
    print("🔧 FIX VIDEO CONVERSION - PROPER GRAYSCALE")
    print("=" * 50)
    
    # Input and output directories
    input_dir = Path("data/training set 17.9.25")
    output_dir = Path("data/training set 17.9.25/preview_videos_fixed")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Check if FFmpeg is available
    has_ffmpeg = check_ffmpeg()
    print(f"🔧 FFmpeg available: {has_ffmpeg}")
    
    # Get all NPY files
    npy_files = list(input_dir.glob("*.npy"))
    
    if not npy_files:
        print("❌ No NPY files found!")
        return
    
    print(f"📊 Found {len(npy_files)} NPY files")
    
    successful = 0
    
    for npy_file in npy_files:
        print(f"\n🔄 Converting: {npy_file.name}")
        
        # Create output video name
        video_name = npy_file.stem.replace("_gentle_v5", "_fixed") + ".mp4"
        output_path = output_dir / video_name
        
        # Try FFmpeg first, then OpenCV fallback
        success = False
        if has_ffmpeg:
            success = npy_to_mp4_fixed(npy_file, output_path)
        
        if not success:
            print("   Trying OpenCV fallback...")
            success = npy_to_mp4_opencv_fixed(npy_file, output_path)
        
        if success:
            successful += 1
    
    print(f"\n✅ CONVERSION COMPLETE")
    print(f"📊 Successfully converted: {successful}/{len(npy_files)}")
    print(f"📁 Fixed videos saved to: {output_dir}")
    
    print(f"\n🎯 THESE SHOULD BE PROPER GRAYSCALE:")
    print(f"   • Open the preview_videos_fixed folder")
    print(f"   • Videos should appear as proper grayscale (no green)")
    print(f"   • Each shows the exact preprocessing result")

if __name__ == "__main__":
    main()
