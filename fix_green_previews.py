#!/usr/bin/env python3
"""
Fix Green Preview Videos
========================

Recreate preview videos from existing NPY files using proper grayscale AVI->MP4 conversion
to eliminate green color artifacts.
"""

import cv2
import numpy as np
import os
import subprocess
from pathlib import Path
from tqdm import tqdm

def create_fixed_preview_video(processed_frames, output_path):
    """Create a preview video using AVI->MP4 conversion to fix green artifacts."""
    try:
        # Convert from [-1,1] to [0,255] for visualization
        preview_frames = ((processed_frames + 1) * 127.5).astype(np.uint8)
        preview_frames = np.clip(preview_frames, 0, 255)

        # First create AVI file (more reliable for grayscale)
        avi_path = str(output_path).replace('.mp4', '.avi')
        
        # Create AVI video writer with grayscale settings
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec for AVI
        fps = 8  # Slow playback for inspection
        h, w = preview_frames.shape[1], preview_frames.shape[2]

        out = cv2.VideoWriter(avi_path, fourcc, fps, (w, h), isColor=False)
        
        if not out.isOpened():
            print(f"❌ Could not open AVI video writer for {avi_path}")
            return False

        for frame in preview_frames:
            out.write(frame)

        out.release()
        
        # Convert AVI to MP4 using FFmpeg if available
        try:
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-i', avi_path,  # input AVI
                '-c:v', 'libx264',  # H.264 codec
                '-pix_fmt', 'gray',  # grayscale pixel format
                '-crf', '23',  # quality setting
                str(output_path)  # output MP4
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Remove temporary AVI file
                os.remove(avi_path)
                return True
            else:
                print(f"⚠️  FFmpeg conversion failed, keeping AVI: {avi_path}")
                return True  # AVI file still created successfully
                
        except (FileNotFoundError, subprocess.SubprocessError):
            print(f"⚠️  FFmpeg not available, keeping AVI: {avi_path}")
            return True  # AVI file still created successfully

    except Exception as e:
        print(f"❌ Failed to create preview for {output_path}: {e}")
        return False

def main():
    """Fix all preview videos in the additional 50 per class dataset."""
    print("🔧 FIXING GREEN PREVIEW VIDEOS")
    print("=" * 50)
    
    # Directories
    data_dir = Path("data/training set 17.9.25/additional 50 per class")
    preview_dir = data_dir / "preview_videos"
    
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Preview directory: {preview_dir}")
    
    # Get all NPY files
    npy_files = []
    for class_dir in ['doctor', 'glasses', 'help', 'phone', 'pillow']:
        class_path = data_dir / class_dir
        if class_path.exists():
            npy_files.extend(list(class_path.glob("*.npy")))
    
    if not npy_files:
        print("❌ No NPY files found!")
        return
    
    print(f"📊 Found {len(npy_files)} NPY files to process")
    
    # Process each NPY file
    successful = 0
    failed = 0
    
    for npy_file in tqdm(npy_files, desc="Creating fixed previews"):
        try:
            # Load the processed frames
            processed_frames = np.load(npy_file)
            
            # Generate preview filename
            base_name = npy_file.stem.replace('_gentle_v5', '')
            preview_filename = f"{base_name}_preview.mp4"
            preview_path = preview_dir / preview_filename
            
            # Create fixed preview video
            if create_fixed_preview_video(processed_frames, preview_path):
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"❌ Error processing {npy_file.name}: {e}")
            failed += 1
    
    print(f"\n✅ PREVIEW FIX COMPLETE")
    print(f"📊 Successfully fixed: {successful}/{len(npy_files)}")
    print(f"📊 Failed: {failed}")
    print(f"📁 Fixed previews saved to: {preview_dir}")
    
    print(f"\n🎯 THESE SHOULD NOW BE PROPER GRAYSCALE:")
    print(f"   • Open the preview_videos folder")
    print(f"   • Videos should appear as proper grayscale (no green)")
    print(f"   • Each shows the exact preprocessing result")

if __name__ == "__main__":
    main()
