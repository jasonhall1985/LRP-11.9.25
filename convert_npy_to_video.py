#!/usr/bin/env python3
"""
Convert NPY files back to viewable MP4 videos for visual inspection
"""

import cv2
import numpy as np
from pathlib import Path

def npy_to_video(npy_path, output_path, fps=10):
    """Convert numpy array back to MP4 video."""
    try:
        # Load the numpy array
        frames = np.load(npy_path)
        print(f"📹 Loaded {npy_path.name}: {frames.shape}")

        # Convert from [-1, 1] back to [0, 255]
        frames_uint8 = ((frames + 1) * 127.5).astype(np.uint8)
        frames_uint8 = np.clip(frames_uint8, 0, 255)

        # Get dimensions
        num_frames, height, width = frames_uint8.shape

        # Create video writer - use MP4V codec with color=True but write grayscale as RGB
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=True)

        # Write frames - convert grayscale to RGB for proper MP4 compatibility
        for frame in frames_uint8:
            # Convert grayscale to RGB (3 channels, same values)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(rgb_frame)

        out.release()

        print(f"✅ Saved video: {output_path.name}")
        print(f"   Frames: {num_frames}, Size: {width}x{height}, FPS: {fps}")

        return True

    except Exception as e:
        print(f"❌ Error converting {npy_path.name}: {e}")
        return False

def main():
    """Convert all NPY files to MP4 for visual inspection."""
    print("🔧 CONVERT NPY TO VIDEO FOR INSPECTION")
    print("=" * 50)
    
    # Input and output directories
    input_dir = Path("data/training set 17.9.25")
    output_dir = Path("data/training set 17.9.25/preview_videos")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    
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
        video_name = npy_file.stem.replace("_gentle_v5", "_preview") + ".mp4"
        output_path = output_dir / video_name
        
        if npy_to_video(npy_file, output_path, fps=8):
            successful += 1
    
    print(f"\n✅ CONVERSION COMPLETE")
    print(f"📊 Successfully converted: {successful}/{len(npy_files)}")
    print(f"📁 Preview videos saved to: {output_dir}")
    
    print(f"\n🎯 NOW YOU CAN VISUALLY INSPECT:")
    print(f"   • Open the preview_videos folder in Finder")
    print(f"   • Preview any .mp4 file directly in Finder")
    print(f"   • Check crop area, lip positioning, and grayscale quality")
    print(f"   • Each video shows exactly what the model will see")
    print(f"   • Videos are proper grayscale MP4 files")

if __name__ == "__main__":
    main()
