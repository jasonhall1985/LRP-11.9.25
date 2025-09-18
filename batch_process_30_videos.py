#!/usr/bin/env python3
"""
Batch Processing Script for 50 Random Videos
============================================

Uses the lip_centered_64x96_multimodel_preprocessing.py pipeline
to process 50 randomly selected videos from data/13.9.25top7dataset_cropped
and save them as MP4 files in data/dataset 17.9.25

Output Format:
- 96×64 landscape orientation (32 frames)
- MP4 format with proper grayscale encoding
- Current lip positioning implementation
"""

import os
import sys
from pathlib import Path
import numpy as np
from typing import List
import time
import random

# Import the existing preprocessing pipeline
sys.path.append('/Users/client/Desktop/LRP classifier 11.9.25')
from lip_centered_64x96_multimodel_preprocessing import MultiModelLipPreprocessor

def get_random_50_videos(source_dir: str) -> List[Path]:
    """Get 50 randomly selected video files from source directory."""
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return []

    # Get all video files (mp4, avi, mov, etc.)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    video_files = []

    for file_path in source_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)

    # Randomly select 50 videos
    random.shuffle(video_files)
    random_50 = video_files[:50]

    print(f"📁 Found {len(video_files)} total videos in {source_dir}")
    print(f"🎯 Processing {len(random_50)} randomly selected videos")

    return random_50

def process_video_to_mp4(preprocessor: MultiModelLipPreprocessor, 
                        input_video: Path, 
                        output_dir: Path) -> bool:
    """Process a single video and save as MP4."""
    try:
        print(f"\n📹 Processing: {input_video.name}")
        
        # Process video with the existing pipeline
        processed_frames, processing_report = preprocessor.process_video_sequence(str(input_video))
        
        if processed_frames is None:
            print(f"❌ Failed to process: {input_video.name}")
            return False
        
        # Create output filename
        output_filename = f"{input_video.stem}_96x64_processed.mp4"
        output_path = output_dir / output_filename
        
        # Save as temporary numpy file first
        temp_npy = output_dir / f"{input_video.stem}_temp.npy"
        np.save(temp_npy, processed_frames)
        
        # Convert to MP4 using the existing FFmpeg function
        success = preprocessor.npy_to_mp4_ffmpeg(temp_npy, output_path)
        
        # Clean up temporary file
        if temp_npy.exists():
            temp_npy.unlink()
        
        if success:
            print(f"✅ Saved: {output_filename}")
            print(f"   Shape: {processed_frames.shape}")
            print(f"   Method: {processing_report['frame_consistency']['method_distribution']}")
            return True
        else:
            print(f"❌ FFmpeg conversion failed for: {input_video.name}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {input_video.name}: {e}")
        return False

def main():
    """Main batch processing function."""
    print("🚀 BATCH PROCESSING: 50 RANDOM VIDEOS")
    print("=" * 60)
    print("Source: data/13.9.25top7dataset_cropped")
    print("Target: data/dataset 17.9.25")
    print("Format: 96×64 landscape MP4 (32 frames)")
    print("Selection: 50 randomly selected videos")
    print()
    
    # Define paths
    source_dir = "data/13.9.25top7dataset_cropped"
    output_dir = Path("data/dataset 17.9.25")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Get 50 random videos
    video_files = get_random_50_videos(source_dir)
    if not video_files:
        print("❌ No videos found to process")
        return
    
    # Initialize preprocessor
    print("\n🔧 Initializing preprocessing pipeline...")
    preprocessor = MultiModelLipPreprocessor()
    
    # Process each video
    print(f"\n📊 PROCESSING {len(video_files)} VIDEOS")
    print("=" * 60)
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i:2d}/{len(video_files)}] {video_file.name}")
        
        if process_video_to_mp4(preprocessor, video_file, output_dir):
            successful += 1
        else:
            failed += 1
    
    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n🏁 BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"✅ Successful: {successful}/{len(video_files)}")
    print(f"❌ Failed: {failed}/{len(video_files)}")
    print(f"⏱️  Total Time: {elapsed_time:.1f} seconds")
    print(f"📁 Output Location: {output_dir}")
    
    if successful > 0:
        print(f"\n🎉 {successful} videos successfully processed to 96×64 landscape MP4 format!")
        print("All files are ready for use with 32-frame temporal sampling.")
    
    if failed > 0:
        print(f"\n⚠️  {failed} videos failed processing - check individual error messages above.")

if __name__ == "__main__":
    main()
