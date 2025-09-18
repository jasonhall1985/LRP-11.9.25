#!/usr/bin/env python3
"""
FULL DATASET BATCH PROCESSING SCRIPT
====================================
Process ALL videos from data/13.9.25top7dataset_cropped using the 10% expanded crop pipeline.
Outputs 96×64 landscape MP4 videos with perfectly centered lips and 32-frame temporal sampling.

Author: Augment Agent
Date: 2025-09-17
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import traceback

# Add current directory to Python path for imports
sys.path.append('/Users/client/Desktop/LRP classifier 11.9.25')

from lip_centered_64x96_multimodel_preprocessing import MultiModelLipPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_dataset_processing.log'),
        logging.StreamHandler()
    ]
)

def get_all_video_files(source_dir: str) -> List[Path]:
    """Get ALL video files from source directory."""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Get all video files (mp4, avi, mov, etc.)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    video_files = []
    
    for file_path in source_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    # Sort for consistent processing order
    video_files.sort()
    
    logging.info(f"Found {len(video_files)} video files in {source_dir}")
    return video_files

def create_output_directory(target_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory ready: {target_path}")
    return target_path

def get_output_filename(input_path: Path) -> str:
    """Generate output filename with _96x64_processed suffix."""
    stem = input_path.stem
    return f"{stem}_96x64_processed.mp4"

def check_already_processed(output_path: Path) -> bool:
    """Check if video has already been processed."""
    return output_path.exists() and output_path.stat().st_size > 0

def process_single_video(preprocessor: MultiModelLipPreprocessor, 
                        input_path: Path, 
                        output_path: Path) -> Tuple[bool, Dict]:
    """Process a single video file."""
    try:
        # Process video sequence
        processed_frames, report = preprocessor.process_video_sequence(str(input_path))
        
        if processed_frames is None:
            return False, {"error": "Processing returned None"}
        
        # Save as MP4
        temp_npy = output_path.with_suffix('.npy')
        import numpy as np
        np.save(temp_npy, processed_frames)
        
        success = preprocessor.npy_to_mp4_ffmpeg(str(temp_npy), str(output_path))
        
        # Clean up temporary file
        if temp_npy.exists():
            temp_npy.unlink()
        
        if success:
            return True, report
        else:
            return False, {"error": "MP4 conversion failed"}
            
    except Exception as e:
        error_msg = f"Exception during processing: {str(e)}"
        logging.error(f"Error processing {input_path.name}: {error_msg}")
        return False, {"error": error_msg}

def main():
    """Main batch processing function."""
    
    # Configuration
    SOURCE_DIR = "data/13.9.25top7dataset_cropped"
    TARGET_DIR = "data/dataset 17.9.25"
    
    print("🚀 FULL DATASET BATCH PROCESSING")
    print("=" * 60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    print(f"Format: 96×64 landscape MP4 (32 frames)")
    print(f"Pipeline: 10% EXPANDED CROP AREAS with perfect lip centering")
    print()
    
    # Initialize
    try:
        # Get all video files
        video_files = get_all_video_files(SOURCE_DIR)
        total_videos = len(video_files)
        
        if total_videos == 0:
            print("❌ No video files found in source directory!")
            return
        
        # Create output directory
        target_path = create_output_directory(TARGET_DIR)
        
        # Initialize preprocessor
        print("🔧 Initializing preprocessing pipeline with 10% EXPANDED CROP...")
        preprocessor = MultiModelLipPreprocessor()
        
        # Processing statistics
        successful = 0
        failed = 0
        skipped = 0
        start_time = time.time()
        
        print(f"\n📊 PROCESSING {total_videos} VIDEOS")
        print("=" * 60)
        print()
        
        # Process each video
        for i, input_path in enumerate(video_files, 1):
            output_filename = get_output_filename(input_path)
            output_path = target_path / output_filename
            
            # Check if already processed
            if check_already_processed(output_path):
                print(f"[{i:4d}/{total_videos}] ⏭️  SKIPPED: {input_path.name} (already exists)")
                skipped += 1
                continue
            
            print(f"[{i:4d}/{total_videos}] {input_path.name}")
            print(f"📹 Processing: {input_path.name}")
            
            # Process video
            success, report = process_single_video(preprocessor, input_path, output_path)
            
            if success:
                successful += 1
                method_dist = report.get("frame_consistency", {}).get("method_distribution", {})
                print(f"✅ Saved: {output_filename}")
                print(f"   Shape: (32, 64, 96)")
                print(f"   Method: {method_dist}")
            else:
                failed += 1
                error_msg = report.get("error", "Unknown error")
                print(f"❌ FAILED: {input_path.name}")
                print(f"   Error: {error_msg}")
                logging.error(f"Failed to process {input_path.name}: {error_msg}")
            
            print()
            
            # Progress update every 50 videos
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = total_videos - i
                eta = remaining / rate if rate > 0 else 0
                
                print(f"📈 PROGRESS UPDATE: {i}/{total_videos} ({i/total_videos*100:.1f}%)")
                print(f"   ✅ Successful: {successful}")
                print(f"   ❌ Failed: {failed}")
                print(f"   ⏭️  Skipped: {skipped}")
                print(f"   ⏱️  Rate: {rate:.1f} videos/sec")
                print(f"   🕐 ETA: {eta/60:.1f} minutes")
                print()
        
        # Final summary
        total_time = time.time() - start_time
        
        print("🏁 FULL DATASET PROCESSING COMPLETE")
        print("=" * 60)
        print(f"✅ Successful: {successful}/{total_videos}")
        print(f"❌ Failed: {failed}/{total_videos}")
        print(f"⏭️  Skipped: {skipped}/{total_videos}")
        print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
        print(f"📁 Output Location: {TARGET_DIR}")
        print()
        
        if successful > 0:
            print(f"🎉 {successful} videos successfully processed to 96×64 landscape MP4 format!")
            print("All files have 10% EXPANDED CROP AREAS with perfectly centered lips!")
            print("Ready for lip-reading model training with 32-frame temporal sampling.")
        
        if failed > 0:
            print(f"⚠️  {failed} videos failed processing - check full_dataset_processing.log for details")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        logging.error(f"Critical error in main(): {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
