#!/usr/bin/env python3
"""
Filter Successful Videos for Mouth Cropping
===========================================

Creates a filtered dataset containing only videos that passed MediaPipe lip detection.
This prepares the input for the mouth cropping pipeline.

Usage:
    python filter_successful_videos.py MEDIAPIPE_CSV INPUT_DIR OUTPUT_DIR

Example:
    python filter_successful_videos.py mediapipe_cropped_face_reports/mediapipe_cropped_face_detailed_20250913_223941.csv data/grid/13.9.25top7dataset data/successful_videos
"""

import sys
import csv
import shutil
import pathlib
from tqdm import tqdm
import logging
from datetime import datetime

def setup_logging(output_dir):
    """Setup logging for the filtering process."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = pathlib.Path(output_dir) / f"video_filtering_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def filter_successful_videos(mediapipe_csv: str, input_dir: str, output_dir: str):
    """
    Filter and copy successful videos based on MediaPipe analysis results.
    
    Args:
        mediapipe_csv: Path to MediaPipe analysis CSV file
        input_dir: Directory containing original videos
        output_dir: Directory for filtered successful videos
    """
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info("Starting video filtering based on MediaPipe results")
    logger.info(f"MediaPipe CSV: {mediapipe_csv}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Read MediaPipe results
    successful_videos = []
    total_videos = 0
    
    with open(mediapipe_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total_videos += 1
            if row['has_sufficient_detection'] == 'True':
                successful_videos.append({
                    'filename': row['filename'],
                    'class': row['class'],
                    'detection_rate': float(row['lip_detection_rate']),
                    'quality_rate': float(row['good_quality_rate'])
                })
    
    logger.info(f"Found {len(successful_videos)} successful videos out of {total_videos} total")
    logger.info(f"Success rate: {len(successful_videos)/total_videos*100:.1f}%")
    
    # Copy successful videos
    copied_count = 0
    missing_count = 0
    class_counts = {}
    
    for video_info in tqdm(successful_videos, desc="Copying successful videos"):
        filename = video_info['filename']
        class_label = video_info['class']
        
        # Track class distribution
        class_counts[class_label] = class_counts.get(class_label, 0) + 1
        
        # Find source file
        source_file = input_path / filename
        if not source_file.exists():
            logger.warning(f"Source file not found: {filename}")
            missing_count += 1
            continue
        
        # Copy to output directory
        dest_file = output_path / filename
        try:
            shutil.copy2(source_file, dest_file)
            copied_count += 1
        except Exception as e:
            logger.error(f"Failed to copy {filename}: {str(e)}")
    
    # Generate summary report
    summary_file = output_path / "filtering_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("SUCCESSFUL VIDEO FILTERING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total videos analyzed: {total_videos}\n")
        f.write(f"Successful videos identified: {len(successful_videos)}\n")
        f.write(f"Videos successfully copied: {copied_count}\n")
        f.write(f"Missing source files: {missing_count}\n")
        f.write(f"Success rate: {len(successful_videos)/total_videos*100:.1f}%\n\n")
        
        f.write("CLASS DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        for class_label, count in sorted(class_counts.items()):
            f.write(f"{class_label}: {count} videos\n")
        
        f.write(f"\nFiltered videos ready for mouth cropping pipeline.\n")
        f.write(f"Output directory: {output_dir}\n")
    
    # Print summary
    print("\n" + "="*60)
    print("VIDEO FILTERING COMPLETED")
    print("="*60)
    print(f"Total videos analyzed: {total_videos}")
    print(f"Successful videos identified: {len(successful_videos)}")
    print(f"Videos successfully copied: {copied_count}")
    print(f"Missing source files: {missing_count}")
    print(f"Success rate: {len(successful_videos)/total_videos*100:.1f}%")
    
    print(f"\nClass distribution:")
    for class_label, count in sorted(class_counts.items()):
        print(f"  {class_label}: {count} videos")
    
    print(f"\nFiltered videos saved to: {output_dir}")
    print(f"Summary report: {summary_file}")
    print("Ready for mouth cropping pipeline!")
    print("="*60)
    
    logger.info("Video filtering completed successfully")
    return copied_count

def main():
    """Main CLI interface."""
    if len(sys.argv) != 4:
        print("Usage: python filter_successful_videos.py MEDIAPIPE_CSV INPUT_DIR OUTPUT_DIR")
        print("\nExample:")
        print("  python filter_successful_videos.py \\")
        print("    mediapipe_cropped_face_reports/mediapipe_cropped_face_detailed_20250913_223941.csv \\")
        print("    data/grid/13.9.25top7dataset \\")
        print("    data/successful_videos")
        sys.exit(1)
    
    mediapipe_csv = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Validate inputs
    if not pathlib.Path(mediapipe_csv).exists():
        print(f"Error: MediaPipe CSV file '{mediapipe_csv}' does not exist")
        sys.exit(1)
    
    if not pathlib.Path(input_dir).exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)
    
    # Run filtering
    copied_count = filter_successful_videos(mediapipe_csv, input_dir, output_dir)
    
    if copied_count == 0:
        print("Warning: No videos were successfully copied!")
        sys.exit(1)
    
    print(f"\nSuccess! {copied_count} videos ready for mouth cropping.")

if __name__ == "__main__":
    main()
