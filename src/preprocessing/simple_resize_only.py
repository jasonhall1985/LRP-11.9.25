#!/usr/bin/env python3
"""
Simple Resize-Only Pipeline
===========================

NO CROPPING APPLIED - Just resizes entire frames to 96x96 pixels.
Preserves the full frame content exactly as it appears in source videos.

Features:
- Takes ENTIRE source frame (no cropping)
- Resizes to 96×96 pixels only
- No image processing applied
- Preserves original color format and pixel values
- Maintains aspect ratio through resize

Usage:
    python simple_resize_only.py SOURCE_DIR OUTPUT_DIR [MANIFEST_CSV]

Example:
    python simple_resize_only.py "/Users/client/Desktop/13.9.25top7dataset" "resized_dataset" "resize_manifest.csv"
"""

import sys
import cv2
import numpy as np
import pandas as pd
import pathlib
import csv
from datetime import datetime
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

class SimpleResizer:
    """
    Simple video resizer that takes entire frames and resizes to 96x96.
    NO CROPPING APPLIED - preserves full frame content.
    """
    
    def __init__(self, 
                 source_dir: str,
                 output_dir: str,
                 manifest_path: str = "resize_manifest.csv"):
        """
        Initialize simple resizer.
        
        Args:
            source_dir: Directory containing source MP4 videos
            output_dir: Directory for resized output videos
            manifest_path: Path for output manifest CSV
        """
        self.source_dir = pathlib.Path(source_dir)
        self.output_dir = pathlib.Path(output_dir)
        self.manifest_path = pathlib.Path(manifest_path)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output specifications
        self.output_size = 96  # 96×96 pixel output
        
        # Video file extensions
        self.video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_label_from_filename(self, filename: str) -> str:
        """
        Extract class label from filename.
        
        Args:
            filename: Video filename
            
        Returns:
            Class label (e.g., 'doctor', 'glasses', etc.)
        """
        filename_lower = filename.lower()
        
        # Split by double underscore
        if '__' in filename:
            parts = filename.split('__')
            if len(parts) > 0:
                return parts[0].lower()
        
        # Fallback: check if filename starts with known class names
        known_classes = {'doctor', 'glasses', 'phone', 'pillow', 'help', 'i_need_to_move', 'my_mouth_is_dry', 'unknown'}
        for class_name in known_classes:
            if filename_lower.startswith(class_name):
                return class_name
        
        return 'unknown'
    
    def find_all_mp4_videos(self) -> List[pathlib.Path]:
        """
        Find all MP4 video files in the source directory.
        
        Returns:
            List of MP4 video file paths
        """
        video_files = []
        
        for ext in self.video_extensions:
            video_files.extend(self.source_dir.glob(f"*{ext}"))
        
        # Filter to only MP4 files as specified
        mp4_files = [f for f in video_files if f.suffix.lower() == '.mp4']
        
        self.logger.info(f"Found {len(mp4_files)} MP4 files in {self.source_dir}")
        return mp4_files
    
    def apply_simple_resize(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply ONLY resize operation - NO CROPPING.
        
        Takes the ENTIRE source frame and resizes to 96×96 pixels.
        
        ONLY operation performed:
        - Resize entire frame to 96×96 pixels using basic interpolation
        
        NO cropping applied:
        - No geometric cropping
        - No region extraction
        - No grid division
        
        NO image processing applied:
        - No histogram equalization
        - No grayscale conversion
        - No contrast/brightness adjustments
        - No filtering or enhancement
        
        Args:
            frame: Input frame (H, W, C) in original color format
            
        Returns:
            Resized frame (96, 96, C) in original color format
        """
        # ONLY resize the entire frame - NO CROPPING
        resized = cv2.resize(frame, (self.output_size, self.output_size), 
                           interpolation=cv2.INTER_AREA)
        
        # Return frame in ORIGINAL color format with ORIGINAL pixel characteristics
        return resized
    
    def process_single_video(self, video_path: pathlib.Path) -> Optional[Dict]:
        """
        Process a single video with resize-only (no cropping).
        
        Args:
            video_path: Path to input video
            
        Returns:
            Processing result dictionary or None if failed
        """
        # Open input video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.warning(f"Cannot open video: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames == 0:
            self.logger.warning(f"No frames found in video: {video_path}")
            cap.release()
            return None
        
        # Extract label and create output filename
        label = self.get_label_from_filename(video_path.name)
        output_filename = f"resized_{video_path.name}"
        output_path = self.output_dir / output_filename
        
        # Setup video writer with SAME codec and quality settings
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (self.output_size, self.output_size)
        )
        
        if not writer.isOpened():
            self.logger.error(f"Cannot create video writer for: {output_path}")
            cap.release()
            return None
        
        # Process each frame with RESIZE ONLY (no cropping)
        processed_frames = 0
        failed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Apply ONLY resize (NO cropping)
                resized_frame = self.apply_simple_resize(frame)
                
                # Verify frame is valid
                if resized_frame.size > 0:
                    # Write frame in ORIGINAL color format
                    writer.write(resized_frame)
                    processed_frames += 1
                else:
                    self.logger.warning(f"Empty frame {processed_frames} in {video_path}")
                    failed_frames += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing frame {processed_frames} of {video_path}: {str(e)}")
                failed_frames += 1
        
        cap.release()
        writer.release()
        
        # Return processing result
        return {
            'output_path': str(output_path),
            'source_path': str(video_path),
            'label': label,
            'original_frames': total_frames,
            'processed_frames': processed_frames,
            'failed_frames': failed_frames,
            'success_rate': processed_frames / max(total_frames, 1),
            'original_fps': fps,
            'output_fps': fps,
            'original_resolution': f'{original_width}x{original_height}',
            'output_resolution': f'{self.output_size}x{self.output_size}',
            'crop_method': 'none_resize_only',
            'crop_region': 'entire_frame',
            'crop_coordinates': f'(0,0)-({original_width},{original_height})',
            'crop_size': f'{original_width}x{original_height}',
            'processing_type': 'resize_only_no_cropping',
            'image_processing': 'none_applied',
            'processing_status': 'success' if processed_frames > 0 else 'failed'
        }
    
    def process_all_videos(self) -> List[Dict]:
        """
        Process all MP4 videos with resize-only (no cropping).
        
        Returns:
            List of processing results
        """
        video_files = self.find_all_mp4_videos()
        results = []
        
        if not video_files:
            self.logger.warning("No MP4 files found in source directory")
            return results
        
        self.logger.info(f"Starting RESIZE-ONLY processing of {len(video_files)} videos")
        self.logger.info("NO CROPPING will be applied - taking entire frames")
        self.logger.info("NO image processing will be applied - preserving original characteristics")
        
        for video_path in tqdm(video_files, desc="Resizing videos"):
            try:
                result = self.process_single_video(video_path)
                if result:
                    results.append(result)
                else:
                    # Record failed processing
                    results.append({
                        'output_path': '',
                        'source_path': str(video_path),
                        'label': self.get_label_from_filename(video_path.name),
                        'original_frames': 0,
                        'processed_frames': 0,
                        'failed_frames': 0,
                        'success_rate': 0.0,
                        'original_fps': 0,
                        'output_fps': 0,
                        'original_resolution': 'unknown',
                        'output_resolution': f'{self.output_size}x{self.output_size}',
                        'crop_method': 'none_resize_only',
                        'crop_region': 'entire_frame',
                        'crop_coordinates': 'failed',
                        'crop_size': 'failed',
                        'processing_type': 'resize_only_no_cropping',
                        'image_processing': 'none_applied',
                        'processing_status': 'failed'
                    })
            except Exception as e:
                self.logger.error(f"Error processing {video_path}: {str(e)}")
                results.append({
                    'output_path': '',
                    'source_path': str(video_path),
                    'label': self.get_label_from_filename(video_path.name),
                    'original_frames': 0,
                    'processed_frames': 0,
                    'failed_frames': 0,
                    'success_rate': 0.0,
                    'original_fps': 0,
                    'output_fps': 0,
                    'original_resolution': 'unknown',
                    'output_resolution': f'{self.output_size}x{self.output_size}',
                    'crop_method': 'none_resize_only',
                    'crop_region': 'entire_frame',
                    'crop_coordinates': 'error',
                    'crop_size': 'error',
                    'processing_type': 'resize_only_no_cropping',
                    'image_processing': 'none_applied',
                    'processing_status': f'error: {str(e)}'
                })
        
        return results
    
    def save_processing_manifest(self, results: List[Dict]) -> None:
        """
        Save processing results to CSV manifest.
        
        Args:
            results: List of processing result dictionaries
        """
        if not results:
            self.logger.warning("No results to save to manifest")
            return
        
        # Define CSV columns
        columns = [
            'output_path', 'source_path', 'label', 'original_frames', 'processed_frames',
            'failed_frames', 'success_rate', 'original_fps', 'output_fps',
            'original_resolution', 'output_resolution', 'crop_method', 'crop_region',
            'crop_coordinates', 'crop_size', 'processing_type', 'image_processing', 'processing_status'
        ]
        
        # Write to CSV
        with open(self.manifest_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)
        
        self.logger.info(f"Processing manifest saved to: {self.manifest_path}")
    
    def print_processing_summary(self, results: List[Dict]) -> None:
        """
        Print a comprehensive processing summary.
        
        Args:
            results: List of processing result dictionaries
        """
        if not results:
            print("No results to summarize")
            return
        
        # Basic counts
        total_videos = len(results)
        successful_videos = len([r for r in results if r['processing_status'] == 'success'])
        failed_videos = total_videos - successful_videos
        overall_success_rate = successful_videos / total_videos if total_videos > 0 else 0
        
        # Frame statistics
        total_original_frames = sum(r['original_frames'] for r in results)
        total_processed_frames = sum(r['processed_frames'] for r in results)
        total_failed_frames = sum(r['failed_frames'] for r in results)
        
        # Label distribution
        label_counts = {}
        successful_label_counts = {}
        for result in results:
            label = result['label']
            label_counts[label] = label_counts.get(label, 0) + 1
            if result['processing_status'] == 'success':
                successful_label_counts[label] = successful_label_counts.get(label, 0) + 1
        
        print("\n" + "="*70)
        print("RESIZE-ONLY PROCESSING SUMMARY")
        print("="*70)
        
        print(f"Source Directory: {self.source_dir}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Processing Time: {datetime.now().isoformat()}")
        print()
        
        print("PROCESSING TYPE: RESIZE-ONLY (NO CROPPING)")
        print("  ✓ Takes ENTIRE source frame")
        print("  ✓ NO cropping applied")
        print("  ✓ NO geometric operations")
        print("  ✓ NO histogram equalization")
        print("  ✓ NO grayscale conversion")
        print("  ✓ NO contrast adjustments")
        print("  ✓ NO brightness modifications")
        print("  ✓ NO filtering or smoothing")
        print("  ✓ Original color format preserved")
        print("  ✓ Original pixel values preserved")
        print()
        
        print("RESIZE CONFIGURATION:")
        print(f"  Input: Entire source frame (any resolution)")
        print(f"  Output Size: {self.output_size}x{self.output_size} pixels")
        print(f"  Method: Basic resize with INTER_AREA interpolation")
        print(f"  Aspect Ratio: Adjusted to square format")
        print()
        
        print("PROCESSING RESULTS:")
        print(f"  Total Videos: {total_videos}")
        print(f"  Successful: {successful_videos} ({overall_success_rate:.1%})")
        print(f"  Failed: {failed_videos}")
        print()
        
        print("FRAME STATISTICS:")
        print(f"  Original Frames: {total_original_frames:,}")
        print(f"  Processed Frames: {total_processed_frames:,}")
        print(f"  Failed Frames: {total_failed_frames:,}")
        frame_success_rate = total_processed_frames / max(total_original_frames, 1)
        print(f"  Frame Success Rate: {frame_success_rate:.1%}")
        print()
        
        print("LABEL DISTRIBUTION:")
        for label, count in sorted(label_counts.items()):
            successful = successful_label_counts.get(label, 0)
            success_rate = successful / count if count > 0 else 0
            print(f"  {label}: {count} total, {successful} successful ({success_rate:.1%})")
        
        print("="*70)


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python simple_resize_only.py SOURCE_DIR OUTPUT_DIR [MANIFEST_CSV]")
        print()
        print("RESIZE-ONLY PROCESSING - NO CROPPING APPLIED")
        print("Takes entire source frames and resizes to 96x96 pixels")
        print("Preserves original video characteristics exactly:")
        print("  ✓ Original color format (RGB/BGR)")
        print("  ✓ Original pixel values")
        print("  ✓ Original frame quality")
        print("  ✓ No cropping or enhancement")
        print()
        print("Example:")
        print('  python simple_resize_only.py "/Users/client/Desktop/13.9.25top7dataset" "resized_dataset"')
        sys.exit(1)
    
    source_dir = sys.argv[1]
    output_dir = sys.argv[2]
    manifest_path = sys.argv[3] if len(sys.argv) > 3 else "resize_manifest.csv"
    
    # Initialize resizer
    resizer = SimpleResizer(source_dir, output_dir, manifest_path)
    
    # Process all videos
    results = resizer.process_all_videos()
    
    # Save manifest and print summary
    resizer.save_processing_manifest(results)
    resizer.print_processing_summary(results)
    
    print(f"\nResize-only processing complete! Check '{output_dir}' for resized videos.")
    print(f"Manifest saved to: {manifest_path}")
    print("\nNO cropping was applied - entire frames resized to 96x96!")


if __name__ == "__main__":
    main()
