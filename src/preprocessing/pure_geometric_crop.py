#!/usr/bin/env python3
"""
Pure Geometric Cropping Pipeline
================================

A simplified geometric cropping pipeline that preserves original video characteristics
exactly while applying ONLY spatial cropping and resizing operations.

NO IMAGE PROCESSING APPLIED:
- No histogram equalization
- No grayscale conversion  
- No contrast adjustments
- No brightness modifications
- No filtering or smoothing
- No compression artifacts

ONLY GEOMETRIC OPERATIONS:
- 3×2 grid division
- Top-middle cell extraction
- Basic resize to 96×96 pixels
- Preserve original color format and pixel values

Usage:
    python pure_geometric_crop.py SOURCE_DIR OUTPUT_DIR [MANIFEST_CSV]

Example:
    python pure_geometric_crop.py "/Users/client/Desktop/13.9.25top7dataset" "pure_cropped_dataset" "pure_processing_manifest.csv"
"""

import sys
import cv2
import numpy as np
import pandas as pd
import pathlib
import csv
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

class PureGeometricCropper:
    """
    Pure geometric cropping tool that applies ONLY spatial operations.
    
    Preserves original video characteristics exactly:
    - Original color format (RGB/BGR)
    - Original pixel values
    - Original frame quality
    - No image enhancement or processing
    """
    
    def __init__(self, 
                 source_dir: str,
                 output_dir: str,
                 manifest_path: str = "pure_processing_manifest.csv"):
        """
        Initialize pure geometric cropper.
        
        Args:
            source_dir: Directory containing source MP4 videos
            output_dir: Directory for cropped output videos
            manifest_path: Path for output manifest CSV
        """
        self.source_dir = pathlib.Path(source_dir)
        self.output_dir = pathlib.Path(output_dir)
        self.manifest_path = pathlib.Path(manifest_path)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Grid configuration (3×2 grid)
        self.grid_cols = 3
        self.grid_rows = 2
        self.target_col = 1  # Middle column (0-indexed)
        self.target_row = 0  # Top row (0-indexed)
        
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
    
    def calculate_grid_crop_region(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """
        Calculate the expanded crop region for the top-middle cell of a 3×2 grid.
        Expands the base cell boundaries by 10% in all directions for wider field of view.

        Args:
            frame_width: Width of the input frame
            frame_height: Height of the input frame

        Returns:
            Tuple of (x_start, y_start, x_end, y_end) for expanded crop region
        """
        # Calculate base cell dimensions
        cell_width = frame_width // self.grid_cols
        cell_height = frame_height // self.grid_rows

        # Calculate base top-middle cell boundaries
        base_x_start = self.target_col * cell_width
        base_x_end = base_x_start + cell_width
        base_y_start = self.target_row * cell_height
        base_y_end = base_y_start + cell_height

        # Calculate expansion amounts (10% of cell dimensions)
        expansion_x = int(cell_width * 0.10)
        expansion_y = int(cell_height * 0.10)

        # Apply expansion in all directions
        x_start = base_x_start - expansion_x
        x_end = base_x_end + expansion_x
        y_start = base_y_start - expansion_y
        y_end = base_y_end + expansion_y

        # Ensure boundaries are within frame limits
        x_start = max(0, x_start)
        x_end = min(frame_width, x_end)
        y_start = max(0, y_start)
        y_end = min(frame_height, y_end)

        return x_start, y_start, x_end, y_end
    
    def apply_pure_geometric_crop(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply PURE geometric cropping with NO image processing.
        Uses expanded crop region (10% larger than base 3×2 grid cell).

        ONLY operations performed:
        1. Extract expanded top-middle cell from 3×2 grid (10% expansion)
        2. Resize to 96×96 pixels using basic interpolation

        NO image processing applied:
        - No histogram equalization
        - No grayscale conversion
        - No contrast/brightness adjustments
        - No filtering or enhancement

        Args:
            frame: Input frame (H, W, C) in original color format

        Returns:
            Cropped and resized frame in original color format with wider field of view
        """
        height, width = frame.shape[:2]
        
        # Step 1: Calculate crop region (PURE GEOMETRIC OPERATION)
        x_start, y_start, x_end, y_end = self.calculate_grid_crop_region(width, height)
        
        # Step 2: Extract crop region (PURE SPATIAL OPERATION)
        cropped = frame[y_start:y_end, x_start:x_end]
        
        # Step 3: Resize to target size (BASIC INTERPOLATION ONLY)
        # Using INTER_AREA for downsampling to preserve quality
        resized = cv2.resize(cropped, (self.output_size, self.output_size), 
                           interpolation=cv2.INTER_AREA)
        
        # Return frame in ORIGINAL color format with ORIGINAL pixel characteristics
        return resized
    
    def process_single_video(self, video_path: pathlib.Path) -> Optional[Dict]:
        """
        Process a single video with pure geometric cropping.
        
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
        output_filename = f"pure_cropped_{video_path.name}"
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
        
        # Process each frame with PURE geometric operations only
        processed_frames = 0
        failed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Apply PURE geometric cropping (NO image processing)
                cropped_resized = self.apply_pure_geometric_crop(frame)
                
                # Verify frame is valid
                if cropped_resized.size > 0:
                    # Write frame in ORIGINAL color format
                    writer.write(cropped_resized)
                    processed_frames += 1
                else:
                    self.logger.warning(f"Empty crop region in frame {processed_frames} of {video_path}")
                    failed_frames += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing frame {processed_frames} of {video_path}: {str(e)}")
                failed_frames += 1
        
        cap.release()
        writer.release()
        
        # Calculate crop region info for metadata
        x_start, y_start, x_end, y_end = self.calculate_grid_crop_region(original_width, original_height)
        crop_width = x_end - x_start
        crop_height = y_end - y_start
        
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
            'crop_method': f'{self.grid_cols}x{self.grid_rows}_grid_pure_expanded',
            'crop_region': 'top_middle_cell_expanded_10pct',
            'crop_coordinates': f'({x_start},{y_start})-({x_end},{y_end})',
            'crop_size': f'{crop_width}x{crop_height}',
            'processing_type': 'pure_geometric_only',
            'image_processing': 'none_applied',
            'processing_status': 'success' if processed_frames > 0 else 'failed'
        }

    def process_all_videos(self) -> List[Dict]:
        """
        Process all MP4 videos in the source directory with pure geometric cropping.

        Returns:
            List of processing results
        """
        video_files = self.find_all_mp4_videos()
        results = []

        if not video_files:
            self.logger.warning("No MP4 files found in source directory")
            return results

        self.logger.info(f"Starting PURE geometric cropping of {len(video_files)} videos")
        self.logger.info("NO image processing will be applied - preserving original characteristics")

        for video_path in tqdm(video_files, desc="Processing videos"):
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
                        'crop_method': f'{self.grid_cols}x{self.grid_rows}_grid_pure_expanded',
                        'crop_region': 'top_middle_cell_expanded_10pct',
                        'crop_coordinates': 'failed',
                        'crop_size': 'failed',
                        'processing_type': 'pure_geometric_only',
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
                    'crop_method': f'{self.grid_cols}x{self.grid_rows}_grid_pure_expanded',
                    'crop_region': 'top_middle_cell_expanded_10pct',
                    'crop_coordinates': 'error',
                    'crop_size': 'error',
                    'processing_type': 'pure_geometric_only',
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
        print("PURE GEOMETRIC CROPPING PROCESSING SUMMARY")
        print("="*70)

        print(f"Source Directory: {self.source_dir}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Processing Time: {datetime.now().isoformat()}")
        print()

        print("PROCESSING TYPE: PURE GEOMETRIC ONLY")
        print("  ✓ NO histogram equalization")
        print("  ✓ NO grayscale conversion")
        print("  ✓ NO contrast adjustments")
        print("  ✓ NO brightness modifications")
        print("  ✓ NO filtering or smoothing")
        print("  ✓ Original color format preserved")
        print("  ✓ Original pixel values preserved")
        print()

        print("GRID CONFIGURATION:")
        print(f"  Grid Layout: {self.grid_cols}x{self.grid_rows} (3 columns × 2 rows)")
        print(f"  Target Cell: row_{self.target_row}_col_{self.target_col} (top-middle)")
        print(f"  Crop Expansion: 10% in all directions for wider field of view")
        print(f"  Output Size: {self.output_size}x{self.output_size} pixels")
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
        print("Usage: python pure_geometric_crop.py SOURCE_DIR OUTPUT_DIR [MANIFEST_CSV]")
        print()
        print("PURE GEOMETRIC CROPPING - NO IMAGE PROCESSING APPLIED")
        print("Preserves original video characteristics exactly:")
        print("  ✓ Original color format (RGB/BGR)")
        print("  ✓ Original pixel values")
        print("  ✓ Original frame quality")
        print("  ✓ No enhancement or modification")
        print()
        print("Example:")
        print('  python pure_geometric_crop.py "/Users/client/Desktop/13.9.25top7dataset" "pure_cropped_dataset"')
        sys.exit(1)

    source_dir = sys.argv[1]
    output_dir = sys.argv[2]
    manifest_path = sys.argv[3] if len(sys.argv) > 3 else "pure_processing_manifest.csv"

    # Initialize cropper
    cropper = PureGeometricCropper(source_dir, output_dir, manifest_path)

    # Process all videos
    results = cropper.process_all_videos()

    # Save manifest and print summary
    cropper.save_processing_manifest(results)
    cropper.print_processing_summary(results)

    print(f"\nPure geometric cropping complete! Check '{output_dir}' for cropped videos.")
    print(f"Manifest saved to: {manifest_path}")
    print("\nNO image processing was applied - original characteristics preserved!")


if __name__ == "__main__":
    main()
