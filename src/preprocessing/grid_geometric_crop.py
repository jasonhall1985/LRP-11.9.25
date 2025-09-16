#!/usr/bin/env python3
"""
Grid Dataset Geometric Cropping Pipeline
========================================

Processes all MP4 videos in the "13.9.25top7dataset" folder using a 3×2 grid-based approach.
Extracts the top-middle cell of the grid for consistent mouth region cropping.

Features:
- 3×2 grid division (3 columns × 2 rows = 6 regions)
- Top-middle cell extraction (top row, middle column)
- Maintains original frame rate and video format
- Resizes to standard 96×96 pixels
- Preserves original video length and frame count
- Comprehensive processing manifest and statistics

Usage:
    python grid_geometric_crop.py SOURCE_DIR OUTPUT_DIR [MANIFEST_CSV]

Example:
    python grid_geometric_crop.py "/Users/client/Desktop/13.9.25top7dataset" "grid_cropped_dataset" "grid_processing_manifest.csv"
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

class GridGeometricCropper:
    """
    Grid-based geometric cropping tool for video datasets.
    
    Divides each frame into a 3×2 grid and extracts the top-middle region
    where relevant visual information (mouth/lip region) is concentrated.
    """
    
    def __init__(self, 
                 source_dir: str,
                 output_dir: str,
                 manifest_path: str = "grid_processing_manifest.csv"):
        """
        Initialize grid geometric cropper.
        
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
        self.output_size = 96           # 96×96 pixel output
        self.maintain_aspect = False    # Force square output
        
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
        # Expected format: "doctor__useruser01__18to39__male__caucasian__20250723T042153.mp4"
        filename_lower = filename.lower()
        
        # Split by double underscore
        if '__' in filename:
            parts = filename.split('__')
            if len(parts) > 0:
                return parts[0].lower()
        
        # Fallback: check if filename starts with known class names
        known_classes = {'doctor', 'glasses', 'phone', 'pillow', 'help', 'unknown'}
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
        Calculate the crop region for the top-middle cell of a 3×2 grid.
        
        Args:
            frame_width: Width of the input frame
            frame_height: Height of the input frame
            
        Returns:
            Tuple of (x_start, y_start, x_end, y_end) for crop region
        """
        # Calculate cell dimensions
        cell_width = frame_width // self.grid_cols
        cell_height = frame_height // self.grid_rows
        
        # Calculate top-middle cell boundaries
        x_start = self.target_col * cell_width
        x_end = x_start + cell_width
        
        y_start = self.target_row * cell_height
        y_end = y_start + cell_height
        
        # Ensure boundaries are within frame
        x_start = max(0, x_start)
        x_end = min(frame_width, x_end)
        y_start = max(0, y_start)
        y_end = min(frame_height, y_end)
        
        return x_start, y_start, x_end, y_end
    
    def apply_grid_crop_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply 3×2 grid crop to extract top-middle region from a frame.
        
        Args:
            frame: Input frame (H, W, C)
            
        Returns:
            Cropped frame region
        """
        height, width = frame.shape[:2]
        
        # Calculate crop region
        x_start, y_start, x_end, y_end = self.calculate_grid_crop_region(width, height)
        
        # Extract crop region
        cropped = frame[y_start:y_end, x_start:x_end]
        
        return cropped
    
    def process_single_video(self, video_path: pathlib.Path) -> Optional[Dict]:
        """
        Process a single video with grid-based geometric cropping.
        
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
        output_filename = f"cropped_{video_path.name}"
        output_path = self.output_dir / output_filename
        
        # Setup video writer
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
        
        # Process each frame
        processed_frames = 0
        failed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Apply grid crop
                cropped = self.apply_grid_crop_to_frame(frame)
                
                # Resize to target size
                if cropped.size > 0:
                    resized = cv2.resize(cropped, (self.output_size, self.output_size), 
                                       interpolation=cv2.INTER_AREA)
                    
                    # Optional: Apply histogram equalization for better contrast
                    if len(resized.shape) == 3:
                        # Convert to grayscale for equalization, then back to BGR
                        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                        equalized = cv2.equalizeHist(gray)
                        resized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
                    
                    writer.write(resized)
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
            'crop_method': f'{self.grid_cols}x{self.grid_rows}_grid',
            'crop_region': f'top_middle_cell',
            'crop_coordinates': f'({x_start},{y_start})-({x_end},{y_end})',
            'crop_size': f'{crop_width}x{crop_height}',
            'processing_status': 'success' if processed_frames > 0 else 'failed'
        }
    
    def process_all_videos(self) -> List[Dict]:
        """
        Process all MP4 videos in the source directory.

        Returns:
            List of processing results
        """
        video_files = self.find_all_mp4_videos()
        results = []

        if not video_files:
            self.logger.warning("No MP4 files found in source directory")
            return results

        self.logger.info(f"Starting grid geometric cropping of {len(video_files)} videos")

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
                        'crop_method': f'{self.grid_cols}x{self.grid_rows}_grid',
                        'crop_region': 'top_middle_cell',
                        'crop_coordinates': 'failed',
                        'crop_size': 'failed',
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
                    'crop_method': f'{self.grid_cols}x{self.grid_rows}_grid',
                    'crop_region': 'top_middle_cell',
                    'crop_coordinates': 'error',
                    'crop_size': 'error',
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
            'crop_coordinates', 'crop_size', 'processing_status'
        ]

        # Write to CSV
        with open(self.manifest_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)

        self.logger.info(f"Processing manifest saved to: {self.manifest_path}")

    def generate_processing_statistics(self, results: List[Dict]) -> Dict:
        """
        Generate comprehensive processing statistics.

        Args:
            results: List of processing result dictionaries

        Returns:
            Statistics dictionary
        """
        if not results:
            return {
                'total_videos': 0,
                'successful_videos': 0,
                'failed_videos': 0,
                'overall_success_rate': 0,
                'total_original_frames': 0,
                'total_processed_frames': 0,
                'total_failed_frames': 0,
                'frame_success_rate': 0,
                'label_distribution': {},
                'successful_label_distribution': {},
                'average_fps': 0,
                'grid_configuration': f'{self.grid_cols}x{self.grid_rows}',
                'target_cell': f'row_{self.target_row}_col_{self.target_col}',
                'output_size': f'{self.output_size}x{self.output_size}',
                'source_directory': str(self.source_dir),
                'output_directory': str(self.output_dir),
                'processing_timestamp': datetime.now().isoformat()
            }

        # Basic counts
        total_videos = len(results)
        successful_videos = len([r for r in results if r['processing_status'] == 'success'])
        failed_videos = total_videos - successful_videos

        # Success rate
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

        # FPS statistics (for successful videos)
        successful_results = [r for r in results if r['processing_status'] == 'success']
        fps_values = [r['original_fps'] for r in successful_results if r['original_fps'] > 0]
        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0

        return {
            'total_videos': total_videos,
            'successful_videos': successful_videos,
            'failed_videos': failed_videos,
            'overall_success_rate': overall_success_rate,
            'total_original_frames': total_original_frames,
            'total_processed_frames': total_processed_frames,
            'total_failed_frames': total_failed_frames,
            'frame_success_rate': total_processed_frames / max(total_original_frames, 1),
            'label_distribution': label_counts,
            'successful_label_distribution': successful_label_counts,
            'average_fps': avg_fps,
            'grid_configuration': f'{self.grid_cols}x{self.grid_rows}',
            'target_cell': f'row_{self.target_row}_col_{self.target_col}',
            'output_size': f'{self.output_size}x{self.output_size}',
            'source_directory': str(self.source_dir),
            'output_directory': str(self.output_dir),
            'processing_timestamp': datetime.now().isoformat()
        }

    def print_processing_summary(self, results: List[Dict]) -> None:
        """
        Print a comprehensive processing summary.

        Args:
            results: List of processing result dictionaries
        """
        stats = self.generate_processing_statistics(results)

        print("\n" + "="*60)
        print("GRID GEOMETRIC CROPPING PROCESSING SUMMARY")
        print("="*60)

        print(f"Source Directory: {stats['source_directory']}")
        print(f"Output Directory: {stats['output_directory']}")
        print(f"Processing Time: {stats['processing_timestamp']}")
        print()

        print("GRID CONFIGURATION:")
        print(f"  Grid Layout: {stats['grid_configuration']} (3 columns × 2 rows)")
        print(f"  Target Cell: {stats['target_cell']} (top-middle)")
        print(f"  Output Size: {stats['output_size']} pixels")
        print()

        print("PROCESSING RESULTS:")
        print(f"  Total Videos: {stats['total_videos']}")
        print(f"  Successful: {stats['successful_videos']} ({stats['overall_success_rate']:.1%})")
        print(f"  Failed: {stats['failed_videos']}")
        print()

        print("FRAME STATISTICS:")
        print(f"  Original Frames: {stats['total_original_frames']:,}")
        print(f"  Processed Frames: {stats['total_processed_frames']:,}")
        print(f"  Failed Frames: {stats['total_failed_frames']:,}")
        print(f"  Frame Success Rate: {stats['frame_success_rate']:.1%}")
        print()

        print("LABEL DISTRIBUTION:")
        for label, count in sorted(stats['label_distribution'].items()):
            successful = stats['successful_label_distribution'].get(label, 0)
            success_rate = successful / count if count > 0 else 0
            print(f"  {label}: {count} total, {successful} successful ({success_rate:.1%})")
        print()

        if stats['average_fps'] > 0:
            print(f"Average FPS: {stats['average_fps']:.1f}")

        print("="*60)


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python grid_geometric_crop.py SOURCE_DIR OUTPUT_DIR [MANIFEST_CSV]")
        print()
        print("Example:")
        print('  python grid_geometric_crop.py "/Users/client/Desktop/13.9.25top7dataset" "grid_cropped_dataset"')
        sys.exit(1)

    source_dir = sys.argv[1]
    output_dir = sys.argv[2]
    manifest_path = sys.argv[3] if len(sys.argv) > 3 else "grid_processing_manifest.csv"

    # Initialize cropper
    cropper = GridGeometricCropper(source_dir, output_dir, manifest_path)

    # Process all videos
    results = cropper.process_all_videos()

    # Save manifest and print summary
    cropper.save_processing_manifest(results)
    cropper.print_processing_summary(results)

    print(f"\nProcessing complete! Check '{output_dir}' for cropped videos.")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
