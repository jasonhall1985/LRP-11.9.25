#!/usr/bin/env python3
"""
Geometric Mouth Cropping Tool
============================

Simple, deterministic geometric cropping approach for ICU lip-reading dataset.
Processes all original videos using a 3×2 grid-based crop targeting the top-middle
region where lips consistently appear.

Features:
- Processes all original videos (no MediaPipe detection required)
- Deterministic geometric crop: top 50% height, middle 33% width
- Faster and more consistent than landmark-based approaches
- Standardized 96×96 output with 32 frames per video
- Comprehensive manifest generation

Usage:
    python geometric_crop.py INPUT_DIR OUTPUT_DIR [MANIFEST_CSV]

Example:
    python geometric_crop.py "/Users/client/Desktop/TRAINING SET 2.9.25" geometric_crops_96x96_32f geometric_manifest.csv
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

class GeometricCropper:
    """
    Geometric mouth cropping tool for ICU lip-reading videos.
    
    Uses deterministic 3×2 grid-based cropping to extract the top-middle region
    where lips consistently appear in ICU dataset videos.
    """
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 manifest_path: str = "geometric_manifest.csv"):
        """
        Initialize geometric cropper.
        
        Args:
            input_dir: Directory containing original videos
            output_dir: Directory for cropped output videos
            manifest_path: Path for output manifest CSV
        """
        self.input_dir = pathlib.Path(input_dir)
        self.output_dir = pathlib.Path(output_dir)
        self.manifest_path = pathlib.Path(manifest_path)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cropping parameters
        self.crop_height_ratio = 0.5    # Top 50% of frame height
        self.crop_width_start = 1/3     # Start at 33% of frame width
        self.crop_width_end = 2/3       # End at 67% of frame width
        
        # Output specifications
        self.output_size = 96           # 96×96 pixel output
        self.target_frames = 32         # Fixed 32 frames per video
        self.output_fps = 25            # Standard output FPS
        
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
        # Handle different filename formats
        filename_lower = filename.lower()

        # Format 1: "doctor__useruser01__18to39__male__caucasian__20250723T042153.mp4"
        if '__' in filename:
            parts = filename.split('__')
            if len(parts) > 0:
                return parts[0].lower()

        # Format 2: "doctor 1.mp4", "glasses 15.mp4", etc.
        if ' ' in filename:
            parts = filename.split(' ')
            if len(parts) > 0:
                return parts[0].lower()

        # Format 3: Check if filename starts with known class names
        known_classes = {'doctor', 'glasses', 'phone', 'pillow', 'help'}
        for class_name in known_classes:
            if filename_lower.startswith(class_name):
                return class_name

        return 'unknown'
    
    def find_all_videos(self) -> List[pathlib.Path]:
        """
        Find all video files in the input directory recursively.
        
        Returns:
            List of video file paths
        """
        video_files = []
        
        for ext in self.video_extensions:
            video_files.extend(self.input_dir.rglob(f"*{ext}"))
        
        self.logger.info(f"Found {len(video_files)} video files in {self.input_dir}")
        return video_files
    
    def geometric_crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply geometric crop to a single frame.
        
        Crops to top-middle region of 3×2 grid:
        - Vertical: Top 50% of frame height
        - Horizontal: Middle 33% of frame width (from 33% to 67%)
        
        Args:
            frame: Input frame (H, W, C)
            
        Returns:
            Cropped frame region
        """
        height, width = frame.shape[:2]
        
        # Calculate crop boundaries
        y_start = 0
        y_end = int(height * self.crop_height_ratio)
        
        x_start = int(width * self.crop_width_start)
        x_end = int(width * self.crop_width_end)
        
        # Ensure valid boundaries
        y_end = max(y_start + 1, min(y_end, height))
        x_end = max(x_start + 1, min(x_end, width))
        
        # Extract crop region
        cropped = frame[y_start:y_end, x_start:x_end]
        
        return cropped
    
    def load_and_resample_frames(self, video_path: pathlib.Path) -> Optional[List[np.ndarray]]:
        """
        Load video frames and resample to target frame count.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of resampled frames or None if loading fails
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.warning(f"Cannot open video: {video_path}")
            return None
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            self.logger.warning(f"No frames found in video: {video_path}")
            return None
        
        # Resample to target frame count
        if len(frames) == self.target_frames:
            return frames
        elif len(frames) > self.target_frames:
            # Uniform sampling
            indices = np.linspace(0, len(frames) - 1, self.target_frames, dtype=int)
            return [frames[i] for i in indices]
        else:
            # Pad with last frame
            resampled = frames.copy()
            while len(resampled) < self.target_frames:
                resampled.append(frames[-1])
            return resampled
    
    def process_video(self, video_path: pathlib.Path) -> Optional[Dict]:
        """
        Process a single video with geometric cropping.
        
        Args:
            video_path: Path to input video
            
        Returns:
            Processing result dictionary or None if failed
        """
        # Load and resample frames
        frames = self.load_and_resample_frames(video_path)
        if frames is None:
            return None
        
        # Extract label and create output filename
        label = self.get_label_from_filename(video_path.name)
        output_filename = f"{label}__geocrop__{video_path.stem}.mp4"
        output_path = self.output_dir / output_filename
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.output_fps,
            (self.output_size, self.output_size)
        )
        
        if not writer.isOpened():
            self.logger.error(f"Cannot create video writer for: {output_path}")
            return None
        
        # Process each frame
        processed_frames = 0
        for frame in frames:
            # Apply geometric crop
            cropped = self.geometric_crop_frame(frame)
            
            # Resize to target size
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
        
        writer.release()
        
        # Return processing result
        return {
            'path': str(output_path),
            'label': label,
            'frames': processed_frames,
            'fps': self.output_fps,
            'crop_method': 'geometric',
            'crop_region': f'top_{self.crop_height_ratio:.0%}_middle_{int((self.crop_width_end - self.crop_width_start) * 100)}%',
            'src': str(video_path),
            'output_size': f'{self.output_size}x{self.output_size}',
            'processing_status': 'success'
        }
    
    def process_all_videos(self) -> List[Dict]:
        """
        Process all videos in the input directory.
        
        Returns:
            List of processing results
        """
        video_files = self.find_all_videos()
        results = []
        
        self.logger.info(f"Starting geometric cropping of {len(video_files)} videos")
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                result = self.process_video(video_path)
                if result:
                    results.append(result)
                else:
                    # Record failed processing
                    results.append({
                        'path': '',
                        'label': self.get_label_from_filename(video_path.name),
                        'frames': 0,
                        'fps': 0,
                        'crop_method': 'geometric',
                        'crop_region': 'failed',
                        'src': str(video_path),
                        'output_size': f'{self.output_size}x{self.output_size}',
                        'processing_status': 'failed'
                    })
            except Exception as e:
                self.logger.error(f"Error processing {video_path}: {str(e)}")
                results.append({
                    'path': '',
                    'label': self.get_label_from_filename(video_path.name),
                    'frames': 0,
                    'fps': 0,
                    'crop_method': 'geometric',
                    'crop_region': 'error',
                    'src': str(video_path),
                    'output_size': f'{self.output_size}x{self.output_size}',
                    'processing_status': f'error: {str(e)}'
                })
        
        return results
    
    def generate_manifest(self, results: List[Dict]) -> str:
        """
        Generate processing manifest CSV.
        
        Args:
            results: List of processing results
            
        Returns:
            Path to generated manifest file
        """
        # Filter successful results
        successful_results = [r for r in results if r['processing_status'] == 'success']
        
        # Create DataFrame
        df = pd.DataFrame(successful_results)
        
        # Add summary statistics
        total_videos = len(results)
        successful_videos = len(successful_results)
        failed_videos = total_videos - successful_videos
        
        # Generate class distribution
        if successful_results:
            class_counts = df['label'].value_counts().to_dict()
        else:
            class_counts = {}
        
        # Save manifest
        df.to_csv(self.manifest_path, index=False)
        
        # Log summary
        self.logger.info(f"Processing Summary:")
        self.logger.info(f"  Total videos processed: {total_videos}")
        self.logger.info(f"  Successful: {successful_videos}")
        self.logger.info(f"  Failed: {failed_videos}")
        self.logger.info(f"  Success rate: {successful_videos/total_videos:.1%}")
        self.logger.info(f"  Class distribution: {class_counts}")
        self.logger.info(f"  Manifest saved: {self.manifest_path}")
        
        return str(self.manifest_path)

    def run_complete_pipeline(self) -> Tuple[str, Dict]:
        """
        Run the complete geometric cropping pipeline.

        Returns:
            Tuple of (manifest_path, summary_stats)
        """
        start_time = datetime.now()

        self.logger.info("=" * 60)
        self.logger.info("GEOMETRIC MOUTH CROPPING PIPELINE")
        self.logger.info("=" * 60)
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Crop strategy: Top {self.crop_height_ratio:.0%} height, Middle {int((self.crop_width_end - self.crop_width_start) * 100)}% width")
        self.logger.info(f"Output format: {self.output_size}×{self.output_size} pixels, {self.target_frames} frames")

        # Process all videos
        results = self.process_all_videos()

        # Generate manifest
        manifest_path = self.generate_manifest(results)

        # Calculate summary statistics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        successful_results = [r for r in results if r['processing_status'] == 'success']

        summary_stats = {
            'total_videos': len(results),
            'successful_videos': len(successful_results),
            'failed_videos': len(results) - len(successful_results),
            'success_rate': len(successful_results) / len(results) if results else 0,
            'processing_time_seconds': processing_time,
            'videos_per_second': len(results) / processing_time if processing_time > 0 else 0,
            'output_directory': str(self.output_dir),
            'manifest_path': manifest_path,
            'crop_method': 'geometric_3x2_grid',
            'crop_region': f'top_{self.crop_height_ratio:.0%}_middle_{int((self.crop_width_end - self.crop_width_start) * 100)}%'
        }

        if successful_results:
            df = pd.DataFrame(successful_results)
            summary_stats['class_distribution'] = df['label'].value_counts().to_dict()
        else:
            summary_stats['class_distribution'] = {}

        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 60)
        self.logger.info(f"Processing time: {processing_time:.1f} seconds")
        self.logger.info(f"Processing rate: {summary_stats['videos_per_second']:.2f} videos/second")

        return manifest_path, summary_stats


def main():
    """Main CLI interface for geometric cropping tool."""
    if len(sys.argv) < 3:
        print("Usage: python geometric_crop.py INPUT_DIR OUTPUT_DIR [MANIFEST_CSV]")
        print("\nDescription:")
        print("  Geometric mouth cropping tool for ICU lip-reading dataset.")
        print("  Uses deterministic 3×2 grid-based cropping (top-middle region).")
        print("\nArguments:")
        print("  INPUT_DIR    - Directory containing original videos")
        print("  OUTPUT_DIR   - Directory for cropped output videos")
        print("  MANIFEST_CSV - Output manifest file (optional, default: geometric_manifest.csv)")
        print("\nExamples:")
        print('  python geometric_crop.py "/Users/client/Desktop/TRAINING SET 2.9.25" geometric_crops_96x96_32f')
        print('  python geometric_crop.py "/Users/client/Desktop/VAL SET" val_geometric_crops val_manifest.csv')
        print("\nCrop Strategy:")
        print("  - Vertical: Top 50% of frame height (removes chin/lower face)")
        print("  - Horizontal: Middle 33% of frame width (centers on lip region)")
        print("  - Output: 96×96 pixels, 32 frames per video")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    manifest_csv = sys.argv[3] if len(sys.argv) > 3 else "geometric_manifest.csv"

    # Validate input directory
    if not pathlib.Path(input_dir).exists():
        print(f"❌ Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)

    try:
        # Initialize cropper
        cropper = GeometricCropper(input_dir, output_dir, manifest_csv)

        # Run complete pipeline
        manifest_path, summary_stats = cropper.run_complete_pipeline()

        print("\n" + "=" * 60)
        print("🎉 GEOMETRIC CROPPING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Processed: {summary_stats['successful_videos']}/{summary_stats['total_videos']} videos")
        print(f"⚡ Success rate: {summary_stats['success_rate']:.1%}")
        print(f"⏱️  Processing time: {summary_stats['processing_time_seconds']:.1f} seconds")
        print(f"🚀 Processing rate: {summary_stats['videos_per_second']:.2f} videos/second")
        print(f"📁 Output directory: {summary_stats['output_directory']}")
        print(f"📄 Manifest file: {manifest_path}")

        if summary_stats['class_distribution']:
            print(f"\n📈 Class Distribution:")
            for class_name, count in summary_stats['class_distribution'].items():
                print(f"   {class_name}: {count} videos")

        print(f"\n🔧 Crop Configuration:")
        print(f"   Method: {summary_stats['crop_method']}")
        print(f"   Region: {summary_stats['crop_region']}")
        print(f"   Output: 96×96 pixels, 32 frames per video")

        print(f"\n✅ Ready for classifier training!")

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
