#!/usr/bin/env python3
"""
Mouth ROI Pipeline - Main CLI Tool
==================================

Standardizes mouth ROIs in videos using MediaPipe Face Mesh and auto-re-crops clips 
where the lips are too small. Provides robust logging and fast batching.

Features:
- MediaPipe Face Mesh lip detection with configurable thresholds
- Automatic size analysis and flagging of too-small ROIs
- Smart re-cropping strategy for flagged videos
- EMA smoothing for stable bounding boxes
- Comprehensive CSV reporting and debug visualizations
- Multi-threaded processing with timeout protection

Usage:
    python mouth_roi_pipeline.py --in_dir INPUT --out_dir OUTPUT [OPTIONS]

Author: Augment Agent
Date: 2025-09-14
"""

import argparse
import os
import sys
import logging
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import signal
import time

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from roi_utils import (
    MediaPipeLipDetector, ROIGeometry, BBoxSmoother, RecropCalculator,
    create_debug_visualization
)


class MouthROIPipeline:
    """
    Main pipeline for mouth ROI standardization and analysis.
    """
    
    def __init__(self, args):
        """Initialize pipeline with command line arguments."""
        self.args = args
        self.setup_logging()
        self.setup_directories()
        
        # Video processing parameters
        self.supported_extensions = {'.mp4', '.mov', '.webm', '.mkv'}
        
        # Detector will be initialized per worker process
        self.detector = None
        
        self.logger.info("Mouth ROI Pipeline initialized")
        self.logger.info(f"Input directory: {self.args.in_dir}")
        self.logger.info(f"Output directory: {self.args.out_dir}")
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.args.verbose else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'mouth_roi_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create output directory structure."""
        self.out_dir = Path(self.args.out_dir)
        
        if not self.args.dry_run:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / 'keep').mkdir(exist_ok=True)
            (self.out_dir / 'recrop').mkdir(exist_ok=True)
            (self.out_dir / 'failed').mkdir(exist_ok=True)
            (self.out_dir / 'debug').mkdir(exist_ok=True)
            
            self.logger.info(f"Created output directories in {self.out_dir}")
    
    def find_video_files(self) -> List[Path]:
        """Find all supported video files recursively."""
        video_files = []
        in_dir = Path(self.args.in_dir)
        
        for ext in self.supported_extensions:
            video_files.extend(in_dir.rglob(f'*{ext}'))
        
        self.logger.info(f"Found {len(video_files)} video files")
        return sorted(video_files)
    
    def sample_frames(self, video_path: Path) -> Optional[List[np.ndarray]]:
        """Sample frames from video at specified FPS."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0 or fps == 0:
            cap.release()
            return None
        
        # Calculate frame sampling
        frame_interval = max(1, int(fps / self.args.fps_sample))
        sampled_frames = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                sampled_frames.append(frame.copy())
                
            frame_idx += 1
        
        cap.release()
        
        self.logger.debug(f"Sampled {len(sampled_frames)} frames from {video_path.name}")
        return sampled_frames if sampled_frames else None
    
    def analyze_video_roi(self, video_path: Path) -> Dict[str, Any]:
        """
        Analyze video for mouth ROI size and quality.

        Returns:
            Analysis results dictionary
        """
        # Initialize detector if not already done (for multiprocessing)
        if self.detector is None:
            self.detector = MediaPipeLipDetector(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        result = {
            'input_path': str(video_path),
            'status': 'unknown',
            'reason': '',
            'width': 0,
            'height': 0,
            'fps': 0,
            'n_frames_sampled': 0,
            'area_ratio_med': 0.0,
            'h_ratio_med': 0.0,
            'w_ratio_med': 0.0,
            'out_path': '',
            'notes': ''
        }
        
        try:
            # Sample frames
            frames = self.sample_frames(video_path)
            if not frames:
                result.update({
                    'status': 'failed',
                    'reason': 'Cannot read video or no frames',
                    'notes': 'Video file corrupted or unsupported format'
                })
                return result
            
            # Get video properties
            cap = cv2.VideoCapture(str(video_path))
            result['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            result['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            result['fps'] = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            result['n_frames_sampled'] = len(frames)
            
            # Analyze each frame
            ratios_list = []
            smoother = BBoxSmoother(alpha=self.args.ema)
            detection_failures = 0
            
            for i, frame in enumerate(frames):
                # Detect landmarks
                landmarks = self.detector.detect_lip_landmarks(frame)
                
                if landmarks is None:
                    detection_failures += 1
                    continue
                
                # Calculate tight bbox
                tight_bbox = ROIGeometry.calculate_tight_bbox(landmarks)
                
                # Add padding
                padded_bbox = ROIGeometry.add_padding(
                    tight_bbox, self.args.pad, frame.shape[:2]
                )
                
                # Smooth bbox
                smoothed_bbox = smoother.smooth(padded_bbox)
                
                # Calculate ratios
                ratios = ROIGeometry.calculate_size_ratios(
                    smoothed_bbox, frame.shape[:2]
                )
                ratios_list.append(ratios)
                
                # Save debug image for first frame
                if i == 0 and not self.args.dry_run:
                    debug_frame = create_debug_visualization(
                        frame, landmarks, smoothed_bbox, None, ratios
                    )
                    debug_path = self.out_dir / 'debug' / f"{video_path.stem}_debug.jpg"
                    cv2.imwrite(str(debug_path), debug_frame)
            
            # Check failure tolerance
            failure_rate = detection_failures / len(frames)
            if failure_rate >= self.args.fail_tol:
                result.update({
                    'status': 'failed',
                    'reason': f'High detection failure rate: {failure_rate:.2%}',
                    'notes': f'Failed on {detection_failures}/{len(frames)} frames'
                })
                return result
            
            if not ratios_list:
                result.update({
                    'status': 'failed',
                    'reason': 'No successful detections',
                    'notes': 'MediaPipe failed to detect lips in all frames'
                })
                return result
            
            # Calculate median ratios
            area_ratios = [r['area_ratio'] for r in ratios_list]
            h_ratios = [r['h_ratio'] for r in ratios_list]
            w_ratios = [r['w_ratio'] for r in ratios_list]
            
            result['area_ratio_med'] = np.median(area_ratios)
            result['h_ratio_med'] = np.median(h_ratios)
            result['w_ratio_med'] = np.median(w_ratios)
            
            # Determine if ROI is too small
            is_too_small = (
                result['area_ratio_med'] < self.args.min_area_ratio or
                result['h_ratio_med'] < self.args.min_h_ratio or
                result['w_ratio_med'] < self.args.min_w_ratio
            )
            
            if is_too_small:
                result.update({
                    'status': 'too_small',
                    'reason': 'ROI below size thresholds',
                    'notes': f"area:{result['area_ratio_med']:.3f} h:{result['h_ratio_med']:.3f} w:{result['w_ratio_med']:.3f}"
                })
            else:
                result.update({
                    'status': 'pass',
                    'reason': 'ROI meets size requirements',
                    'notes': f"area:{result['area_ratio_med']:.3f} h:{result['h_ratio_med']:.3f} w:{result['w_ratio_med']:.3f}"
                })
            
        except Exception as e:
            self.logger.error(f"Error analyzing {video_path}: {e}")
            result.update({
                'status': 'failed',
                'reason': f'Processing error: {str(e)}',
                'notes': 'Unexpected error during analysis'
            })
        
        return result
    
    def process_single_video(self, video_path: Path) -> Dict[str, Any]:
        """Process a single video file."""
        try:
            # Set timeout alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(self.args.timeout_s)
            
            result = self.analyze_video_roi(video_path)
            
            if not self.args.dry_run:
                if result['status'] == 'pass':
                    # Copy to keep directory
                    if self.args.normalize_keep:
                        result['out_path'] = self.recrop_video(video_path, result, 'keep')
                    else:
                        result['out_path'] = self.copy_video(video_path, 'keep')
                        
                elif result['status'] == 'too_small':
                    # Recrop video
                    result['out_path'] = self.recrop_video(video_path, result, 'recrop')
                    
                elif result['status'] == 'failed':
                    # Copy to failed directory
                    result['out_path'] = self.copy_video(video_path, 'failed')
            
            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {video_path}: {e}")
            return {
                'input_path': str(video_path),
                'status': 'failed',
                'reason': f'Processing error: {str(e)}',
                'out_path': '',
                'notes': 'Timeout or unexpected error'
            }
    
    def copy_video(self, video_path: Path, subdir: str) -> str:
        """Copy video to output subdirectory."""
        import shutil
        
        out_path = self.out_dir / subdir / video_path.name
        shutil.copy2(video_path, out_path)
        return str(out_path)
    
    def recrop_video(self, video_path: Path, analysis: Dict, subdir: str) -> str:
        """Recrop video with optimized framing."""
        try:
            # Initialize detector if not already done (for multiprocessing)
            if self.detector is None:
                self.detector = MediaPipeLipDetector(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            # Open input video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"Cannot open video for recropping: {video_path}")
                return self.copy_video(video_path, 'failed')

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Setup output video
            output_filename = f"recropped_{video_path.name}"
            output_path = self.out_dir / subdir / output_filename

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path), fourcc, fps,
                (self.args.out_size, self.args.out_size)
            )

            # Initialize processing components
            smoother = BBoxSmoother(alpha=self.args.ema)
            crop_smoother = BBoxSmoother(alpha=self.args.ema * 0.8)  # Slower for crop window

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect landmarks
                landmarks = self.detector.detect_lip_landmarks(frame)

                if landmarks is not None:
                    # Calculate and smooth ROI bbox
                    tight_bbox = ROIGeometry.calculate_tight_bbox(landmarks)
                    padded_bbox = ROIGeometry.add_padding(
                        tight_bbox, self.args.pad, frame.shape[:2]
                    )
                    smoothed_bbox = smoother.smooth(padded_bbox)

                    # Calculate recrop window
                    crop_window = RecropCalculator.calculate_recrop_window(
                        smoothed_bbox,
                        self.args.target_h_ratio,
                        self.args.target_w_ratio,
                        frame.shape[:2],
                        safety_margin=0.05
                    )

                    # Smooth crop window
                    crop_window = crop_smoother.smooth(crop_window)

                else:
                    # Use previous crop window if detection fails
                    if crop_smoother.smoothed_bbox is not None:
                        crop_window = crop_smoother.smoothed_bbox
                    else:
                        # Fallback to center crop
                        h, w = frame.shape[:2]
                        size = min(h, w)
                        x1 = (w - size) // 2
                        y1 = (h - size) // 2
                        crop_window = (x1, y1, x1 + size, y1 + size)

                # Apply crop and resize
                x1, y1, x2, y2 = crop_window
                cropped = frame[y1:y2, x1:x2]

                if cropped.size > 0:
                    resized = cv2.resize(cropped, (self.args.out_size, self.args.out_size))
                    writer.write(resized)

                frame_count += 1

            cap.release()
            writer.release()

            self.logger.debug(f"Recropped {frame_count} frames for {video_path.name}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"Error recropping {video_path}: {e}")
            return self.copy_video(video_path, 'failed')
    
    def run_pipeline(self) -> str:
        """Run the complete pipeline."""
        start_time = datetime.now()
        
        self.logger.info("=" * 60)
        self.logger.info("MOUTH ROI STANDARDIZATION PIPELINE")
        self.logger.info("=" * 60)
        
        # Find videos
        video_files = self.find_video_files()
        if not video_files:
            self.logger.error("No video files found!")
            return ""
        
        # Process videos
        if self.args.workers == 1:
            # Single-threaded processing
            results = []
            for video_path in tqdm(video_files, desc="Processing videos"):
                result = self.process_single_video(video_path)
                results.append(result)
        else:
            # Multi-threaded processing
            with mp.Pool(self.args.workers) as pool:
                results = list(tqdm(
                    pool.imap(self.process_single_video, video_files),
                    total=len(video_files),
                    desc="Processing videos"
                ))
        
        # Generate report
        report_path = self.generate_report(results)
        
        # Log summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total videos processed: {len(results)}")
        self.logger.info(f"Processing time: {duration}")
        self.logger.info(f"Report saved: {report_path}")
        
        return report_path
    
    def generate_report(self, results: List[Dict]) -> str:
        """Generate CSV report from results."""
        if self.args.dry_run:
            report_path = "mouth_roi_report_dry_run.csv"
        else:
            report_path = str(self.out_dir / "mouth_roi_report.csv")
        
        df = pd.DataFrame(results)
        df.to_csv(report_path, index=False)
        
        # Log summary statistics
        status_counts = df['status'].value_counts()
        self.logger.info("Status summary:")
        for status, count in status_counts.items():
            self.logger.info(f"  {status}: {count}")
        
        return report_path


def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutError("Video processing timeout")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mouth ROI Pipeline - Standardize mouth ROIs using MediaPipe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # I/O arguments
    parser.add_argument('--in_dir', type=str, 
                       default='/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped',
                       help='Input directory containing videos')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='Output directory for processed videos')
    
    # Size threshold arguments
    parser.add_argument('--min_area_ratio', type=float, default=0.30,
                       help='Minimum area ratio threshold')
    parser.add_argument('--min_h_ratio', type=float, default=0.40,
                       help='Minimum height ratio threshold')
    parser.add_argument('--min_w_ratio', type=float, default=0.40,
                       help='Minimum width ratio threshold')
    
    # Recrop target arguments
    parser.add_argument('--target_h_ratio', type=float, default=0.50,
                       help='Target height ratio for recropping')
    parser.add_argument('--target_w_ratio', type=float, default=0.50,
                       help='Target width ratio for recropping')
    
    # Processing arguments
    parser.add_argument('--out_size', type=int, default=96,
                       help='Output video size (square)')
    parser.add_argument('--fps_sample', type=int, default=5,
                       help='Frame sampling rate for analysis')
    parser.add_argument('--pad', type=float, default=0.12,
                       help='Padding ratio around detected ROI')
    parser.add_argument('--ema', type=float, default=0.6,
                       help='EMA smoothing factor for bbox')
    parser.add_argument('--fail_tol', type=float, default=0.30,
                       help='Failure tolerance (30%% = 0.30)')
    
    # System arguments
    parser.add_argument('--workers', type=int, default=mp.cpu_count(),
                       help='Number of worker processes')
    parser.add_argument('--timeout_s', type=int, default=120,
                       help='Timeout per video in seconds')
    
    # Control arguments
    parser.add_argument('--dry_run', action='store_true',
                       help='Analyze only, do not write files')
    parser.add_argument('--normalize_keep', action='store_true',
                       help='Apply light standardization to passing videos')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup timeout handler
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
    
    # Run pipeline
    pipeline = MouthROIPipeline(args)
    report_path = pipeline.run_pipeline()
    
    print(f"\nPipeline completed! Report saved to: {report_path}")


if __name__ == "__main__":
    main()
