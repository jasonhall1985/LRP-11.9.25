#!/usr/bin/env python3
"""
Fixed Mouth ROI Pipeline - Production Version
=============================================

Production-ready mouth ROI standardization pipeline with adaptive detection
that works with both full face videos and cropped face videos (like ICU dataset).

Key improvements:
- Automatic detection of video type (full face vs cropped face)
- Geometric lip detection for cropped face videos
- MediaPipe Face Mesh for full face videos
- Robust error handling and logging

Author: Augment Agent
Date: 2025-09-14
"""

import cv2
import numpy as np
import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from multiprocessing import Pool, cpu_count
import time
from tqdm import tqdm

# Import our improved utilities
from improved_roi_utils import AdaptiveLipDetector
from roi_utils import ROIGeometry, BBoxSmoother, RecropCalculator


class FixedMouthROIProcessor:
    """
    Fixed mouth ROI processor with adaptive detection for different video types.
    """
    
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(__name__)
        
        # Initialize detector (will be done per worker process)
        self.detector = None
        
    def process_single_video(self, video_info: Tuple[str, str]) -> Dict[str, Any]:
        """
        Process a single video file with adaptive lip detection.
        
        Args:
            video_info: Tuple of (input_path, output_path)
            
        Returns:
            Processing results dictionary
        """
        input_path, relative_path = video_info
        
        # Initialize detector if not already done (for multiprocessing)
        if self.detector is None:
            self.detector = AdaptiveLipDetector(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                auto_detect_mode=True
            )
        
        try:
            return self._process_video_adaptive(input_path, relative_path)
        except Exception as e:
            self.logger.error(f"Error processing {input_path}: {str(e)}")
            return {
                'input_path': input_path,
                'status': 'error',
                'reason': str(e),
                'output_path': None
            }
    
    def _process_video_adaptive(self, input_path: str, relative_path: str) -> Dict[str, Any]:
        """Process video with adaptive detection method."""
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return {
                'input_path': input_path,
                'status': 'failed',
                'reason': 'Could not open video file',
                'output_path': None
            }
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Sample frames for analysis
        sample_interval = max(1, total_frames // self.args.fps_sample)
        
        # Initialize smoother
        smoother = BBoxSmoother(alpha=self.args.ema)
        
        # Statistics
        detected_frames = 0
        failed_frames = 0
        bboxes = []
        
        # Process frames
        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect lip landmarks
            landmarks = self.detector.detect_lip_landmarks(frame)
            
            if landmarks is not None:
                detected_frames += 1
                
                # Calculate bounding box
                tight_bbox = ROIGeometry.calculate_tight_bbox(landmarks)
                padded_bbox = ROIGeometry.add_padding(
                    tight_bbox, self.args.pad, frame.shape[:2]
                )
                smoothed_bbox = smoother.smooth(padded_bbox)
                bboxes.append(smoothed_bbox)
                
            else:
                failed_frames += 1
                
            frame_idx += sample_interval
            
        cap.release()
        
        # Analyze results
        if detected_frames == 0:
            return {
                'input_path': input_path,
                'status': 'failed',
                'reason': 'No lip landmarks detected in any frame',
                'detection_mode': self.detector.get_detection_mode(),
                'output_path': None
            }
        
        detection_rate = detected_frames / (detected_frames + failed_frames)
        
        if detection_rate < 0.3:  # Less than 30% detection rate
            return {
                'input_path': input_path,
                'status': 'failed',
                'reason': f'Low detection rate: {detection_rate:.1%}',
                'detection_mode': self.detector.get_detection_mode(),
                'output_path': None
            }
        
        # Calculate average ROI size
        if bboxes:
            avg_bbox = np.mean(bboxes, axis=0).astype(int)
            ratios = ROIGeometry.calculate_size_ratios(avg_bbox, (height, width))
            
            # Check if ROI meets size requirements
            if (ratios['area_ratio'] >= self.args.min_area_ratio and
                ratios['h_ratio'] >= self.args.min_h_ratio and
                ratios['w_ratio'] >= self.args.min_w_ratio):
                
                # ROI is good - copy to keep folder
                output_path = self._get_output_path(relative_path, 'keep')
                if not self.args.dry_run:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    # Copy original file
                    import shutil
                    shutil.copy2(input_path, output_path)
                
                return {
                    'input_path': input_path,
                    'status': 'keep',
                    'reason': 'ROI meets size requirements',
                    'detection_mode': self.detector.get_detection_mode(),
                    'detection_rate': detection_rate,
                    'ratios': ratios,
                    'output_path': output_path
                }
            else:
                # ROI is too small - recrop
                output_path = self._get_output_path(relative_path, 'recrop')
                if not self.args.dry_run:
                    success = self._recrop_video(input_path, output_path, avg_bbox)
                    if not success:
                        return {
                            'input_path': input_path,
                            'status': 'failed',
                            'reason': 'Failed to recrop video',
                            'detection_mode': self.detector.get_detection_mode(),
                            'output_path': None
                        }
                
                return {
                    'input_path': input_path,
                    'status': 'too_small',
                    'reason': 'ROI below size thresholds - recropped',
                    'detection_mode': self.detector.get_detection_mode(),
                    'detection_rate': detection_rate,
                    'ratios': ratios,
                    'output_path': output_path
                }
    
    def _get_output_path(self, relative_path: str, category: str) -> str:
        """Get output path for processed video."""
        filename = os.path.basename(relative_path)
        if category == 'recrop':
            filename = f"recropped_{filename}"
        elif category == 'failed':
            filename = f"failed_{filename}"
            
        return os.path.join(self.args.out_dir, category, filename)
    
    def _recrop_video(self, input_path: str, output_path: str, avg_bbox: np.ndarray) -> bool:
        """Recrop video to focus on lip region."""
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return False
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate crop window
            frame_shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
                          int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            
            crop_window = RecropCalculator.calculate_recrop_window(
                tuple(avg_bbox),
                self.args.target_h_ratio,
                self.args.target_w_ratio,
                frame_shape
            )
            
            # Setup video writer
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (self.args.out_size, self.args.out_size))
            
            # Process all frames
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Crop frame
                x1, y1, x2, y2 = crop_window
                cropped = frame[y1:y2, x1:x2]
                
                if cropped.size > 0:
                    # Resize to target size
                    resized = cv2.resize(cropped, (self.args.out_size, self.args.out_size))
                    out.write(resized)
                    frame_count += 1
                    
            cap.release()
            out.release()
            
            self.logger.info(f"Recropped {frame_count} frames: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error recropping video {input_path}: {str(e)}")
            return False


def worker_init():
    """Initialize worker process."""
    pass


def worker_process_video(args_and_video):
    """Worker function for multiprocessing."""
    args, video_info = args_and_video
    processor = FixedMouthROIProcessor(args)
    return processor.process_single_video(video_info)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fixed Mouth ROI Pipeline with Adaptive Detection")
    
    # Input/Output
    parser.add_argument('--in_dir', type=str, required=True, help='Input directory with videos')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    
    # Size thresholds
    parser.add_argument('--min_area_ratio', type=float, default=0.30, help='Minimum area ratio')
    parser.add_argument('--min_h_ratio', type=float, default=0.40, help='Minimum height ratio')
    parser.add_argument('--min_w_ratio', type=float, default=0.40, help='Minimum width ratio')
    
    # Recrop parameters
    parser.add_argument('--target_h_ratio', type=float, default=0.50, help='Target height ratio for recrop')
    parser.add_argument('--target_w_ratio', type=float, default=0.50, help='Target width ratio for recrop')
    parser.add_argument('--out_size', type=int, default=96, help='Output video size (square)')
    
    # Processing parameters
    parser.add_argument('--fps_sample', type=int, default=5, help='Frame sampling rate')
    parser.add_argument('--pad', type=float, default=0.12, help='Padding ratio around ROI')
    parser.add_argument('--ema', type=float, default=0.3, help='EMA smoothing factor')
    parser.add_argument('--workers', type=int, default=6, help='Number of worker processes')
    
    # Options
    parser.add_argument('--dry_run', action='store_true', help='Dry run - no output files')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Fixed Mouth ROI Pipeline with Adaptive Detection")
    logger.info(f"Input directory: {args.in_dir}")
    logger.info(f"Output directory: {args.out_dir}")
    
    # Find all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
    video_files = []
    
    for root, dirs, files in os.walk(args.in_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, args.in_dir)
                video_files.append((full_path, rel_path))
    
    logger.info(f"Found {len(video_files)} video files")
    
    if not video_files:
        logger.error("No video files found!")
        return
    
    # Create output directories
    if not args.dry_run:
        for subdir in ['keep', 'recrop', 'failed']:
            os.makedirs(os.path.join(args.out_dir, subdir), exist_ok=True)
    
    # Process videos
    start_time = time.time()
    
    if args.workers > 1:
        # Multiprocessing
        with Pool(processes=args.workers, initializer=worker_init) as pool:
            tasks = [(args, video_info) for video_info in video_files]
            results = list(tqdm(
                pool.imap(worker_process_video, tasks),
                total=len(video_files),
                desc="Processing videos"
            ))
    else:
        # Single process
        processor = FixedMouthROIProcessor(args)
        results = []
        for video_info in tqdm(video_files, desc="Processing videos"):
            result = processor.process_single_video(video_info)
            results.append(result)
    
    # Generate report
    processing_time = time.time() - start_time
    
    # Analyze results
    status_counts = {}
    detection_modes = {}
    
    for result in results:
        status = result.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
        
        mode = result.get('detection_mode')
        if mode:
            detection_modes[mode] = detection_modes.get(mode, 0) + 1
    
    # Save detailed report
    report_path = os.path.join(args.out_dir, 'fixed_pipeline_report.json')
    report = {
        'processing_time': processing_time,
        'total_videos': len(video_files),
        'status_counts': status_counts,
        'detection_modes': detection_modes,
        'results': results
    }
    
    if not args.dry_run:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total videos: {len(video_files)}")
    logger.info(f"Processing time: {processing_time:.1f} seconds")
    logger.info(f"Speed: {len(video_files)/processing_time:.1f} videos/second")
    
    logger.info("\nStatus breakdown:")
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")
    
    logger.info("\nDetection modes:")
    for mode, count in detection_modes.items():
        logger.info(f"  {mode}: {count}")
    
    if not args.dry_run:
        logger.info(f"\nDetailed report saved: {report_path}")


if __name__ == "__main__":
    main()
