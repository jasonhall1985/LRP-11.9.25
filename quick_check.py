#!/usr/bin/env python3
"""
Quick Check - Fast Scan-Only Summary
====================================

Fast scan-only tool for mouth ROI analysis without writing any files.
Provides quick statistics and summary of video dataset quality.

Features:
- Fast MediaPipe-based ROI analysis
- No file writing (scan-only mode)
- Summary statistics and distribution analysis
- Configurable thresholds for different quality levels
- Progress tracking with tqdm

Usage:
    python quick_check.py --in_dir INPUT_DIR [OPTIONS]

Example:
    python quick_check.py --in_dir '/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped'

Author: Augment Agent
Date: 2025-09-14
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from roi_utils import MediaPipeLipDetector, ROIGeometry, BBoxSmoother


class QuickChecker:
    """
    Fast scan-only analyzer for mouth ROI quality assessment.
    """
    
    def __init__(self, args):
        """Initialize quick checker with arguments."""
        self.args = args
        self.setup_logging()
        
        # Video processing parameters
        self.supported_extensions = {'.mp4', '.mov', '.webm', '.mkv'}
        
        # Initialize detector
        self.detector = MediaPipeLipDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.logger.info("Quick Checker initialized")
        self.logger.info(f"Input directory: {self.args.in_dir}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.args.verbose else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def find_video_files(self) -> List[Path]:
        """Find all supported video files recursively."""
        video_files = []
        in_dir = Path(self.args.in_dir)
        
        for ext in self.supported_extensions:
            video_files.extend(in_dir.rglob(f'*{ext}'))
        
        self.logger.info(f"Found {len(video_files)} video files")
        return sorted(video_files)
    
    def quick_analyze_video(self, video_path: Path) -> Dict[str, Any]:
        """
        Quick analysis of a single video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Analysis results dictionary
        """
        import cv2
        
        result = {
            'path': str(video_path),
            'name': video_path.name,
            'status': 'unknown',
            'area_ratio': 0.0,
            'h_ratio': 0.0,
            'w_ratio': 0.0,
            'detection_rate': 0.0,
            'frame_count': 0
        }
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                result['status'] = 'failed'
                return result
            
            # Get basic properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0 or fps == 0:
                cap.release()
                result['status'] = 'failed'
                return result
            
            # Sample frames for analysis
            frame_interval = max(1, int(fps / self.args.fps_sample))
            ratios_list = []
            detection_count = 0
            frame_count = 0
            
            smoother = BBoxSmoother(alpha=0.6)
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    frame_count += 1
                    
                    # Detect landmarks
                    landmarks = self.detector.detect_lip_landmarks(frame)
                    
                    if landmarks is not None:
                        detection_count += 1
                        
                        # Calculate ROI ratios
                        tight_bbox = ROIGeometry.calculate_tight_bbox(landmarks)
                        padded_bbox = ROIGeometry.add_padding(
                            tight_bbox, self.args.pad, frame.shape[:2]
                        )
                        smoothed_bbox = smoother.smooth(padded_bbox)
                        
                        ratios = ROIGeometry.calculate_size_ratios(
                            smoothed_bbox, frame.shape[:2]
                        )
                        ratios_list.append(ratios)
                
                frame_idx += 1
            
            cap.release()
            
            # Calculate results
            result['frame_count'] = frame_count
            result['detection_rate'] = detection_count / frame_count if frame_count > 0 else 0.0
            
            if ratios_list:
                area_ratios = [r['area_ratio'] for r in ratios_list]
                h_ratios = [r['h_ratio'] for r in ratios_list]
                w_ratios = [r['w_ratio'] for r in ratios_list]
                
                result['area_ratio'] = np.median(area_ratios)
                result['h_ratio'] = np.median(h_ratios)
                result['w_ratio'] = np.median(w_ratios)
                
                # Determine status
                if result['detection_rate'] < (1 - self.args.fail_tol):
                    result['status'] = 'failed'
                elif (result['area_ratio'] < self.args.min_area_ratio or
                      result['h_ratio'] < self.args.min_h_ratio or
                      result['w_ratio'] < self.args.min_w_ratio):
                    result['status'] = 'too_small'
                else:
                    result['status'] = 'pass'
            else:
                result['status'] = 'failed'
        
        except Exception as e:
            self.logger.debug(f"Error analyzing {video_path}: {e}")
            result['status'] = 'failed'
        
        return result
    
    def run_quick_check(self) -> Dict[str, Any]:
        """
        Run quick check analysis on all videos.
        
        Returns:
            Summary statistics dictionary
        """
        start_time = datetime.now()
        
        self.logger.info("=" * 60)
        self.logger.info("QUICK CHECK - MOUTH ROI ANALYSIS")
        self.logger.info("=" * 60)
        
        # Find videos
        video_files = self.find_video_files()
        if not video_files:
            self.logger.error("No video files found!")
            return {}
        
        # Analyze videos
        results = []
        for video_path in tqdm(video_files, desc="Analyzing videos"):
            result = self.quick_analyze_video(video_path)
            results.append(result)
        
        # Generate summary
        summary = self.generate_summary(results)
        
        # Log results
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.logger.info("=" * 60)
        self.logger.info("QUICK CHECK COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total videos analyzed: {len(results)}")
        self.logger.info(f"Analysis time: {duration}")
        
        self.print_summary(summary)
        
        return summary
    
    def generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        df = pd.DataFrame(results)
        
        # Status counts
        status_counts = df['status'].value_counts().to_dict()
        
        # Filter successful detections for ratio analysis
        successful = df[df['status'].isin(['pass', 'too_small'])]
        
        summary = {
            'total_videos': len(results),
            'status_counts': status_counts,
            'success_rate': len(successful) / len(results) if results else 0.0,
        }
        
        if len(successful) > 0:
            summary.update({
                'area_ratio_stats': {
                    'mean': successful['area_ratio'].mean(),
                    'median': successful['area_ratio'].median(),
                    'std': successful['area_ratio'].std(),
                    'min': successful['area_ratio'].min(),
                    'max': successful['area_ratio'].max()
                },
                'h_ratio_stats': {
                    'mean': successful['h_ratio'].mean(),
                    'median': successful['h_ratio'].median(),
                    'std': successful['h_ratio'].std(),
                    'min': successful['h_ratio'].min(),
                    'max': successful['h_ratio'].max()
                },
                'w_ratio_stats': {
                    'mean': successful['w_ratio'].mean(),
                    'median': successful['w_ratio'].median(),
                    'std': successful['w_ratio'].std(),
                    'min': successful['w_ratio'].min(),
                    'max': successful['w_ratio'].max()
                },
                'detection_rate_stats': {
                    'mean': successful['detection_rate'].mean(),
                    'median': successful['detection_rate'].median(),
                    'min': successful['detection_rate'].min(),
                    'max': successful['detection_rate'].max()
                }
            })
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print formatted summary to console."""
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        
        print(f"Total videos: {summary['total_videos']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        
        print("\nStatus Distribution:")
        for status, count in summary['status_counts'].items():
            percentage = count / summary['total_videos'] * 100
            print(f"  {status:>10}: {count:>4} ({percentage:>5.1f}%)")
        
        if 'area_ratio_stats' in summary:
            print(f"\nArea Ratio Statistics:")
            stats = summary['area_ratio_stats']
            print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  Median: {stats['median']:.3f}")
            print(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
            
            print(f"\nHeight Ratio Statistics:")
            stats = summary['h_ratio_stats']
            print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  Median: {stats['median']:.3f}")
            print(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
            
            print(f"\nWidth Ratio Statistics:")
            stats = summary['w_ratio_stats']
            print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  Median: {stats['median']:.3f}")
            print(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
            
            print(f"\nDetection Rate Statistics:")
            stats = summary['detection_rate_stats']
            print(f"  Mean: {stats['mean']:.1%}")
            print(f"  Median: {stats['median']:.1%}")
            print(f"  Range: {stats['min']:.1%} - {stats['max']:.1%}")
        
        print("\nThreshold Analysis:")
        print(f"  Current thresholds:")
        print(f"    Min area ratio: {self.args.min_area_ratio:.2f}")
        print(f"    Min height ratio: {self.args.min_h_ratio:.2f}")
        print(f"    Min width ratio: {self.args.min_w_ratio:.2f}")
        
        if 'too_small' in summary['status_counts']:
            too_small_count = summary['status_counts']['too_small']
            too_small_pct = too_small_count / summary['total_videos'] * 100
            print(f"  Videos flagged as too small: {too_small_count} ({too_small_pct:.1f}%)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quick Check - Fast scan-only mouth ROI analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # I/O arguments
    parser.add_argument('--in_dir', type=str, required=True,
                       help='Input directory containing videos')
    
    # Threshold arguments
    parser.add_argument('--min_area_ratio', type=float, default=0.30,
                       help='Minimum area ratio threshold')
    parser.add_argument('--min_h_ratio', type=float, default=0.40,
                       help='Minimum height ratio threshold')
    parser.add_argument('--min_w_ratio', type=float, default=0.40,
                       help='Minimum width ratio threshold')
    
    # Processing arguments
    parser.add_argument('--fps_sample', type=int, default=5,
                       help='Frame sampling rate for analysis')
    parser.add_argument('--pad', type=float, default=0.12,
                       help='Padding ratio around detected ROI')
    parser.add_argument('--fail_tol', type=float, default=0.30,
                       help='Failure tolerance (30%% = 0.30)')
    
    # Control arguments
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Run quick check
    checker = QuickChecker(args)
    summary = checker.run_quick_check()
    
    print(f"\nQuick check completed!")


if __name__ == "__main__":
    main()
