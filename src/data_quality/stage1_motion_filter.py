#!/usr/bin/env python3
"""
Stage 1: Motion Detection Filter for ICU Lip-Reading Dataset
Removes videos with insufficient motion (static frames, corrupted files, non-speaking content)
"""

import os
import cv2
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import json

class MotionDetectionFilter:
    """
    Filters videos based on frame-to-frame motion analysis.
    Removes videos with insufficient motion indicating static content or corruption.
    """
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 motion_threshold: float = 0.10,  # 10% of frames must show significant motion
                 pixel_change_threshold: float = 0.05,  # 5% pixel change threshold
                 log_dir: str = "logs"):
        """
        Initialize motion detection filter.
        
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory for filtered videos
            motion_threshold: Minimum fraction of frames with significant motion
            pixel_change_threshold: Minimum pixel change ratio for motion detection
            log_dir: Directory for log files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.motion_threshold = motion_threshold
        self.pixel_change_threshold = pixel_change_threshold
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Results storage
        self.results = []
        self.class_stats = {}
        
    def setup_logging(self):
        """Setup detailed logging for motion analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"stage1_motion_filter_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Motion Detection Filter initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Motion threshold: {self.motion_threshold}")
        self.logger.info(f"Pixel change threshold: {self.pixel_change_threshold}")
        
    def extract_class_from_filename(self, filename: str) -> str:
        """Extract class label from filename."""
        classes = ['doctor', 'glasses', 'phone', 'pillow', 'help']
        for cls in classes:
            if filename.startswith(cls + '__'):
                return cls
        return 'unknown'
    
    def calculate_optical_flow_motion(self, video_path: str) -> Tuple[float, int, str]:
        """
        Calculate motion score using optical flow analysis.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (motion_score, total_frames, analysis_status)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0, 0, "failed_to_open"
            
            # Read first frame
            ret, prev_frame = cap.read()
            if not ret:
                cap.release()
                return 0.0, 0, "no_frames"
            
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            motion_frames = 0
            total_frames = 1
            
            while True:
                ret, curr_frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate frame difference
                frame_diff = cv2.absdiff(prev_gray, curr_gray)
                
                # Calculate percentage of changed pixels
                total_pixels = frame_diff.shape[0] * frame_diff.shape[1]
                changed_pixels = np.count_nonzero(frame_diff > 30)  # Threshold for significant change
                change_ratio = changed_pixels / total_pixels
                
                # Check if motion exceeds threshold
                if change_ratio > self.pixel_change_threshold:
                    motion_frames += 1
                
                prev_gray = curr_gray
                total_frames += 1
            
            cap.release()
            
            # Calculate motion score
            motion_score = motion_frames / total_frames if total_frames > 0 else 0.0
            
            return motion_score, total_frames, "success"
            
        except Exception as e:
            self.logger.error(f"Error analyzing {video_path}: {str(e)}")
            return 0.0, 0, f"error: {str(e)}"
    
    def analyze_video(self, video_path: Path) -> Dict:
        """
        Analyze a single video for motion content.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with analysis results
        """
        filename = video_path.name
        class_label = self.extract_class_from_filename(filename)
        
        self.logger.info(f"Analyzing: {filename}")
        
        # Calculate motion score
        motion_score, total_frames, status = self.calculate_optical_flow_motion(str(video_path))
        
        # Determine if video passes filter
        passes_filter = (motion_score >= self.motion_threshold and 
                        status == "success" and 
                        total_frames > 5)  # Minimum frame requirement
        
        # Determine removal reason
        removal_reason = None
        if not passes_filter:
            if status != "success":
                removal_reason = f"analysis_failed: {status}"
            elif total_frames <= 5:
                removal_reason = "insufficient_frames"
            elif motion_score < self.motion_threshold:
                removal_reason = f"insufficient_motion: {motion_score:.3f} < {self.motion_threshold}"
        
        result = {
            'filename': filename,
            'class': class_label,
            'motion_score': motion_score,
            'total_frames': total_frames,
            'analysis_status': status,
            'passes_filter': passes_filter,
            'removal_reason': removal_reason,
            'file_size_mb': video_path.stat().st_size / (1024 * 1024)
        }
        
        self.logger.info(f"  Motion score: {motion_score:.3f}, Frames: {total_frames}, "
                        f"Passes: {passes_filter}, Reason: {removal_reason}")
        
        return result

    def process_videos(self) -> None:
        """Process all videos in the input directory."""
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov']
        video_files = []

        for ext in video_extensions:
            video_files.extend(list(self.input_dir.glob(f'*{ext}')))

        self.logger.info(f"Found {len(video_files)} video files to process")

        # Initialize class statistics
        classes = ['doctor', 'glasses', 'phone', 'pillow', 'help']
        for cls in classes:
            self.class_stats[cls] = {'total': 0, 'passed': 0, 'removed': 0}

        # Process each video with progress bar
        for video_path in tqdm(video_files, desc="Analyzing videos"):
            result = self.analyze_video(video_path)
            self.results.append(result)

            # Update class statistics
            class_label = result['class']
            if class_label in self.class_stats:
                self.class_stats[class_label]['total'] += 1
                if result['passes_filter']:
                    self.class_stats[class_label]['passed'] += 1
                else:
                    self.class_stats[class_label]['removed'] += 1

        self.logger.info("Video analysis completed")

    def copy_filtered_videos(self) -> None:
        """Copy videos that passed the filter to output directory."""
        self.logger.info("Copying filtered videos to output directory...")

        copied_count = 0
        for result in tqdm(self.results, desc="Copying videos"):
            if result['passes_filter']:
                src_path = self.input_dir / result['filename']
                dst_path = self.output_dir / result['filename']

                try:
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to copy {result['filename']}: {str(e)}")

        self.logger.info(f"Successfully copied {copied_count} videos")

    def generate_reports(self) -> None:
        """Generate detailed reports and CSV manifest."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results to CSV
        df = pd.DataFrame(self.results)
        csv_path = self.log_dir / f"stage1_detailed_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Detailed results saved to: {csv_path}")

        # Generate summary report
        report_path = self.log_dir / f"stage1_summary_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("STAGE 1: MOTION DETECTION FILTER - SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Motion threshold: {self.motion_threshold}\n")
            f.write(f"Pixel change threshold: {self.pixel_change_threshold}\n\n")

            # Overall statistics
            total_videos = len(self.results)
            passed_videos = sum(1 for r in self.results if r['passes_filter'])
            removed_videos = total_videos - passed_videos

            f.write("OVERALL STATISTICS:\n")
            f.write(f"Total videos analyzed: {total_videos}\n")
            f.write(f"Videos passed filter: {passed_videos} ({passed_videos/total_videos*100:.1f}%)\n")
            f.write(f"Videos removed: {removed_videos} ({removed_videos/total_videos*100:.1f}%)\n\n")

            # Class-wise statistics
            f.write("CLASS-WISE STATISTICS:\n")
            f.write("-" * 40 + "\n")
            for cls, stats in self.class_stats.items():
                if stats['total'] > 0:
                    pass_rate = stats['passed'] / stats['total'] * 100
                    f.write(f"{cls.upper()}:\n")
                    f.write(f"  Total: {stats['total']}\n")
                    f.write(f"  Passed: {stats['passed']} ({pass_rate:.1f}%)\n")
                    f.write(f"  Removed: {stats['removed']}\n\n")

            # Removal reasons analysis
            f.write("REMOVAL REASONS:\n")
            f.write("-" * 40 + "\n")
            removal_reasons = {}
            for result in self.results:
                if not result['passes_filter'] and result['removal_reason']:
                    reason = result['removal_reason'].split(':')[0]  # Get main reason
                    removal_reasons[reason] = removal_reasons.get(reason, 0) + 1

            for reason, count in sorted(removal_reasons.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{reason}: {count} videos\n")

            f.write(f"\nRecommendations:\n")
            f.write("-" * 40 + "\n")

            # Motion score distribution analysis
            motion_scores = [r['motion_score'] for r in self.results if r['analysis_status'] == 'success']
            if motion_scores:
                avg_motion = np.mean(motion_scores)
                std_motion = np.std(motion_scores)
                f.write(f"Average motion score: {avg_motion:.3f} ± {std_motion:.3f}\n")

                if avg_motion < self.motion_threshold * 1.5:
                    f.write("⚠️  Consider lowering motion threshold - many videos have low motion\n")
                if removed_videos > total_videos * 0.3:
                    f.write("⚠️  High removal rate - review threshold settings\n")

                # Check for class imbalance after filtering
                min_class_count = min(stats['passed'] for stats in self.class_stats.values())
                if min_class_count < 10:
                    f.write("⚠️  Some classes have very few samples after filtering\n")

        self.logger.info(f"Summary report saved to: {report_path}")

        # Save configuration for reproducibility
        config_path = self.log_dir / f"stage1_config_{timestamp}.json"
        config = {
            'motion_threshold': self.motion_threshold,
            'pixel_change_threshold': self.pixel_change_threshold,
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'timestamp': timestamp,
            'total_processed': total_videos,
            'total_passed': passed_videos
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Configuration saved to: {config_path}")

    def run_filter(self) -> Dict:
        """
        Run the complete motion detection filter pipeline.

        Returns:
            Dictionary with summary statistics
        """
        self.logger.info("Starting Stage 1: Motion Detection Filter")

        # Process all videos
        self.process_videos()

        # Copy filtered videos
        self.copy_filtered_videos()

        # Generate reports
        self.generate_reports()

        # Return summary statistics
        total_videos = len(self.results)
        passed_videos = sum(1 for r in self.results if r['passes_filter'])

        summary = {
            'total_videos': total_videos,
            'passed_videos': passed_videos,
            'removed_videos': total_videos - passed_videos,
            'pass_rate': passed_videos / total_videos if total_videos > 0 else 0,
            'class_stats': self.class_stats.copy()
        }

        self.logger.info("Stage 1 Motion Detection Filter completed successfully!")
        self.logger.info(f"Summary: {passed_videos}/{total_videos} videos passed "
                        f"({summary['pass_rate']*100:.1f}% pass rate)")

        return summary


def main():
    """Main execution function for Stage 1 Motion Detection Filter."""
    import argparse

    parser = argparse.ArgumentParser(description='Stage 1: Motion Detection Filter')
    parser.add_argument('--input_dir', type=str,
                       default='data/grid/13.9.25top7dataset',
                       help='Input directory containing videos')
    parser.add_argument('--output_dir', type=str,
                       default='cleaned_dataset/stage1_motion_filtered',
                       help='Output directory for filtered videos')
    parser.add_argument('--motion_threshold', type=float, default=0.10,
                       help='Minimum fraction of frames with significant motion')
    parser.add_argument('--pixel_change_threshold', type=float, default=0.05,
                       help='Minimum pixel change ratio for motion detection')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for log files')

    args = parser.parse_args()

    # Initialize and run filter
    filter_obj = MotionDetectionFilter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        motion_threshold=args.motion_threshold,
        pixel_change_threshold=args.pixel_change_threshold,
        log_dir=args.log_dir
    )

    # Run the filtering process
    summary = filter_obj.run_filter()

    # Print final summary
    print("\n" + "="*60)
    print("STAGE 1 MOTION DETECTION FILTER - COMPLETED")
    print("="*60)
    print(f"Total videos processed: {summary['total_videos']}")
    print(f"Videos passed filter: {summary['passed_videos']}")
    print(f"Videos removed: {summary['removed_videos']}")
    print(f"Pass rate: {summary['pass_rate']*100:.1f}%")
    print("\nClass-wise results:")
    for cls, stats in summary['class_stats'].items():
        if stats['total'] > 0:
            print(f"  {cls}: {stats['passed']}/{stats['total']} passed "
                  f"({stats['passed']/stats['total']*100:.1f}%)")

    print(f"\nFiltered videos saved to: {args.output_dir}")
    print(f"Detailed logs saved to: {args.log_dir}")
    print("\n⚠️  IMPORTANT: Review the summary report before proceeding to Stage 2!")

    return summary


if __name__ == "__main__":
    main()
