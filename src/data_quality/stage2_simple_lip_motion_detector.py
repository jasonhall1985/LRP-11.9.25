#!/usr/bin/env python3
"""
Stage 2: Simple Lip Motion Detection for ICU Lip-Reading Dataset
Detects sustained movement in the lip region using simple motion detection.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt

class SimpleLipMotionDetector:
    """
    Simple lip motion detector that focuses on sustained movement in the lower face region.
    Much simpler than facial landmark detection - just looks for consistent motion.
    """
    
    def __init__(self, 
                 input_dir: str,
                 motion_threshold: float = 0.15,  # 15% of frames must show lip motion
                 pixel_change_threshold: float = 0.02,  # 2% pixel change for motion
                 log_dir: str = "simple_lip_motion_reports"):
        """
        Initialize simple lip motion detector.
        
        Args:
            input_dir: Directory containing input videos
            motion_threshold: Minimum fraction of frames with lip motion
            pixel_change_threshold: Minimum pixel change ratio for motion detection
            log_dir: Directory for analysis reports
        """
        self.input_dir = Path(input_dir)
        self.motion_threshold = motion_threshold
        self.pixel_change_threshold = pixel_change_threshold
        self.log_dir = Path(log_dir)
        
        # Create report directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Results storage
        self.results = []
        self.class_stats = {}
        
    def setup_logging(self):
        """Setup detailed logging for lip motion analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"simple_lip_motion_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Simple Lip Motion Detector initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Motion threshold: {self.motion_threshold}")
        self.logger.info(f"Pixel change threshold: {self.pixel_change_threshold}")
        self.logger.info("⚠️  NO FILE OPERATIONS WILL BE PERFORMED - ANALYSIS ONLY")
        
    def extract_class_from_filename(self, filename: str) -> str:
        """Extract class label from filename."""
        classes = ['doctor', 'glasses', 'phone', 'pillow', 'help']
        for cls in classes:
            if filename.startswith(cls + '__'):
                return cls
        return 'unknown'
    
    def detect_lip_region_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[float, Dict]:
        """
        Detect motion in the lip region (lower center portion of frame).
        
        Args:
            frame1: First frame (grayscale)
            frame2: Second frame (grayscale)
            
        Returns:
            Tuple of (motion_score, motion_info)
        """
        h, w = frame1.shape
        
        # Define lip region (lower center portion of frame)
        # This is where lip movement would typically occur
        lip_region_top = int(h * 0.6)      # Start at 60% down from top
        lip_region_bottom = int(h * 0.9)   # End at 90% down from top
        lip_region_left = int(w * 0.25)    # Start at 25% from left
        lip_region_right = int(w * 0.75)   # End at 75% from left
        
        # Extract lip regions from both frames
        lip1 = frame1[lip_region_top:lip_region_bottom, lip_region_left:lip_region_right]
        lip2 = frame2[lip_region_top:lip_region_bottom, lip_region_left:lip_region_right]
        
        # Calculate frame difference in lip region
        if lip1.shape != lip2.shape or lip1.size == 0:
            return 0.0, {'region_valid': False, 'region_size': 0}
        
        # Calculate absolute difference
        diff = cv2.absdiff(lip1, lip2)
        
        # Count pixels with significant change
        total_pixels = diff.shape[0] * diff.shape[1]
        changed_pixels = np.count_nonzero(diff > 20)  # Threshold for significant change
        
        # Calculate motion score
        motion_score = changed_pixels / total_pixels if total_pixels > 0 else 0.0
        
        motion_info = {
            'region_valid': True,
            'region_size': total_pixels,
            'changed_pixels': changed_pixels,
            'motion_score': motion_score,
            'region_bounds': (lip_region_top, lip_region_bottom, lip_region_left, lip_region_right)
        }
        
        return motion_score, motion_info
    
    def analyze_video_lip_motion(self, video_path: Path) -> Dict:
        """
        Analyze lip motion for a single video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with lip motion analysis results
        """
        filename = video_path.name
        class_label = self.extract_class_from_filename(filename)
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return self._create_failed_result(filename, class_label, "failed_to_open")
            
            # Read first frame
            ret, prev_frame = cap.read()
            if not ret:
                cap.release()
                return self._create_failed_result(filename, class_label, "no_frames")
            
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            # Analysis variables
            total_frames = 1
            motion_frames = 0
            motion_scores = []
            sustained_motion_sequences = 0
            current_sequence_length = 0
            
            while True:
                ret, curr_frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                
                # Detect lip region motion
                motion_score, motion_info = self.detect_lip_region_motion(prev_gray, curr_gray)
                
                if motion_info['region_valid']:
                    motion_scores.append(motion_score)
                    
                    # Check if this frame has significant motion
                    if motion_score > self.pixel_change_threshold:
                        motion_frames += 1
                        current_sequence_length += 1
                    else:
                        # End of motion sequence
                        if current_sequence_length >= 3:  # At least 3 consecutive frames
                            sustained_motion_sequences += 1
                        current_sequence_length = 0
                
                prev_gray = curr_gray
                total_frames += 1
            
            # Check for final sequence
            if current_sequence_length >= 3:
                sustained_motion_sequences += 1
            
            cap.release()
            
            # Calculate statistics
            lip_motion_rate = motion_frames / total_frames if total_frames > 0 else 0.0
            has_sufficient_motion = lip_motion_rate >= self.motion_threshold
            
            # Categorize motion quality
            if lip_motion_rate >= 0.30:
                motion_category = "excellent"
            elif lip_motion_rate >= self.motion_threshold:
                motion_category = "good"
            elif lip_motion_rate >= 0.05:
                motion_category = "moderate"
            elif lip_motion_rate > 0.0:
                motion_category = "poor"
            else:
                motion_category = "none"
            
            result = {
                'filename': filename,
                'class': class_label,
                'total_frames': total_frames,
                'motion_frames': motion_frames,
                'lip_motion_rate': lip_motion_rate,
                'has_sufficient_motion': has_sufficient_motion,
                'motion_category': motion_category,
                'sustained_sequences': sustained_motion_sequences,
                'mean_motion_score': np.mean(motion_scores) if motion_scores else 0,
                'max_motion_score': np.max(motion_scores) if motion_scores else 0,
                'std_motion_score': np.std(motion_scores) if motion_scores else 0,
                'analysis_status': 'success',
                'file_size_mb': video_path.stat().st_size / (1024 * 1024)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {video_path}: {str(e)}")
            return self._create_failed_result(filename, class_label, f"error: {str(e)}")
    
    def _create_failed_result(self, filename: str, class_label: str, error_reason: str) -> Dict:
        """Create result dictionary for failed analysis."""
        return {
            'filename': filename,
            'class': class_label,
            'total_frames': 0,
            'motion_frames': 0,
            'lip_motion_rate': 0.0,
            'has_sufficient_motion': False,
            'motion_category': 'failed',
            'sustained_sequences': 0,
            'mean_motion_score': 0,
            'max_motion_score': 0,
            'std_motion_score': 0,
            'analysis_status': error_reason,
            'file_size_mb': 0
        }
    
    def process_all_videos(self) -> None:
        """Process all videos in the input directory for lip motion analysis."""
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(list(self.input_dir.glob(f'*{ext}')))
        
        self.logger.info(f"Found {len(video_files)} video files to analyze")
        
        # Initialize class statistics
        classes = ['doctor', 'glasses', 'phone', 'pillow', 'help']
        for cls in classes:
            self.class_stats[cls] = {
                'total': 0,
                'excellent_motion': 0,
                'good_motion': 0,
                'moderate_motion': 0,
                'poor_motion': 0,
                'no_motion': 0,
                'sufficient_motion': 0,
                'motion_rates': [],
                'sustained_sequences': []
            }
        
        # Process each video with progress bar
        for video_path in tqdm(video_files, desc="Analyzing lip motion"):
            result = self.analyze_video_lip_motion(video_path)
            self.results.append(result)
            
            # Update class statistics
            class_label = result['class']
            if class_label in self.class_stats:
                stats = self.class_stats[class_label]
                stats['total'] += 1
                stats['motion_rates'].append(result['lip_motion_rate'])
                stats['sustained_sequences'].append(result['sustained_sequences'])
                
                if result['has_sufficient_motion']:
                    stats['sufficient_motion'] += 1
                
                # Categorize by motion quality
                category = result['motion_category']
                if category == 'excellent':
                    stats['excellent_motion'] += 1
                elif category == 'good':
                    stats['good_motion'] += 1
                elif category == 'moderate':
                    stats['moderate_motion'] += 1
                elif category == 'poor':
                    stats['poor_motion'] += 1
                else:
                    stats['no_motion'] += 1
        
        self.logger.info("Simple lip motion analysis completed - NO FILES WERE MODIFIED")
    
    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive lip motion analysis report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to CSV
        df = pd.DataFrame(self.results)
        csv_path = self.log_dir / f"simple_lip_motion_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Detailed results saved to: {csv_path}")
        
        # Generate comprehensive report
        report_path = self.log_dir / f"simple_lip_motion_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("STAGE 2: SIMPLE LIP MOTION ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Motion threshold: {self.motion_threshold} ({self.motion_threshold*100:.0f}% of frames)\n")
            f.write(f"Pixel change threshold: {self.pixel_change_threshold} ({self.pixel_change_threshold*100:.0f}% pixel change)\n")
            f.write("⚠️  NO FILES WERE MODIFIED - ANALYSIS ONLY\n\n")
            
            # Overall statistics
            total_videos = len(self.results)
            sufficient_motion = sum(1 for r in self.results if r['has_sufficient_motion'])
            successful_analysis = sum(1 for r in self.results if r['analysis_status'] == 'success')
            
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total videos analyzed: {total_videos}\n")
            f.write(f"Successful analysis: {successful_analysis} ({successful_analysis/total_videos*100:.1f}%)\n")
            f.write(f"Videos with sufficient lip motion: {sufficient_motion} ({sufficient_motion/total_videos*100:.1f}%)\n")
            f.write(f"Videos with insufficient lip motion: {total_videos - sufficient_motion} ({(total_videos - sufficient_motion)/total_videos*100:.1f}%)\n\n")
            
            # Motion rate statistics
            motion_rates = [r['lip_motion_rate'] for r in self.results if r['analysis_status'] == 'success']
            if motion_rates:
                f.write("LIP MOTION RATE STATISTICS:\n")
                f.write("-" * 50 + "\n")
                f.write(f"Mean motion rate: {np.mean(motion_rates):.4f}\n")
                f.write(f"Median motion rate: {np.median(motion_rates):.4f}\n")
                f.write(f"Standard deviation: {np.std(motion_rates):.4f}\n")
                f.write(f"Min motion rate: {np.min(motion_rates):.4f}\n")
                f.write(f"Max motion rate: {np.max(motion_rates):.4f}\n\n")
            
            # Class-wise detailed statistics
            f.write("CLASS-WISE DETAILED ANALYSIS:\n")
            f.write("=" * 50 + "\n")
            for cls, stats in self.class_stats.items():
                if stats['total'] > 0:
                    f.write(f"\n{cls.upper()}:\n")
                    f.write(f"  Total videos: {stats['total']}\n")
                    f.write(f"  Sufficient motion: {stats['sufficient_motion']} ({stats['sufficient_motion']/stats['total']*100:.1f}%)\n")
                    f.write(f"  Excellent motion (≥30%): {stats['excellent_motion']}\n")
                    f.write(f"  Good motion (15-30%): {stats['good_motion']}\n")
                    f.write(f"  Moderate motion (5-15%): {stats['moderate_motion']}\n")
                    f.write(f"  Poor motion (0-5%): {stats['poor_motion']}\n")
                    f.write(f"  No motion (0%): {stats['no_motion']}\n")
                    
                    if stats['motion_rates']:
                        rates = stats['motion_rates']
                        f.write(f"  Mean motion rate: {np.mean(rates):.4f}\n")
                        f.write(f"  Median motion rate: {np.median(rates):.4f}\n")
                    
                    if stats['sustained_sequences']:
                        sequences = stats['sustained_sequences']
                        f.write(f"  Mean sustained sequences: {np.mean(sequences):.1f}\n")
            
            # Threshold analysis
            f.write(f"\nTHRESHOLD ANALYSIS & RECOMMENDATIONS:\n")
            f.write("=" * 50 + "\n")
            
            # Calculate pass rates at different thresholds
            thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
            f.write("Pass rates at different lip motion thresholds:\n")
            for thresh in thresholds:
                pass_count = sum(1 for r in self.results if r['lip_motion_rate'] >= thresh and r['analysis_status'] == 'success')
                pass_rate = pass_count / total_videos * 100
                f.write(f"  {thresh:.2f}: {pass_count}/{total_videos} videos ({pass_rate:.1f}%)\n")
            
            f.write(f"\nRECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            
            avg_motion = np.mean(motion_rates) if motion_rates else 0
            if avg_motion < 0.10:
                f.write("• Dataset has low lip motion overall - consider threshold of 0.05-0.10\n")
            elif avg_motion < 0.20:
                f.write("• Dataset has moderate lip motion - current threshold may be appropriate\n")
            else:
                f.write("• Dataset has good lip motion - current threshold is suitable\n")
            
            if sufficient_motion < total_videos * 0.1:
                f.write("• Very few videos pass current threshold - consider lowering significantly\n")
            elif sufficient_motion < total_videos * 0.3:
                f.write("• Low pass rate - consider moderate threshold reduction\n")
        
        self.logger.info(f"Comprehensive report saved to: {report_path}")
    
    def run_analysis(self) -> Dict:
        """Run the complete simple lip motion analysis pipeline."""
        self.logger.info("Starting Simple Lip Motion Analysis - NO FILE OPERATIONS")
        
        # Process all videos
        self.process_all_videos()
        
        # Generate comprehensive reports
        self.generate_comprehensive_report()
        
        # Return summary statistics
        total_videos = len(self.results)
        sufficient_motion = sum(1 for r in self.results if r['has_sufficient_motion'])
        
        summary = {
            'total_videos': total_videos,
            'sufficient_motion': sufficient_motion,
            'insufficient_motion': total_videos - sufficient_motion,
            'pass_rate': sufficient_motion / total_videos if total_videos > 0 else 0,
            'class_stats': self.class_stats.copy(),
            'files_modified': False,
            'analysis_only': True
        }
        
        self.logger.info("Simple Lip Motion Analysis completed successfully!")
        self.logger.info(f"Summary: {sufficient_motion}/{total_videos} videos have sufficient lip motion "
                        f"({summary['pass_rate']*100:.1f}% pass rate)")
        self.logger.info("⚠️  NO FILES WERE MODIFIED - ALL ORIGINAL DATA PRESERVED")
        
        return summary


def main():
    """Main execution function for Simple Lip Motion Analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Lip Motion Analysis')
    parser.add_argument('--input_dir', type=str, 
                       default='data/grid/13.9.25top7dataset',
                       help='Input directory containing videos')
    parser.add_argument('--motion_threshold', type=float, default=0.15,
                       help='Minimum fraction of frames with lip motion')
    parser.add_argument('--pixel_change_threshold', type=float, default=0.02,
                       help='Minimum pixel change ratio for motion detection')
    parser.add_argument('--log_dir', type=str, default='simple_lip_motion_reports',
                       help='Directory for analysis reports')
    
    args = parser.parse_args()
    
    # Initialize and run analyzer
    analyzer = SimpleLipMotionDetector(
        input_dir=args.input_dir,
        motion_threshold=args.motion_threshold,
        pixel_change_threshold=args.pixel_change_threshold,
        log_dir=args.log_dir
    )
    
    # Run the analysis process
    summary = analyzer.run_analysis()
    
    # Print final summary
    print("\n" + "="*70)
    print("SIMPLE LIP MOTION ANALYSIS - COMPLETED")
    print("="*70)
    print(f"Total videos analyzed: {summary['total_videos']}")
    print(f"Videos with sufficient lip motion: {summary['sufficient_motion']}")
    print(f"Videos with insufficient lip motion: {summary['insufficient_motion']}")
    print(f"Pass rate: {summary['pass_rate']*100:.1f}%")
    print("\nClass-wise results:")
    for cls, stats in summary['class_stats'].items():
        if stats['total'] > 0:
            print(f"  {cls}: {stats['sufficient_motion']}/{stats['total']} sufficient motion "
                  f"({stats['sufficient_motion']/stats['total']*100:.1f}%)")
    
    print(f"\nDetailed reports saved to: {args.log_dir}")
    print("⚠️  NO FILES WERE MODIFIED - ALL ORIGINAL DATA PRESERVED")
    
    return summary


if __name__ == "__main__":
    main()
