#!/usr/bin/env python3
"""
Motion Analysis Report Generator for ICU Lip-Reading Dataset
Analyzes motion characteristics WITHOUT removing or copying any videos.
Generates comprehensive reports for informed decision-making.
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
import seaborn as sns

class MotionAnalysisReporter:
    """
    Analyzes video motion characteristics and generates comprehensive reports
    WITHOUT performing any file operations (no copying, moving, or deleting).
    """
    
    def __init__(self, 
                 input_dir: str,
                 motion_threshold: float = 0.10,  # 10% of frames must show significant motion
                 pixel_change_threshold: float = 0.05,  # 5% pixel change threshold
                 log_dir: str = "motion_analysis_reports"):
        """
        Initialize motion analysis reporter.
        
        Args:
            input_dir: Directory containing input videos
            motion_threshold: Minimum fraction of frames with significant motion
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
        """Setup detailed logging for motion analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"motion_analysis_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Motion Analysis Reporter initialized")
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
        
        # Calculate motion score
        motion_score, total_frames, status = self.calculate_optical_flow_motion(str(video_path))
        
        # Determine motion category
        has_sufficient_motion = (motion_score >= self.motion_threshold and 
                               status == "success" and 
                               total_frames > 5)
        
        # Categorize motion level
        if motion_score >= 0.20:
            motion_category = "high"
        elif motion_score >= self.motion_threshold:
            motion_category = "medium"
        elif motion_score >= 0.05:
            motion_category = "low"
        else:
            motion_category = "minimal"
        
        result = {
            'filename': filename,
            'class': class_label,
            'motion_score': motion_score,
            'total_frames': total_frames,
            'analysis_status': status,
            'has_sufficient_motion': has_sufficient_motion,
            'motion_category': motion_category,
            'file_size_mb': video_path.stat().st_size / (1024 * 1024)
        }
        
        return result
    
    def process_all_videos(self) -> None:
        """Process all videos in the input directory for motion analysis."""
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
                'high_motion': 0, 
                'medium_motion': 0, 
                'low_motion': 0, 
                'minimal_motion': 0,
                'sufficient_motion': 0,
                'motion_scores': []
            }
        
        # Process each video with progress bar
        for video_path in tqdm(video_files, desc="Analyzing video motion"):
            result = self.analyze_video(video_path)
            self.results.append(result)
            
            # Update class statistics
            class_label = result['class']
            if class_label in self.class_stats:
                stats = self.class_stats[class_label]
                stats['total'] += 1
                stats['motion_scores'].append(result['motion_score'])
                
                if result['has_sufficient_motion']:
                    stats['sufficient_motion'] += 1
                
                # Categorize by motion level
                category = result['motion_category']
                if category == 'high':
                    stats['high_motion'] += 1
                elif category == 'medium':
                    stats['medium_motion'] += 1
                elif category == 'low':
                    stats['low_motion'] += 1
                else:
                    stats['minimal_motion'] += 1
        
        self.logger.info("Motion analysis completed - NO FILES WERE MODIFIED")
    
    def generate_motion_distribution_plots(self) -> None:
        """Generate motion score distribution plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Motion Score Analysis - ICU Lip-Reading Dataset', fontsize=16)
        
        # Overall distribution
        motion_scores = [r['motion_score'] for r in self.results if r['analysis_status'] == 'success']
        axes[0, 0].hist(motion_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.motion_threshold, color='red', linestyle='--', 
                          label=f'Threshold ({self.motion_threshold})')
        axes[0, 0].set_title('Overall Motion Score Distribution')
        axes[0, 0].set_xlabel('Motion Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Class-wise distributions
        classes = ['doctor', 'glasses', 'phone', 'pillow', 'help']
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, (cls, color) in enumerate(zip(classes, colors)):
            if i < 5:  # We have 5 remaining subplot positions
                row = (i + 1) // 3
                col = (i + 1) % 3
                
                class_scores = self.class_stats[cls]['motion_scores']
                if class_scores:
                    axes[row, col].hist(class_scores, bins=20, alpha=0.7, color=color, edgecolor='black')
                    axes[row, col].axvline(self.motion_threshold, color='red', linestyle='--')
                    axes[row, col].set_title(f'{cls.upper()} Motion Scores')
                    axes[row, col].set_xlabel('Motion Score')
                    axes[row, col].set_ylabel('Frequency')
        
        plt.tight_layout()
        plot_path = self.log_dir / f"motion_distribution_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Motion distribution plots saved to: {plot_path}")
    
    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive motion analysis report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to CSV
        df = pd.DataFrame(self.results)
        csv_path = self.log_dir / f"detailed_motion_analysis_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Detailed results saved to: {csv_path}")
        
        # Generate comprehensive report
        report_path = self.log_dir / f"motion_analysis_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("MOTION ANALYSIS REPORT - ICU LIP-READING DATASET\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Motion threshold: {self.motion_threshold} (10% of frames)\n")
            f.write(f"Pixel change threshold: {self.pixel_change_threshold} (5% pixel change)\n")
            f.write("⚠️  NO FILES WERE MODIFIED - ANALYSIS ONLY\n\n")
            
            # Overall statistics
            total_videos = len(self.results)
            sufficient_motion = sum(1 for r in self.results if r['has_sufficient_motion'])
            
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total videos analyzed: {total_videos}\n")
            f.write(f"Videos with sufficient motion: {sufficient_motion} ({sufficient_motion/total_videos*100:.1f}%)\n")
            f.write(f"Videos with insufficient motion: {total_videos - sufficient_motion} ({(total_videos - sufficient_motion)/total_videos*100:.1f}%)\n\n")
            
            # Motion score statistics
            motion_scores = [r['motion_score'] for r in self.results if r['analysis_status'] == 'success']
            if motion_scores:
                f.write("MOTION SCORE STATISTICS:\n")
                f.write("-" * 50 + "\n")
                f.write(f"Mean motion score: {np.mean(motion_scores):.4f}\n")
                f.write(f"Median motion score: {np.median(motion_scores):.4f}\n")
                f.write(f"Standard deviation: {np.std(motion_scores):.4f}\n")
                f.write(f"Min motion score: {np.min(motion_scores):.4f}\n")
                f.write(f"Max motion score: {np.max(motion_scores):.4f}\n")
                f.write(f"25th percentile: {np.percentile(motion_scores, 25):.4f}\n")
                f.write(f"75th percentile: {np.percentile(motion_scores, 75):.4f}\n\n")
            
            # Class-wise detailed statistics
            f.write("CLASS-WISE DETAILED ANALYSIS:\n")
            f.write("=" * 50 + "\n")
            for cls, stats in self.class_stats.items():
                if stats['total'] > 0:
                    f.write(f"\n{cls.upper()}:\n")
                    f.write(f"  Total videos: {stats['total']}\n")
                    f.write(f"  Sufficient motion: {stats['sufficient_motion']} ({stats['sufficient_motion']/stats['total']*100:.1f}%)\n")
                    f.write(f"  High motion (≥20%): {stats['high_motion']}\n")
                    f.write(f"  Medium motion (10-20%): {stats['medium_motion']}\n")
                    f.write(f"  Low motion (5-10%): {stats['low_motion']}\n")
                    f.write(f"  Minimal motion (<5%): {stats['minimal_motion']}\n")
                    
                    if stats['motion_scores']:
                        scores = stats['motion_scores']
                        f.write(f"  Mean motion score: {np.mean(scores):.4f}\n")
                        f.write(f"  Median motion score: {np.median(scores):.4f}\n")
                        f.write(f"  Std deviation: {np.std(scores):.4f}\n")
            
            # Threshold recommendations
            f.write(f"\nTHRESHOLD ANALYSIS & RECOMMENDATIONS:\n")
            f.write("=" * 50 + "\n")
            
            # Calculate pass rates at different thresholds
            thresholds = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
            f.write("Pass rates at different motion thresholds:\n")
            for thresh in thresholds:
                pass_count = sum(1 for r in self.results if r['motion_score'] >= thresh and r['analysis_status'] == 'success')
                pass_rate = pass_count / total_videos * 100
                f.write(f"  {thresh:.2f}: {pass_count}/{total_videos} videos ({pass_rate:.1f}%)\n")
            
            f.write(f"\nRECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            
            avg_motion = np.mean(motion_scores) if motion_scores else 0
            if avg_motion < 0.05:
                f.write("• Dataset has very low motion overall - consider threshold of 0.03 or lower\n")
            elif avg_motion < 0.10:
                f.write("• Dataset has moderate motion - consider threshold of 0.05-0.07\n")
            else:
                f.write("• Dataset has good motion - current threshold of 0.10 is appropriate\n")
            
            if sufficient_motion < total_videos * 0.1:
                f.write("• Very few videos pass current threshold - consider lowering significantly\n")
            elif sufficient_motion < total_videos * 0.3:
                f.write("• Low pass rate - consider moderate threshold reduction\n")
            
            # Check for class imbalance
            min_sufficient = min(stats['sufficient_motion'] for stats in self.class_stats.values())
            if min_sufficient == 0:
                f.write("• Some classes have NO videos with sufficient motion - threshold too high\n")
            elif min_sufficient < 5:
                f.write("• Some classes have very few high-motion videos - consider class-specific thresholds\n")
        
        self.logger.info(f"Comprehensive report saved to: {report_path}")
        
        # Generate motion distribution plots
        self.generate_motion_distribution_plots()
        
        # Save configuration
        config_path = self.log_dir / f"analysis_config_{timestamp}.json"
        config = {
            'motion_threshold': self.motion_threshold,
            'pixel_change_threshold': self.pixel_change_threshold,
            'input_dir': str(self.input_dir),
            'timestamp': timestamp,
            'total_analyzed': total_videos,
            'sufficient_motion_count': sufficient_motion,
            'analysis_only': True,
            'no_files_modified': True
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Analysis configuration saved to: {config_path}")
    
    def run_analysis(self) -> Dict:
        """
        Run the complete motion analysis pipeline WITHOUT modifying any files.
        
        Returns:
            Dictionary with summary statistics
        """
        self.logger.info("Starting Motion Analysis - NO FILE OPERATIONS")
        
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
        
        self.logger.info("Motion Analysis completed successfully!")
        self.logger.info(f"Summary: {sufficient_motion}/{total_videos} videos have sufficient motion "
                        f"({summary['pass_rate']*100:.1f}% pass rate)")
        self.logger.info("⚠️  NO FILES WERE MODIFIED - ALL ORIGINAL DATA PRESERVED")
        
        return summary


def main():
    """Main execution function for Motion Analysis Report."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Motion Analysis Report Generator')
    parser.add_argument('--input_dir', type=str, 
                       default='data/grid/13.9.25top7dataset',
                       help='Input directory containing videos')
    parser.add_argument('--motion_threshold', type=float, default=0.10,
                       help='Minimum fraction of frames with significant motion')
    parser.add_argument('--pixel_change_threshold', type=float, default=0.05,
                       help='Minimum pixel change ratio for motion detection')
    parser.add_argument('--log_dir', type=str, default='motion_analysis_reports',
                       help='Directory for analysis reports')
    
    args = parser.parse_args()
    
    # Initialize and run analyzer
    analyzer = MotionAnalysisReporter(
        input_dir=args.input_dir,
        motion_threshold=args.motion_threshold,
        pixel_change_threshold=args.pixel_change_threshold,
        log_dir=args.log_dir
    )
    
    # Run the analysis process
    summary = analyzer.run_analysis()
    
    # Print final summary
    print("\n" + "="*70)
    print("MOTION ANALYSIS REPORT - COMPLETED")
    print("="*70)
    print(f"Total videos analyzed: {summary['total_videos']}")
    print(f"Videos with sufficient motion: {summary['sufficient_motion']}")
    print(f"Videos with insufficient motion: {summary['insufficient_motion']}")
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
