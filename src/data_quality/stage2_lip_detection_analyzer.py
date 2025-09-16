#!/usr/bin/env python3
"""
Stage 2: Lip Region Detection Filter for ICU Lip-Reading Dataset
Analyzes lip detectability using MediaPipe Face Mesh WITHOUT modifying any files.
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

class LipDetectionAnalyzer:
    """
    Analyzes lip region detectability using MediaPipe Face Mesh.
    Generates comprehensive reports WITHOUT performing any file operations.
    """
    
    def __init__(self, 
                 input_dir: str,
                 lip_detection_threshold: float = 0.80,  # 80% of frames must have lip detection
                 log_dir: str = "lip_detection_reports"):
        """
        Initialize lip detection analyzer.
        
        Args:
            input_dir: Directory containing input videos
            lip_detection_threshold: Minimum fraction of frames with successful lip detection
            log_dir: Directory for analysis reports
        """
        self.input_dir = Path(input_dir)
        self.lip_detection_threshold = lip_detection_threshold
        self.log_dir = Path(log_dir)
        
        # Create report directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenCV face detection (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize face detection confidence parameters
        self.min_face_size = (30, 30)  # Minimum face size for detection
        self.scale_factor = 1.1
        self.min_neighbors = 5

        # Lip region estimation parameters (as fraction of face bounding box)
        self.lip_region_params = {
            'x_offset': 0.25,      # Start lip region at 25% from left of face
            'width_ratio': 0.50,   # Lip region width is 50% of face width
            'y_offset': 0.65,      # Start lip region at 65% from top of face
            'height_ratio': 0.25   # Lip region height is 25% of face height
        }
        
        # Setup logging
        self.setup_logging()
        
        # Results storage
        self.results = []
        self.class_stats = {}
        
    def setup_logging(self):
        """Setup detailed logging for lip detection analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"lip_detection_analysis_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Lip Detection Analyzer initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Lip detection threshold: {self.lip_detection_threshold}")
        self.logger.info("⚠️  NO FILE OPERATIONS WILL BE PERFORMED - ANALYSIS ONLY")
        
    def extract_class_from_filename(self, filename: str) -> str:
        """Extract class label from filename."""
        classes = ['doctor', 'glasses', 'phone', 'pillow', 'help']
        for cls in classes:
            if filename.startswith(cls + '__'):
                return cls
        return 'unknown'
    
    def extract_lip_region(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Extract lip region from a single frame using face detection + region estimation.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Tuple of (lip_region_coordinates, detection_info)
        """
        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detection_info = {
            'face_detected': False,
            'lip_region_detected': False,
            'bounding_box': None,
            'lip_region_area': 0,
            'detection_confidence': 0.0,
            'face_size': 0
        }

        # Detect faces using Haar Cascade
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size
        )

        if len(faces) > 0:
            # Use the largest face detected
            face = max(faces, key=lambda f: f[2] * f[3])
            fx, fy, fw, fh = face

            detection_info['face_detected'] = True
            detection_info['face_size'] = fw * fh

            # Estimate lip region based on face bounding box
            lip_x = int(fx + fw * self.lip_region_params['x_offset'])
            lip_y = int(fy + fh * self.lip_region_params['y_offset'])
            lip_w = int(fw * self.lip_region_params['width_ratio'])
            lip_h = int(fh * self.lip_region_params['height_ratio'])

            # Ensure lip region is within frame bounds
            h, w = frame.shape[:2]
            lip_x = max(0, min(lip_x, w - lip_w))
            lip_y = max(0, min(lip_y, h - lip_h))
            lip_w = min(lip_w, w - lip_x)
            lip_h = min(lip_h, h - lip_y)

            if lip_w > 10 and lip_h > 10:  # Minimum viable lip region size
                detection_info['lip_region_detected'] = True
                detection_info['bounding_box'] = (lip_x, lip_y, lip_x + lip_w, lip_y + lip_h)
                detection_info['lip_region_area'] = lip_w * lip_h

                # Calculate detection confidence based on face size and region validity
                face_area = fw * fh
                frame_area = w * h
                face_ratio = face_area / frame_area

                # Higher confidence for larger faces (better resolution)
                detection_info['detection_confidence'] = min(1.0, face_ratio * 10)  # Scale factor

                # Return lip region coordinates
                lip_coordinates = np.array([
                    [lip_x, lip_y],                    # Top-left
                    [lip_x + lip_w, lip_y],           # Top-right
                    [lip_x + lip_w, lip_y + lip_h],   # Bottom-right
                    [lip_x, lip_y + lip_h]            # Bottom-left
                ])

                return lip_coordinates, detection_info

        return None, detection_info
    
    def analyze_video_lip_detection(self, video_path: Path) -> Dict:
        """
        Analyze lip detection for a single video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with lip detection analysis results
        """
        filename = video_path.name
        class_label = self.extract_class_from_filename(filename)
        
        self.logger.info(f"Analyzing lip detection: {filename}")
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return self._create_failed_result(filename, class_label, "failed_to_open")
            
            # Analysis variables
            total_frames = 0
            frames_with_lips = 0
            lip_areas = []
            bounding_boxes = []
            detection_confidences = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                total_frames += 1
                
                # Extract lip region
                lip_region, detection_info = self.extract_lip_region(frame)
                
                if lip_region is not None and detection_info['lip_region_detected']:
                    frames_with_lips += 1
                    lip_areas.append(detection_info['lip_region_area'])
                    bounding_boxes.append(detection_info['bounding_box'])
                    detection_confidences.append(detection_info['detection_confidence'])
            
            cap.release()
            
            # Calculate statistics
            lip_detection_rate = frames_with_lips / total_frames if total_frames > 0 else 0.0
            has_sufficient_lip_detection = lip_detection_rate >= self.lip_detection_threshold
            
            # Bounding box statistics
            bbox_stats = self._calculate_bbox_statistics(bounding_boxes)
            
            # Categorize detection quality
            if lip_detection_rate >= 0.90:
                detection_category = "excellent"
            elif lip_detection_rate >= self.lip_detection_threshold:
                detection_category = "good"
            elif lip_detection_rate >= 0.50:
                detection_category = "moderate"
            elif lip_detection_rate > 0.0:
                detection_category = "poor"
            else:
                detection_category = "failed"
            
            result = {
                'filename': filename,
                'class': class_label,
                'total_frames': total_frames,
                'frames_with_lips': frames_with_lips,
                'lip_detection_rate': lip_detection_rate,
                'has_sufficient_lip_detection': has_sufficient_lip_detection,
                'detection_category': detection_category,
                'mean_lip_area': np.mean(lip_areas) if lip_areas else 0,
                'std_lip_area': np.std(lip_areas) if lip_areas else 0,
                'mean_detection_confidence': np.mean(detection_confidences) if detection_confidences else 0,
                'bbox_width_mean': bbox_stats['width_mean'],
                'bbox_height_mean': bbox_stats['height_mean'],
                'bbox_width_std': bbox_stats['width_std'],
                'bbox_height_std': bbox_stats['height_std'],
                'bbox_stability': bbox_stats['stability'],
                'analysis_status': 'success',
                'file_size_mb': video_path.stat().st_size / (1024 * 1024)
            }
            
            self.logger.info(f"  Lip detection rate: {lip_detection_rate:.3f}, "
                           f"Category: {detection_category}, "
                           f"Frames: {frames_with_lips}/{total_frames}")
            
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
            'frames_with_lips': 0,
            'lip_detection_rate': 0.0,
            'has_sufficient_lip_detection': False,
            'detection_category': 'failed',
            'mean_lip_area': 0,
            'std_lip_area': 0,
            'mean_detection_confidence': 0,
            'bbox_width_mean': 0,
            'bbox_height_mean': 0,
            'bbox_width_std': 0,
            'bbox_height_std': 0,
            'bbox_stability': 0,
            'analysis_status': error_reason,
            'file_size_mb': 0
        }
    
    def _calculate_bbox_statistics(self, bounding_boxes: List[Tuple]) -> Dict:
        """Calculate bounding box statistics for stability analysis."""
        if not bounding_boxes:
            return {
                'width_mean': 0, 'height_mean': 0,
                'width_std': 0, 'height_std': 0,
                'stability': 0
            }
        
        widths = [bbox[2] - bbox[0] for bbox in bounding_boxes]
        heights = [bbox[3] - bbox[1] for bbox in bounding_boxes]
        
        width_mean = np.mean(widths)
        height_mean = np.mean(heights)
        width_std = np.std(widths)
        height_std = np.std(heights)
        
        # Calculate stability score (lower std relative to mean = more stable)
        width_cv = width_std / width_mean if width_mean > 0 else float('inf')
        height_cv = height_std / height_mean if height_mean > 0 else float('inf')
        stability = 1.0 / (1.0 + (width_cv + height_cv) / 2.0)  # Normalized stability score
        
        return {
            'width_mean': width_mean,
            'height_mean': height_mean,
            'width_std': width_std,
            'height_std': height_std,
            'stability': stability
        }
    
    def process_all_videos(self) -> None:
        """Process all videos in the input directory for lip detection analysis."""
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
                'excellent_detection': 0,
                'good_detection': 0,
                'moderate_detection': 0,
                'poor_detection': 0,
                'failed_detection': 0,
                'sufficient_detection': 0,
                'detection_rates': [],
                'lip_areas': [],
                'bbox_stabilities': []
            }
        
        # Process each video with progress bar
        for video_path in tqdm(video_files, desc="Analyzing lip detection"):
            result = self.analyze_video_lip_detection(video_path)
            self.results.append(result)
            
            # Update class statistics
            class_label = result['class']
            if class_label in self.class_stats:
                stats = self.class_stats[class_label]
                stats['total'] += 1
                stats['detection_rates'].append(result['lip_detection_rate'])
                
                if result['has_sufficient_lip_detection']:
                    stats['sufficient_detection'] += 1
                
                # Categorize by detection quality
                category = result['detection_category']
                if category == 'excellent':
                    stats['excellent_detection'] += 1
                elif category == 'good':
                    stats['good_detection'] += 1
                elif category == 'moderate':
                    stats['moderate_detection'] += 1
                elif category == 'poor':
                    stats['poor_detection'] += 1
                else:
                    stats['failed_detection'] += 1
                
                # Store additional metrics
                if result['mean_lip_area'] > 0:
                    stats['lip_areas'].append(result['mean_lip_area'])
                if result['bbox_stability'] > 0:
                    stats['bbox_stabilities'].append(result['bbox_stability'])
        
        self.logger.info("Lip detection analysis completed - NO FILES WERE MODIFIED")

    def load_motion_analysis_results(self) -> Optional[pd.DataFrame]:
        """Load previous motion analysis results for cross-reference."""
        try:
            # Find the most recent motion analysis CSV
            motion_reports_dir = Path("motion_analysis_reports")
            if motion_reports_dir.exists():
                csv_files = list(motion_reports_dir.glob("detailed_motion_analysis_*.csv"))
                if csv_files:
                    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                    motion_df = pd.read_csv(latest_csv)
                    self.logger.info(f"Loaded motion analysis results from: {latest_csv}")
                    return motion_df
        except Exception as e:
            self.logger.warning(f"Could not load motion analysis results: {str(e)}")
        return None

    def generate_detection_distribution_plots(self) -> None:
        """Generate lip detection distribution plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Lip Detection Analysis - ICU Lip-Reading Dataset', fontsize=16)

        # Overall detection rate distribution
        detection_rates = [r['lip_detection_rate'] for r in self.results if r['analysis_status'] == 'success']
        axes[0, 0].hist(detection_rates, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 0].axvline(self.lip_detection_threshold, color='red', linestyle='--',
                          label=f'Threshold ({self.lip_detection_threshold})')
        axes[0, 0].set_title('Overall Lip Detection Rate Distribution')
        axes[0, 0].set_xlabel('Lip Detection Rate')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()

        # Class-wise detection rates
        classes = ['doctor', 'glasses', 'phone', 'pillow', 'help']
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for i, (cls, color) in enumerate(zip(classes, colors)):
            if i < 5:  # We have 5 remaining subplot positions
                row = (i + 1) // 3
                col = (i + 1) % 3

                class_rates = self.class_stats[cls]['detection_rates']
                if class_rates:
                    axes[row, col].hist(class_rates, bins=20, alpha=0.7, color=color, edgecolor='black')
                    axes[row, col].axvline(self.lip_detection_threshold, color='red', linestyle='--')
                    axes[row, col].set_title(f'{cls.upper()} Lip Detection Rates')
                    axes[row, col].set_xlabel('Detection Rate')
                    axes[row, col].set_ylabel('Frequency')

        plt.tight_layout()
        plot_path = self.log_dir / f"lip_detection_distribution_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Lip detection distribution plots saved to: {plot_path}")

    def generate_motion_lip_correlation_analysis(self, motion_df: pd.DataFrame) -> Dict:
        """Generate correlation analysis between motion and lip detection."""
        correlation_stats = {}

        try:
            # Merge datasets on filename
            lip_df = pd.DataFrame(self.results)
            merged_df = pd.merge(lip_df, motion_df, on='filename', how='inner', suffixes=('_lip', '_motion'))

            if len(merged_df) > 0:
                # Calculate correlations
                motion_lip_corr = merged_df['motion_score'].corr(merged_df['lip_detection_rate'])

                # Analyze categories
                high_motion_high_lip = len(merged_df[
                    (merged_df['motion_score'] >= 0.05) &
                    (merged_df['lip_detection_rate'] >= self.lip_detection_threshold)
                ])

                high_motion_low_lip = len(merged_df[
                    (merged_df['motion_score'] >= 0.05) &
                    (merged_df['lip_detection_rate'] < self.lip_detection_threshold)
                ])

                low_motion_high_lip = len(merged_df[
                    (merged_df['motion_score'] < 0.05) &
                    (merged_df['lip_detection_rate'] >= self.lip_detection_threshold)
                ])

                correlation_stats = {
                    'correlation_coefficient': motion_lip_corr,
                    'total_analyzed': len(merged_df),
                    'high_motion_high_lip': high_motion_high_lip,
                    'high_motion_low_lip': high_motion_low_lip,
                    'low_motion_high_lip': low_motion_high_lip,
                    'correlation_strength': 'strong' if abs(motion_lip_corr) > 0.7 else
                                          'moderate' if abs(motion_lip_corr) > 0.3 else 'weak'
                }

                self.logger.info(f"Motion-Lip correlation: {motion_lip_corr:.3f} ({correlation_stats['correlation_strength']})")

        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {str(e)}")

        return correlation_stats

    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive lip detection analysis report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load motion analysis for cross-reference
        motion_df = self.load_motion_analysis_results()
        correlation_stats = {}
        if motion_df is not None:
            correlation_stats = self.generate_motion_lip_correlation_analysis(motion_df)

        # Save detailed results to CSV
        df = pd.DataFrame(self.results)
        csv_path = self.log_dir / f"detailed_lip_detection_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Detailed results saved to: {csv_path}")

        # Generate comprehensive report
        report_path = self.log_dir / f"lip_detection_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("STAGE 2: LIP DETECTION ANALYSIS REPORT - ICU LIP-READING DATASET\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Lip detection threshold: {self.lip_detection_threshold} (80% of frames)\n")
            f.write("⚠️  NO FILES WERE MODIFIED - ANALYSIS ONLY\n\n")

            # Overall statistics
            total_videos = len(self.results)
            sufficient_detection = sum(1 for r in self.results if r['has_sufficient_lip_detection'])
            successful_analysis = sum(1 for r in self.results if r['analysis_status'] == 'success')

            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total videos analyzed: {total_videos}\n")
            f.write(f"Successful analysis: {successful_analysis} ({successful_analysis/total_videos*100:.1f}%)\n")
            f.write(f"Videos with sufficient lip detection: {sufficient_detection} ({sufficient_detection/total_videos*100:.1f}%)\n")
            f.write(f"Videos with insufficient lip detection: {total_videos - sufficient_detection} ({(total_videos - sufficient_detection)/total_videos*100:.1f}%)\n\n")

            # Detection rate statistics
            detection_rates = [r['lip_detection_rate'] for r in self.results if r['analysis_status'] == 'success']
            if detection_rates:
                f.write("LIP DETECTION RATE STATISTICS:\n")
                f.write("-" * 60 + "\n")
                f.write(f"Mean detection rate: {np.mean(detection_rates):.4f}\n")
                f.write(f"Median detection rate: {np.median(detection_rates):.4f}\n")
                f.write(f"Standard deviation: {np.std(detection_rates):.4f}\n")
                f.write(f"Min detection rate: {np.min(detection_rates):.4f}\n")
                f.write(f"Max detection rate: {np.max(detection_rates):.4f}\n")
                f.write(f"25th percentile: {np.percentile(detection_rates, 25):.4f}\n")
                f.write(f"75th percentile: {np.percentile(detection_rates, 75):.4f}\n\n")

            # Class-wise detailed statistics
            f.write("CLASS-WISE DETAILED ANALYSIS:\n")
            f.write("=" * 60 + "\n")
            for cls, stats in self.class_stats.items():
                if stats['total'] > 0:
                    f.write(f"\n{cls.upper()}:\n")
                    f.write(f"  Total videos: {stats['total']}\n")
                    f.write(f"  Sufficient detection: {stats['sufficient_detection']} ({stats['sufficient_detection']/stats['total']*100:.1f}%)\n")
                    f.write(f"  Excellent detection (≥90%): {stats['excellent_detection']}\n")
                    f.write(f"  Good detection (80-90%): {stats['good_detection']}\n")
                    f.write(f"  Moderate detection (50-80%): {stats['moderate_detection']}\n")
                    f.write(f"  Poor detection (0-50%): {stats['poor_detection']}\n")
                    f.write(f"  Failed detection (0%): {stats['failed_detection']}\n")

                    if stats['detection_rates']:
                        rates = stats['detection_rates']
                        f.write(f"  Mean detection rate: {np.mean(rates):.4f}\n")
                        f.write(f"  Median detection rate: {np.median(rates):.4f}\n")
                        f.write(f"  Std deviation: {np.std(rates):.4f}\n")

                    if stats['lip_areas']:
                        areas = stats['lip_areas']
                        f.write(f"  Mean lip area: {np.mean(areas):.1f} pixels²\n")

                    if stats['bbox_stabilities']:
                        stabilities = stats['bbox_stabilities']
                        f.write(f"  Mean bbox stability: {np.mean(stabilities):.3f}\n")

            # Threshold analysis
            f.write(f"\nTHRESHOLD ANALYSIS & RECOMMENDATIONS:\n")
            f.write("=" * 60 + "\n")

            # Calculate pass rates at different thresholds
            thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
            f.write("Pass rates at different lip detection thresholds:\n")
            for thresh in thresholds:
                pass_count = sum(1 for r in self.results if r['lip_detection_rate'] >= thresh and r['analysis_status'] == 'success')
                pass_rate = pass_count / total_videos * 100
                f.write(f"  {thresh:.2f}: {pass_count}/{total_videos} videos ({pass_rate:.1f}%)\n")

            # Motion-Lip correlation analysis
            if correlation_stats:
                f.write(f"\nMOTION-LIP DETECTION CORRELATION:\n")
                f.write("-" * 60 + "\n")
                f.write(f"Correlation coefficient: {correlation_stats.get('correlation_coefficient', 'N/A'):.3f}\n")
                f.write(f"Correlation strength: {correlation_stats.get('correlation_strength', 'N/A')}\n")
                f.write(f"High motion + High lip detection: {correlation_stats.get('high_motion_high_lip', 0)} videos\n")
                f.write(f"High motion + Low lip detection: {correlation_stats.get('high_motion_low_lip', 0)} videos\n")
                f.write(f"Low motion + High lip detection: {correlation_stats.get('low_motion_high_lip', 0)} videos\n")

            f.write(f"\nRECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")

            avg_detection = np.mean(detection_rates) if detection_rates else 0
            if avg_detection < 0.50:
                f.write("• Dataset has low lip detection rates overall - consider lowering threshold\n")
            elif avg_detection < 0.80:
                f.write("• Dataset has moderate lip detection - current threshold may be appropriate\n")
            else:
                f.write("• Dataset has good lip detection - current threshold is suitable\n")

            if sufficient_detection < total_videos * 0.1:
                f.write("• Very few videos pass current threshold - consider significant reduction\n")
            elif sufficient_detection < total_videos * 0.3:
                f.write("• Low pass rate - consider moderate threshold reduction\n")

            # Check for class imbalance
            min_sufficient = min(stats['sufficient_detection'] for stats in self.class_stats.values())
            if min_sufficient == 0:
                f.write("• Some classes have NO videos with sufficient lip detection\n")
            elif min_sufficient < 5:
                f.write("• Some classes have very few videos with good lip detection\n")

        self.logger.info(f"Comprehensive report saved to: {report_path}")

        # Generate detection distribution plots
        self.generate_detection_distribution_plots()

        # Save configuration
        config_path = self.log_dir / f"lip_detection_config_{timestamp}.json"
        config = {
            'lip_detection_threshold': self.lip_detection_threshold,
            'input_dir': str(self.input_dir),
            'timestamp': timestamp,
            'total_analyzed': total_videos,
            'sufficient_detection_count': sufficient_detection,
            'analysis_only': True,
            'no_files_modified': True,
            'opencv_settings': {
                'scale_factor': self.scale_factor,
                'min_neighbors': self.min_neighbors,
                'min_face_size': self.min_face_size
            },
            'lip_region_params': self.lip_region_params
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Analysis configuration saved to: {config_path}")

    def run_analysis(self) -> Dict:
        """
        Run the complete lip detection analysis pipeline WITHOUT modifying any files.

        Returns:
            Dictionary with summary statistics
        """
        self.logger.info("Starting Stage 2: Lip Detection Analysis - NO FILE OPERATIONS")

        # Process all videos
        self.process_all_videos()

        # Generate comprehensive reports
        self.generate_comprehensive_report()

        # Return summary statistics
        total_videos = len(self.results)
        sufficient_detection = sum(1 for r in self.results if r['has_sufficient_lip_detection'])

        summary = {
            'total_videos': total_videos,
            'sufficient_detection': sufficient_detection,
            'insufficient_detection': total_videos - sufficient_detection,
            'pass_rate': sufficient_detection / total_videos if total_videos > 0 else 0,
            'class_stats': self.class_stats.copy(),
            'files_modified': False,
            'analysis_only': True
        }

        self.logger.info("Stage 2 Lip Detection Analysis completed successfully!")
        self.logger.info(f"Summary: {sufficient_detection}/{total_videos} videos have sufficient lip detection "
                        f"({summary['pass_rate']*100:.1f}% pass rate)")
        self.logger.info("⚠️  NO FILES WERE MODIFIED - ALL ORIGINAL DATA PRESERVED")

        return summary


def main():
    """Main execution function for Stage 2 Lip Detection Analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Stage 2: Lip Detection Analysis')
    parser.add_argument('--input_dir', type=str,
                       default='data/grid/13.9.25top7dataset',
                       help='Input directory containing videos')
    parser.add_argument('--lip_detection_threshold', type=float, default=0.80,
                       help='Minimum fraction of frames with successful lip detection')
    parser.add_argument('--log_dir', type=str, default='lip_detection_reports',
                       help='Directory for analysis reports')

    args = parser.parse_args()

    # Initialize and run analyzer
    analyzer = LipDetectionAnalyzer(
        input_dir=args.input_dir,
        lip_detection_threshold=args.lip_detection_threshold,
        log_dir=args.log_dir
    )

    # Run the analysis process
    summary = analyzer.run_analysis()

    # Print final summary
    print("\n" + "="*80)
    print("STAGE 2 LIP DETECTION ANALYSIS - COMPLETED")
    print("="*80)
    print(f"Total videos analyzed: {summary['total_videos']}")
    print(f"Videos with sufficient lip detection: {summary['sufficient_detection']}")
    print(f"Videos with insufficient lip detection: {summary['insufficient_detection']}")
    print(f"Pass rate: {summary['pass_rate']*100:.1f}%")
    print("\nClass-wise results:")
    for cls, stats in summary['class_stats'].items():
        if stats['total'] > 0:
            print(f"  {cls}: {stats['sufficient_detection']}/{stats['total']} sufficient detection "
                  f"({stats['sufficient_detection']/stats['total']*100:.1f}%)")

    print(f"\nDetailed reports saved to: {args.log_dir}")
    print("⚠️  NO FILES WERE MODIFIED - ALL ORIGINAL DATA PRESERVED")

    return summary


if __name__ == "__main__":
    main()
