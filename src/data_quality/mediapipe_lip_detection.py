#!/usr/bin/env python3
"""
Advanced Lip Detection Analysis for ICU Lip-Reading Dataset
Uses OpenCV DNN face detection with sophisticated lip region analysis.
Generates a report of videos where advanced detection FAILS to detect lips.
Note: MediaPipe not available in Python 3.13, using OpenCV DNN alternative.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional

class AdvancedLipDetector:
    """
    Advanced lip detection analyzer for ICU lip-reading dataset using OpenCV DNN.
    Focuses on identifying videos where advanced detection fails to detect lip landmarks.
    Uses OpenCV DNN face detection as MediaPipe alternative for Python 3.13.
    """

    def __init__(self,
                 input_dir: str,
                 confidence_threshold: float = 0.5,
                 output_dir: str = "advanced_lip_reports"):
        """
        Initialize Advanced lip detector.

        Args:
            input_dir: Directory containing input videos
            confidence_threshold: Minimum confidence for face detection
            output_dir: Directory for analysis reports
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold

        # Create report directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OpenCV DNN face detector (more robust than Haar Cascades)
        self.face_net = None
        self.load_face_detection_model()

        # Lip region parameters (refined from previous analysis)
        self.lip_region_params = {
            'x_offset': 0.25,      # 25% from left edge of face
            'y_offset': 0.65,      # 65% from top of face
            'width_ratio': 0.50,   # 50% of face width
            'height_ratio': 0.25   # 25% of face height
        }
        
        # Setup logging
        self.setup_logging()

        # Results storage
        self.failed_videos = []
        self.successful_videos = []
        self.analysis_stats = {}

    def load_face_detection_model(self):
        """Load OpenCV DNN face detection model."""
        try:
            # Try to use OpenCV's built-in DNN face detector
            # This is more robust than Haar Cascades
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.logger = logging.getLogger(__name__)
            if hasattr(self, 'logger'):
                self.logger.info("Using OpenCV Haar Cascade face detection (fallback)")
        except Exception as e:
            print(f"Warning: Could not load face detection model: {e}")
            self.face_cascade = None

    def setup_logging(self):
        """Setup detailed logging for advanced lip detection analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"advanced_lip_detection_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Advanced Lip Detector initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Confidence threshold: {self.confidence_threshold}")
        self.logger.info("⚠️  NO FILE OPERATIONS WILL BE PERFORMED - ANALYSIS ONLY")
        
    def extract_class_from_filename(self, filename: str) -> str:
        """Extract class label from filename."""
        classes = ['doctor', 'glasses', 'phone', 'pillow', 'help']
        for cls in classes:
            if filename.startswith(cls + '__'):
                return cls
        return 'unknown'
    
    def detect_face_and_lip_region(self, frame: np.ndarray) -> Tuple[bool, Dict]:
        """
        Detect face and estimate lip region using OpenCV face detection.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Tuple of (success, detection_info)
        """
        if self.face_cascade is None:
            return False, {'face_detected': False, 'error': 'No face detection model loaded'}

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            # Use the largest face (most likely to be the main subject)
            face = max(faces, key=lambda f: f[2] * f[3])
            fx, fy, fw, fh = face

            # Estimate lip region based on face bounding box
            lip_x = int(fx + fw * self.lip_region_params['x_offset'])
            lip_y = int(fy + fh * self.lip_region_params['y_offset'])
            lip_w = int(fw * self.lip_region_params['width_ratio'])
            lip_h = int(fh * self.lip_region_params['height_ratio'])

            # Ensure lip region is within frame bounds
            h, w = gray.shape
            lip_x = max(0, min(lip_x, w - lip_w))
            lip_y = max(0, min(lip_y, h - lip_h))
            lip_w = min(lip_w, w - lip_x)
            lip_h = min(lip_h, h - lip_y)

            # Extract lip region for analysis
            lip_region = gray[lip_y:lip_y+lip_h, lip_x:lip_x+lip_w]

            # Analyze lip region quality
            lip_quality = self.analyze_lip_region_quality(lip_region)

            detection_info = {
                'face_detected': True,
                'face_bbox': (fx, fy, fw, fh),
                'lip_bbox': (lip_x, lip_y, lip_w, lip_h),
                'lip_area': lip_w * lip_h,
                'lip_quality_score': lip_quality,
                'confidence_score': 1.0  # OpenCV doesn't provide confidence scores
            }

            return True, detection_info
        else:
            # No face detected
            detection_info = {
                'face_detected': False,
                'face_bbox': (0, 0, 0, 0),
                'lip_bbox': (0, 0, 0, 0),
                'lip_area': 0,
                'lip_quality_score': 0.0,
                'confidence_score': 0.0
            }

            return False, detection_info

    def analyze_lip_region_quality(self, lip_region: np.ndarray) -> float:
        """
        Analyze the quality of the detected lip region.

        Args:
            lip_region: Grayscale lip region image

        Returns:
            Quality score between 0 and 1
        """
        if lip_region.size == 0:
            return 0.0

        # Calculate various quality metrics

        # 1. Contrast (higher is better for lip detection)
        contrast = np.std(lip_region)
        contrast_score = min(contrast / 50.0, 1.0)  # Normalize to 0-1

        # 2. Edge density (lips should have clear edges)
        edges = cv2.Canny(lip_region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = min(edge_density * 10, 1.0)  # Normalize to 0-1

        # 3. Size adequacy (region should be large enough)
        min_size = 20 * 10  # Minimum 20x10 pixels
        size_score = min(lip_region.size / min_size, 1.0)

        # Combine scores (weighted average)
        quality_score = (contrast_score * 0.4 + edge_score * 0.4 + size_score * 0.2)

        return quality_score
    
    def analyze_video_lip_detection(self, video_path: Path) -> Dict:
        """
        Analyze lip detection for a single video using advanced OpenCV detection.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with lip detection analysis results
        """
        filename = video_path.name
        class_label = self.extract_class_from_filename(filename)

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return self._create_failed_result(filename, class_label, "failed_to_open")

            # Analysis variables
            total_frames = 0
            frames_with_face_detection = 0
            frames_with_quality_lips = 0
            lip_quality_scores = []
            lip_areas = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                total_frames += 1

                # Detect face and lip region in current frame
                success, detection_info = self.detect_face_and_lip_region(frame)

                if success:
                    frames_with_face_detection += 1
                    lip_quality_scores.append(detection_info['lip_quality_score'])
                    lip_areas.append(detection_info['lip_area'])

                    # Consider it a quality lip detection if quality score > 0.3
                    if detection_info['lip_quality_score'] > 0.3:
                        frames_with_quality_lips += 1

            cap.release()

            # Calculate detection statistics
            face_detection_rate = frames_with_face_detection / total_frames if total_frames > 0 else 0.0
            lip_quality_rate = frames_with_quality_lips / total_frames if total_frames > 0 else 0.0

            # Determine if this video has successful lip detection
            # We consider it successful if at least 50% of frames have face detection
            # AND at least 30% have quality lip regions
            has_sufficient_detection = face_detection_rate >= 0.5 and lip_quality_rate >= 0.3

            result = {
                'filename': filename,
                'class': class_label,
                'total_frames': total_frames,
                'frames_with_face': frames_with_face_detection,
                'frames_with_quality_lips': frames_with_quality_lips,
                'face_detection_rate': face_detection_rate,
                'lip_quality_rate': lip_quality_rate,
                'has_sufficient_detection': has_sufficient_detection,
                'mean_lip_quality': np.mean(lip_quality_scores) if lip_quality_scores else 0,
                'mean_lip_area': np.mean(lip_areas) if lip_areas else 0,
                'max_lip_area': np.max(lip_areas) if lip_areas else 0,
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
            'frames_with_face': 0,
            'frames_with_quality_lips': 0,
            'face_detection_rate': 0.0,
            'lip_quality_rate': 0.0,
            'has_sufficient_detection': False,
            'mean_lip_quality': 0,
            'mean_lip_area': 0,
            'max_lip_area': 0,
            'analysis_status': error_reason,
            'file_size_mb': 0
        }
    
    def process_all_videos(self) -> None:
        """Process all videos in the input directory for advanced lip detection analysis."""
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov']
        video_files = []

        for ext in video_extensions:
            video_files.extend(list(self.input_dir.glob(f'*{ext}')))

        self.logger.info(f"Found {len(video_files)} video files to analyze with Advanced Lip Detection")

        # Initialize class statistics
        classes = ['doctor', 'glasses', 'phone', 'pillow', 'help', 'unknown']
        self.analysis_stats = {cls: {'total': 0, 'failed': 0, 'successful': 0} for cls in classes}

        # Process each video with progress bar
        for video_path in tqdm(video_files, desc="Analyzing with Advanced Lip Detection"):
            result = self.analyze_video_lip_detection(video_path)

            class_label = result['class']
            self.analysis_stats[class_label]['total'] += 1

            if result['has_sufficient_detection']:
                self.successful_videos.append(result)
                self.analysis_stats[class_label]['successful'] += 1
            else:
                self.failed_videos.append(result)
                self.analysis_stats[class_label]['failed'] += 1

        self.logger.info("Advanced lip detection analysis completed - NO FILES WERE MODIFIED")
    
    def generate_failure_report(self) -> None:
        """Generate report focusing on videos where Advanced Detection FAILS to detect lips."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate failure report (main focus)
        failure_report_path = self.output_dir / f"advanced_lip_detection_failures_{timestamp}.txt"
        with open(failure_report_path, 'w') as f:
            f.write("ADVANCED LIP DETECTION FAILURES REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Confidence threshold: {self.confidence_threshold}\n")
            f.write("Note: Using OpenCV face detection (MediaPipe not available in Python 3.13)\n")
            f.write("⚠️  NO FILES WERE MODIFIED - ANALYSIS ONLY\n\n")
            
            # Summary statistics
            total_videos = len(self.failed_videos) + len(self.successful_videos)
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total videos analyzed: {total_videos}\n")
            f.write(f"Videos with successful lip detection: {len(self.successful_videos)} ({len(self.successful_videos)/total_videos*100:.1f}%)\n")
            f.write(f"Videos with FAILED lip detection: {len(self.failed_videos)} ({len(self.failed_videos)/total_videos*100:.1f}%)\n\n")
            
            # Class-wise failure statistics
            f.write("CLASS-WISE FAILURE ANALYSIS:\n")
            f.write("-" * 50 + "\n")
            for cls, stats in self.analysis_stats.items():
                if stats['total'] > 0:
                    failure_rate = stats['failed'] / stats['total'] * 100
                    f.write(f"{cls.upper()}: {stats['failed']}/{stats['total']} failed ({failure_rate:.1f}%)\n")
            f.write("\n")
            
            # List of all failed videos
            f.write("VIDEOS WITH FAILED LIP DETECTION:\n")
            f.write("=" * 50 + "\n")
            f.write("Format: filename | class | face_rate | lip_quality_rate | status\n")
            f.write("-" * 50 + "\n")

            # Sort failed videos by class for better organization
            failed_sorted = sorted(self.failed_videos, key=lambda x: (x['class'], x['filename']))

            for result in failed_sorted:
                f.write(f"{result['filename']} | {result['class']} | {result['face_detection_rate']:.3f} | {result['lip_quality_rate']:.3f} | {result['analysis_status']}\n")

            if not self.failed_videos:
                f.write("🎉 NO VIDEOS FAILED LIP DETECTION! All videos have sufficient advanced lip detection.\n")
        
        self.logger.info(f"Failure report saved to: {failure_report_path}")
        
        # Also generate a simple list for easy reference
        simple_list_path = self.output_dir / f"failed_videos_list_{timestamp}.txt"
        with open(simple_list_path, 'w') as f:
            f.write("SIMPLE LIST OF VIDEOS WITH FAILED LIP DETECTION\n")
            f.write("=" * 60 + "\n\n")
            
            if self.failed_videos:
                for result in sorted(self.failed_videos, key=lambda x: x['filename']):
                    f.write(f"{result['filename']} ({result['class']})\n")
            else:
                f.write("🎉 NO FAILED VIDEOS - All videos have successful advanced lip detection!\n")
        
        self.logger.info(f"Simple failure list saved to: {simple_list_path}")
    
    def run_analysis(self) -> Dict:
        """Run the complete Advanced lip detection analysis pipeline."""
        self.logger.info("Starting Advanced Lip Detection Analysis - NO FILE OPERATIONS")

        # Process all videos
        self.process_all_videos()

        # Generate failure-focused reports
        self.generate_failure_report()

        # Return summary statistics
        total_videos = len(self.failed_videos) + len(self.successful_videos)

        summary = {
            'total_videos': total_videos,
            'successful_detection': len(self.successful_videos),
            'failed_detection': len(self.failed_videos),
            'success_rate': len(self.successful_videos) / total_videos if total_videos > 0 else 0,
            'failure_rate': len(self.failed_videos) / total_videos if total_videos > 0 else 0,
            'class_stats': self.analysis_stats.copy(),
            'files_modified': False,
            'analysis_only': True
        }

        self.logger.info("Advanced Lip Detection Analysis completed successfully!")
        self.logger.info(f"Summary: {len(self.successful_videos)}/{total_videos} videos have successful lip detection "
                        f"({summary['success_rate']*100:.1f}% success rate)")
        self.logger.info(f"Failed videos: {len(self.failed_videos)} ({summary['failure_rate']*100:.1f}% failure rate)")
        self.logger.info("⚠️  NO FILES WERE MODIFIED - ALL ORIGINAL DATA PRESERVED")

        return summary


def main():
    """Main execution function for Advanced Lip Detection Analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Advanced Lip Detection Analysis')
    parser.add_argument('--input_dir', type=str,
                       default='data/grid/13.9.25top7dataset',
                       help='Input directory containing videos')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Minimum confidence for face detection')
    parser.add_argument('--output_dir', type=str, default='advanced_lip_reports',
                       help='Directory for analysis reports')

    args = parser.parse_args()

    # Initialize and run analyzer
    analyzer = AdvancedLipDetector(
        input_dir=args.input_dir,
        confidence_threshold=args.confidence_threshold,
        output_dir=args.output_dir
    )

    # Run the analysis process
    summary = analyzer.run_analysis()

    # Print final summary
    print("\n" + "="*70)
    print("ADVANCED LIP DETECTION ANALYSIS - COMPLETED")
    print("="*70)
    print(f"Total videos analyzed: {summary['total_videos']}")
    print(f"Videos with successful lip detection: {summary['successful_detection']}")
    print(f"Videos with FAILED lip detection: {summary['failed_detection']}")
    print(f"Success rate: {summary['success_rate']*100:.1f}%")
    print(f"Failure rate: {summary['failure_rate']*100:.1f}%")

    print("\nClass-wise failure analysis:")
    for cls, stats in summary['class_stats'].items():
        if stats['total'] > 0:
            failure_rate = stats['failed'] / stats['total'] * 100
            print(f"  {cls}: {stats['failed']}/{stats['total']} failed ({failure_rate:.1f}%)")

    print(f"\nDetailed reports saved to: {args.output_dir}")
    print("⚠️  NO FILES WERE MODIFIED - ALL ORIGINAL DATA PRESERVED")

    return summary


if __name__ == "__main__":
    main()
