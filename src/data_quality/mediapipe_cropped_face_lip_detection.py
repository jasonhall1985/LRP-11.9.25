#!/usr/bin/env python3
"""
MediaPipe Lip Detection for Cropped Face Videos (ICU Lip-Reading Dataset)
Specifically designed for videos showing lower half of faces with lips in top-middle region.
Uses MediaPipe Face Mesh for precise facial landmark detection.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd

class MediaPipeCroppedFaceLipDetector:
    """
    MediaPipe-based lip detection analyzer specifically for cropped face videos.
    Optimized for ICU lip-reading dataset format where lips are in top-middle region.
    """
    
    def __init__(self, 
                 input_dir: str,
                 min_detection_confidence: float = 0.3,  # Lower for cropped faces
                 min_tracking_confidence: float = 0.3,   # Lower for cropped faces
                 success_threshold: float = 0.5,         # 50% of frames must have lip detection
                 output_dir: str = "mediapipe_cropped_face_reports"):
        """
        Initialize MediaPipe lip detector for cropped face videos.
        
        Args:
            input_dir: Directory containing input videos
            min_detection_confidence: Minimum confidence for face detection (lowered for cropped faces)
            min_tracking_confidence: Minimum confidence for face tracking (lowered for cropped faces)
            success_threshold: Minimum fraction of frames that must have lip detection
            output_dir: Directory for analysis reports
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.success_threshold = success_threshold
        
        # Create report directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe Face Mesh with settings optimized for cropped faces
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,                    # Expect only one face (cropped)
            refine_landmarks=True,              # Get detailed lip landmarks
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # MediaPipe Face Mesh lip landmark indices (468-point model)
        # These are the specific landmark points that define the lip region
        self.UPPER_LIP_OUTER = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        self.LOWER_LIP_OUTER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
        self.UPPER_LIP_INNER = [13, 82, 81, 80, 78]
        self.LOWER_LIP_INNER = [14, 317, 402, 318, 324]
        
        # Combined lip landmarks for comprehensive detection
        self.ALL_LIP_LANDMARKS = (self.UPPER_LIP_OUTER + self.LOWER_LIP_OUTER + 
                                 self.UPPER_LIP_INNER + self.LOWER_LIP_INNER)
        
        # Remove duplicates and sort
        self.ALL_LIP_LANDMARKS = sorted(list(set(self.ALL_LIP_LANDMARKS)))
        
        # Setup logging
        self.setup_logging()
        
        # Results storage
        self.failed_videos = []
        self.successful_videos = []
        self.analysis_stats = {}
        
    def setup_logging(self):
        """Setup detailed logging for MediaPipe lip detection analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"mediapipe_cropped_face_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"MediaPipe Cropped Face Lip Detector initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Detection confidence: {self.min_detection_confidence}")
        self.logger.info(f"Tracking confidence: {self.min_tracking_confidence}")
        self.logger.info(f"Success threshold: {self.success_threshold}")
        self.logger.info("Optimized for cropped face videos with lips in top-middle region")
        self.logger.info("⚠️  NO FILE OPERATIONS WILL BE PERFORMED - ANALYSIS ONLY")
        
    def extract_class_from_filename(self, filename: str) -> str:
        """Extract class label from filename."""
        classes = ['doctor', 'glasses', 'phone', 'pillow', 'help']
        for cls in classes:
            if filename.startswith(cls + '__'):
                return cls
        return 'unknown'
    
    def detect_lip_landmarks_cropped_face(self, frame: np.ndarray) -> Tuple[bool, Dict]:
        """
        Detect lip landmarks in a cropped face frame using MediaPipe Face Mesh.
        Optimized for videos where lips are in the top-middle region.
        
        Args:
            frame: Input frame (BGR format) - cropped face with lips in top-middle
            
        Returns:
            Tuple of (success, landmark_info)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Get the first (and should be only) face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract lip landmark coordinates
            h, w, _ = frame.shape
            lip_points = []
            
            for idx in self.ALL_LIP_LANDMARKS:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    # Ensure coordinates are within frame bounds
                    x = max(0, min(x, w-1))
                    y = max(0, min(y, h-1))
                    lip_points.append((x, y))
            
            if len(lip_points) > 0:
                # Calculate lip region statistics
                lip_points = np.array(lip_points)
                lip_center = np.mean(lip_points, axis=0)
                lip_width = np.max(lip_points[:, 0]) - np.min(lip_points[:, 0])
                lip_height = np.max(lip_points[:, 1]) - np.min(lip_points[:, 1])
                
                # Calculate lip region quality metrics
                lip_area = lip_width * lip_height
                
                # Check if lip region is in expected location (top-middle for cropped faces)
                expected_y_range = (0, h * 0.6)  # Lips should be in top 60% of cropped face
                expected_x_range = (w * 0.2, w * 0.8)  # Lips should be in middle 60% horizontally
                
                lips_in_expected_region = (
                    expected_y_range[0] <= lip_center[1] <= expected_y_range[1] and
                    expected_x_range[0] <= lip_center[0] <= expected_x_range[1]
                )
                
                landmark_info = {
                    'face_detected': True,
                    'lip_landmarks_count': len(lip_points),
                    'lip_center': lip_center.tolist(),
                    'lip_width': float(lip_width),
                    'lip_height': float(lip_height),
                    'lip_area': float(lip_area),
                    'lips_in_expected_region': lips_in_expected_region,
                    'detection_quality': 'good' if lips_in_expected_region and lip_area > 100 else 'poor'
                }
                
                return True, landmark_info
        
        # No face/lips detected
        landmark_info = {
            'face_detected': False,
            'lip_landmarks_count': 0,
            'lip_center': [0, 0],
            'lip_width': 0,
            'lip_height': 0,
            'lip_area': 0,
            'lips_in_expected_region': False,
            'detection_quality': 'failed'
        }
        
        return False, landmark_info
    
    def analyze_video_lip_detection(self, video_path: Path) -> Dict:
        """
        Analyze lip detection for a single cropped face video using MediaPipe.
        
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
            frames_with_lips = 0
            frames_with_good_quality = 0
            lip_areas = []
            lip_centers = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                total_frames += 1
                
                # Detect lip landmarks in current frame
                success, landmark_info = self.detect_lip_landmarks_cropped_face(frame)
                
                if success:
                    frames_with_lips += 1
                    lip_areas.append(landmark_info['lip_area'])
                    lip_centers.append(landmark_info['lip_center'])
                    
                    if landmark_info['detection_quality'] == 'good':
                        frames_with_good_quality += 1
            
            cap.release()
            
            # Calculate detection statistics
            lip_detection_rate = frames_with_lips / total_frames if total_frames > 0 else 0.0
            good_quality_rate = frames_with_good_quality / total_frames if total_frames > 0 else 0.0
            
            # Determine if this video has sufficient lip detection
            has_sufficient_detection = lip_detection_rate >= self.success_threshold
            
            result = {
                'filename': filename,
                'class': class_label,
                'total_frames': total_frames,
                'frames_with_lips': frames_with_lips,
                'frames_with_good_quality': frames_with_good_quality,
                'lip_detection_rate': lip_detection_rate,
                'good_quality_rate': good_quality_rate,
                'has_sufficient_detection': has_sufficient_detection,
                'mean_lip_area': np.mean(lip_areas) if lip_areas else 0,
                'max_lip_area': np.max(lip_areas) if lip_areas else 0,
                'lip_center_stability': np.std(lip_centers, axis=0).tolist() if len(lip_centers) > 1 else [0, 0],
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
            'frames_with_lips': 0,
            'frames_with_good_quality': 0,
            'lip_detection_rate': 0.0,
            'good_quality_rate': 0.0,
            'has_sufficient_detection': False,
            'mean_lip_area': 0,
            'max_lip_area': 0,
            'lip_center_stability': [0, 0],
            'analysis_status': error_reason,
            'file_size_mb': 0
        }

    def process_all_videos(self) -> None:
        """Process all videos in the input directory for MediaPipe lip detection analysis."""
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov']
        video_files = []

        for ext in video_extensions:
            video_files.extend(list(self.input_dir.glob(f'*{ext}')))

        self.logger.info(f"Found {len(video_files)} video files to analyze with MediaPipe Face Mesh")

        # Initialize class statistics
        classes = ['doctor', 'glasses', 'phone', 'pillow', 'help', 'unknown']
        self.analysis_stats = {cls: {'total': 0, 'failed': 0, 'successful': 0} for cls in classes}

        # Process each video with progress bar
        for video_path in tqdm(video_files, desc="Analyzing with MediaPipe Face Mesh (Cropped Faces)"):
            result = self.analyze_video_lip_detection(video_path)

            class_label = result['class']
            self.analysis_stats[class_label]['total'] += 1

            if result['has_sufficient_detection']:
                self.successful_videos.append(result)
                self.analysis_stats[class_label]['successful'] += 1
            else:
                self.failed_videos.append(result)
                self.analysis_stats[class_label]['failed'] += 1

        self.logger.info("MediaPipe cropped face lip detection analysis completed - NO FILES WERE MODIFIED")

    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive report focusing on MediaPipe lip detection results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results to CSV
        all_results = self.successful_videos + self.failed_videos
        df = pd.DataFrame(all_results)
        csv_path = self.output_dir / f"mediapipe_cropped_face_detailed_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Detailed results saved to: {csv_path}")

        # Generate comprehensive report
        report_path = self.output_dir / f"mediapipe_cropped_face_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("MEDIAPIPE CROPPED FACE LIP DETECTION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Detection confidence threshold: {self.min_detection_confidence}\n")
            f.write(f"Tracking confidence threshold: {self.min_tracking_confidence}\n")
            f.write(f"Success threshold: {self.success_threshold} ({self.success_threshold*100:.0f}% of frames)\n")
            f.write("Optimized for cropped face videos with lips in top-middle region\n")
            f.write("⚠️  NO FILES WERE MODIFIED - ANALYSIS ONLY\n\n")

            # Summary statistics
            total_videos = len(all_results)
            successful_videos = len(self.successful_videos)
            failed_videos = len(self.failed_videos)

            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total videos analyzed: {total_videos}\n")
            f.write(f"Videos with successful lip detection: {successful_videos} ({successful_videos/total_videos*100:.1f}%)\n")
            f.write(f"Videos with FAILED lip detection: {failed_videos} ({failed_videos/total_videos*100:.1f}%)\n\n")

            # Detection rate statistics for successful videos
            if self.successful_videos:
                detection_rates = [v['lip_detection_rate'] for v in self.successful_videos]
                quality_rates = [v['good_quality_rate'] for v in self.successful_videos]

                f.write("SUCCESSFUL VIDEOS STATISTICS:\n")
                f.write("-" * 50 + "\n")
                f.write(f"Mean lip detection rate: {np.mean(detection_rates):.3f}\n")
                f.write(f"Median lip detection rate: {np.median(detection_rates):.3f}\n")
                f.write(f"Min lip detection rate: {np.min(detection_rates):.3f}\n")
                f.write(f"Max lip detection rate: {np.max(detection_rates):.3f}\n")
                f.write(f"Mean good quality rate: {np.mean(quality_rates):.3f}\n")
                f.write(f"Median good quality rate: {np.median(quality_rates):.3f}\n\n")

            # Class-wise detailed statistics
            f.write("CLASS-WISE DETAILED ANALYSIS:\n")
            f.write("=" * 50 + "\n")
            for cls, stats in self.analysis_stats.items():
                if stats['total'] > 0:
                    success_rate = stats['successful'] / stats['total'] * 100
                    failure_rate = stats['failed'] / stats['total'] * 100

                    f.write(f"\n{cls.upper()}:\n")
                    f.write(f"  Total videos: {stats['total']}\n")
                    f.write(f"  Successful detection: {stats['successful']} ({success_rate:.1f}%)\n")
                    f.write(f"  Failed detection: {stats['failed']} ({failure_rate:.1f}%)\n")

                    # Class-specific statistics
                    class_successful = [v for v in self.successful_videos if v['class'] == cls]
                    class_failed = [v for v in self.failed_videos if v['class'] == cls]

                    if class_successful:
                        class_detection_rates = [v['lip_detection_rate'] for v in class_successful]
                        f.write(f"  Mean detection rate (successful): {np.mean(class_detection_rates):.3f}\n")

                    if class_failed:
                        class_failed_rates = [v['lip_detection_rate'] for v in class_failed]
                        f.write(f"  Mean detection rate (failed): {np.mean(class_failed_rates):.3f}\n")

            # Top performing videos
            f.write(f"\nTOP 10 PERFORMING VIDEOS:\n")
            f.write("=" * 50 + "\n")
            if self.successful_videos:
                top_videos = sorted(self.successful_videos, key=lambda x: x['lip_detection_rate'], reverse=True)[:10]
                for i, video in enumerate(top_videos, 1):
                    f.write(f"{i:2d}. {video['filename']}\n")
                    f.write(f"    Class: {video['class']}\n")
                    f.write(f"    Lip detection rate: {video['lip_detection_rate']:.3f}\n")
                    f.write(f"    Good quality rate: {video['good_quality_rate']:.3f}\n")
                    f.write(f"    Total frames: {video['total_frames']}\n\n")
            else:
                f.write("No successful videos found.\n")

        self.logger.info(f"Comprehensive report saved to: {report_path}")

        # Generate simple failure list
        failure_list_path = self.output_dir / f"mediapipe_failed_videos_{timestamp}.txt"
        with open(failure_list_path, 'w') as f:
            f.write("MEDIAPIPE LIP DETECTION FAILURES - SIMPLE LIST\n")
            f.write("=" * 60 + "\n\n")
            f.write("Videos where MediaPipe FAILS to detect lips consistently:\n")
            f.write(f"(Detection rate < {self.success_threshold*100:.0f}% of frames)\n\n")

            if self.failed_videos:
                # Sort by class and filename for easy reference
                failed_sorted = sorted(self.failed_videos, key=lambda x: (x['class'], x['filename']))

                for result in failed_sorted:
                    f.write(f"{result['filename']} ({result['class']}) - Detection: {result['lip_detection_rate']:.1%}\n")

                f.write(f"\nTotal failed videos: {len(self.failed_videos)}\n")
            else:
                f.write("🎉 NO FAILED VIDEOS! All videos have sufficient MediaPipe lip detection.\n")

        self.logger.info(f"Simple failure list saved to: {failure_list_path}")

    def run_analysis(self) -> Dict:
        """Run the complete MediaPipe cropped face lip detection analysis pipeline."""
        self.logger.info("Starting MediaPipe Cropped Face Lip Detection Analysis - NO FILE OPERATIONS")

        # Process all videos
        self.process_all_videos()

        # Generate comprehensive reports
        self.generate_comprehensive_report()

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

        self.logger.info("MediaPipe Cropped Face Lip Detection Analysis completed successfully!")
        self.logger.info(f"Summary: {len(self.successful_videos)}/{total_videos} videos have successful lip detection "
                        f"({summary['success_rate']*100:.1f}% success rate)")
        self.logger.info(f"Failed videos: {len(self.failed_videos)} ({summary['failure_rate']*100:.1f}% failure rate)")
        self.logger.info("⚠️  NO FILES WERE MODIFIED - ALL ORIGINAL DATA PRESERVED")

        return summary


def main():
    """Main execution function for MediaPipe Cropped Face Lip Detection Analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='MediaPipe Cropped Face Lip Detection Analysis')
    parser.add_argument('--input_dir', type=str,
                       default='data/grid/13.9.25top7dataset',
                       help='Input directory containing cropped face videos')
    parser.add_argument('--min_detection_confidence', type=float, default=0.3,
                       help='Minimum confidence for face detection (lowered for cropped faces)')
    parser.add_argument('--min_tracking_confidence', type=float, default=0.3,
                       help='Minimum confidence for face tracking (lowered for cropped faces)')
    parser.add_argument('--success_threshold', type=float, default=0.5,
                       help='Minimum fraction of frames that must have lip detection')
    parser.add_argument('--output_dir', type=str, default='mediapipe_cropped_face_reports',
                       help='Directory for analysis reports')

    args = parser.parse_args()

    # Initialize and run analyzer
    analyzer = MediaPipeCroppedFaceLipDetector(
        input_dir=args.input_dir,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        success_threshold=args.success_threshold,
        output_dir=args.output_dir
    )

    # Run the analysis process
    summary = analyzer.run_analysis()

    # Print final summary
    print("\n" + "="*70)
    print("MEDIAPIPE CROPPED FACE LIP DETECTION ANALYSIS - COMPLETED")
    print("="*70)
    print(f"Total videos analyzed: {summary['total_videos']}")
    print(f"Videos with successful lip detection: {summary['successful_detection']}")
    print(f"Videos with FAILED lip detection: {summary['failed_detection']}")
    print(f"Success rate: {summary['success_rate']*100:.1f}%")
    print(f"Failure rate: {summary['failure_rate']*100:.1f}%")

    print("\nClass-wise analysis:")
    for cls, stats in summary['class_stats'].items():
        if stats['total'] > 0:
            success_rate = stats['successful'] / stats['total'] * 100
            failure_rate = stats['failed'] / stats['total'] * 100
            print(f"  {cls}: {stats['successful']}/{stats['total']} successful ({success_rate:.1f}%), "
                  f"{stats['failed']} failed ({failure_rate:.1f}%)")

    print(f"\nDetailed reports saved to: {args.output_dir}")
    print("⚠️  NO FILES WERE MODIFIED - ALL ORIGINAL DATA PRESERVED")

    return summary


if __name__ == "__main__":
    main()
