#!/usr/bin/env python3
"""
Debug Improved Pipeline - Adaptive Mouth ROI Detection
======================================================

Test the improved adaptive lip detection pipeline that automatically
handles both full face and cropped face videos.

Author: Augment Agent
Date: 2025-09-14
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict, Any

# Import our improved ROI utilities
from improved_roi_utils import AdaptiveLipDetector
from roi_utils import ROIGeometry, BBoxSmoother, RecropCalculator, create_debug_visualization

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedVideoDebugger:
    """Debug the improved adaptive mouth ROI pipeline."""
    
    def __init__(self, video_path: str, output_dir: str = "improved_debug_output"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize adaptive detector
        self.detector = AdaptiveLipDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            auto_detect_mode=True
        )
        
        # Initialize smoother
        self.smoother = BBoxSmoother(alpha=0.3)
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'detected_frames': 0,
            'failed_frames': 0,
            'detection_mode': None,
            'bboxes': [],
            'ratios': []
        }
        
    def analyze_video(self, max_frames: int = 50, save_debug_frames: bool = True) -> Dict[str, Any]:
        """
        Analyze video with the improved adaptive pipeline.
        """
        logger.info(f"Starting improved analysis of: {self.video_path}")
        
        if not os.path.exists(self.video_path):
            logger.error(f"Video file not found: {self.video_path}")
            return {}
            
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {self.video_path}")
            return {}
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
        
        frame_idx = 0
        processed_frames = []
        
        while frame_idx < min(max_frames, total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            self.stats['total_frames'] += 1
            
            # Detect lip landmarks with adaptive method
            landmarks = self.detector.detect_lip_landmarks(frame)
            
            if landmarks is not None:
                self.stats['detected_frames'] += 1
                
                # Calculate tight bounding box around lips
                tight_bbox = ROIGeometry.calculate_tight_bbox(landmarks)
                
                # Add padding
                padded_bbox = ROIGeometry.add_padding(
                    tight_bbox, padding_ratio=0.12, frame_shape=frame.shape[:2]
                )
                
                # Apply smoothing
                smoothed_bbox = self.smoother.smooth(padded_bbox)
                
                # Calculate size ratios
                ratios = ROIGeometry.calculate_size_ratios(smoothed_bbox, frame.shape[:2])
                
                # Store statistics
                self.stats['bboxes'].append(smoothed_bbox)
                self.stats['ratios'].append(ratios)
                
                # Get detection mode
                if self.stats['detection_mode'] is None:
                    self.stats['detection_mode'] = self.detector.get_detection_mode()
                
                # Log detailed frame info
                logger.info(f"Frame {frame_idx}: "
                          f"mode={self.detector.get_detection_mode()}, "
                          f"landmarks={len(landmarks)}, "
                          f"bbox={smoothed_bbox}, "
                          f"ratios={ratios}")
                
                # Create debug visualization
                if save_debug_frames:
                    debug_frame = create_debug_visualization(
                        frame, landmarks, smoothed_bbox, None, ratios
                    )
                    
                    # Add detection mode info
                    mode_text = f"Mode: {self.detector.get_detection_mode() or 'detecting...'}"
                    cv2.putText(debug_frame, mode_text, (10, height - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Save debug frame
                    debug_path = self.output_dir / f"debug_frame_{frame_idx:04d}.jpg"
                    cv2.imwrite(str(debug_path), debug_frame)
                
                # Crop the frame using the smoothed bbox
                x1, y1, x2, y2 = smoothed_bbox
                cropped_frame = frame[y1:y2, x1:x2]
                
                if cropped_frame.size > 0:
                    # Resize to standard size (96x96)
                    resized_frame = cv2.resize(cropped_frame, (96, 96))
                    processed_frames.append(resized_frame)
                    
                    # Save individual cropped frame
                    if save_debug_frames:
                        crop_path = self.output_dir / f"cropped_frame_{frame_idx:04d}.jpg"
                        cv2.imwrite(str(crop_path), resized_frame)
                
            else:
                self.stats['failed_frames'] += 1
                logger.warning(f"Frame {frame_idx}: No landmarks detected")
                
            frame_idx += 1
            
        cap.release()
        
        # Create output video from processed frames
        if processed_frames:
            self._create_output_video(processed_frames, fps)
            
        # Generate summary report
        return self._generate_report()
        
    def _create_output_video(self, frames: List[np.ndarray], fps: float):
        """Create output video from processed frames."""
        if not frames:
            logger.warning("No frames to create output video")
            return
            
        output_path = self.output_dir / "improved_processed_video.mp4"
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (96, 96))
        
        for frame in frames:
            out.write(frame)
            
        out.release()
        logger.info(f"Output video saved: {output_path}")
        
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        if self.stats['total_frames'] == 0:
            return {}
            
        detection_rate = self.stats['detected_frames'] / self.stats['total_frames']
        
        # Calculate average ratios
        avg_ratios = {}
        if self.stats['ratios']:
            for key in ['area_ratio', 'h_ratio', 'w_ratio']:
                values = [r[key] for r in self.stats['ratios']]
                avg_ratios[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        report = {
            'video_path': self.video_path,
            'detection_mode': self.stats['detection_mode'],
            'total_frames': self.stats['total_frames'],
            'detected_frames': self.stats['detected_frames'],
            'failed_frames': self.stats['failed_frames'],
            'detection_rate': detection_rate,
            'average_ratios': avg_ratios
        }
        
        # Save report to file
        import json
        report_path = self.output_dir / "improved_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Analysis report saved: {report_path}")
        return report


def main():
    """Main function to run improved pipeline debugging."""
    
    # Target video path
    video_path = "/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped/doctor__useruser01__18to39__female__aboriginal__20250807T054104_topmid.mp4"
    
    # Create debugger
    debugger = ImprovedVideoDebugger(video_path, "improved_debug_output")
    
    # Run analysis
    logger.info("=" * 60)
    logger.info("STARTING IMPROVED ADAPTIVE PIPELINE DEBUG")
    logger.info("=" * 60)
    
    results = debugger.analyze_video(max_frames=30, save_debug_frames=True)
    
    if results:
        logger.info("=" * 60)
        logger.info("IMPROVED ANALYSIS RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Detection mode: {results['detection_mode']}")
        logger.info(f"Detection rate: {results['detection_rate']:.2%}")
        logger.info(f"Detected frames: {results['detected_frames']}/{results['total_frames']}")
        
        if results['average_ratios']:
            logger.info("Average ROI ratios:")
            for key, stats in results['average_ratios'].items():
                logger.info(f"  {key}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                          f"(range: {stats['min']:.3f} - {stats['max']:.3f})")
        
        logger.info(f"Output directory: improved_debug_output/")
        logger.info("Check the debug frames and processed video for visual inspection.")
    else:
        logger.error("Analysis failed - check video path and dependencies")


if __name__ == "__main__":
    main()
