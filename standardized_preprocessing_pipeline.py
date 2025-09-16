#!/usr/bin/env python3
"""
Standardized Data Preprocessing Pipeline for Lip-Reading Training
================================================================

A comprehensive preprocessing pipeline that processes videos into consistent format
for 5-class classification task (phone, glasses, doctor, help, pillow).

Features:
- MediaPipe Face Mesh for facial landmark detection
- Configured for ICU-style cropped face videos (lower half of faces)
- Geometric cropping: top 50% height, middle 33% width
- Temporal standardization: exactly 32 frames per video
- Grayscale conversion
- No resizing/scaling - maintains original cropped dimensions
- Visual outputs for inspection and validation
- Comprehensive manifest generation

Usage:
    # Phase 1: Single video test
    python standardized_preprocessing_pipeline.py --mode single --input "data/TEST SET/doctor 3.mp4" --output single_video_test_output
    
    # Phase 2: Full dataset processing
    python standardized_preprocessing_pipeline.py --mode batch --input data --output processed_dataset

Author: Augment Agent
Date: 2025-09-16
"""

import cv2
import numpy as np
import os
import sys
import argparse
import json
import logging
import csv
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from datetime import datetime
import time
from tqdm import tqdm

# Import existing utilities
from roi_utils import MediaPipeLipDetector, MEDIAPIPE_AVAILABLE

class StandardizedPreprocessingPipeline:
    """
    Standardized preprocessing pipeline for lip-reading training data.
    
    Processes videos into consistent format with MediaPipe Face Mesh detection,
    geometric cropping, and temporal standardization.
    """
    
    def __init__(self, 
                 output_dir: str,
                 target_frames: int = 32,
                 detection_confidence: float = 0.45,
                 salvage_confidence: float = 0.30,
                 enable_visual_outputs: bool = True):
        """
        Initialize preprocessing pipeline.
        
        Args:
            output_dir: Directory for processed outputs
            target_frames: Target number of frames per video (default: 32)
            detection_confidence: Primary detection threshold (40-50%)
            salvage_confidence: Salvaging threshold (20-40%)
            enable_visual_outputs: Generate visual inspection outputs
        """
        self.output_dir = Path(output_dir)
        self.target_frames = target_frames
        self.detection_confidence = detection_confidence
        self.salvage_confidence = salvage_confidence
        self.enable_visual_outputs = enable_visual_outputs
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "processed_videos").mkdir(exist_ok=True)
        (self.output_dir / "debug_frames").mkdir(exist_ok=True)
        (self.output_dir / "cropped_frames").mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize lip detector (uses existing utilities)
        self.setup_lip_detector()
        
        # Geometric cropping parameters
        self.crop_height_ratio = 0.5    # Top 50% of frame height
        self.crop_width_start = 1/3     # Start at 33% of frame width (middle 33%)
        self.crop_width_end = 2/3       # End at 67% of frame width
        
        # Processing statistics
        self.stats = {
            'total_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'detection_success_rate': 0.0,
            'processing_time': 0.0
        }
        
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_file = self.output_dir / f"preprocessing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_lip_detector(self):
        """Initialize lip detector using existing utilities."""
        # Primary detector
        self.lip_detector = MediaPipeLipDetector(
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.detection_confidence
        )

        # Salvage detector with lower thresholds
        self.lip_detector_salvage = MediaPipeLipDetector(
            min_detection_confidence=self.salvage_confidence,
            min_tracking_confidence=self.salvage_confidence
        )

        detection_method = "MediaPipe Face Mesh" if MEDIAPIPE_AVAILABLE else "OpenCV Haar Cascades"
        self.logger.info(f"Lip detector initialized using: {detection_method}")
        self.logger.info(f"  - Primary detection confidence: {self.detection_confidence}")
        self.logger.info(f"  - Salvage detection confidence: {self.salvage_confidence}")
        
    def detect_lip_landmarks(self, frame: np.ndarray, use_salvage: bool = False) -> Optional[np.ndarray]:
        """
        Detect lip landmarks using existing utilities.

        Args:
            frame: Input frame (BGR format)
            use_salvage: Use lower confidence threshold for salvaging

        Returns:
            Lip landmarks as (N, 2) array or None if no face detected
        """
        if frame is None or frame.size == 0:
            return None

        # Choose appropriate detector based on salvage mode
        detector = self.lip_detector_salvage if use_salvage else self.lip_detector

        # Detect landmarks using existing utilities
        landmarks = detector.detect_lip_landmarks(frame)

        return landmarks

    def apply_lip_aware_crop(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply lip-aware cropping with 10% padding around detected lips.
        Falls back to geometric cropping if no lips detected.

        Args:
            frame: Input frame (H, W, C)

        Returns:
            Cropped frame with adequate lip padding (no resizing applied)
        """
        height, width = frame.shape[:2]

        # Try to detect lips first
        landmarks = self.detect_lip_landmarks(frame, use_salvage=False)
        if landmarks is None:
            # Try salvage detection
            landmarks = self.detect_lip_landmarks(frame, use_salvage=True)

        if landmarks is not None and len(landmarks) > 0:
            # Create bounding box around detected lips with 10% padding
            min_x = int(np.min(landmarks[:, 0]))
            max_x = int(np.max(landmarks[:, 0]))
            min_y = int(np.min(landmarks[:, 1]))
            max_y = int(np.max(landmarks[:, 1]))

            # Calculate lip region dimensions
            lip_width = max_x - min_x
            lip_height = max_y - min_y

            # Add 10% padding around lips
            padding_x = int(lip_width * 0.10)
            padding_y = int(lip_height * 0.10)

            # Apply padding
            x_start = max(0, min_x - padding_x)
            x_end = min(width, max_x + padding_x)
            y_start = max(0, min_y - padding_y)
            y_end = min(height, max_y + padding_y)

            self.logger.debug(f"Lip-aware crop: lips at ({min_x},{min_y})-({max_x},{max_y}), crop ({x_start},{y_start})-({x_end},{y_end})")

        else:
            # Fallback to generous geometric cropping with more lip space
            self.logger.debug("No lips detected, using generous geometric fallback crop")

            # Use more generous cropping to ensure lips aren't cut off
            # Top 60% height (instead of 50%) and middle 50% width (instead of 33%)
            y_start = 0
            y_end = int(height * 0.60)  # More vertical space

            # Center 50% width to ensure lateral lip coverage
            center_x = width // 2
            crop_width = int(width * 0.50)  # 50% width instead of 33%
            x_start = max(0, center_x - crop_width // 2)
            x_end = min(width, center_x + crop_width // 2)

            # Ensure valid boundaries
            y_end = max(y_start + 1, min(y_end, height))
            x_start = max(0, min(x_start, width - 1))
            x_end = max(x_start + 1, min(x_end, width))

            self.logger.debug(f"Generous geometric crop: ({x_start},{y_start})-({x_end},{y_end}) from {width}x{height}")

        # Extract crop region (NO RESIZING)
        cropped = frame[y_start:y_end, x_start:x_end]

        return cropped

    def apply_geometric_crop(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply geometric cropping: top 50% height, middle 33% width.
        
        Args:
            frame: Input frame (H, W, C)
            
        Returns:
            Geometrically cropped frame (no resizing applied)
        """
        height, width = frame.shape[:2]
        
        # Calculate crop boundaries
        y_start = 0
        y_end = int(height * self.crop_height_ratio)
        
        x_start = int(width * self.crop_width_start)
        x_end = int(width * self.crop_width_end)
        
        # Ensure valid boundaries
        y_end = max(y_start + 1, min(y_end, height))
        x_end = max(x_start + 1, min(x_end, width))
        
        # Extract crop region (NO RESIZING)
        cropped = frame[y_start:y_end, x_start:x_end]
        
        return cropped
        
    def extract_temporal_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract exactly target_frames frames using uniform temporal sampling.
        
        Args:
            video_path: Path to input video
            
        Returns:
            List of exactly target_frames frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= self.target_frames:
            # If video has fewer frames, read all and pad/interpolate
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            
            # Pad by repeating last frame if needed
            while len(frames) < self.target_frames:
                frames.append(frames[-1].copy() if frames else np.zeros((480, 640, 3), dtype=np.uint8))
                
            return frames[:self.target_frames]
        else:
            # Uniform sampling for longer videos
            frame_indices = np.linspace(0, total_frames - 1, self.target_frames, dtype=int)
            frames = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    
            cap.release()
            return frames
            
    def convert_to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert frame to grayscale.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Grayscale frame
        """
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def process_single_video(self, video_path: str, video_name: str = None) -> Dict[str, Any]:
        """
        Process a single video through the complete preprocessing pipeline.

        Args:
            video_path: Path to input video
            video_name: Optional custom name for output files

        Returns:
            Processing results dictionary
        """
        if video_name is None:
            video_name = Path(video_path).stem

        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Output name: {video_name}")

        start_time = time.time()

        # Initialize results
        results = {
            'input_path': video_path,
            'video_name': video_name,
            'status': 'unknown',
            'error': None,
            'original_dimensions': None,
            'cropped_dimensions': None,
            'total_frames': 0,
            'processed_frames': 0,
            'detection_success_rate': 0.0,
            'detection_confidence_scores': [],
            'output_files': {},
            'processing_time': 0.0
        }

        try:
            # Get original video properties
            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            original_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_duration = original_total_frames / original_fps if original_fps > 0 else 0
            cap.release()

            self.logger.info(f"Original video: {original_total_frames} frames, {original_fps:.2f} FPS, {original_duration:.2f}s")

            # Step 1: Extract temporal frames
            self.logger.info(f"Step 1: Extracting {self.target_frames} frames...")
            frames = self.extract_temporal_frames(video_path)

            if not frames:
                results['status'] = 'failed'
                results['error'] = 'Could not extract frames from video'
                return results

            results['total_frames'] = len(frames)
            results['original_dimensions'] = f"{frames[0].shape[1]}x{frames[0].shape[0]}"

            # Step 2: Process each frame through the pipeline
            self.logger.info("Step 2: Processing frames through pipeline...")
            processed_frames = []
            detection_scores = []
            debug_info = []

            for i, frame in enumerate(frames):
                frame_result = self.process_single_frame(frame, i, video_name)

                if frame_result['success']:
                    processed_frames.append(frame_result['processed_frame'])
                    detection_scores.append(frame_result['confidence_score'])
                    debug_info.append(frame_result['debug_info'])
                else:
                    # Use lip-aware crop with 10% padding (includes geometric fallback)
                    fallback_frame = self.apply_lip_aware_crop(frame)
                    fallback_frame = self.convert_to_grayscale(fallback_frame)
                    processed_frames.append(fallback_frame)
                    detection_scores.append(0.0)
                    debug_info.append({'method': 'lip_aware_fallback', 'landmarks': None})

            results['processed_frames'] = len(processed_frames)
            results['detection_confidence_scores'] = detection_scores
            results['detection_success_rate'] = sum(1 for score in detection_scores if score > 0) / len(detection_scores)

            if processed_frames:
                results['cropped_dimensions'] = f"{processed_frames[0].shape[1]}x{processed_frames[0].shape[0]}"

            # Step 3: Save processed video
            self.logger.info("Step 3: Saving processed video...")
            output_video_path = self.save_processed_video(processed_frames, video_name, original_fps, original_duration)
            results['output_files']['processed_video'] = str(output_video_path)

            # Step 4: Generate visual outputs if enabled
            if self.enable_visual_outputs:
                self.logger.info("Step 4: Generating visual outputs...")
                visual_outputs = self.generate_visual_outputs(frames, processed_frames, debug_info, video_name)
                results['output_files'].update(visual_outputs)

            # Step 5: Create manifest entry
            self.logger.info("Step 5: Creating manifest entry...")
            manifest_entry = self.create_manifest_entry(results)
            manifest_path = self.save_manifest_entry(manifest_entry, video_name)
            results['output_files']['manifest'] = str(manifest_path)

            results['status'] = 'success'
            results['processing_time'] = time.time() - start_time

            self.logger.info(f"✅ Successfully processed {video_name}")
            self.logger.info(f"   - Detection success rate: {results['detection_success_rate']:.1%}")
            self.logger.info(f"   - Processing time: {results['processing_time']:.2f}s")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            results['processing_time'] = time.time() - start_time
            self.logger.error(f"❌ Failed to process {video_name}: {e}")

        return results

    def process_single_frame(self, frame: np.ndarray, frame_idx: int, video_name: str) -> Dict[str, Any]:
        """
        Process a single frame through the pipeline.

        Args:
            frame: Input frame
            frame_idx: Frame index for debugging
            video_name: Video name for debug outputs

        Returns:
            Frame processing results
        """
        result = {
            'success': False,
            'processed_frame': None,
            'confidence_score': 0.0,
            'debug_info': {}
        }

        # Try primary detection first
        landmarks = self.detect_lip_landmarks(frame, use_salvage=False)
        confidence_score = self.detection_confidence

        # If primary detection fails, try salvage detection
        if landmarks is None:
            landmarks = self.detect_lip_landmarks(frame, use_salvage=True)
            confidence_score = self.salvage_confidence if landmarks is not None else 0.0

        # Apply lip-aware cropping with 10% padding
        cropped_frame = self.apply_lip_aware_crop(frame)

        # Convert to grayscale
        gray_frame = self.convert_to_grayscale(cropped_frame)

        result['processed_frame'] = gray_frame
        result['confidence_score'] = confidence_score
        result['success'] = True
        result['debug_info'] = {
            'frame_idx': frame_idx,
            'landmarks_detected': landmarks is not None,
            'num_landmarks': len(landmarks) if landmarks is not None else 0,
            'detection_method': 'primary' if confidence_score == self.detection_confidence else 'salvage' if landmarks is not None else 'lip_aware_crop',
            'landmarks': landmarks.tolist() if landmarks is not None else None
        }

        return result

    def save_processed_video(self, frames: List[np.ndarray], video_name: str, original_fps: float = None, original_duration: float = None) -> Path:
        """
        Save processed frames as a video file.

        Args:
            frames: List of processed frames
            video_name: Name for output video
            original_fps: Original video FPS
            original_duration: Original video duration in seconds

        Returns:
            Path to saved video file
        """
        output_path = self.output_dir / "processed_videos" / f"{video_name}_processed.mp4"

        if not frames:
            raise ValueError("No frames to save")

        # Get frame dimensions
        height, width = frames[0].shape[:2]

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Calculate appropriate FPS to preserve original video duration
        if original_duration is not None and len(frames) > 0:
            # Calculate FPS so that 32 frames play for the same duration as original video
            fps = len(frames) / original_duration
            self.logger.info(f"Calculated FPS: {fps:.2f} (to preserve {original_duration:.2f}s duration with {len(frames)} frames)")
        elif original_fps is not None:
            fps = original_fps
        else:
            fps = 30.0  # Fallback FPS

        # Always write as color video (convert grayscale to BGR)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)

        for frame in frames:
            if len(frame.shape) == 2:
                # Convert grayscale to BGR for video writer
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                out.write(frame_bgr)
            else:
                out.write(frame)

        out.release()

        self.logger.info(f"Saved processed video: {output_path}")
        return output_path

    def generate_visual_outputs(self, original_frames: List[np.ndarray],
                              processed_frames: List[np.ndarray],
                              debug_info: List[Dict],
                              video_name: str) -> Dict[str, str]:
        """
        Generate visual outputs for inspection.

        Args:
            original_frames: Original video frames
            processed_frames: Processed frames
            debug_info: Debug information for each frame
            video_name: Video name for output files

        Returns:
            Dictionary of generated output file paths
        """
        outputs = {}

        # Save sample frames with landmarks
        sample_indices = [0, len(original_frames)//4, len(original_frames)//2,
                         3*len(original_frames)//4, len(original_frames)-1]

        for i, idx in enumerate(sample_indices):
            if idx < len(original_frames):
                # Original frame with landmarks
                original_with_landmarks = self.draw_landmarks_on_frame(
                    original_frames[idx], debug_info[idx])
                landmark_path = self.output_dir / "debug_frames" / f"{video_name}_landmarks_sample_{i}.jpg"
                cv2.imwrite(str(landmark_path), original_with_landmarks)

                # Processed frame
                processed_path = self.output_dir / "cropped_frames" / f"{video_name}_processed_sample_{i}.jpg"
                if len(processed_frames[idx].shape) == 2:
                    cv2.imwrite(str(processed_path), processed_frames[idx])
                else:
                    cv2.imwrite(str(processed_path), processed_frames[idx])

                # Before/after comparison
                comparison = self.create_before_after_comparison(
                    original_frames[idx], processed_frames[idx])
                comparison_path = self.output_dir / "debug_frames" / f"{video_name}_comparison_sample_{i}.jpg"
                cv2.imwrite(str(comparison_path), comparison)

        outputs['sample_frames'] = str(self.output_dir / "debug_frames")
        outputs['cropped_frames'] = str(self.output_dir / "cropped_frames")

        # Create preview video (first 10 processed frames)
        preview_frames = processed_frames[:min(10, len(processed_frames))]
        preview_path = self.save_preview_video(preview_frames, video_name)
        outputs['preview_video'] = str(preview_path)

        return outputs

    def draw_landmarks_on_frame(self, frame: np.ndarray, debug_info: Dict) -> np.ndarray:
        """
        Draw detected landmarks on frame for visualization.

        Args:
            frame: Original frame
            debug_info: Debug information containing landmarks

        Returns:
            Frame with landmarks drawn
        """
        frame_copy = frame.copy()

        if debug_info.get('landmarks') is not None:
            landmarks = np.array(debug_info['landmarks'])

            # Draw landmarks as small circles
            for point in landmarks:
                cv2.circle(frame_copy, tuple(point.astype(int)), 2, (0, 255, 0), -1)

            # Draw bounding box around landmarks
            if len(landmarks) > 0:
                x_coords = landmarks[:, 0]
                y_coords = landmarks[:, 1]
                x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Add detection method text
        method = debug_info.get('detection_method', 'unknown')
        cv2.putText(frame_copy, f"Method: {method}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame_copy

    def create_before_after_comparison(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """
        Create side-by-side before/after comparison image.

        Args:
            original: Original frame
            processed: Processed frame

        Returns:
            Side-by-side comparison image
        """
        # Resize original to match processed height for comparison
        orig_h, orig_w = original.shape[:2]
        proc_h, proc_w = processed.shape[:2]

        # Scale original to match processed height
        scale = proc_h / orig_h
        new_orig_w = int(orig_w * scale)
        original_resized = cv2.resize(original, (new_orig_w, proc_h))

        # Convert processed to BGR if grayscale
        if len(processed.shape) == 2:
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            processed_bgr = processed

        # Create side-by-side comparison
        comparison = np.hstack([original_resized, processed_bgr])

        # Add labels
        cv2.putText(comparison, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "Processed", (new_orig_w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return comparison

    def save_preview_video(self, frames: List[np.ndarray], video_name: str) -> Path:
        """
        Save a short preview video of processed frames.

        Args:
            frames: Processed frames for preview
            video_name: Video name for output

        Returns:
            Path to preview video
        """
        output_path = self.output_dir / f"{video_name}_preview.mp4"

        if not frames:
            return output_path

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 5.0  # Slow FPS for preview

        # Always write as color video
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)

        for frame in frames:
            if len(frame.shape) == 2:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                out.write(frame_bgr)
            else:
                out.write(frame)

        out.release()
        return output_path

    def create_manifest_entry(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create manifest entry with comprehensive metadata.

        Args:
            results: Processing results dictionary

        Returns:
            Manifest entry dictionary
        """
        return {
            'video_name': results['video_name'],
            'input_path': results['input_path'],
            'output_path': results['output_files'].get('processed_video', ''),
            'status': results['status'],
            'error': results.get('error', ''),
            'original_dimensions': results.get('original_dimensions', ''),
            'cropped_dimensions': results.get('cropped_dimensions', ''),
            'total_frames': results.get('total_frames', 0),
            'processed_frames': results.get('processed_frames', 0),
            'target_frames': self.target_frames,
            'detection_success_rate': results.get('detection_success_rate', 0.0),
            'avg_confidence_score': np.mean(results.get('detection_confidence_scores', [0.0])),
            'min_confidence_score': np.min(results.get('detection_confidence_scores', [0.0])),
            'max_confidence_score': np.max(results.get('detection_confidence_scores', [0.0])),
            'processing_time': results.get('processing_time', 0.0),
            'crop_method': 'geometric_top50_middle33',
            'detection_method': 'mediapipe_face_mesh',
            'primary_confidence_threshold': self.detection_confidence,
            'salvage_confidence_threshold': self.salvage_confidence,
            'timestamp': datetime.now().isoformat()
        }

    def save_manifest_entry(self, manifest_entry: Dict[str, Any], video_name: str) -> Path:
        """
        Save manifest entry to CSV file.

        Args:
            manifest_entry: Manifest entry dictionary
            video_name: Video name for output file

        Returns:
            Path to manifest file
        """
        manifest_path = self.output_dir / f"{video_name}_manifest.csv"

        # Write manifest entry
        with open(manifest_path, 'w', newline='') as csvfile:
            fieldnames = list(manifest_entry.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(manifest_entry)

        return manifest_path

    def generate_comprehensive_report(self, results: Dict[str, Any]) -> Path:
        """
        Generate comprehensive test report.

        Args:
            results: Processing results

        Returns:
            Path to report file
        """
        report_path = self.output_dir / "comprehensive_test_report.json"

        # Create comprehensive report
        report = {
            'pipeline_info': {
                'version': '1.0.0',
                'target_frames': self.target_frames,
                'detection_confidence': self.detection_confidence,
                'salvage_confidence': self.salvage_confidence,
                'crop_method': 'geometric_top50_middle33',
                'output_format': 'grayscale'
            },
            'processing_results': results,
            'success_criteria': {
                'video_processed': results['status'] == 'success',
                'frames_extracted': results.get('processed_frames', 0) == self.target_frames,
                'detection_working': results.get('detection_success_rate', 0.0) > 0.0,
                'outputs_generated': len(results.get('output_files', {})) > 0
            },
            'recommendations': self._generate_recommendations(results),
            'timestamp': datetime.now().isoformat()
        }

        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Comprehensive report saved: {report_path}")
        return report_path

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on processing results."""
        recommendations = []

        detection_rate = results.get('detection_success_rate', 0.0)

        if detection_rate < 0.3:
            recommendations.append("Low detection success rate. Consider adjusting confidence thresholds.")
        elif detection_rate < 0.6:
            recommendations.append("Moderate detection success rate. Pipeline working but could be optimized.")
        else:
            recommendations.append("Good detection success rate. Pipeline performing well.")

        if results['status'] == 'success':
            recommendations.append("Video processed successfully. Ready for Phase 2 full dataset processing.")
        else:
            recommendations.append("Processing failed. Review error logs before proceeding to Phase 2.")

        return recommendations


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Standardized Preprocessing Pipeline for Lip-Reading')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single',
                       help='Processing mode: single video test or batch processing')
    parser.add_argument('--input', required=True,
                       help='Input video file (single mode) or directory (batch mode)')
    parser.add_argument('--output', required=True,
                       help='Output directory for processed files')
    parser.add_argument('--target-frames', type=int, default=32,
                       help='Target number of frames per video (default: 32)')
    parser.add_argument('--detection-confidence', type=float, default=0.45,
                       help='Primary detection confidence threshold (default: 0.45)')
    parser.add_argument('--salvage-confidence', type=float, default=0.30,
                       help='Salvage detection confidence threshold (default: 0.30)')
    parser.add_argument('--no-visual-outputs', action='store_true',
                       help='Disable visual output generation')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = StandardizedPreprocessingPipeline(
        output_dir=args.output,
        target_frames=args.target_frames,
        detection_confidence=args.detection_confidence,
        salvage_confidence=args.salvage_confidence,
        enable_visual_outputs=not args.no_visual_outputs
    )

    if args.mode == 'single':
        # Phase 1: Single video test
        print("="*80)
        print("PHASE 1: SINGLE VIDEO TEST")
        print("="*80)

        if not os.path.exists(args.input):
            print(f"Error: Input video not found: {args.input}")
            return 1

        # Process single video
        results = pipeline.process_single_video(args.input)

        # Generate comprehensive report
        report_path = pipeline.generate_comprehensive_report(results)

        # Print summary
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Status: {results['status']}")
        print(f"Detection Success Rate: {results.get('detection_success_rate', 0.0):.1%}")
        print(f"Processing Time: {results.get('processing_time', 0.0):.2f}s")
        print(f"Output Directory: {args.output}")
        print(f"Comprehensive Report: {report_path}")

        if results['status'] == 'success':
            print("\n✅ Phase 1 completed successfully!")
            print("Review visual outputs and proceed to Phase 2 if satisfied.")
            return 0
        else:
            print(f"\n❌ Phase 1 failed: {results.get('error', 'Unknown error')}")
            return 1

    else:
        # Phase 2: Batch processing (placeholder)
        print("="*80)
        print("PHASE 2: BATCH PROCESSING")
        print("="*80)
        print("Phase 2 implementation pending Phase 1 approval...")
        return 0


if __name__ == "__main__":
    sys.exit(main())
