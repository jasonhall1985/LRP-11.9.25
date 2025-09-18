#!/usr/bin/env python3
"""
MediaPipe-driven Mouth ROI Cropper for Lip-Reading Training Data
================================================================

Battle-tested pipeline for creating training-ready mouth crops from ICU lip-reading videos.
Uses MediaPipe Face Mesh for landmark-driven ROI detection with smoothing and bridging.

Features:
- Tolerant of partial detections (40-50% threshold)
- Bridges short gaps with exponential moving average smoothing
- Fixed 32-frame, 96x96 output for consistent training data
- Generates manifest CSV for clean train/test splits
- Optimized for cropped face videos (ICU dataset format)

Usage:
    python crop_mouth.py INPUT_DIR OUTPUT_DIR [MANIFEST_CSV]

Example:
    python crop_mouth.py data/grid/13.9.25top7dataset mouth_crops_96x96_32f manifest.csv
"""

import sys
import csv
import math
import json
import pathlib
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# Configuration parameters - tuned for ICU lip-reading dataset
PASS_MIN_FRAC = 0.40        # Accept if >= 40% frames have mouth detection
SALVAGE_MIN_FRAC = 0.20     # Salvage if >= 20% with bridging
BRIDGE_MAX = 6              # Max consecutive frames to fill from last ROI
EMA_ALPHA = 0.6             # ROI smoothing strength (exponential moving average)
MOUTH_MIN_W = 40            # Minimum mouth width in pixels
EXPAND_W, EXPAND_H = 1.6, 1.8  # Expand mouth bbox for chin/upper lip context
OUT_SIZE = 96               # Output crop size (96x96)
OUT_FPS = 25                # Output MP4 fps
TARGET_FRAMES = 32          # Fixed clip length for training

class MouthCropper:
    """
    MediaPipe-based mouth cropper for lip-reading training data preparation.
    Handles partial detections with smoothing and bridging for robust ROI extraction.
    """
    
    def __init__(self, input_dir: str, output_dir: str, manifest_path: str = "manifest.csv"):
        """
        Initialize the mouth cropper.
        
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory for cropped output videos
            manifest_path: Path for manifest CSV file
        """
        self.input_dir = pathlib.Path(input_dir)
        self.output_dir = pathlib.Path(output_dir)
        self.manifest_path = manifest_path
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe Face Mesh with settings optimized for cropped faces
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.3,  # Lower for cropped faces
            min_tracking_confidence=0.3    # Lower for cropped faces
        )
        
        # MediaPipe lip landmark indices (outer + inner subset for robust detection)
        # These are the key landmarks that define the mouth region
        self.LIP_IDXS = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,  # Outer lip contour
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308    # Inner lip details
        ]
        
        # Setup logging
        self.setup_logging()
        
        # Results tracking
        self.processed_videos = []
        self.stats = {
            'total_videos': 0,
            'kept_videos': 0,
            'salvaged_videos': 0,
            'dropped_videos': 0,
            'detection_rates': []
        }
    
    def setup_logging(self):
        """Setup logging for the cropping process."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"mouth_cropping_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("MediaPipe Mouth Cropper initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Pass threshold: {PASS_MIN_FRAC*100:.0f}% detection rate")
        self.logger.info(f"Salvage threshold: {SALVAGE_MIN_FRAC*100:.0f}% detection rate")
        self.logger.info(f"Target output: {TARGET_FRAMES} frames, {OUT_SIZE}x{OUT_SIZE} pixels")
    
    def get_label_from_filename(self, filename: str) -> str:
        """
        Extract class label from filename.
        Expected format: "doctor__user...mp4" -> "doctor"
        """
        parts = filename.split("__")
        if len(parts) > 0:
            label = parts[0].lower()
            # Map known classes
            known_classes = ['doctor', 'glasses', 'phone', 'pillow', 'help']
            if label in known_classes:
                return label
        return 'unknown'
    
    def extract_mouth_bbox(self, landmarks, width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Extract mouth bounding box from MediaPipe face landmarks.
        
        Args:
            landmarks: MediaPipe face landmarks
            width: Frame width
            height: Frame height
            
        Returns:
            Tuple of (x1, y1, x2, y2) bounding box coordinates
        """
        # Extract lip landmark coordinates
        xs = [int(landmarks[i].x * width) for i in self.LIP_IDXS]
        ys = [int(landmarks[i].y * height) for i in self.LIP_IDXS]
        
        # Calculate bounding box with safety bounds
        x1 = max(min(xs), 0)
        x2 = min(max(xs), width - 1)
        y1 = max(min(ys), 0)
        y2 = min(max(ys), height - 1)
        
        return x1, y1, x2, y2
    
    def expand_and_clip_bbox(self, x1: int, y1: int, x2: int, y2: int, 
                           width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Expand mouth bounding box to include chin and upper lip context.
        
        Args:
            x1, y1, x2, y2: Original bounding box
            width, height: Frame dimensions
            
        Returns:
            Expanded and clipped bounding box
        """
        # Calculate center and current dimensions
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        
        # Expand dimensions
        new_width = bw * EXPAND_W
        new_height = bh * EXPAND_H
        
        # Calculate new coordinates with clipping
        nx1 = int(max(0, cx - new_width / 2))
        ny1 = int(max(0, cy - new_height / 2))
        nx2 = int(min(width - 1, cx + new_width / 2))
        ny2 = int(min(height - 1, cy + new_height / 2))
        
        return nx1, ny1, nx2, ny2
    
    def apply_exponential_moving_average(self, previous_roi: Optional[Tuple[int, int, int, int]], 
                                       current_roi: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Apply exponential moving average smoothing to ROI coordinates.
        
        Args:
            previous_roi: Previous smoothed ROI (None for first frame)
            current_roi: Current detected ROI
            
        Returns:
            Smoothed ROI coordinates
        """
        if previous_roi is None:
            return current_roi
        
        # Apply EMA to each coordinate
        smoothed = tuple(
            int(EMA_ALPHA * current_roi[i] + (1 - EMA_ALPHA) * previous_roi[i])
            for i in range(4)
        )
        
        return smoothed
    
    def generate_uniform_frame_indices(self, total_frames: int, target_frames: int) -> List[int]:
        """
        Generate uniformly distributed frame indices for resampling.
        
        Args:
            total_frames: Total number of available frames
            target_frames: Target number of frames to sample
            
        Returns:
            List of frame indices
        """
        if total_frames <= 0:
            return []
        
        if total_frames == target_frames:
            return list(range(total_frames))
        
        if total_frames >= target_frames:
            # Uniform sampling
            indices = [int(round(i * (total_frames - 1) / (target_frames - 1))) 
                      for i in range(target_frames)]
        else:
            # Pad with repetition of last frame
            indices = list(range(total_frames))
            while len(indices) < target_frames:
                indices.append(indices[-1] if indices else 0)
        
        return indices

    def process_single_video(self, video_path: pathlib.Path) -> Optional[Dict[str, Any]]:
        """
        Process a single video file to extract mouth ROI crops.

        Args:
            video_path: Path to input video file

        Returns:
            Dictionary with processing results or None if failed/dropped
        """
        label = self.get_label_from_filename(video_path.name)

        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            if total_frames == 0:
                self.logger.warning(f"No frames found in {video_path.name}")
                cap.release()
                return None

            # Process all frames
            frames = []
            rois = []
            detection_hits = 0
            last_valid_roi = None
            consecutive_misses = 0

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                height, width = frame.shape[:2]

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                current_roi = None
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    x1, y1, x2, y2 = self.extract_mouth_bbox(landmarks, width, height)

                    # Validate mouth detection quality
                    mouth_width = x2 - x1
                    mouth_height = y2 - y1

                    if mouth_width >= MOUTH_MIN_W and mouth_height >= MOUTH_MIN_W // 2:
                        current_roi = (x1, y1, x2, y2)
                        last_valid_roi = current_roi
                        consecutive_misses = 0
                        detection_hits += 1
                    else:
                        consecutive_misses += 1
                else:
                    consecutive_misses += 1

                # Handle missing detections with bridging
                if current_roi is None:
                    if last_valid_roi is not None and consecutive_misses <= BRIDGE_MAX:
                        # Bridge with last valid ROI
                        current_roi = last_valid_roi
                    # else: remains None, will be handled later

                frames.append(frame)
                rois.append(current_roi)
                frame_idx += 1

            cap.release()

            # Calculate detection statistics
            total_processed_frames = len(frames)
            detection_rate = detection_hits / max(total_processed_frames, 1)

            # Decide whether to keep, salvage, or drop
            keep_video = detection_rate >= PASS_MIN_FRAC
            salvage_video = (not keep_video) and (detection_rate >= SALVAGE_MIN_FRAC)

            if not (keep_video or salvage_video):
                self.logger.debug(f"Dropping {video_path.name}: detection rate {detection_rate:.3f} too low")
                self.stats['dropped_videos'] += 1
                return None

            # Fill remaining None ROIs with fallback strategy
            smoothed_rois = []
            previous_smoothed = None

            for i, (frame, roi) in enumerate(zip(frames, rois)):
                height, width = frame.shape[:2]

                if roi is None:
                    if previous_smoothed is not None:
                        # Use previous smoothed ROI
                        roi = previous_smoothed
                    else:
                        # Fallback: central lower third of frame (typical for cropped faces)
                        x1 = int(width * 0.25)
                        y1 = int(height * 0.5)
                        x2 = int(width * 0.75)
                        y2 = int(height * 0.95)
                        roi = (x1, y1, x2, y2)

                # Expand ROI for context
                expanded_roi = self.expand_and_clip_bbox(*roi, width, height)

                # Apply exponential moving average smoothing
                smoothed_roi = self.apply_exponential_moving_average(previous_smoothed, expanded_roi)
                previous_smoothed = smoothed_roi
                smoothed_rois.append(smoothed_roi)

            # Resample to target frame count
            frame_indices = self.generate_uniform_frame_indices(len(frames), TARGET_FRAMES)

            # Generate output filename
            output_filename = f"{label}__crop__{video_path.stem}.mp4"
            output_path = self.output_dir / output_filename

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                OUT_FPS,
                (OUT_SIZE, OUT_SIZE)
            )

            if not video_writer.isOpened():
                self.logger.error(f"Failed to create video writer for {output_path}")
                return None

            # Write cropped frames
            for idx in frame_indices:
                frame = frames[idx]
                x1, y1, x2, y2 = smoothed_rois[idx]

                # Extract crop
                crop = frame[y1:y2, x1:x2]

                # Safety check for empty crop
                if crop.size == 0:
                    crop = frame

                # Resize to target size
                crop = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)

                # Optional: Apply histogram equalization for better contrast
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                video_writer.write(crop)

            video_writer.release()

            # Update statistics
            if keep_video:
                self.stats['kept_videos'] += 1
            else:
                self.stats['salvaged_videos'] += 1

            self.stats['detection_rates'].append(detection_rate)

            # Return processing result
            result = {
                'path': str(output_path),
                'label': label,
                'frames': TARGET_FRAMES,
                'fps': OUT_FPS,
                'detect_frac': round(float(detection_rate), 3),
                'src': str(video_path),
                'status': 'kept' if keep_video else 'salvaged'
            }

            return result

        except Exception as e:
            self.logger.error(f"Error processing {video_path}: {str(e)}")
            return None

    def process_all_videos(self) -> List[Dict[str, Any]]:
        """
        Process all videos in the input directory.

        Returns:
            List of successfully processed video results
        """
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
        video_files = []

        for ext in video_extensions:
            video_files.extend(list(self.input_dir.glob(f'*{ext}')))
            video_files.extend(list(self.input_dir.glob(f'*{ext.upper()}')))

        self.logger.info(f"Found {len(video_files)} video files to process")
        self.stats['total_videos'] = len(video_files)

        # Process videos with progress bar
        results = []
        for video_path in tqdm(video_files, desc="Cropping mouth ROIs"):
            result = self.process_single_video(video_path)
            if result is not None:
                results.append(result)
                self.processed_videos.append(result)

        return results

    def generate_manifest(self, results: List[Dict[str, Any]]) -> None:
        """
        Generate manifest CSV file with processing results.

        Args:
            results: List of processing results
        """
        if not results:
            self.logger.warning("No results to write to manifest")
            return

        # Write manifest CSV
        fieldnames = ['path', 'label', 'frames', 'fps', 'detect_frac', 'src', 'status']

        with open(self.manifest_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        self.logger.info(f"Manifest written to {self.manifest_path}")

    def print_summary_statistics(self) -> None:
        """Print comprehensive summary of processing results."""
        total = self.stats['total_videos']
        kept = self.stats['kept_videos']
        salvaged = self.stats['salvaged_videos']
        dropped = self.stats['dropped_videos']

        print("\n" + "="*70)
        print("MOUTH CROPPING PIPELINE - SUMMARY STATISTICS")
        print("="*70)
        print(f"Total videos processed: {total}")
        print(f"Videos kept (≥{PASS_MIN_FRAC*100:.0f}% detection): {kept} ({kept/total*100:.1f}%)")
        print(f"Videos salvaged ({SALVAGE_MIN_FRAC*100:.0f}%-{PASS_MIN_FRAC*100:.0f}% detection): {salvaged} ({salvaged/total*100:.1f}%)")
        print(f"Videos dropped (<{SALVAGE_MIN_FRAC*100:.0f}% detection): {dropped} ({dropped/total*100:.1f}%)")
        print(f"Total usable videos: {kept + salvaged} ({(kept + salvaged)/total*100:.1f}%)")

        if self.stats['detection_rates']:
            detection_rates = self.stats['detection_rates']
            print(f"\nDetection rate statistics:")
            print(f"  Mean: {np.mean(detection_rates):.3f}")
            print(f"  Median: {np.median(detection_rates):.3f}")
            print(f"  Min: {np.min(detection_rates):.3f}")
            print(f"  Max: {np.max(detection_rates):.3f}")

        # Class distribution
        class_counts = {}
        for result in self.processed_videos:
            label = result['label']
            class_counts[label] = class_counts.get(label, 0) + 1

        if class_counts:
            print(f"\nClass distribution:")
            for label, count in sorted(class_counts.items()):
                print(f"  {label}: {count} videos")

        print(f"\nOutput directory: {self.output_dir}")
        print(f"Manifest file: {self.manifest_path}")
        print(f"Output format: {TARGET_FRAMES} frames, {OUT_SIZE}x{OUT_SIZE} pixels, {OUT_FPS} fps")
        print("="*70)

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete mouth cropping pipeline.

        Returns:
            Summary statistics dictionary
        """
        self.logger.info("Starting MediaPipe mouth cropping pipeline")

        # Process all videos
        results = self.process_all_videos()

        # Generate manifest
        self.generate_manifest(results)

        # Print summary
        self.print_summary_statistics()

        # Return summary for programmatic use
        summary = {
            'total_videos': self.stats['total_videos'],
            'kept_videos': self.stats['kept_videos'],
            'salvaged_videos': self.stats['salvaged_videos'],
            'dropped_videos': self.stats['dropped_videos'],
            'usable_videos': self.stats['kept_videos'] + self.stats['salvaged_videos'],
            'detection_rates': self.stats['detection_rates'].copy(),
            'output_dir': str(self.output_dir),
            'manifest_path': self.manifest_path
        }

        self.logger.info("Mouth cropping pipeline completed successfully!")
        return summary


def main():
    """Main CLI interface for the mouth cropping pipeline."""
    if len(sys.argv) < 3:
        print("Usage: python crop_mouth.py INPUT_DIR OUTPUT_DIR [MANIFEST_CSV]")
        print("\nExample:")
        print("  python crop_mouth.py data/grid/13.9.25top7dataset mouth_crops_96x96_32f manifest.csv")
        print("\nDescription:")
        print("  Creates training-ready mouth crops from lip-reading videos using MediaPipe.")
        print("  Handles partial detections with smoothing and bridging for robust ROI extraction.")
        print(f"  Output: {TARGET_FRAMES} frames, {OUT_SIZE}x{OUT_SIZE} pixels, {OUT_FPS} fps")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    manifest_path = sys.argv[3] if len(sys.argv) > 3 else "manifest.csv"

    # Validate input directory
    if not pathlib.Path(input_dir).exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)

    # Initialize and run cropper
    cropper = MouthCropper(input_dir, output_dir, manifest_path)
    summary = cropper.run_pipeline()

    # Exit with appropriate code
    if summary['usable_videos'] == 0:
        print("Warning: No usable videos were produced!")
        sys.exit(1)

    print(f"\nSuccess! {summary['usable_videos']} usable training videos created.")
    return summary


if __name__ == "__main__":
    main()
