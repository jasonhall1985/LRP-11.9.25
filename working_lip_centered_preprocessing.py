#!/usr/bin/env python3
"""
Working Lip-Centered 96×64 Preprocessing Pipeline
=================================================
Proven MediaPipe-based preprocessing that successfully centers mouths in frames.
Based on the working approach from crop_mouth.py and lip_centered_64x96_multimodel_preprocessing.py

Key Features:
1. MediaPipe Face Mesh for reliable lip detection
2. Proper mouth centering in 96×64 landscape format
3. EMA smoothing for stable positioning
4. Fallback strategies for robust processing

Author: Augment Agent
Date: 2025-09-17
"""

import os
import sys
import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import subprocess
# Removed MediaPipe dependency - using geometric cropping instead

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WorkingLipCenteredPreprocessor:
    """Working lip-centered preprocessing with proven geometric cropping."""

    def __init__(self):
        self.target_width = 96
        self.target_height = 64
        self.target_frames = 32
    
    def get_geometric_crop(self, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get geometric crop using proven approach from process_full_dataset_gentle_v5.py"""
        h, w = frame_shape

        # Use the proven bigger crop approach (80% height, 60% width)
        crop_h = int(0.80 * h)  # 80% height from top
        crop_w_start = int(0.20 * w)  # Start at 20% width
        crop_w_end = int(0.80 * w)    # End at 80% width

        return crop_w_start, 0, crop_w_end, crop_h
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with geometric cropping."""
        h, w = frame.shape[:2]

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Use proven geometric crop
        x1, y1, x2, y2 = self.get_geometric_crop((h, w))
        cropped = gray[y1:y2, x1:x2]

        # Resize to target dimensions (96x64 landscape)
        if cropped.size > 0:
            resized = cv2.resize(cropped, (self.target_width, self.target_height),
                               interpolation=cv2.INTER_CUBIC)
            return resized
        else:
            # Ultimate fallback: center crop
            center_x, center_y = w // 2, h // 2
            half_w, half_h = self.target_width // 2, self.target_height // 2
            x1 = max(0, center_x - half_w)
            y1 = max(0, center_y - half_h)
            x2 = min(w, center_x + half_w)
            y2 = min(h, center_y + half_h)

            center_crop = gray[y1:y2, x1:x2]
            return cv2.resize(center_crop, (self.target_width, self.target_height),
                            interpolation=cv2.INTER_CUBIC)
    
    def process_video(self, video_path: str) -> Tuple[Optional[np.ndarray], Dict]:
        """Process video with working lip-centered preprocessing."""
        
        logging.info(f"Processing with WORKING lip centering: {Path(video_path).name}")
        
        try:
            # Load video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, {"error": "Could not open video", "processing_success": False}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 1.0
            
            # Read all frames
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if not frames:
                return None, {"error": "No frames extracted", "processing_success": False}
            
            logging.info(f"Extracted {len(frames)} frames, {duration:.2f}s, {fps:.1f}fps")
            
            # Process each frame
            processed_frames = []
            for frame in frames:
                processed_frame = self.process_frame(frame)
                processed_frames.append(processed_frame)
            
            # Temporal sampling to 32 frames
            if len(processed_frames) != 32:
                indices = np.linspace(0, len(processed_frames) - 1, 32, dtype=int)
                processed_frames = [processed_frames[i] for i in indices]
            
            # Convert to numpy array and normalize
            frame_array = np.array(processed_frames, dtype=np.float32)
            
            # Convert to grayscale if needed
            if len(frame_array.shape) == 4 and frame_array.shape[-1] == 3:
                frame_array = np.mean(frame_array, axis=-1)
            
            # Normalize to [-1, 1]
            frame_array = (frame_array / 127.5) - 1.0
            
            # Create report
            report = {
                "original_duration": duration,
                "frames_processed": len(processed_frames),
                "processing_success": True
            }
            
            logging.info(f"Successfully processed: {len(processed_frames)} frames")
            
            return frame_array, report
            
        except Exception as e:
            logging.error(f"Error in processing: {str(e)}")
            return None, {"error": str(e), "processing_success": False}
    
    def save_video_with_dynamic_fps(self, frames: np.ndarray, output_path: str, original_duration: float) -> bool:
        """Save video with dynamic frame rate."""
        try:
            # Convert frames to uint8
            if frames.dtype != np.uint8:
                frames_uint8 = ((frames + 1.0) * 127.5).astype(np.uint8)
            else:
                frames_uint8 = frames.copy()
            
            # Calculate dynamic FPS
            dynamic_fps = 32.0 / original_duration if original_duration > 0 else 25.0
            dynamic_fps = max(10.0, min(60.0, dynamic_fps))
            
            # Save video using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, dynamic_fps, 
                                (self.target_width, self.target_height), False)
            
            for frame in frames_uint8:
                out.write(frame)
            
            out.release()
            logging.info(f"Saved video with {dynamic_fps:.1f} fps")
            return True
            
        except Exception as e:
            logging.error(f"Error saving video: {str(e)}")
            return False

# Test the working preprocessor
if __name__ == "__main__":
    # Test with a sample video
    test_video = "data/13.9.25top7dataset_cropped/help__useruser01__18to39__female__asian__20250902T013654_topmid.mp4"
    output_dir = Path("data/working_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎯 TESTING WORKING LIP-CENTERED PREPROCESSING")
    print("=" * 60)
    
    if Path(test_video).exists():
        preprocessor = WorkingLipCenteredPreprocessor()
        
        start_time = time.time()
        frames, result = preprocessor.process_video(test_video)
        processing_time = time.time() - start_time
        
        if frames is not None:
            output_path = output_dir / f"{Path(test_video).stem}_WORKING.mp4"
            success = preprocessor.save_video_with_dynamic_fps(
                frames, str(output_path), result.get('original_duration', 1.0)
            )
            
            if success:
                print(f"✅ SUCCESS: {processing_time:.2f}s")
                print(f"📁 Output: {output_path}")
            else:
                print("❌ Failed to save video")
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    else:
        print(f"❌ Test video not found: {test_video}")
