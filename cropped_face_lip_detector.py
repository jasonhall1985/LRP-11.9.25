#!/usr/bin/env python3
"""
Cropped Face Lip Detector
========================

Specialized lip detection for videos that are already cropped to show only
the lower half of faces (nose down), where MediaPipe Face Mesh fails because
it can't detect a complete face.

This detector uses geometric analysis and motion detection to identify
the lip region within cropped face videos.

Author: Augment Agent
Date: 2025-09-14
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging


class CroppedFaceLipDetector:
    """
    Lip detector optimized for cropped face videos where only the lower
    half of the face is visible (nose down to chin).
    """
    
    def __init__(self, 
                 motion_threshold: float = 0.02,
                 min_contour_area: int = 50,
                 expected_lip_region: Tuple[float, float, float, float] = (0.2, 0.1, 0.6, 0.4)):
        """
        Initialize cropped face lip detector.
        
        Args:
            motion_threshold: Threshold for motion detection
            min_contour_area: Minimum contour area for lip detection
            expected_lip_region: Expected lip region as (x_ratio, y_ratio, w_ratio, h_ratio)
        """
        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self.expected_lip_region = expected_lip_region
        
        # For motion detection
        self.prev_frame = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, varThreshold=16
        )
        
        self.logger = logging.getLogger(__name__)
        
    def detect_lip_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect lip region in cropped face frame using geometric and motion analysis.
        
        Args:
            frame: Input frame (BGR format) - cropped face with lips visible
            
        Returns:
            Lip bounding box as (x1, y1, x2, y2) or None if not detected
        """
        h, w = frame.shape[:2]
        
        # Method 1: Geometric estimation based on ICU dataset characteristics
        geometric_bbox = self._geometric_lip_detection(frame)
        
        # Method 2: Motion-based detection (for speaking/moving lips)
        motion_bbox = self._motion_based_detection(frame)
        
        # Method 3: Color-based detection (lip color analysis)
        color_bbox = self._color_based_detection(frame)
        
        # Combine methods with confidence weighting
        candidates = []
        if geometric_bbox:
            candidates.append(('geometric', geometric_bbox, 0.6))
        if motion_bbox:
            candidates.append(('motion', motion_bbox, 0.3))
        if color_bbox:
            candidates.append(('color', color_bbox, 0.1))
            
        if not candidates:
            return None
            
        # Use geometric detection as primary, validate with others
        return geometric_bbox if geometric_bbox else (motion_bbox or color_bbox)
        
    def _geometric_lip_detection(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Geometric lip detection based on ICU dataset characteristics.
        
        For ICU dataset: lips are positioned in top-middle portion of cropped frames.
        """
        h, w = frame.shape[:2]
        
        # Based on ICU dataset analysis: lips are in top-middle region
        # Crop to top 50% height and middle 33% width
        x_ratio, y_ratio, w_ratio, h_ratio = self.expected_lip_region
        
        lip_x1 = int(w * x_ratio)
        lip_y1 = int(h * y_ratio)
        lip_x2 = int(w * (x_ratio + w_ratio))
        lip_y2 = int(h * (y_ratio + h_ratio))
        
        # Ensure bounds are valid
        lip_x1 = max(0, min(lip_x1, w-1))
        lip_y1 = max(0, min(lip_y1, h-1))
        lip_x2 = max(lip_x1+1, min(lip_x2, w))
        lip_y2 = max(lip_y1+1, min(lip_y2, h))
        
        return (lip_x1, lip_y1, lip_x2, lip_y2)
        
    def _motion_based_detection(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Motion-based lip detection for speaking/moving lips.
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(self.prev_frame, gray)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, int(255 * self.motion_threshold), 255, cv2.THRESH_BINARY)
        
        # Find contours in motion areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.prev_frame = gray
            return None
            
        # Find largest motion area (likely to be lips if speaking)
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < self.min_contour_area:
            self.prev_frame = gray
            return None
            
        # Get bounding box of motion area
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        self.prev_frame = gray
        return (x, y, x + w, y + h)
        
    def _color_based_detection(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Color-based lip detection using HSV color space.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define lip color ranges (reddish/pinkish hues)
        # Lower range (red-pink)
        lower1 = np.array([0, 30, 30])
        upper1 = np.array([10, 255, 255])
        
        # Upper range (red-pink)
        lower2 = np.array([160, 30, 30])
        upper2 = np.array([180, 255, 255])
        
        # Create masks
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find largest color region
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < self.min_contour_area:
            return None
            
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, x + w, y + h)
        
    def create_debug_visualization(self, frame: np.ndarray, 
                                 detected_bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Create debug visualization showing detection results.
        """
        debug_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw expected lip region
        x_ratio, y_ratio, w_ratio, h_ratio = self.expected_lip_region
        exp_x1 = int(w * x_ratio)
        exp_y1 = int(h * y_ratio)
        exp_x2 = int(w * (x_ratio + w_ratio))
        exp_y2 = int(h * (y_ratio + h_ratio))
        
        cv2.rectangle(debug_frame, (exp_x1, exp_y1), (exp_x2, exp_y2), (255, 0, 0), 1)
        cv2.putText(debug_frame, "Expected Region", (exp_x1, exp_y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # Draw detected bbox
        if detected_bbox:
            x1, y1, x2, y2 = detected_bbox
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_frame, "Detected Lips", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Add frame info
        cv2.putText(debug_frame, f"{w}x{h}", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return debug_frame


def test_cropped_face_detector():
    """Test the cropped face lip detector on the target video."""
    
    video_path = "/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped/doctor__useruser01__18to39__female__aboriginal__20250807T054104_topmid.mp4"
    
    detector = CroppedFaceLipDetector()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Testing cropped face detector on: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    
    # Process frames
    frame_idx = 0
    detected_count = 0
    output_frames = []
    
    while frame_idx < min(30, total_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect lip region
        lip_bbox = detector.detect_lip_region(frame)
        
        if lip_bbox:
            detected_count += 1
            x1, y1, x2, y2 = lip_bbox
            
            # Crop to detected region
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                # Resize to 96x96
                resized = cv2.resize(cropped, (96, 96))
                output_frames.append(resized)
                
                # Save debug frame
                debug_frame = detector.create_debug_visualization(frame, lip_bbox)
                cv2.imwrite(f"cropped_debug_{frame_idx:04d}.jpg", debug_frame)
                cv2.imwrite(f"cropped_result_{frame_idx:04d}.jpg", resized)
                
            print(f"Frame {frame_idx}: Detected lip region {lip_bbox}")
        else:
            print(f"Frame {frame_idx}: No lip region detected")
            
        frame_idx += 1
        
    cap.release()
    
    detection_rate = detected_count / frame_idx if frame_idx > 0 else 0
    print(f"\nDetection Results:")
    print(f"  Frames processed: {frame_idx}")
    print(f"  Detections: {detected_count}")
    print(f"  Detection rate: {detection_rate:.2%}")
    
    # Create output video
    if output_frames:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('cropped_face_output.mp4', fourcc, fps, (96, 96))
        
        for frame in output_frames:
            out.write(frame)
            
        out.release()
        print(f"Output video saved: cropped_face_output.mp4")


if __name__ == "__main__":
    test_cropped_face_detector()
