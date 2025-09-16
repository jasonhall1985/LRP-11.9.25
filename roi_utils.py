#!/usr/bin/env python3
"""
ROI Utilities for Mouth ROI Pipeline
====================================

MediaPipe Face Mesh and geometry helpers for mouth ROI extraction and standardization.
Includes landmark detection, bounding box calculation, smoothing, and size analysis.

Author: Augment Agent
Date: 2025-09-14
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging

# Try to import MediaPipe, fall back to OpenCV if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Falling back to OpenCV face detection.")
    print("To install MediaPipe: pip install mediapipe")

# MediaPipe Face Mesh lip landmark indices
# Outer lip contour (key points for robust detection)
OUTER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Inner lip contour (for detailed mouth region)
INNER_LIP_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Combined lip indices for ROI calculation
ALL_LIP_INDICES = OUTER_LIP_INDICES + INNER_LIP_INDICES


class MediaPipeLipDetector:
    """
    MediaPipe Face Mesh-based lip landmark detector optimized for mouth ROI extraction.
    Falls back to OpenCV face detection if MediaPipe is not available.
    """

    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Face Mesh detector or OpenCV fallback.

        Args:
            static_image_mode: Whether to treat input as static images
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks around lips/eyes
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
        """
        self.logger = logging.getLogger(__name__)

        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.use_mediapipe = True
        else:
            # Fallback to OpenCV face detection
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.use_mediapipe = False
            self.logger.warning("Using OpenCV fallback for face detection")
    
    def detect_lip_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect lip landmarks in a frame using MediaPipe Face Mesh or OpenCV fallback.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Lip landmarks as (N, 2) array or None if no face detected
        """
        if self.use_mediapipe:
            return self._detect_with_mediapipe(frame)
        else:
            return self._detect_with_opencv(frame)

    def _detect_with_mediapipe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect landmarks using MediaPipe Face Mesh."""
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]

        # Extract lip landmarks
        h, w = frame.shape[:2]
        lip_landmarks = []

        for idx in ALL_LIP_INDICES:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            lip_landmarks.append([x, y])

        return np.array(lip_landmarks)

    def _detect_with_opencv(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect face region using OpenCV and estimate lip landmarks."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return None

        # Use the first detected face
        x, y, w, h = faces[0]

        # Estimate lip region (lower third of face)
        lip_y_start = y + int(h * 0.6)
        lip_y_end = y + int(h * 0.9)
        lip_x_start = x + int(w * 0.2)
        lip_x_end = x + int(w * 0.8)

        # Create approximate lip landmarks (rectangular region)
        lip_landmarks = np.array([
            [lip_x_start, lip_y_start],
            [lip_x_end, lip_y_start],
            [lip_x_end, lip_y_end],
            [lip_x_start, lip_y_end],
            # Add center points for better bbox calculation
            [(lip_x_start + lip_x_end) // 2, lip_y_start],
            [(lip_x_start + lip_x_end) // 2, lip_y_end],
            [lip_x_start, (lip_y_start + lip_y_end) // 2],
            [lip_x_end, (lip_y_start + lip_y_end) // 2]
        ])

        return lip_landmarks
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_mesh') and self.use_mediapipe:
            self.face_mesh.close()


class ROIGeometry:
    """
    Geometry utilities for mouth ROI calculation and manipulation.
    """
    
    @staticmethod
    def calculate_tight_bbox(landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calculate tight bounding box around landmarks.
        
        Args:
            landmarks: Landmark points as (N, 2) array
            
        Returns:
            Bounding box as (x1, y1, x2, y2)
        """
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
        x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
        
        return x1, y1, x2, y2
    
    @staticmethod
    def add_padding(bbox: Tuple[int, int, int, int], 
                   padding_ratio: float,
                   frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Add padding to bounding box.
        
        Args:
            bbox: Original bounding box (x1, y1, x2, y2)
            padding_ratio: Padding ratio (e.g., 0.12 for 12% padding)
            frame_shape: Frame shape (height, width)
            
        Returns:
            Padded bounding box (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bbox
        h, w = frame_shape
        
        # Calculate padding
        bbox_w, bbox_h = x2 - x1, y2 - y1
        pad_w = int(bbox_w * padding_ratio)
        pad_h = int(bbox_h * padding_ratio)
        
        # Apply padding with bounds checking
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(w, x2 + pad_w)
        y2_pad = min(h, y2 + pad_h)
        
        return x1_pad, y1_pad, x2_pad, y2_pad
    
    @staticmethod
    def calculate_size_ratios(bbox: Tuple[int, int, int, int],
                            frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Calculate size ratios for ROI analysis.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            frame_shape: Frame shape (height, width)
            
        Returns:
            Dictionary with area_ratio, h_ratio, w_ratio
        """
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame_shape
        
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        bbox_area = bbox_w * bbox_h
        frame_area = frame_w * frame_h
        
        return {
            'area_ratio': bbox_area / frame_area if frame_area > 0 else 0.0,
            'h_ratio': bbox_h / frame_h if frame_h > 0 else 0.0,
            'w_ratio': bbox_w / frame_w if frame_w > 0 else 0.0
        }


class BBoxSmoother:
    """
    Exponential Moving Average (EMA) smoother for bounding box coordinates.
    """
    
    def __init__(self, alpha: float = 0.6):
        """
        Initialize bbox smoother.
        
        Args:
            alpha: EMA smoothing factor (0 < alpha <= 1)
        """
        self.alpha = alpha
        self.smoothed_bbox = None
    
    def smooth(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Apply EMA smoothing to bounding box.
        
        Args:
            bbox: Current bounding box (x1, y1, x2, y2)
            
        Returns:
            Smoothed bounding box (x1, y1, x2, y2)
        """
        if self.smoothed_bbox is None:
            # Initialize with first bbox
            self.smoothed_bbox = bbox
            return bbox
        
        # Apply EMA smoothing
        x1, y1, x2, y2 = bbox
        sx1, sy1, sx2, sy2 = self.smoothed_bbox
        
        smoothed = (
            int(self.alpha * x1 + (1 - self.alpha) * sx1),
            int(self.alpha * y1 + (1 - self.alpha) * sy1),
            int(self.alpha * x2 + (1 - self.alpha) * sx2),
            int(self.alpha * y2 + (1 - self.alpha) * sy2)
        )
        
        self.smoothed_bbox = smoothed
        return smoothed
    
    def reset(self):
        """Reset smoother state."""
        self.smoothed_bbox = None

    def get_last_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the last smoothed bounding box."""
        return self.smoothed_bbox


class RecropCalculator:
    """
    Calculator for determining optimal recrop parameters for small mouth ROIs.
    """
    
    @staticmethod
    def calculate_recrop_window(bbox: Tuple[int, int, int, int],
                              target_h_ratio: float,
                              target_w_ratio: float,
                              frame_shape: Tuple[int, int],
                              safety_margin: float = 0.05) -> Tuple[int, int, int, int]:
        """
        Calculate optimal recrop window to achieve target ratios.
        
        Args:
            bbox: Current mouth bounding box (x1, y1, x2, y2)
            target_h_ratio: Target height ratio (e.g., 0.50)
            target_w_ratio: Target width ratio (e.g., 0.50)
            frame_shape: Frame shape (height, width)
            safety_margin: Additional safety margin (e.g., 0.05 for 5%)
            
        Returns:
            Recrop window (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame_shape
        
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        # Calculate desired crop dimensions
        desired_w = bbox_w / target_w_ratio
        desired_h = bbox_h / target_h_ratio
        
        # Use the larger dimension to maintain aspect ratio
        crop_size = max(desired_w, desired_h)
        
        # Add safety margin
        crop_size *= (1 + safety_margin)
        
        # Calculate crop window centered on bbox
        bbox_center_x = (x1 + x2) // 2
        bbox_center_y = (y1 + y2) // 2
        
        half_crop = int(crop_size // 2)
        
        crop_x1 = max(0, bbox_center_x - half_crop)
        crop_y1 = max(0, bbox_center_y - half_crop)
        crop_x2 = min(frame_w, bbox_center_x + half_crop)
        crop_y2 = min(frame_h, bbox_center_y + half_crop)
        
        return crop_x1, crop_y1, crop_x2, crop_y2


def create_debug_visualization(frame: np.ndarray,
                             landmarks: Optional[np.ndarray],
                             smoothed_bbox: Optional[Tuple[int, int, int, int]],
                             crop_window: Optional[Tuple[int, int, int, int]],
                             ratios: Optional[Dict[str, float]]) -> np.ndarray:
    """
    Create debug visualization with landmarks, bboxes, and ratio information.
    
    Args:
        frame: Input frame
        landmarks: Detected landmarks
        smoothed_bbox: Smoothed bounding box
        crop_window: Final crop window
        ratios: Size ratios
        
    Returns:
        Annotated frame
    """
    debug_frame = frame.copy()
    
    # Draw landmarks
    if landmarks is not None:
        for point in landmarks:
            cv2.circle(debug_frame, tuple(point), 2, (0, 255, 0), -1)
    
    # Draw smoothed bbox (green)
    if smoothed_bbox is not None:
        x1, y1, x2, y2 = smoothed_bbox
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug_frame, "Smoothed ROI", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw crop window (yellow)
    if crop_window is not None:
        x1, y1, x2, y2 = crop_window
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(debug_frame, "Crop Window", (x1, y1-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Add ratio text
    if ratios is not None:
        text_y = 30
        for key, value in ratios.items():
            text = f"{key}: {value:.3f}"
            cv2.putText(debug_frame, text, (10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            text_y += 25
    
    return debug_frame
