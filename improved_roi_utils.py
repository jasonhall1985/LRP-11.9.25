#!/usr/bin/env python3
"""
Improved ROI Utilities for Mouth ROI Pipeline
=============================================

Enhanced MediaPipe Face Mesh and geometry helpers that work with both
full faces and cropped face videos (like the ICU dataset).

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
OUTER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
INNER_LIP_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
ALL_LIP_INDICES = OUTER_LIP_INDICES + INNER_LIP_INDICES


class AdaptiveLipDetector:
    """
    Adaptive lip detector that automatically chooses the best detection method
    based on the input video characteristics.
    """

    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 auto_detect_mode: bool = True):
        """
        Initialize adaptive lip detector.

        Args:
            static_image_mode: Whether to treat input as static images
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks around lips/eyes
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            auto_detect_mode: Whether to automatically detect cropped vs full face videos
        """
        self.logger = logging.getLogger(__name__)
        self.auto_detect_mode = auto_detect_mode
        self.detection_mode = None  # Will be set automatically
        
        # MediaPipe setup
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
            
        # Cropped face detection parameters (for ICU dataset)
        self.expected_lip_region = (0.2, 0.1, 0.6, 0.4)  # (x_ratio, y_ratio, w_ratio, h_ratio)
        
        # Mode detection statistics
        self.frame_count = 0
        self.mediapipe_success_count = 0
        self.mode_detection_frames = 10  # Frames to analyze for mode detection
        
    def detect_lip_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect lip landmarks using the most appropriate method.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Lip landmarks as (N, 2) array or None if no detection
        """
        # Auto-detect mode on first few frames
        if self.auto_detect_mode and self.detection_mode is None:
            self._auto_detect_mode(frame)
            
        # Use appropriate detection method
        if self.detection_mode == 'cropped_face':
            return self._detect_cropped_face_landmarks(frame)
        elif self.use_mediapipe:
            return self._detect_with_mediapipe(frame)
        else:
            return self._detect_with_opencv(frame)
            
    def _auto_detect_mode(self, frame: np.ndarray):
        """
        Automatically detect whether this is a cropped face or full face video.
        """
        self.frame_count += 1
        
        # Try MediaPipe detection
        if self.use_mediapipe:
            landmarks = self._detect_with_mediapipe(frame)
            if landmarks is not None:
                self.mediapipe_success_count += 1
                
        # Make decision after analyzing several frames
        if self.frame_count >= self.mode_detection_frames:
            success_rate = self.mediapipe_success_count / self.frame_count
            
            if success_rate < 0.3:  # Less than 30% success with MediaPipe
                self.detection_mode = 'cropped_face'
                self.logger.info(f"Auto-detected cropped face mode (MediaPipe success: {success_rate:.1%})")
            else:
                self.detection_mode = 'full_face'
                self.logger.info(f"Auto-detected full face mode (MediaPipe success: {success_rate:.1%})")
                
    def _detect_cropped_face_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect lip landmarks in cropped face videos using geometric estimation.
        """
        h, w = frame.shape[:2]
        
        # Use geometric estimation for cropped faces
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
        
        # Create synthetic landmarks around the estimated lip region
        # This creates a rectangular pattern of landmarks for compatibility
        landmarks = []
        
        # Top edge
        for i in range(5):
            x = lip_x1 + (lip_x2 - lip_x1) * i / 4
            landmarks.append([x, lip_y1])
            
        # Right edge
        for i in range(1, 4):
            y = lip_y1 + (lip_y2 - lip_y1) * i / 3
            landmarks.append([lip_x2, y])
            
        # Bottom edge
        for i in range(4, -1, -1):
            x = lip_x1 + (lip_x2 - lip_x1) * i / 4
            landmarks.append([x, lip_y2])
            
        # Left edge
        for i in range(2, 0, -1):
            y = lip_y1 + (lip_y2 - lip_y1) * i / 3
            landmarks.append([lip_x1, y])
            
        return np.array(landmarks, dtype=np.int32)
        
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
        """Detect landmarks using OpenCV face detection fallback."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
            
        # Use largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = face
        
        # Estimate lip region within face
        lip_x = fx + int(fw * 0.25)
        lip_y = fy + int(fh * 0.65)
        lip_w = int(fw * 0.5)
        lip_h = int(fh * 0.25)
        
        # Create synthetic landmarks
        landmarks = []
        for i in range(8):
            angle = 2 * np.pi * i / 8
            x = lip_x + lip_w//2 + int((lip_w//2) * np.cos(angle))
            y = lip_y + lip_h//2 + int((lip_h//2) * np.sin(angle))
            landmarks.append([x, y])
            
        return np.array(landmarks)
        
    def get_detection_mode(self) -> Optional[str]:
        """Get the current detection mode."""
        return self.detection_mode
        
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_mesh') and self.use_mediapipe:
            self.face_mesh.close()


# Import other classes from original roi_utils.py
from roi_utils import ROIGeometry, BBoxSmoother, RecropCalculator, create_debug_visualization
