#!/usr/bin/env python3
"""
Improved ROI Utils V2 - Fixed Vertical Positioning
==================================================

Enhanced version of the adaptive lip detector with corrected vertical positioning
for better mouth ROI centering in cropped face videos.

Key improvements:
- Adjusted geometric detection parameters for better vertical centering
- Increased crop area by 25% for more tolerance
- Added vertical offset adjustments
- Enhanced debug visualization

Author: Augment Agent
Date: 2025-09-14
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Optional, Tuple, List
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveLipDetectorV2:
    """
    Enhanced adaptive lip detector with improved vertical positioning for cropped face videos.
    """
    
    def __init__(self, min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5,
                 auto_detect_mode: bool = True):
        """
        Initialize the enhanced adaptive lip detector.
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.auto_detect_mode = auto_detect_mode
        self.detection_mode = None  # Will be 'cropped_face' or 'full_face'
        self.logger = logger
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.use_mediapipe = True
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe: {e}")
            # Fallback to OpenCV
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.use_mediapipe = False
                self.logger.warning("Using OpenCV fallback for face detection")
            except Exception as e2:
                self.logger.error(f"Failed to initialize OpenCV cascade: {e2}")
                raise RuntimeError("Could not initialize any face detection method")
            
        # IMPROVED: Enhanced cropped face detection parameters for better vertical positioning
        # Previous: (0.2, 0.1, 0.6, 0.4) - too high and narrow
        # New: Better centered with 25% larger area for tolerance
        self.expected_lip_region = (0.15, 0.25, 0.7, 0.5)  # (x_ratio, y_ratio, w_ratio, h_ratio)
        
        # Additional vertical adjustment parameters
        self.vertical_offset_ratio = 0.05  # Fine-tune vertical positioning
        self.area_expansion_factor = 1.25  # 25% larger area for tolerance
        
        # Mode detection statistics
        self.frame_count = 0
        self.mediapipe_success_count = 0
        self.mode_detection_frames = 10  # Frames to analyze for mode detection
        
        self.logger.info(f"Enhanced AdaptiveLipDetectorV2 initialized")
        self.logger.info(f"Improved lip region: {self.expected_lip_region}")
        self.logger.info(f"Area expansion: {self.area_expansion_factor}x")
        
    def detect_lip_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect lip landmarks using the most appropriate method with improved positioning.
        """
        if frame is None or frame.size == 0:
            return None
            
        # Auto-detect mode if enabled
        if self.auto_detect_mode and self.detection_mode is None:
            self._auto_detect_mode(frame)
            
        # Use appropriate detection method
        if self.detection_mode == 'cropped_face':
            return self._detect_cropped_face_landmarks_v2(frame)
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
            if success_rate < 0.3:  # Low success rate indicates cropped faces
                self.detection_mode = 'cropped_face'
                self.logger.info(f"Auto-detected cropped face mode (MediaPipe success: {success_rate:.1%})")
            else:
                self.detection_mode = 'full_face'
                self.logger.info(f"Auto-detected full face mode (MediaPipe success: {success_rate:.1%})")
                
    def _detect_cropped_face_landmarks_v2(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Enhanced geometric lip detection with improved vertical positioning.
        """
        h, w = frame.shape[:2]
        
        # Use improved geometric estimation for cropped faces
        x_ratio, y_ratio, w_ratio, h_ratio = self.expected_lip_region
        
        # Apply vertical offset adjustment for fine-tuning
        adjusted_y_ratio = y_ratio + self.vertical_offset_ratio
        
        # Calculate base lip region
        base_lip_x1 = int(w * x_ratio)
        base_lip_y1 = int(h * adjusted_y_ratio)
        base_lip_x2 = int(w * (x_ratio + w_ratio))
        base_lip_y2 = int(h * (adjusted_y_ratio + h_ratio))
        
        # Apply area expansion for better tolerance
        expansion = self.area_expansion_factor - 1.0  # 0.25 for 25% expansion
        
        # Expand horizontally and vertically
        h_expand = int((base_lip_x2 - base_lip_x1) * expansion / 2)
        v_expand = int((base_lip_y2 - base_lip_y1) * expansion / 2)
        
        lip_x1 = base_lip_x1 - h_expand
        lip_y1 = base_lip_y1 - v_expand
        lip_x2 = base_lip_x2 + h_expand
        lip_y2 = base_lip_y2 + v_expand
        
        # Ensure bounds are valid
        lip_x1 = max(0, min(lip_x1, w-1))
        lip_y1 = max(0, min(lip_y1, h-1))
        lip_x2 = max(lip_x1+1, min(lip_x2, w))
        lip_y2 = max(lip_y1+1, min(lip_y2, h))
        
        # Create enhanced synthetic landmarks around the estimated lip region
        # Generate more landmarks for better bounding box calculation
        landmarks = []
        
        # Create a grid of landmarks within the lip region for better coverage
        for i in range(5):  # 5 rows
            for j in range(3):  # 3 columns
                x = lip_x1 + (lip_x2 - lip_x1) * j / 2
                y = lip_y1 + (lip_y2 - lip_y1) * i / 4
                landmarks.append([x, y])
                
        # Add corner landmarks for precise bounding
        landmarks.extend([
            [lip_x1, lip_y1],  # Top-left
            [lip_x2, lip_y1],  # Top-right
            [lip_x1, lip_y2],  # Bottom-left
            [lip_x2, lip_y2],  # Bottom-right
            [(lip_x1 + lip_x2) / 2, (lip_y1 + lip_y2) / 2]  # Center
        ])
        
        return np.array(landmarks, dtype=np.float32)
        
    def _detect_with_mediapipe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect lip landmarks using MediaPipe Face Mesh.
        """
        if not self.use_mediapipe:
            return None
            
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Extract lip landmarks (MediaPipe indices for lips)
                lip_indices = [
                    61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
                    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
                ]
                
                h, w = frame.shape[:2]
                lip_landmarks = []
                
                for idx in lip_indices:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        lip_landmarks.append([x, y])
                
                if lip_landmarks:
                    return np.array(lip_landmarks, dtype=np.float32)
                    
        except Exception as e:
            self.logger.debug(f"MediaPipe detection failed: {e}")
            
        return None
        
    def _detect_with_opencv(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback detection using OpenCV Haar cascades.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Use the largest face
                face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = face
                
                # Estimate lip region within the face
                lip_y = y + int(h * 0.6)  # Lips are typically in lower part of face
                lip_h = int(h * 0.3)
                lip_x = x + int(w * 0.2)
                lip_w = int(w * 0.6)
                
                # Create synthetic landmarks
                landmarks = []
                for i in range(3):
                    for j in range(5):
                        lx = lip_x + (lip_w * j / 4)
                        ly = lip_y + (lip_h * i / 2)
                        landmarks.append([lx, ly])
                        
                return np.array(landmarks, dtype=np.float32)
                
        except Exception as e:
            self.logger.debug(f"OpenCV detection failed: {e}")
            
        return None
        
    def get_detection_mode(self) -> Optional[str]:
        """
        Get the current detection mode.
        """
        return self.detection_mode
        
    def create_debug_visualization_v2(self, frame: np.ndarray, landmarks: np.ndarray, 
                                     bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Create enhanced debug visualization showing improved positioning.
        """
        debug_frame = frame.copy()
        
        if landmarks is not None:
            # Draw landmarks
            for landmark in landmarks:
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(debug_frame, (x, y), 2, (0, 255, 0), -1)
                
        # Draw bounding box
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Add positioning info
            cv2.putText(debug_frame, f"V2: {self.detection_mode}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(debug_frame, f"BBox: {bbox}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Show lip region parameters
            params_text = f"Lip region: {self.expected_lip_region}"
            cv2.putText(debug_frame, params_text, (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            expansion_text = f"Expansion: {self.area_expansion_factor}x"
            cv2.putText(debug_frame, expansion_text, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
        return debug_frame


# Enhanced debug visualization function
def create_debug_visualization_v2(frame: np.ndarray, landmarks: Optional[np.ndarray], 
                                 bbox: Optional[Tuple[int, int, int, int]], 
                                 detector: Optional[AdaptiveLipDetectorV2] = None,
                                 ratios: Optional[dict] = None) -> np.ndarray:
    """
    Create enhanced debug visualization with improved positioning indicators.
    """
    debug_frame = frame.copy()
    
    if landmarks is not None:
        # Draw landmarks with different colors for better visibility
        for i, landmark in enumerate(landmarks):
            x, y = int(landmark[0]), int(landmark[1])
            color = (0, 255, 0) if i < 15 else (255, 0, 0)  # Green for main, red for corners
            cv2.circle(debug_frame, (x, y), 2, color, -1)
            
    # Draw bounding box with enhanced styling
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw center cross for positioning reference
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(debug_frame, (center_x - 10, center_y), (center_x + 10, center_y), (255, 0, 255), 2)
        cv2.line(debug_frame, (center_x, center_y - 10), (center_x, center_y + 10), (255, 0, 255), 2)
        
        # Add enhanced info text
        cv2.putText(debug_frame, "V2: Enhanced Positioning", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if detector:
            mode_text = f"Mode: {detector.get_detection_mode()}"
            cv2.putText(debug_frame, mode_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        # Show ratios if available
        if ratios:
            ratio_text = f"Area: {ratios.get('area_ratio', 0):.3f}"
            cv2.putText(debug_frame, ratio_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
    return debug_frame
