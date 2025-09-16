#!/usr/bin/env python3
"""
Improved ROI Utils V3 - Aggressive Vertical Positioning Fix
===========================================================

V3 version with much more aggressive vertical positioning corrections based on
ICU dataset analysis. This version uses empirically-derived positioning that
centers the mouth region more accurately.

Key V3 improvements:
- Empirically-derived lip region parameters based on ICU dataset analysis
- Much more aggressive vertical centering (y_ratio=0.35 vs 0.25)
- Larger crop area (50% expansion vs 25%)
- Better fallback positioning for edge cases

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


class AdaptiveLipDetectorV3:
    """
    V3 adaptive lip detector with aggressive vertical positioning corrections.
    """
    
    def __init__(self, min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5,
                 auto_detect_mode: bool = True):
        """
        Initialize the V3 adaptive lip detector with aggressive positioning.
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
            
        # V3: AGGRESSIVE vertical positioning corrections based on ICU dataset analysis
        # Previous V2: (0.15, 0.25, 0.7, 0.5) - still too high
        # V3: Much more aggressive centering with empirically-derived parameters
        self.expected_lip_region = (0.1, 0.35, 0.8, 0.6)  # (x_ratio, y_ratio, w_ratio, h_ratio)
        
        # V3: More aggressive expansion and positioning adjustments
        self.vertical_offset_ratio = 0.1  # Larger vertical adjustment
        self.area_expansion_factor = 1.5  # 50% larger area (vs 25% in V2)
        
        # Mode detection statistics
        self.frame_count = 0
        self.mediapipe_success_count = 0
        self.mode_detection_frames = 10  # Frames to analyze for mode detection
        
        self.logger.info(f"V3 AdaptiveLipDetectorV3 initialized with AGGRESSIVE positioning")
        self.logger.info(f"V3 lip region: {self.expected_lip_region}")
        self.logger.info(f"V3 area expansion: {self.area_expansion_factor}x")
        self.logger.info(f"V3 vertical offset: {self.vertical_offset_ratio}")
        
    def detect_lip_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect lip landmarks using V3 aggressive positioning method.
        """
        if frame is None or frame.size == 0:
            return None
            
        # Auto-detect mode if enabled
        if self.auto_detect_mode and self.detection_mode is None:
            self._auto_detect_mode(frame)
            
        # Use appropriate detection method
        if self.detection_mode == 'cropped_face':
            return self._detect_cropped_face_landmarks_v3(frame)
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
                self.logger.info(f"V3 Auto-detected cropped face mode (MediaPipe success: {success_rate:.1%})")
            else:
                self.detection_mode = 'full_face'
                self.logger.info(f"V3 Auto-detected full face mode (MediaPipe success: {success_rate:.1%})")
                
    def _detect_cropped_face_landmarks_v3(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        V3 aggressive geometric lip detection with empirically-derived positioning.
        """
        h, w = frame.shape[:2]
        
        # V3: Use aggressive geometric estimation for cropped faces
        x_ratio, y_ratio, w_ratio, h_ratio = self.expected_lip_region
        
        # V3: Apply larger vertical offset adjustment for aggressive centering
        adjusted_y_ratio = y_ratio + self.vertical_offset_ratio
        
        # V3: Calculate base lip region with aggressive positioning
        base_lip_x1 = int(w * x_ratio)
        base_lip_y1 = int(h * adjusted_y_ratio)
        base_lip_x2 = int(w * (x_ratio + w_ratio))
        base_lip_y2 = int(h * (adjusted_y_ratio + h_ratio))
        
        # V3: Apply much larger area expansion (50% vs 25%)
        expansion = self.area_expansion_factor - 1.0  # 0.5 for 50% expansion
        
        # V3: Expand horizontally and vertically with larger margins
        h_expand = int((base_lip_x2 - base_lip_x1) * expansion / 2)
        v_expand = int((base_lip_y2 - base_lip_y1) * expansion / 2)
        
        lip_x1 = base_lip_x1 - h_expand
        lip_y1 = base_lip_y1 - v_expand
        lip_x2 = base_lip_x2 + h_expand
        lip_y2 = base_lip_y2 + v_expand
        
        # V3: Ensure bounds are valid with better edge handling
        lip_x1 = max(0, min(lip_x1, w-1))
        lip_y1 = max(0, min(lip_y1, h-1))
        lip_x2 = max(lip_x1+1, min(lip_x2, w))
        lip_y2 = max(lip_y1+1, min(lip_y2, h))
        
        # V3: Create enhanced synthetic landmarks with better coverage
        landmarks = []
        
        # V3: Create a denser grid of landmarks for better bounding box calculation
        for i in range(7):  # 7 rows (vs 5 in V2)
            for j in range(5):  # 5 columns (vs 3 in V2)
                x = lip_x1 + (lip_x2 - lip_x1) * j / 4
                y = lip_y1 + (lip_y2 - lip_y1) * i / 6
                landmarks.append([x, y])
                
        # V3: Add more corner and edge landmarks for precise bounding
        landmarks.extend([
            [lip_x1, lip_y1],  # Top-left
            [lip_x2, lip_y1],  # Top-right
            [lip_x1, lip_y2],  # Bottom-left
            [lip_x2, lip_y2],  # Bottom-right
            [(lip_x1 + lip_x2) / 2, (lip_y1 + lip_y2) / 2],  # Center
            # V3: Additional edge points for better coverage
            [(lip_x1 + lip_x2) / 2, lip_y1],  # Top-center
            [(lip_x1 + lip_x2) / 2, lip_y2],  # Bottom-center
            [lip_x1, (lip_y1 + lip_y2) / 2],  # Left-center
            [lip_x2, (lip_y1 + lip_y2) / 2],  # Right-center
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
        
    def create_debug_visualization_v3(self, frame: np.ndarray, landmarks: np.ndarray, 
                                     bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Create V3 debug visualization showing aggressive positioning improvements.
        """
        debug_frame = frame.copy()
        
        if landmarks is not None:
            # Draw landmarks with V3 color coding
            for i, landmark in enumerate(landmarks):
                x, y = int(landmark[0]), int(landmark[1])
                if i < 35:  # Main grid points
                    color = (0, 255, 0)  # Green
                else:  # Edge/corner points
                    color = (255, 0, 255)  # Magenta
                cv2.circle(debug_frame, (x, y), 2, color, -1)
                
        # Draw bounding box with V3 styling
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            
            # V3: Draw aggressive positioning indicators
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.line(debug_frame, (center_x - 15, center_y), (center_x + 15, center_y), (255, 0, 255), 3)
            cv2.line(debug_frame, (center_x, center_y - 15), (center_x, center_y + 15), (255, 0, 255), 3)
            
            # V3: Add enhanced info text
            cv2.putText(debug_frame, "V3: AGGRESSIVE POSITIONING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(debug_frame, f"Mode: {self.detection_mode}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # V3: Show aggressive parameters
            params_text = f"V3 Lip region: {self.expected_lip_region}"
            cv2.putText(debug_frame, params_text, (10, frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            expansion_text = f"V3 Expansion: {self.area_expansion_factor}x"
            cv2.putText(debug_frame, expansion_text, (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            offset_text = f"V3 Vertical offset: {self.vertical_offset_ratio}"
            cv2.putText(debug_frame, offset_text, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
        return debug_frame


# V3 Enhanced debug visualization function
def create_debug_visualization_v3(frame: np.ndarray, landmarks: Optional[np.ndarray], 
                                 bbox: Optional[Tuple[int, int, int, int]], 
                                 detector: Optional[AdaptiveLipDetectorV3] = None,
                                 ratios: Optional[dict] = None) -> np.ndarray:
    """
    Create V3 debug visualization with aggressive positioning indicators.
    """
    debug_frame = frame.copy()
    
    if landmarks is not None:
        # V3: Draw landmarks with enhanced visibility
        for i, landmark in enumerate(landmarks):
            x, y = int(landmark[0]), int(landmark[1])
            if i < 35:  # Main grid points
                color = (0, 255, 0)  # Green
                radius = 2
            else:  # Edge/corner points
                color = (255, 0, 255)  # Magenta
                radius = 3
            cv2.circle(debug_frame, (x, y), radius, color, -1)
            
    # V3: Draw bounding box with aggressive styling
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        # V3: Draw enhanced center cross for positioning reference
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(debug_frame, (center_x - 15, center_y), (center_x + 15, center_y), (255, 0, 255), 3)
        cv2.line(debug_frame, (center_x, center_y - 15), (center_x, center_y + 15), (255, 0, 255), 3)
        
        # V3: Add aggressive positioning info
        cv2.putText(debug_frame, "V3: AGGRESSIVE POSITIONING", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if detector:
            mode_text = f"Mode: {detector.get_detection_mode()}"
            cv2.putText(debug_frame, mode_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        # V3: Show ratios if available
        if ratios:
            ratio_text = f"Area: {ratios.get('area_ratio', 0):.3f}"
            cv2.putText(debug_frame, ratio_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
    return debug_frame
