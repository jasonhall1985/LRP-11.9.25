"""
MediaPipe-based lip ROI extraction for consistent lip reading preprocessing.
Provides robust lip detection and standardized ROI extraction for both training and inference.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import MediaPipe, fall back to simple face detection if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    logger.warning("MediaPipe not available, falling back to OpenCV face detection")
    MEDIAPIPE_AVAILABLE = False

class LipROIExtractor:
    """
    Robust lip ROI extraction using MediaPipe Face Mesh.
    Ensures consistent preprocessing for training and inference.
    """
    
    def __init__(self,
                 roi_size: Tuple[int, int] = (96, 96),
                 confidence_threshold: float = 0.5,
                 max_num_faces: int = 1):
        """
        Initialize the lip ROI extractor.

        Args:
            roi_size: Target size for extracted lip ROI (width, height)
            confidence_threshold: Minimum confidence for face detection
            max_num_faces: Maximum number of faces to detect
        """
        self.roi_size = roi_size
        self.confidence_threshold = confidence_threshold
        self.max_num_faces = max_num_faces

        if MEDIAPIPE_AVAILABLE:
            # Initialize MediaPipe Face Mesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=confidence_threshold,
                min_tracking_confidence=confidence_threshold
            )
        else:
            # Initialize OpenCV face detector as fallback
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Lip landmark indices for MediaPipe Face Mesh (468 landmarks)
        # Outer lip contour
        self.OUTER_LIP_INDICES = [
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267,
            269, 270, 267, 271, 272, 271, 272
        ]
        
        # Inner lip contour  
        self.INNER_LIP_INDICES = [
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
            310, 311, 312, 13, 82, 81, 80, 78
        ]
        
        # Combined lip landmarks for ROI calculation
        self.LIP_INDICES = list(set(self.OUTER_LIP_INDICES + self.INNER_LIP_INDICES))
        
    def extract_lip_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract lip landmarks from a single image.

        Args:
            image: Input image (BGR format)

        Returns:
            Lip landmarks as (N, 2) array or None if no face detected
        """
        if MEDIAPIPE_AVAILABLE:
            return self._extract_lip_landmarks_mediapipe(image)
        else:
            return self._extract_lip_landmarks_opencv(image)

    def _extract_lip_landmarks_mediapipe(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract lip landmarks using MediaPipe."""
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        # Get the first face (assuming single face)
        face_landmarks = results.multi_face_landmarks[0]

        # Extract lip landmarks
        h, w = image.shape[:2]
        lip_landmarks = []

        for idx in self.LIP_INDICES:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            lip_landmarks.append([x, y])

        return np.array(lip_landmarks)

    def _extract_lip_landmarks_opencv(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract lip region using OpenCV face detection (fallback)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

        # Return corner points of estimated lip region
        lip_landmarks = np.array([
            [lip_x_start, lip_y_start],
            [lip_x_end, lip_y_start],
            [lip_x_end, lip_y_end],
            [lip_x_start, lip_y_end]
        ])

        return lip_landmarks
    
    def calculate_lip_roi_bbox(self, landmarks: np.ndarray, 
                              margin_factor: float = 0.3) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box for lip ROI with margin.
        
        Args:
            landmarks: Lip landmarks as (N, 2) array
            margin_factor: Margin factor to expand the bounding box
            
        Returns:
            Bounding box as (x1, y1, x2, y2)
        """
        x_min, y_min = landmarks.min(axis=0)
        x_max, y_max = landmarks.max(axis=0)
        
        # Add margin
        width = x_max - x_min
        height = y_max - y_min
        
        margin_x = int(width * margin_factor)
        margin_y = int(height * margin_factor)
        
        x1 = max(0, x_min - margin_x)
        y1 = max(0, y_min - margin_y)
        x2 = x_max + margin_x
        y2 = y_max + margin_y
        
        return x1, y1, x2, y2
    
    def extract_lip_roi(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract and resize lip ROI from image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Extracted lip ROI resized to target size, or None if extraction fails
        """
        # Extract lip landmarks
        landmarks = self.extract_lip_landmarks(image)
        if landmarks is None:
            return None
            
        # Calculate ROI bounding box
        x1, y1, x2, y2 = self.calculate_lip_roi_bbox(landmarks)
        
        # Ensure bounding box is within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract ROI
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
            
        # Resize to target size
        roi_resized = cv2.resize(roi, self.roi_size, interpolation=cv2.INTER_AREA)
        
        return roi_resized
    
    def process_video_frames(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Process a sequence of video frames to extract lip ROIs.
        
        Args:
            frames: List of video frames (BGR format)
            
        Returns:
            Array of lip ROIs with shape (num_frames, height, width, channels)
            or None if processing fails
        """
        lip_rois = []
        failed_frames = 0
        
        for i, frame in enumerate(frames):
            roi = self.extract_lip_roi(frame)
            if roi is not None:
                lip_rois.append(roi)
            else:
                failed_frames += 1
                # Use previous frame if available, otherwise skip
                if lip_rois:
                    lip_rois.append(lip_rois[-1])  # Duplicate last successful frame
                else:
                    logger.warning(f"Failed to extract ROI from frame {i}")
                    
        if failed_frames > len(frames) * 0.5:  # More than 50% failed
            logger.error(f"Too many failed frames: {failed_frames}/{len(frames)}")
            return None
            
        if not lip_rois:
            return None
            
        return np.array(lip_rois)
    
    def standardize_sequence_length(self, frames: np.ndarray, 
                                  target_length: int = 24) -> np.ndarray:
        """
        Standardize video sequence to target length.
        
        Args:
            frames: Input frames with shape (num_frames, height, width, channels)
            target_length: Target number of frames
            
        Returns:
            Standardized sequence with exactly target_length frames
        """
        current_length = len(frames)
        
        if current_length == target_length:
            return frames
        elif current_length < target_length:
            # Pad with last frame
            padding_needed = target_length - current_length
            last_frame = frames[-1:].repeat(padding_needed, axis=0)
            return np.concatenate([frames, last_frame], axis=0)
        else:
            # Center crop to target length
            start_idx = (current_length - target_length) // 2
            return frames[start_idx:start_idx + target_length]
    
    def preprocess_for_model(self, frames: np.ndarray, 
                           grayscale: bool = True,
                           normalize: bool = True) -> np.ndarray:
        """
        Final preprocessing for model input.
        
        Args:
            frames: Input frames with shape (num_frames, height, width, channels)
            grayscale: Convert to grayscale
            normalize: Normalize pixel values to [0, 1]
            
        Returns:
            Preprocessed frames ready for model input
        """
        processed = frames.copy()
        
        if grayscale and processed.shape[-1] == 3:
            # Convert to grayscale
            processed = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                                for frame in processed])
            processed = np.expand_dims(processed, axis=-1)  # Add channel dimension
            
        if normalize:
            processed = processed.astype(np.float32) / 255.0
            
        return processed
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if MEDIAPIPE_AVAILABLE and hasattr(self, 'face_mesh'):
            self.face_mesh.close()


def load_video_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Load video frames from file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load
        
    Returns:
        List of video frames (BGR format)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frames.append(frame)
        frame_count += 1
        
        if max_frames and frame_count >= max_frames:
            break
            
    cap.release()
    return frames


if __name__ == "__main__":
    # Example usage
    extractor = LipROIExtractor()
    
    # Test with a sample video
    video_path = "sample_video.mp4"  # Replace with actual path
    frames = load_video_frames(video_path)
    
    if frames:
        # Extract lip ROIs
        lip_rois = extractor.process_video_frames(frames)
        
        if lip_rois is not None:
            # Standardize sequence length
            standardized = extractor.standardize_sequence_length(lip_rois, target_length=24)
            
            # Preprocess for model
            model_input = extractor.preprocess_for_model(standardized)
            
            print(f"Original frames: {len(frames)}")
            print(f"Extracted ROIs: {lip_rois.shape}")
            print(f"Standardized: {standardized.shape}")
            print(f"Model input: {model_input.shape}")
        else:
            print("Failed to extract lip ROIs")
    else:
        print("Failed to load video frames")
