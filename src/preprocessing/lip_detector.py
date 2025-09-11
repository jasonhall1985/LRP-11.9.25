"""
Lip Detection and Preprocessing Module

This module uses MediaPipe Face Mesh to detect and extract lip regions from video frames.
It provides functionality to convert videos to image sequences and preprocess them for ML training.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional
import os
from PIL import Image


class LipDetector:
    """
    A class for detecting and extracting lip regions from video frames using MediaPipe.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (64, 64)):
        """
        Initialize the LipDetector.
        
        Args:
            target_size: Target size for lip region crops (width, height)
        """
        self.target_size = target_size
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Lip landmark indices for MediaPipe Face Mesh
        self.UPPER_LIP_INDICES = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        self.LOWER_LIP_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
        self.LIP_INDICES = self.UPPER_LIP_INDICES + self.LOWER_LIP_INDICES
    
    def detect_lips_in_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract lip region from a single frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            Cropped and resized lip region as grayscale numpy array, or None if no face detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get the first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract lip landmarks
        h, w = frame.shape[:2]
        lip_points = []
        
        for idx in self.LIP_INDICES:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            lip_points.append([x, y])
        
        lip_points = np.array(lip_points)
        
        # Get bounding box around lip region with padding
        x_min, y_min = np.min(lip_points, axis=0)
        x_max, y_max = np.max(lip_points, axis=0)
        
        # Add padding (20% of lip region size)
        lip_width = x_max - x_min
        lip_height = y_max - y_min
        padding_x = int(lip_width * 0.2)
        padding_y = int(lip_height * 0.2)
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(w, x_max + padding_x)
        y_max = min(h, y_max + padding_y)
        
        # Crop lip region
        lip_region = frame[y_min:y_max, x_min:x_max]
        
        if lip_region.size == 0:
            return None
        
        # Convert to grayscale
        if len(lip_region.shape) == 3:
            lip_region = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        lip_region = cv2.resize(lip_region, self.target_size)
        
        # Normalize pixel values to [0, 1]
        lip_region = lip_region.astype(np.float32) / 255.0
        
        return lip_region
    
    def process_video(self, video_path: str, max_frames: int = 30) -> Optional[np.ndarray]:
        """
        Process a video file and extract lip regions from all frames.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to process
            
        Returns:
            Array of shape (frames, height, width) containing lip regions, or None if processing failed
        """
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return None
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return None
        
        lip_sequences = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            lip_region = self.detect_lips_in_frame(frame)
            
            if lip_region is not None:
                lip_sequences.append(lip_region)
                frame_count += 1
        
        cap.release()
        
        if len(lip_sequences) == 0:
            print(f"No lip regions detected in video: {video_path}")
            return None
        
        # Convert to numpy array and pad/truncate to max_frames
        lip_sequences = np.array(lip_sequences)
        
        # If we have fewer frames than max_frames, pad with the last frame
        if len(lip_sequences) < max_frames:
            last_frame = lip_sequences[-1]
            padding_needed = max_frames - len(lip_sequences)
            padding = np.tile(last_frame[np.newaxis, :, :], (padding_needed, 1, 1))
            lip_sequences = np.concatenate([lip_sequences, padding], axis=0)
        
        return lip_sequences
    
    def process_video_folder(self, folder_path: str, word_label: str, max_frames: int = 30) -> Tuple[List[np.ndarray], List[str]]:
        """
        Process all videos in a folder and extract lip sequences.
        
        Args:
            folder_path: Path to folder containing video files
            word_label: Label for the word being spoken in these videos
            max_frames: Maximum number of frames to process per video
            
        Returns:
            Tuple of (lip_sequences, labels) where lip_sequences is a list of numpy arrays
            and labels is a list of corresponding word labels
        """
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return [], []
        
        lip_sequences = []
        labels = []
        
        # Get all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = [f for f in os.listdir(folder_path) 
                      if any(f.lower().endswith(ext) for ext in video_extensions)]
        
        print(f"Processing {len(video_files)} videos in {folder_path}")
        
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            lip_sequence = self.process_video(video_path, max_frames)
            
            if lip_sequence is not None:
                lip_sequences.append(lip_sequence)
                labels.append(word_label)
                print(f"✓ Processed: {video_file}")
            else:
                print(f"✗ Failed to process: {video_file}")
        
        return lip_sequences, labels


def main():
    """
    Example usage of the LipDetector class.
    """
    detector = LipDetector()
    
    # Example: Process a single video
    # lip_sequence = detector.process_video("path/to/video.mp4")
    # if lip_sequence is not None:
    #     print(f"Extracted {len(lip_sequence)} frames of size {lip_sequence.shape[1:]}")
    
    # Example: Process a folder of videos
    # sequences, labels = detector.process_video_folder("data/training_set/doctor", "doctor")
    # print(f"Processed {len(sequences)} videos for word 'doctor'")


if __name__ == "__main__":
    main()
