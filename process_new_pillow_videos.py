#!/usr/bin/env python3
"""
PHASE 1: New Video Integration and Preprocessing
Process 10 new pillow videos from 'data/extra videos 22.9.25/' directory
Apply training-compatible preprocessing pipeline to match existing training data format
"""

import os
import cv2
import numpy as np
import torch
from datetime import datetime
import glob

class TrainingCompatibleLipDetector:
    """
    Training-compatible lip detection and preprocessing system
    Matches the exact preprocessing pipeline used for training data
    """
    
    def __init__(self):
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def is_preprocessed_mouth_crop(self, frame):
        """
        Detect if frame is already a preprocessed mouth crop (96x64)
        Returns True if already processed, False if needs processing
        """
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        
        # Training data mouth crops are 96x64 (aspect ratio 1.5)
        if abs(aspect_ratio - 1.5) < 0.1 and height < 100:
            return True
        return False
    
    def detect_and_crop_mouth(self, frame):
        """
        Detect face and crop mouth region using training-compatible parameters
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            # Fallback: center crop 60% of frame
            h, w = frame.shape[:2]
            crop_h, crop_w = int(h * 0.6), int(w * 0.6)
            start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
            return frame[start_y:start_y + crop_h, start_x:start_x + crop_w]
        
        # Use largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Calculate mouth region (65-88% down face, 25-75% from sides)
        mouth_y_start = y + int(h * 0.65)
        mouth_y_end = y + int(h * 0.88)
        mouth_x_start = x + int(w * 0.25)
        mouth_x_end = x + int(w * 0.75)
        
        # Apply 1.3x padding expansion
        padding_factor = 1.3
        mouth_h = mouth_y_end - mouth_y_start
        mouth_w = mouth_x_end - mouth_x_start
        
        expand_h = int((mouth_h * padding_factor - mouth_h) / 2)
        expand_w = int((mouth_w * padding_factor - mouth_w) / 2)
        
        # Expand with bounds checking
        frame_h, frame_w = frame.shape[:2]
        final_y_start = max(0, mouth_y_start - expand_h)
        final_y_end = min(frame_h, mouth_y_end + expand_h)
        final_x_start = max(0, mouth_x_start - expand_w)
        final_x_end = min(frame_w, mouth_x_end + expand_w)
        
        return frame[final_y_start:final_y_end, final_x_start:final_x_end]
    
    def preprocess_video(self, video_path, output_path):
        """
        Apply complete training-compatible preprocessing pipeline
        """
        print(f"Processing: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames found in video: {video_path}")
        
        print(f"  Original frames: {len(frames)}")
        
        # Process each frame
        processed_frames = []
        for frame in frames:
            # Check if already preprocessed
            if self.is_preprocessed_mouth_crop(frame):
                # Already processed, just convert to grayscale
                if len(frame.shape) == 3:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray_frame = frame
            else:
                # Detect and crop mouth region
                mouth_crop = self.detect_and_crop_mouth(frame)
                # Convert BGR to Grayscale
                gray_frame = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY)
            
            # Resize to 64x96 (training data format)
            resized_frame = cv2.resize(gray_frame, (96, 64))
            
            # Normalize to [0, 1]
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            
            processed_frames.append(normalized_frame)
        
        # Extract 32 contiguous center frames
        total_frames = len(processed_frames)
        if total_frames >= 32:
            start_idx = (total_frames - 32) // 2
            final_frames = processed_frames[start_idx:start_idx + 32]
        else:
            # Pad with repeated frames if less than 32
            final_frames = processed_frames[:]
            while len(final_frames) < 32:
                final_frames.extend(processed_frames)
            final_frames = final_frames[:32]
        
        print(f"  Final frames: {len(final_frames)}")
        
        # Convert to tensor format and save as video
        tensor_frames = np.stack(final_frames)  # Shape: (32, 64, 96)
        
        # Save as MP4 video (convert back to 0-255 range)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (96, 64), isColor=False)
        
        for frame in final_frames:
            # Convert back to 0-255 range for video saving
            frame_uint8 = (frame * 255).astype(np.uint8)
            out.write(frame_uint8)
        
        out.release()
        print(f"  Saved: {output_path}")
        
        return tensor_frames

def main():
    """
    Process all 10 new pillow videos from 'data/extra videos 22.9.25/' directory
    """
    print("🎯 PHASE 1: New Video Integration and Preprocessing")
    print("=" * 60)
    
    # Initialize detector
    detector = TrainingCompatibleLipDetector()
    
    # Input and output directories
    input_dir = "data/extra videos 22.9.25"
    output_dir = "data/the_best_videos_so_far"
    
    # Find all pillow videos
    video_files = glob.glob(os.path.join(input_dir, "pillow_*.mp4"))
    video_files.sort()
    
    print(f"Found {len(video_files)} pillow videos to process:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {os.path.basename(video_file)}")
    
    if len(video_files) != 10:
        print(f"⚠️  WARNING: Expected 10 videos, found {len(video_files)}")
    
    # Process each video
    processed_count = 0
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    
    for i, video_path in enumerate(video_files, 1):
        try:
            # Generate output filename with consistent naming convention
            # Format: pillow__useruser01__65plus__female__caucasian__[timestamp]_topmid_96x64_processed.mp4
            output_filename = f"pillow__useruser01__65plus__female__caucasian__{timestamp}_{i:02d}_topmid_96x64_processed.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # Process video
            tensor_frames = detector.preprocess_video(video_path, output_path)
            
            # Verify output
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"  ✅ Success: {output_filename} ({file_size:,} bytes)")
                processed_count += 1
            else:
                print(f"  ❌ Failed: {output_filename}")
                
        except Exception as e:
            print(f"  ❌ Error processing {os.path.basename(video_path)}: {e}")
    
    print("=" * 60)
    print(f"📊 PHASE 1 RESULTS:")
    print(f"  Videos processed: {processed_count}/10")
    print(f"  Success rate: {processed_count/10*100:.1f}%")
    
    if processed_count == 10:
        print("✅ PHASE 1 COMPLETE: All 10 new pillow videos successfully processed!")
        print("🎯 Ready for PHASE 2: Dataset Balancing and Stratification")
    else:
        print(f"⚠️  PHASE 1 INCOMPLETE: {10-processed_count} videos failed processing")
    
    return processed_count == 10

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
