#!/usr/bin/env python3
"""
Comprehensive Video Preprocessing Pipeline for 104 Unprocessed Videos
Process all unprocessed videos using training-compatible preprocessing pipeline
Target: 85-per-class balanced dataset for ≥82% cross-demographic validation accuracy
"""

import os
import cv2
import numpy as np
import glob
import pandas as pd
from datetime import datetime
import re
from collections import defaultdict

class TrainingCompatibleLipDetector:
    """
    Training-compatible lip detection and preprocessing system
    Matches the exact preprocessing pipeline used for training data
    """
    
    def __init__(self):
        # Initialize face cascade for lip detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Preprocessing parameters matching training data
        self.target_width = 96
        self.target_height = 64
        self.target_frames = 32
        
        # Mouth region parameters (percentage of face)
        self.mouth_y_start = 0.65  # 65% down the face
        self.mouth_y_end = 0.88    # 88% down the face
        self.mouth_x_start = 0.25  # 25% from left
        self.mouth_x_end = 0.75    # 75% from left
        
        # Padding expansion
        self.padding_horizontal = 0.4  # 40% horizontal expansion
        self.padding_vertical = 0.5    # 50% vertical expansion
    
    def detect_face_and_extract_mouth(self, frame):
        """Detect face and extract mouth region"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Use the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Calculate mouth region within face
            mouth_y_start = int(y + h * self.mouth_y_start)
            mouth_y_end = int(y + h * self.mouth_y_end)
            mouth_x_start = int(x + w * self.mouth_x_start)
            mouth_x_end = int(x + w * self.mouth_x_end)
            
            # Extract mouth region
            mouth_region = frame[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
            
            # Apply padding expansion
            mouth_h, mouth_w = mouth_region.shape[:2]
            pad_h = int(mouth_h * self.padding_vertical)
            pad_w = int(mouth_w * self.padding_horizontal)
            
            # Expand region with padding
            expanded_y_start = max(0, mouth_y_start - pad_h)
            expanded_y_end = min(frame.shape[0], mouth_y_end + pad_h)
            expanded_x_start = max(0, mouth_x_start - pad_w)
            expanded_x_end = min(frame.shape[1], mouth_x_end + pad_w)
            
            expanded_mouth = frame[expanded_y_start:expanded_y_end, expanded_x_start:expanded_x_end]
            
            return expanded_mouth
        else:
            # Fallback: center crop to approximate mouth region
            h, w = frame.shape[:2]
            center_y, center_x = h // 2, w // 2
            crop_h, crop_w = int(h * 0.6), int(w * 0.6)
            
            y_start = max(0, center_y - crop_h // 2)
            y_end = min(h, center_y + crop_h // 2)
            x_start = max(0, center_x - crop_w // 2)
            x_end = min(w, center_x + crop_w // 2)
            
            return frame[y_start:y_end, x_start:x_end]
    
    def preprocess_video(self, video_path, output_path):
        """
        Preprocess video with training-compatible pipeline
        BGR→Grayscale, [0,1] normalization, 64×96 resize, 32 center frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return False
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract mouth region
            mouth_crop = self.detect_face_and_extract_mouth(frame)
            
            # Convert BGR to Grayscale
            if len(mouth_crop.shape) == 3:
                gray_frame = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = mouth_crop
            
            # Resize to target dimensions (96×64)
            resized_frame = cv2.resize(gray_frame, (self.target_width, self.target_height))
            
            # Normalize to [0, 1]
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            
            frames.append(normalized_frame)
            frame_count += 1
        
        cap.release()
        
        if len(frames) == 0:
            print(f"❌ No frames extracted from: {video_path}")
            return False
        
        # Extract exactly 32 contiguous center frames
        if len(frames) >= self.target_frames:
            start_idx = (len(frames) - self.target_frames) // 2
            selected_frames = frames[start_idx:start_idx + self.target_frames]
        else:
            # Pad with repeated frames if too short
            selected_frames = frames.copy()
            while len(selected_frames) < self.target_frames:
                selected_frames.extend(frames[:min(len(frames), self.target_frames - len(selected_frames))])
            selected_frames = selected_frames[:self.target_frames]
        
        # Save processed video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (self.target_width, self.target_height), False)
        
        for frame in selected_frames:
            # Convert back to uint8 for video writer
            frame_uint8 = (frame * 255).astype(np.uint8)
            out.write(frame_uint8)
        
        out.release()
        
        print(f"✅ Processed: {os.path.basename(video_path)} -> {os.path.basename(output_path)}")
        return True

def normalize_class_name(raw_class_name):
    """Normalize class name variations to standard format"""
    if not raw_class_name:
        return None
    
    # Convert to lowercase and replace spaces with underscores
    normalized = raw_class_name.lower().replace(' ', '_')
    
    # Handle specific variations
    class_mappings = {
        'my_mouth_is_dry': 'my_mouth_is_dry',
        'i_need_to_move': 'i_need_to_move',
        'doctor': 'doctor',
        'pillow': 'pillow',
        # Handle variations
        'my mouth is dry': 'my_mouth_is_dry',
        'i need to move': 'i_need_to_move',
    }
    
    return class_mappings.get(normalized, normalized)

def extract_class_from_filename(filename):
    """Extract and normalize class name from unprocessed video filename"""
    filename = os.path.basename(filename)
    
    # Try various patterns to extract class name
    potential_class = None
    
    # Pattern 1: Standard underscore format
    if filename.startswith('pillow_'):
        potential_class = 'pillow'
    elif filename.startswith('doctor_'):
        potential_class = 'doctor'
    elif filename.startswith('i_need_to_move_'):
        potential_class = 'i_need_to_move'
    elif filename.startswith('my_mouth_is_dry_'):
        potential_class = 'my_mouth_is_dry'
    
    # Pattern 2: Space-separated format
    elif 'my mouth is dry' in filename.lower():
        potential_class = 'my mouth is dry'
    elif 'i need to move' in filename.lower():
        potential_class = 'i need to move'
    
    # Pattern 3: Mixed case variations
    elif filename.lower().startswith('my mouth is dry'):
        potential_class = 'my mouth is dry'
    elif filename.lower().startswith('i need to move'):
        potential_class = 'i need to move'
    
    # Normalize the extracted class name
    if potential_class:
        normalized_class = normalize_class_name(potential_class)
        if normalized_class in ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']:
            return normalized_class
    
    return None

def extract_demographic_info(filename):
    """Extract demographic information from filename"""
    # Pattern: class_gender_age_ethnicity_video_info.mp4
    # Example: "my mouth is dry_female_18-39_caucasian_video 8.mp4"
    
    # Default values
    age_group = 'unknown'
    gender = 'unknown'
    ethnicity = 'unknown'
    
    filename_lower = filename.lower()
    
    # Extract gender
    if 'female' in filename_lower:
        gender = 'female'
    elif 'male' in filename_lower:
        gender = 'male'
    
    # Extract age group
    if '18-39' in filename_lower or '18-29' in filename_lower:
        age_group = '18to39'
    elif '65plus' in filename_lower:
        age_group = '65plus'
    elif '40-64' in filename_lower:
        age_group = '40to64'
    
    # Extract ethnicity
    if 'caucasian' in filename_lower or 'causasian' in filename_lower:  # Handle typo
        ethnicity = 'caucasian'
    elif 'asian' in filename_lower:
        ethnicity = 'asian'
    elif 'aboriginal' in filename_lower:
        ethnicity = 'aboriginal'
    
    return age_group, gender, ethnicity

def generate_processed_filename(original_filename, class_name, age_group, gender, ethnicity):
    """Generate standardized processed filename"""
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    
    # Create unique identifier from original filename
    base_name = os.path.splitext(original_filename)[0]
    # Remove spaces and special characters for unique ID
    unique_id = re.sub(r'[^a-zA-Z0-9]', '', base_name)[:10]
    
    processed_filename = f"{class_name}__useruser01__{age_group}__{gender}__{ethnicity}__{timestamp}_{unique_id}_topmid_96x64_processed.mp4"
    
    return processed_filename

def main():
    """Execute comprehensive video preprocessing pipeline"""
    print("🚀 COMPREHENSIVE VIDEO PREPROCESSING PIPELINE")
    print("=" * 70)
    print("Objective: Process 104 unprocessed videos for 85-per-class balanced training")
    print("Target: ≥82% cross-demographic validation accuracy")
    
    # Initialize preprocessing system
    detector = TrainingCompatibleLipDetector()
    
    # Input and output directories
    input_dir = "data/extra videos 22.9.25"
    output_dir = "data/the_best_videos_so_far"
    
    # Find all unprocessed videos
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    print(f"\n📁 Found {len(video_files)} unprocessed videos")
    
    # Process each video
    processed_count = 0
    failed_count = 0
    class_counts = defaultdict(int)
    processing_log = []
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        
        # Extract class and demographic info
        class_name = extract_class_from_filename(filename)
        age_group, gender, ethnicity = extract_demographic_info(filename)
        
        if class_name:
            # Generate processed filename
            processed_filename = generate_processed_filename(filename, class_name, age_group, gender, ethnicity)
            output_path = os.path.join(output_dir, processed_filename)
            
            # Skip if already processed
            if os.path.exists(output_path):
                print(f"⏭️  Already processed: {filename}")
                continue
            
            # Process video
            success = detector.preprocess_video(video_path, output_path)
            
            if success:
                processed_count += 1
                class_counts[class_name] += 1
                processing_log.append({
                    'original_filename': filename,
                    'processed_filename': processed_filename,
                    'class': class_name,
                    'age_group': age_group,
                    'gender': gender,
                    'ethnicity': ethnicity,
                    'status': 'success'
                })
            else:
                failed_count += 1
                processing_log.append({
                    'original_filename': filename,
                    'processed_filename': processed_filename,
                    'class': class_name,
                    'age_group': age_group,
                    'gender': gender,
                    'ethnicity': ethnicity,
                    'status': 'failed'
                })
        else:
            print(f"❌ Could not extract class from: {filename}")
            failed_count += 1
    
    # Save processing log
    log_df = pd.DataFrame(processing_log)
    log_path = f"preprocessing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_df.to_csv(log_path, index=False)
    
    # Results summary
    print("\n" + "=" * 70)
    print("📊 PREPROCESSING RESULTS:")
    print(f"  ✅ Successfully processed: {processed_count} videos")
    print(f"  ❌ Failed to process: {failed_count} videos")
    print(f"  📄 Processing log saved: {log_path}")
    
    print(f"\n📊 PROCESSED VIDEOS BY CLASS:")
    for class_name in sorted(class_counts.keys()):
        count = class_counts[class_name]
        print(f"  {class_name}: {count} videos")
    
    if processed_count > 0:
        print(f"\n🎯 Phase 1 Complete: {processed_count} videos processed with training-compatible pipeline")
        print("✅ Ready for Phase 2: Balanced Dataset Creation")
        return True
    else:
        print("\n❌ Phase 1 Failed: No videos were successfully processed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
