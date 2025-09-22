#!/usr/bin/env python3
"""
Process 140 New Diverse Videos Using Training-Compatible Pipeline
Apply TrainingCompatibleLipDetector system for consistent preprocessing
"""

import os
import cv2
import numpy as np
import glob
from datetime import datetime
import json
from collections import defaultdict

class TrainingCompatibleLipDetector:
    """Training-compatible lip detection and preprocessing system"""
    
    def __init__(self):
        self.target_width = 96
        self.target_height = 64
        self.target_frames = 32
        
        # Mouth region detection parameters (matching training data)
        self.mouth_y_start = 0.65  # 65% down the face
        self.mouth_y_end = 0.88    # 88% down the face
        self.mouth_x_start = 0.25  # 25% from left
        self.mouth_x_end = 0.75    # 75% from left
        
        # Padding expansion (matching training data)
        self.padding_horizontal = 0.4  # 40% horizontal expansion
        self.padding_vertical = 0.5    # 50% vertical expansion
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print(f"🔧 TrainingCompatibleLipDetector initialized:")
        print(f"   Target resolution: {self.target_width}×{self.target_height}")
        print(f"   Target frames: {self.target_frames}")
        print(f"   Mouth region: {self.mouth_y_start:.0%}-{self.mouth_y_end:.0%} vertical, {self.mouth_x_start:.0%}-{self.mouth_x_end:.0%} horizontal")
    
    def detect_face_and_mouth_region(self, frame):
        """Detect face and estimate mouth region"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Use the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Calculate mouth region within face
            mouth_x1 = int(x + w * self.mouth_x_start)
            mouth_x2 = int(x + w * self.mouth_x_end)
            mouth_y1 = int(y + h * self.mouth_y_start)
            mouth_y2 = int(y + h * self.mouth_y_end)
            
            # Apply padding expansion
            padding_x = int(w * self.padding_horizontal / 2)
            padding_y = int(h * self.padding_vertical / 2)
            
            mouth_x1 = max(0, mouth_x1 - padding_x)
            mouth_x2 = min(frame.shape[1], mouth_x2 + padding_x)
            mouth_y1 = max(0, mouth_y1 - padding_y)
            mouth_y2 = min(frame.shape[0], mouth_y2 + padding_y)
            
            return mouth_x1, mouth_y1, mouth_x2, mouth_y2, True
        
        # Fallback: center crop (60% of frame)
        h, w = frame.shape[:2]
        crop_h = int(h * 0.6)
        crop_w = int(w * 0.6)
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2
        
        return start_x, start_y, start_x + crop_w, start_y + crop_h, False
    
    def process_video(self, input_path, output_path):
        """Process video with training-compatible preprocessing"""
        try:
            cap = cv2.VideoCapture(input_path)
            
            if not cap.isOpened():
                return False, "Could not open video file"
            
            # Read all frames
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                return False, "No frames found in video"
            
            # Process frames
            processed_frames = []
            
            for frame in frames:
                # Detect mouth region
                x1, y1, x2, y2, face_detected = self.detect_face_and_mouth_region(frame)
                
                # Crop to mouth region
                cropped = frame[y1:y2, x1:x2]
                
                if cropped.size == 0:
                    continue
                
                # Convert to grayscale
                if len(cropped.shape) == 3:
                    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                
                # Resize to target resolution
                resized = cv2.resize(cropped, (self.target_width, self.target_height))
                
                # Normalize to [0, 1]
                normalized = resized.astype(np.float32) / 255.0
                
                processed_frames.append(normalized)
            
            if len(processed_frames) == 0:
                return False, "No valid frames after processing"
            
            # Extract exactly 32 contiguous center frames
            if len(processed_frames) >= self.target_frames:
                start_idx = (len(processed_frames) - self.target_frames) // 2
                final_frames = processed_frames[start_idx:start_idx + self.target_frames]
            else:
                # Repeat frames to reach target count
                final_frames = processed_frames.copy()
                while len(final_frames) < self.target_frames:
                    final_frames.extend(processed_frames[:min(len(processed_frames), self.target_frames - len(final_frames))])
                final_frames = final_frames[:self.target_frames]
            
            # Save processed video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (self.target_width, self.target_height), False)
            
            for frame in final_frames:
                # Convert back to uint8 for saving
                frame_uint8 = (frame * 255).astype(np.uint8)
                out.write(frame_uint8)
            
            out.release()
            
            return True, f"Successfully processed {len(final_frames)} frames"
            
        except Exception as e:
            return False, f"Error processing video: {str(e)}"

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
    """Extract and normalize class name from video filename"""
    filename = os.path.basename(filename)
    filename_lower = filename.lower()
    
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
    
    # Pattern 2: Capital I variations (I_need_to_move)
    elif filename.startswith('I_need_to_move_') or filename_lower.startswith('i_need_to_move_'):
        potential_class = 'i_need_to_move'
    
    # Pattern 3: Check if filename contains class keywords anywhere
    if not potential_class:
        if 'pillow' in filename_lower:
            potential_class = 'pillow'
        elif 'doctor' in filename_lower:
            potential_class = 'doctor'
        elif 'i_need_to_move' in filename_lower or 'i need to move' in filename_lower:
            potential_class = 'i_need_to_move'
        elif 'my_mouth_is_dry' in filename_lower or 'my mouth is dry' in filename_lower:
            potential_class = 'my_mouth_is_dry'
    
    # Normalize the extracted class name
    if potential_class:
        normalized_class = normalize_class_name(potential_class)
        if normalized_class in ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']:
            return normalized_class
    
    return None

def extract_demographic_info(filename):
    """Extract demographic information from filename"""
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
    if '18-39' in filename_lower or '18-29' in filename_lower or '30-39' in filename_lower:
        age_group = '18to39'
    elif '40-64' in filename_lower or '40-49' in filename_lower or '50-64' in filename_lower:
        age_group = '40to64'
    elif '65plus' in filename_lower or '65+' in filename_lower:
        age_group = '65plus'
    
    # Extract ethnicity
    if 'caucasian' in filename_lower:
        ethnicity = 'caucasian'
    elif 'asian' in filename_lower:
        ethnicity = 'asian'
    elif 'aboriginal' in filename_lower:
        ethnicity = 'aboriginal'
    elif 'african' in filename_lower:
        ethnicity = 'african'
    elif 'hispanic' in filename_lower:
        ethnicity = 'hispanic'
    
    return age_group, gender, ethnicity

def generate_processed_filename(original_filename, class_name, age_group, gender, ethnicity):
    """Generate standardized processed filename"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = abs(hash(original_filename)) % 10000
    
    # Format: class__useruser01__age__gender__ethnicity__timestamp_uniqueid_topmid_96x64_processed.mp4
    processed_filename = f"{class_name}__useruser01__{age_group}__{gender}__{ethnicity}__{timestamp}_{unique_id:04d}_topmid_96x64_processed.mp4"
    
    return processed_filename

def main():
    """Execute comprehensive video preprocessing pipeline"""
    print("🎯 PROCESSING 140 NEW DIVERSE VIDEOS")
    print("=" * 70)
    print("Using Training-Compatible Lip Detection Pipeline")
    
    # Setup paths
    input_dir = "data/extra videos 22.9.25/extra videos 11pm"
    output_dir = "data/the_best_videos_so_far"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    print(f"\n📁 Found {len(video_files)} MP4 files to process")
    
    if len(video_files) == 0:
        print("❌ No MP4 files found")
        return False
    
    # Initialize processor
    processor = TrainingCompatibleLipDetector()
    
    # Process each video
    results = {
        'successful': [],
        'failed': [],
        'class_counts': defaultdict(int),
        'demographic_counts': defaultdict(int)
    }
    
    print(f"\n🔄 PROCESSING VIDEOS:")
    print("=" * 70)
    
    for i, video_path in enumerate(video_files, 1):
        filename = os.path.basename(video_path)
        
        # Extract metadata
        class_name = extract_class_from_filename(filename)
        age_group, gender, ethnicity = extract_demographic_info(filename)
        
        if not class_name:
            print(f"  {i:3d}/{len(video_files)} ❌ Could not extract class from: {filename}")
            results['failed'].append({'filename': filename, 'reason': 'Class extraction failed'})
            continue
        
        # Generate output filename
        output_filename = generate_processed_filename(filename, class_name, age_group, gender, ethnicity)
        output_path = os.path.join(output_dir, output_filename)
        
        # Process video
        success, message = processor.process_video(video_path, output_path)
        
        if success:
            print(f"  {i:3d}/{len(video_files)} ✅ {filename}")
            print(f"      → {output_filename}")
            print(f"      Class: {class_name}, Demographics: {age_group}_{gender}_{ethnicity}")
            
            results['successful'].append({
                'original_filename': filename,
                'processed_filename': output_filename,
                'class': class_name,
                'age_group': age_group,
                'gender': gender,
                'ethnicity': ethnicity,
                'demographic_group': f"{age_group}_{gender}_{ethnicity}"
            })
            
            results['class_counts'][class_name] += 1
            results['demographic_counts'][f"{age_group}_{gender}_{ethnicity}"] += 1
        else:
            print(f"  {i:3d}/{len(video_files)} ❌ Failed: {filename}")
            print(f"      Error: {message}")
            results['failed'].append({'filename': filename, 'reason': message})
    
    # Summary statistics
    total_processed = len(results['successful'])
    total_failed = len(results['failed'])
    success_rate = (total_processed / len(video_files)) * 100 if len(video_files) > 0 else 0
    
    print("\n" + "=" * 70)
    print("📊 PROCESSING SUMMARY")
    print(f"✅ Successfully processed: {total_processed} videos ({success_rate:.1f}%)")
    print(f"❌ Failed: {total_failed} videos")
    
    print(f"\n📊 PROCESSED VIDEOS BY CLASS:")
    target_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    for class_name in sorted(target_classes):
        count = results['class_counts'][class_name]
        print(f"  {class_name}: {count} videos")
    
    print(f"\n🌍 DEMOGRAPHIC DIVERSITY:")
    print(f"  Unique demographic groups: {len(results['demographic_counts'])}")
    for demo, count in sorted(results['demographic_counts'].items()):
        print(f"  {demo}: {count} videos")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f"new_diverse_videos_processing_results_{timestamp}.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Processing results saved: {results_path}")
    
    if total_processed > 0:
        print("🎉 NEW DIVERSE VIDEOS SUCCESSFULLY INTEGRATED!")
        print("🚀 Ready for Phase 2: Enhanced Balanced Dataset Creation")
        return True, results_path
    else:
        print("❌ No videos were successfully processed")
        return False, None

if __name__ == "__main__":
    success, results_path = main()
    exit(0 if success else 1)
