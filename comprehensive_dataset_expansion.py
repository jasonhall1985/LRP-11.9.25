#!/usr/bin/env python3
"""
Comprehensive Dataset Expansion Pipeline
=======================================

Expands the lip-reading dataset by processing 250 new videos (50 per class) from 
`data/13.9.25top7dataset_cropped` using the exact same gentle_v5 preprocessing 
pipeline that achieved 57.14% validation accuracy.

Key Features:
- Processes only MP4 format videos
- Demographic-aware splitting to prevent data leakage
- Exact gentle_v5 preprocessing parameters
- Quality control and verification
- Balanced class distribution (50 videos per class)
- Preview video generation for inspection
"""

import cv2
import numpy as np
import os
import json
import random
import subprocess
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ComprehensiveDatasetExpander:
    def __init__(self, source_dir, target_dir):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.target_classes = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        self.videos_per_class = 50
        self.target_frames = 32
        
        # Quality checks (same as preview_videos_fixed processing)
        # - Shape must be (32, 96, 96)
        # - Range: min between -1.1 and -0.8, max between 0.8 and 1.1
        
        # Demographic tracking for split assignment
        self.demographic_splits = {}
        self.processing_log = []
        
        # Create output directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create the required directory structure."""
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # Create class directories
        for class_name in self.target_classes:
            (self.target_dir / class_name).mkdir(exist_ok=True)
        
        # Create preview directory
        (self.target_dir / "preview_videos").mkdir(exist_ok=True)
        
        print(f"📁 Created directory structure at: {self.target_dir}")
    
    def analyze_source_data(self):
        """Analyze source data and extract demographic information."""
        print("🔍 Analyzing source data...")
        
        class_demographics = defaultdict(list)
        
        # Scan all MP4 files
        for file_path in self.source_dir.glob("*.mp4"):
            filename = file_path.name
            parts = filename.split('__')
            
            if len(parts) >= 5:
                class_name = parts[0]
                age_group = parts[2]
                gender = parts[3]
                ethnicity = parts[4].split('__')[0]
                
                if class_name in self.target_classes:
                    demographic_key = f"{age_group}_{gender}_{ethnicity}"
                    class_demographics[class_name].append({
                        'file_path': file_path,
                        'demographic': demographic_key,
                        'filename': filename
                    })
        
        # Print analysis
        print("\n📊 Source Data Analysis:")
        print("=" * 50)
        total_available = 0
        for class_name in sorted(class_demographics.keys()):
            videos = class_demographics[class_name]
            unique_demographics = set(v['demographic'] for v in videos)
            print(f"{class_name:8}: {len(videos):3d} MP4 videos, {len(unique_demographics):2d} demographics")
            total_available += len(videos)
        
        print(f"{'Total':<8}: {total_available:3d} MP4 videos")
        
        return class_demographics
    
    def assign_demographic_splits(self, class_demographics):
        """Assign demographic groups to train/val/test splits to prevent data leakage."""
        print("\n🎯 Assigning demographic splits...")
        
        all_demographics = set()
        for class_videos in class_demographics.values():
            for video_info in class_videos:
                all_demographics.add(video_info['demographic'])
        
        # Randomly assign demographics to splits (70/20/10)
        demographics_list = list(all_demographics)
        random.shuffle(demographics_list)
        
        n_train = int(0.7 * len(demographics_list))
        n_val = int(0.2 * len(demographics_list))
        
        train_demographics = set(demographics_list[:n_train])
        val_demographics = set(demographics_list[n_train:n_train + n_val])
        test_demographics = set(demographics_list[n_train + n_val:])
        
        # Store assignments
        for demo in train_demographics:
            self.demographic_splits[demo] = 'train'
        for demo in val_demographics:
            self.demographic_splits[demo] = 'val'
        for demo in test_demographics:
            self.demographic_splits[demo] = 'test'
        
        print(f"   Train demographics: {len(train_demographics)}")
        print(f"   Val demographics: {len(val_demographics)}")
        print(f"   Test demographics: {len(test_demographics)}")
        
        return train_demographics, val_demographics, test_demographics
    
    def select_balanced_videos(self, class_demographics):
        """Select exactly 50 videos per class with balanced demographic representation."""
        print("\n⚖️  Selecting balanced videos...")
        
        selected_videos = {}
        
        for class_name in self.target_classes:
            videos = class_demographics[class_name]
            
            if len(videos) < self.videos_per_class:
                print(f"⚠️  Warning: {class_name} has only {len(videos)} videos, need {self.videos_per_class}")
                selected_videos[class_name] = videos
            else:
                # Shuffle and select first 50
                random.shuffle(videos)
                selected_videos[class_name] = videos[:self.videos_per_class]
            
            print(f"   {class_name}: Selected {len(selected_videos[class_name])} videos")
        
        return selected_videos
    
    def apply_gentle_v5_preprocessing(self, frames):
        """
        Apply the exact gentle_v5 preprocessing used for the 57.14% accuracy model.
        
        Parameters:
        - Minimal CLAHE: clipLimit=1.5, tileGridSize=(8,8)
        - Conservative percentile: (p1,p99)
        - Minimal gamma: 1.02
        - Brightness standardization to 0.5
        - Normalization to [-1,1]
        """
        frames = frames.astype(np.float32) / 255.0
        
        processed_frames = []
        for frame in frames:
            frame_uint8 = (frame * 255).astype(np.uint8)
            
            # MINIMAL CLAHE enhancement
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(frame_uint8).astype(np.float32) / 255.0
            
            # CONSERVATIVE percentile normalization
            p1, p99 = np.percentile(enhanced, [1, 99])
            if p99 > p1:
                enhanced = np.clip((enhanced - p1) / (p99 - p1), 0, 1)
            
            # MINIMAL gamma correction
            gamma = 1.02
            enhanced = np.power(enhanced, 1.0 / gamma)
            
            # Brightness standardization
            target_brightness = 0.5
            current_brightness = np.mean(enhanced)
            if current_brightness > 0:
                brightness_factor = target_brightness / current_brightness
                enhanced = np.clip(enhanced * brightness_factor, 0, 1)
            
            processed_frames.append(enhanced)
        
        frames = np.array(processed_frames)
        frames = (frames - 0.5) / 0.5  # Normalize to [-1, 1]
        
        return frames
    
    def load_and_crop_video(self, video_path):
        """
        Load MP4 video using OPTIMIZED cropping parameters for complete lip visibility.
        - Crop: 65% height × 40% width (30% to 70% horizontally)
        - Vertical positioning: Start from 10% down to center mouth region
        - Resize: to 96×96 pixels (CRITICAL for preventing green artifacts)
        - 32-frame temporal sampling using np.linspace()
        Uses OPTIMIZED parameters for better lip capture and positioning.
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale using ITU-R BT.709 weights
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use OPTIMIZED cropping parameters for complete lip capture
            h, w = gray.shape

            # Optimized crop for complete lip capture (65% height, centered mouth)
            crop_h = int(0.65 * h)  # 65% height (instead of 80%)

            # Vertical positioning: start from 10% down to center mouth region
            crop_v_start = int(0.10 * h)
            crop_v_end = crop_v_start + crop_h

            # Horizontal positioning: middle 40% for better mouth capture
            crop_w_start = int(0.30 * w)  # 30% to 70% (40% width)
            crop_w_end = int(0.70 * w)

            # Ensure crop doesn't exceed frame boundaries
            crop_v_end = min(crop_v_end, h)
            crop_w_end = min(crop_w_end, w)

            cropped = gray[crop_v_start:crop_v_end, crop_w_start:crop_w_end]

            # CRITICAL: Resize to 96x96 (same as preview_videos_fixed processing)
            resized = cv2.resize(cropped, (96, 96))
            frames.append(resized)
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        # 32-frame temporal sampling using np.linspace()
        if len(frames) >= self.target_frames:
            indices = np.linspace(0, len(frames)-1, self.target_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            # Repeat frames if not enough
            while len(frames) < self.target_frames:
                frames.extend(frames[:min(len(frames), self.target_frames - len(frames))])
        
        return np.array(frames[:self.target_frames])

    def verify_quality(self, processed_frames, filename):
        """Apply EXACT same quality verification as preview_videos_fixed processing."""
        quality_metrics = {
            'filename': filename,
            'shape': processed_frames.shape,
            'min_val': float(processed_frames.min()),
            'max_val': float(processed_frames.max()),
            'mean_val': float(processed_frames.mean()),
            'std_val': float(processed_frames.std()),
            'has_nan': bool(np.isnan(processed_frames).any()),
            'has_inf': bool(np.isinf(processed_frames).any()),
            'quality_passed': False
        }

        # EXACT same quality checks as process_full_dataset_gentle_v5.py
        quality_passed = True
        failure_reasons = []

        # Shape check: must be (32, 96, 96)
        if processed_frames.shape != (32, 96, 96):
            quality_passed = False
            failure_reasons.append(f"Wrong shape: {processed_frames.shape} != (32, 96, 96)")

        # Range check: min between -1.1 and -0.8, max between 0.8 and 1.1
        min_val = processed_frames.min()
        max_val = processed_frames.max()
        if not (-1.1 <= min_val <= -0.8 and 0.8 <= max_val <= 1.1):
            quality_passed = False
            failure_reasons.append(f"Values out of range: min={min_val:.3f}, max={max_val:.3f}")

        quality_metrics['quality_passed'] = quality_passed
        quality_metrics['failure_reasons'] = failure_reasons

        return quality_metrics

    def create_preview_video(self, processed_frames, output_path):
        """Create a preview video for visual inspection using AVI->MP4 conversion."""
        try:
            # Convert from [-1,1] to [0,255] for visualization
            preview_frames = ((processed_frames + 1) * 127.5).astype(np.uint8)
            preview_frames = np.clip(preview_frames, 0, 255)

            # First create AVI file (more reliable for grayscale)
            avi_path = str(output_path).replace('.mp4', '.avi')

            # Create AVI video writer with grayscale settings
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec for AVI
            fps = 8  # Slow playback for inspection
            h, w = preview_frames.shape[1], preview_frames.shape[2]

            out = cv2.VideoWriter(avi_path, fourcc, fps, (w, h), isColor=False)

            if not out.isOpened():
                print(f"❌ Could not open AVI video writer for {avi_path}")
                return False

            for frame in preview_frames:
                out.write(frame)

            out.release()

            # Convert AVI to MP4 using FFmpeg if available
            try:
                import subprocess
                cmd = [
                    'ffmpeg', '-y',  # -y to overwrite output file
                    '-i', avi_path,  # input AVI
                    '-c:v', 'libx264',  # H.264 codec
                    '-pix_fmt', 'gray',  # grayscale pixel format
                    '-crf', '23',  # quality setting
                    str(output_path)  # output MP4
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    # Remove temporary AVI file
                    os.remove(avi_path)
                    return True
                else:
                    print(f"⚠️  FFmpeg conversion failed, keeping AVI: {avi_path}")
                    return True  # AVI file still created successfully

            except (FileNotFoundError, subprocess.SubprocessError):
                print(f"⚠️  FFmpeg not available, keeping AVI: {avi_path}")
                return True  # AVI file still created successfully

        except Exception as e:
            print(f"❌ Failed to create preview for {output_path}: {e}")
            return False

    def process_single_video(self, video_info, class_name):
        """Process a single video with full pipeline."""
        video_path = video_info['file_path']
        filename = video_info['filename']
        demographic = video_info['demographic']

        try:
            # Load and crop video
            frames = self.load_and_crop_video(video_path)

            if frames is None:
                return None, f"Failed to load video: {filename}"

            # Apply gentle_v5 preprocessing
            processed_frames = self.apply_gentle_v5_preprocessing(frames)

            # Quality verification
            quality_metrics = self.verify_quality(processed_frames, filename)

            if not quality_metrics['quality_passed']:
                return None, f"Quality check failed: {', '.join(quality_metrics['failure_reasons'])}"

            # Generate output filename
            base_name = Path(filename).stem
            output_filename = f"{base_name}_gentle_v5.npy"
            output_path = self.target_dir / class_name / output_filename

            # Save processed video
            np.save(output_path, processed_frames)

            # Create preview video
            preview_filename = f"{base_name}_preview.mp4"
            preview_path = self.target_dir / "preview_videos" / preview_filename
            preview_created = self.create_preview_video(processed_frames, preview_path)

            # Add demographic split info
            split_assignment = self.demographic_splits.get(demographic, 'unknown')

            # Create processing log entry
            log_entry = {
                'input_video': str(video_path),
                'output_video': str(output_path),
                'preview_video': str(preview_path) if preview_created else None,
                'class': class_name,
                'demographic': demographic,
                'split_assignment': split_assignment,
                'quality_metrics': quality_metrics,
                'preview_created': preview_created,
                'processing_success': True
            }

            return log_entry, None

        except Exception as e:
            error_msg = f"Processing error for {filename}: {str(e)}"
            log_entry = {
                'input_video': str(video_path),
                'output_video': None,
                'preview_video': None,
                'class': class_name,
                'demographic': demographic,
                'split_assignment': 'unknown',
                'quality_metrics': None,
                'preview_created': False,
                'processing_success': False,
                'error_message': error_msg
            }
            return log_entry, error_msg

    def process_all_videos(self, selected_videos):
        """Process all selected videos with progress tracking."""
        print("\n🔄 Processing videos with gentle_v5 preprocessing...")

        total_videos = sum(len(videos) for videos in selected_videos.values())
        processed_count = 0
        failed_count = 0

        for class_name in self.target_classes:
            videos = selected_videos[class_name]
            print(f"\n📹 Processing {class_name}: {len(videos)} videos")

            class_processed = 0
            class_failed = 0

            for video_info in tqdm(videos, desc=f"Processing {class_name}"):
                log_entry, error_msg = self.process_single_video(video_info, class_name)

                if log_entry:
                    self.processing_log.append(log_entry)

                    if log_entry['processing_success']:
                        processed_count += 1
                        class_processed += 1
                    else:
                        failed_count += 1
                        class_failed += 1
                        if error_msg:
                            print(f"   ❌ {error_msg}")
                else:
                    failed_count += 1
                    class_failed += 1
                    if error_msg:
                        print(f"   ❌ {error_msg}")

            print(f"   ✅ {class_name}: {class_processed} processed, {class_failed} failed")

        print(f"\n📊 Processing Summary:")
        print(f"   Total processed: {processed_count}/{total_videos}")
        print(f"   Success rate: {processed_count/total_videos*100:.1f}%")

        return processed_count, failed_count

    def save_processing_log(self):
        """Save comprehensive processing log with demographic split assignments."""
        log_file = self.target_dir / "processing_log.json"

        # Create summary statistics
        summary = {
            'total_videos_processed': len([log for log in self.processing_log if log['processing_success']]),
            'total_videos_failed': len([log for log in self.processing_log if not log['processing_success']]),
            'videos_per_class': {},
            'demographic_splits': self.demographic_splits,
            'split_distribution': {'train': 0, 'val': 0, 'test': 0, 'unknown': 0},
            'quality_summary': {
                'avg_min_value': 0.0,
                'avg_max_value': 0.0,
                'videos_with_quality_issues': 0
            }
        }

        # Calculate per-class statistics
        for class_name in self.target_classes:
            class_logs = [log for log in self.processing_log if log['class'] == class_name and log['processing_success']]
            summary['videos_per_class'][class_name] = len(class_logs)

        # Calculate split distribution
        for log in self.processing_log:
            if log['processing_success']:
                split = log['split_assignment']
                summary['split_distribution'][split] = summary['split_distribution'].get(split, 0) + 1

        # Calculate quality statistics
        successful_logs = [log for log in self.processing_log if log['processing_success'] and log['quality_metrics']]
        if successful_logs:
            min_values = [log['quality_metrics']['min_val'] for log in successful_logs]
            max_values = [log['quality_metrics']['max_val'] for log in successful_logs]
            summary['quality_summary']['avg_min_value'] = np.mean(min_values)
            summary['quality_summary']['avg_max_value'] = np.mean(max_values)

        # Save complete log
        log_data = {
            'summary': summary,
            'processing_log': self.processing_log,
            'parameters': {
                'source_directory': str(self.source_dir),
                'target_directory': str(self.target_dir),
                'target_classes': self.target_classes,
                'videos_per_class': self.videos_per_class,
                'target_frames': self.target_frames,
                'preprocessing': 'gentle_v5',
                'geometric_cropping': 'optimized_65%_height_x_middle_40%_width_centered',
                'temporal_sampling': '32_frames_np_linspace',
                'resize_to_96x96': True,
                'mp4_format_only': True
            }
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"📋 Processing log saved: {log_file}")
        return log_file

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE DATASET EXPANSION SUMMARY")
        print("="*60)

        successful_logs = [log for log in self.processing_log if log['processing_success']]
        failed_logs = [log for log in self.processing_log if not log['processing_success']]

        print(f"\n🎯 Processing Results:")
        print(f"   Total videos processed: {len(successful_logs)}")
        print(f"   Total videos failed: {len(failed_logs)}")
        print(f"   Success rate: {len(successful_logs)/(len(successful_logs)+len(failed_logs))*100:.1f}%")

        print(f"\n📈 Class Distribution:")
        for class_name in self.target_classes:
            class_count = len([log for log in successful_logs if log['class'] == class_name])
            print(f"   {class_name:8}: {class_count:2d} videos")

        print(f"\n🎭 Demographic Split Distribution:")
        split_counts = {'train': 0, 'val': 0, 'test': 0, 'unknown': 0}
        for log in successful_logs:
            split = log['split_assignment']
            split_counts[split] = split_counts.get(split, 0) + 1

        for split, count in split_counts.items():
            percentage = count / len(successful_logs) * 100 if successful_logs else 0
            print(f"   {split:7}: {count:3d} videos ({percentage:.1f}%)")

        print(f"\n✅ Quality Metrics:")
        if successful_logs:
            quality_metrics = [log['quality_metrics'] for log in successful_logs if log['quality_metrics']]
            if quality_metrics:
                avg_min = np.mean([qm['min_val'] for qm in quality_metrics])
                avg_max = np.mean([qm['max_val'] for qm in quality_metrics])
                print(f"   Average min value: {avg_min:.3f}")
                print(f"   Average max value: {avg_max:.3f}")
                print(f"   Expected range: min [-1.1, -0.8], max [0.8, 1.1]")

        print(f"\n📁 Output Structure:")
        print(f"   Target directory: {self.target_dir}")
        print(f"   Class directories: {len(self.target_classes)} created")
        print(f"   Preview videos: {self.target_dir}/preview_videos/")
        print(f"   Processing log: {self.target_dir}/processing_log.json")

        print(f"\n🔧 Processing Parameters:")
        print(f"   Source format: MP4 only")
        print(f"   Preprocessing: gentle_v5 (CLAHE=1.5, p1-p99, gamma=1.02)")
        print(f"   Geometric crop: optimized 65% height × middle 40% width (centered)")
        print(f"   Temporal sampling: 32 frames (np.linspace)")
        print(f"   Resize to 96x96: Yes (same as preview_videos_fixed)")
        print(f"   Demographic splitting: Prevents data leakage")

        if failed_logs:
            print(f"\n❌ Failed Videos:")
            for log in failed_logs[:5]:  # Show first 5 failures
                print(f"   {log.get('error_message', 'Unknown error')}")
            if len(failed_logs) > 5:
                print(f"   ... and {len(failed_logs) - 5} more failures")

        print("\n" + "="*60)
        print("✨ Dataset expansion complete! Ready for inspection and integration.")
        print("="*60)

    def run_expansion(self):
        """Execute the complete dataset expansion pipeline."""
        print("🚀 Starting Comprehensive Dataset Expansion")
        print("="*60)

        # Set random seed for reproducible demographic splits
        random.seed(42)
        np.random.seed(42)

        # Step 1: Analyze source data
        class_demographics = self.analyze_source_data()

        # Step 2: Assign demographic splits
        self.assign_demographic_splits(class_demographics)

        # Step 3: Select balanced videos
        selected_videos = self.select_balanced_videos(class_demographics)

        # Step 4: Process all videos
        processed_count, failed_count = self.process_all_videos(selected_videos)

        # Step 5: Save processing log
        self.save_processing_log()

        # Step 6: Generate summary report
        self.generate_summary_report()

        return processed_count, failed_count


def main():
    """Main execution function."""
    source_dir = "data/13.9.25top7dataset_cropped"
    target_dir = "data/training set 17.9.25/additional 50 per class"

    # Verify source directory exists
    if not Path(source_dir).exists():
        print(f"❌ Source directory not found: {source_dir}")
        return

    # Create and run expander
    expander = ComprehensiveDatasetExpander(source_dir, target_dir)
    processed_count, failed_count = expander.run_expansion()

    print(f"\n🎉 Expansion complete!")
    print(f"   Processed: {processed_count} videos")
    print(f"   Failed: {failed_count} videos")
    print(f"   Output: {target_dir}")

    return processed_count, failed_count


if __name__ == "__main__":
    main()
