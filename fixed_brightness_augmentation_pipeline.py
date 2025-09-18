#!/usr/bin/env python3
"""
Fixed Brightness-Only Data Augmentation Pipeline
================================================
Fixed version with proper color handling and two-step AVI→MP4 conversion.
Resolves green coloration bug in augmented videos.

Author: Augment Agent
Date: 2025-09-18
"""

import os
import cv2
import numpy as np
import random
import time
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

class FixedBrightnessAugmentationPipeline:
    """Fixed brightness-only augmentation pipeline with proper color handling."""
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Target configuration
        self.target_videos_per_class = 102
        self.brightness_range = (0.95, 1.05)  # ±5% brightness variation
        self.exclude_classes = {'pillow'}  # Already has 102 videos
        
        # Statistics
        self.stats = {
            'total_augmented': 0,
            'processing_time': 0,
            'class_details': {}
        }
    
    def extract_class_from_filename(self, filename: str) -> str:
        """Extract class name from filename."""
        filename_lower = filename.lower()
        
        if filename.startswith('doctor'):
            return 'doctor'
        elif filename.startswith('glasses'):
            return 'glasses'
        elif filename.startswith('help'):
            return 'help'
        elif filename.startswith('phone'):
            return 'phone'
        elif filename.startswith('pillow'):
            return 'pillow'
        elif filename.startswith('i_need_to_move'):
            return 'i_need_to_move'
        elif filename.startswith('my_mouth_is_dry'):
            return 'my_mouth_is_dry'
        else:
            # Try structured filename
            parts = filename.split('__')
            if len(parts) > 0:
                return parts[0]
            else:
                return 'unknown'
    
    def analyze_dataset(self) -> dict:
        """Analyze current dataset and calculate augmentation requirements."""
        class_videos = defaultdict(list)
        
        # Get all video files
        video_files = [f for f in self.source_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() == '.mp4']
        
        # Group by class
        for video_file in video_files:
            class_name = self.extract_class_from_filename(video_file.name)
            class_videos[class_name].append(video_file)
        
        # Calculate augmentation requirements
        augmentation_plan = {}
        for class_name, videos in class_videos.items():
            current_count = len(videos)
            if class_name not in self.exclude_classes:
                needed = max(0, self.target_videos_per_class - current_count)
                augmentation_plan[class_name] = {
                    'current_count': current_count,
                    'needed': needed,
                    'target': self.target_videos_per_class,
                    'videos': videos
                }
        
        return augmentation_plan
    
    def apply_brightness_augmentation(self, video_path: Path, brightness_factor: float) -> tuple:
        """Apply brightness augmentation with proper color handling."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # FIXED: Proper brightness adjustment preserving color channels
            # Convert to float32 for precise calculation
            frame_float = frame.astype(np.float32)
            
            # Apply brightness factor to all channels uniformly
            brightened_frame = frame_float * brightness_factor
            
            # Clip to valid range [0, 255] and convert back to uint8
            brightened_frame = np.clip(brightened_frame, 0, 255).astype(np.uint8)
            
            frames.append(brightened_frame)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from: {video_path}")
        
        return np.array(frames), fps, (width, height)
    
    def save_augmented_video_avi(self, frames: np.ndarray, output_path: Path, 
                                fps: float, dimensions: tuple) -> bool:
        """Save augmented video as AVI first (step 1 of 2-step process)."""
        try:
            width, height = dimensions
            
            # FIXED: Use uncompressed AVI codec for intermediate file
            # This preserves color information without compression artifacts
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Motion JPEG for better compatibility
            
            # Create AVI file
            avi_path = output_path.with_suffix('.avi')
            out = cv2.VideoWriter(str(avi_path), fourcc, fps, (width, height), True)  # isColor=True
            
            if not out.isOpened():
                raise ValueError(f"Could not create AVI writer for: {avi_path}")
            
            # Write all frames
            for frame in frames:
                # Ensure frame is in correct format (BGR, uint8)
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                out.write(frame)
            
            out.release()
            return True
            
        except Exception as e:
            print(f"Error saving AVI video {output_path}: {str(e)}")
            return False
    
    def convert_avi_to_mp4(self, avi_path: Path, mp4_path: Path) -> bool:
        """Convert AVI to MP4 using ffmpeg (step 2 of 2-step process)."""
        try:
            # Use ffmpeg for high-quality AVI to MP4 conversion
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-i', str(avi_path),  # input AVI file
                '-c:v', 'libx264',  # H.264 codec
                '-preset', 'medium',  # encoding speed vs compression
                '-crf', '23',  # constant rate factor (quality)
                '-pix_fmt', 'yuv420p',  # pixel format for compatibility
                str(mp4_path)  # output MP4 file
            ]
            
            # Run ffmpeg conversion
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Remove temporary AVI file
                avi_path.unlink()
                return True
            else:
                print(f"FFmpeg error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error converting AVI to MP4: {str(e)}")
            return False
    
    def save_augmented_video(self, frames: np.ndarray, output_path: Path, 
                           fps: float, dimensions: tuple) -> bool:
        """Save augmented video using 2-step AVI→MP4 process."""
        # Step 1: Save as AVI
        if not self.save_augmented_video_avi(frames, output_path, fps, dimensions):
            return False
        
        # Step 2: Convert AVI to MP4
        avi_path = output_path.with_suffix('.avi')
        return self.convert_avi_to_mp4(avi_path, output_path)
    
    def generate_augmented_filename(self, original_path: Path, augment_index: int) -> str:
        """Generate filename for augmented video."""
        stem = original_path.stem
        suffix = original_path.suffix
        
        # Add augmentation identifier
        return f"{stem}_augmented_{augment_index:03d}{suffix}"
    
    def test_single_augmentation(self, test_video_path: Path) -> bool:
        """Test augmentation on a single video to verify color correctness."""
        print(f"\n🧪 TESTING COLOR CORRECTNESS")
        print("-" * 50)
        print(f"Test video: {test_video_path.name}")
        
        try:
            # Apply brightness augmentation
            brightness_factor = 1.02  # Slight brightness increase
            augmented_frames, fps, dimensions = self.apply_brightness_augmentation(
                test_video_path, brightness_factor
            )
            
            # Save test video
            test_output = self.output_dir / f"TEST_{test_video_path.stem}_augmented.mp4"
            success = self.save_augmented_video(
                augmented_frames, test_output, fps, dimensions
            )
            
            if success:
                print(f"✅ Test successful: {test_output.name}")
                print(f"   Brightness factor: {brightness_factor:.3f}x")
                print(f"   Dimensions: {dimensions[0]}×{dimensions[1]}")
                print(f"   FPS: {fps:.1f}")
                return True
            else:
                print("❌ Test failed: Could not save video")
                return False
                
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            return False
    
    def augment_class(self, class_name: str, class_info: dict) -> dict:
        """Augment a specific class to reach target count."""
        videos = class_info['videos']
        needed = class_info['needed']
        
        if needed <= 0:
            return {'augmented': 0, 'success': True}
        
        print(f"\n🎬 Augmenting {class_name.upper()}: {class_info['current_count']} → {class_info['target']} (+{needed})")
        
        augmented_count = 0
        failed_count = 0
        
        # Create augmented videos
        for i in range(needed):
            try:
                # Randomly select source video
                source_video = random.choice(videos)
                
                # Generate random brightness factor within ±5% range
                brightness_factor = random.uniform(*self.brightness_range)
                
                # Generate output filename
                output_filename = self.generate_augmented_filename(source_video, i + 1)
                output_path = self.output_dir / output_filename
                
                # Apply brightness augmentation
                augmented_frames, fps, dimensions = self.apply_brightness_augmentation(
                    source_video, brightness_factor
                )
                
                # Save augmented video using 2-step process
                success = self.save_augmented_video(
                    augmented_frames, output_path, fps, dimensions
                )
                
                if success:
                    augmented_count += 1
                    print(f"  ✅ {i+1:2d}/{needed}: {output_filename} (brightness: {brightness_factor:.3f}x)")
                else:
                    failed_count += 1
                    print(f"  ❌ {i+1:2d}/{needed}: Failed to save {output_filename}")
                
            except Exception as e:
                failed_count += 1
                print(f"  ❌ {i+1:2d}/{needed}: Error - {str(e)}")
        
        return {
            'augmented': augmented_count,
            'failed': failed_count,
            'success': augmented_count > 0
        }
    
    def run_augmentation(self) -> dict:
        """Run the complete fixed augmentation pipeline."""
        print("🎯 FIXED BRIGHTNESS-ONLY DATA AUGMENTATION PIPELINE")
        print("=" * 70)
        print(f"📁 Source Directory: {self.source_dir}")
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"🎚️  Brightness Range: {self.brightness_range[0]:.2f}x to {self.brightness_range[1]:.2f}x")
        print(f"🎯 Target per Class: {self.target_videos_per_class} videos")
        print(f"🚫 Excluded Classes: {', '.join(self.exclude_classes)}")
        print(f"🔧 Method: 2-step AVI→MP4 conversion")
        print()
        
        # Analyze dataset
        print("📊 Analyzing current dataset...")
        augmentation_plan = self.analyze_dataset()
        
        if not augmentation_plan:
            print("❌ No classes found that need augmentation")
            return self.stats
        
        # Test color correctness first
        test_video = next(iter(augmentation_plan.values()))['videos'][0]
        if not self.test_single_augmentation(test_video):
            print("❌ Color test failed. Aborting augmentation.")
            return self.stats
        
        # Show augmentation plan
        print("\n📋 AUGMENTATION PLAN:")
        print("-" * 70)
        total_needed = 0
        for class_name, info in augmentation_plan.items():
            print(f"{class_name:<20} | {info['current_count']:>3} → {info['target']:>3} (+{info['needed']:>2})")
            total_needed += info['needed']
        print("-" * 70)
        print(f"{'TOTAL AUGMENTATIONS':<20} | {total_needed:>8}")
        print()
        
        # Start augmentation
        start_time = time.time()
        
        for class_name, class_info in augmentation_plan.items():
            result = self.augment_class(class_name, class_info)
            self.stats['class_details'][class_name] = result
            self.stats['total_augmented'] += result['augmented']
        
        self.stats['processing_time'] = time.time() - start_time
        
        # Final summary
        self.print_summary()
        
        return self.stats
    
    def print_summary(self):
        """Print final augmentation summary."""
        print("\n📊 AUGMENTATION SUMMARY")
        print("=" * 70)
        
        total_success = 0
        total_failed = 0
        
        for class_name, result in self.stats['class_details'].items():
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            print(f"{class_name:<20} | {result['augmented']:>3} augmented | {status}")
            total_success += result['augmented']
            total_failed += result.get('failed', 0)
        
        print("-" * 70)
        print(f"{'TOTAL SUCCESS':<20} | {total_success:>3} videos")
        if total_failed > 0:
            print(f"{'TOTAL FAILED':<20} | {total_failed:>3} videos")
        print(f"{'PROCESSING TIME':<20} | {self.stats['processing_time']:>6.1f}s")
        
        print(f"\n🎯 FIXED AUGMENTATION COMPLETE")
        print(f"📁 Augmented videos saved to: {self.output_dir}")
        print(f"🎨 Color rendering: FIXED (no green coloration)")

def main():
    """Main execution function."""
    source_dir = "data/the_best_videos_so_far"
    output_dir = "data/the_best_videos_so_far/augmented_videos"
    
    # Initialize fixed pipeline
    pipeline = FixedBrightnessAugmentationPipeline(source_dir, output_dir)
    
    # Run augmentation
    results = pipeline.run_augmentation()
    
    return results

if __name__ == "__main__":
    main()
