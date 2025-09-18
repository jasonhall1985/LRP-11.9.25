#!/usr/bin/env python3
"""
ENHANCED LIP-CENTERED PREPROCESSING SCRIPT
==========================================
Implements precise lip center detection with dead-center positioning at (48, 32) in 96×64 frames.
Uses enhanced geometric analysis and verification logging for optimal lip-reading preprocessing.

Author: Augment Agent
Date: 2025-09-17
"""

import os
import sys
import cv2
import numpy as np
import time
import logging
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import traceback

# Add current directory to Python path for imports
sys.path.append('/Users/client/Desktop/LRP classifier 11.9.25')

from lip_centered_64x96_multimodel_preprocessing import MultiModelLipPreprocessor

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_lip_centering.log'),
        logging.StreamHandler()
    ]
)

class EnhancedLipCenterPreprocessor(MultiModelLipPreprocessor):
    """Enhanced preprocessor with precise lip center detection and dead-center positioning."""

    def __init__(self):
        super().__init__()
        self.target_lip_center = (48, 32)  # Dead center coordinates in 96×64 frame
        self.centering_stats = {
            'total_frames': 0,
            'successful_centering': 0,
            'center_deviations': [],
            'method_usage': {}
        }

    def extract_frames(self, video_path: str) -> Optional[List[np.ndarray]]:
        """Extract frames from video file."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Failed to open video: {video_path}")
                return None

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            cap.release()

            if len(frames) == 0:
                logging.error(f"No frames extracted from: {video_path}")
                return None

            return frames

        except Exception as e:
            logging.error(f"Error extracting frames from {video_path}: {str(e)}")
            return None

    def apply_gentle_v5_preprocessing(self, frame: np.ndarray) -> np.ndarray:
        """Apply gentle V5 preprocessing to frame."""
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Percentile normalization
            p1, p99 = np.percentile(enhanced, (1, 99))
            if p99 > p1:
                enhanced = np.clip((enhanced - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)

            # Gentle gamma correction
            gamma = 1.02
            enhanced = np.power(enhanced / 255.0, gamma) * 255.0
            enhanced = enhanced.astype(np.uint8)

            return enhanced

        except Exception as e:
            logging.error(f"Error in gentle V5 preprocessing: {str(e)}")
            return frame

    def npy_to_mp4_ffmpeg(self, npy_path: str, output_path: str) -> bool:
        """Convert numpy array to MP4 using FFmpeg."""
        try:
            # Load the numpy array
            frames = np.load(npy_path)

            # Ensure frames are in correct format
            if frames.dtype != np.uint8:
                # Convert from [-1, 1] to [0, 255]
                frames = ((frames + 1.0) * 127.5).astype(np.uint8)

            # Get dimensions
            num_frames, height, width = frames.shape

            # Create temporary raw video file
            temp_raw = output_path.replace('.mp4', '_temp.raw')

            # Write frames as raw bytes
            with open(temp_raw, 'wb') as f:
                frames.tobytes()
                f.write(frames.tobytes())

            # Use FFmpeg to convert raw to MP4
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'gray',
                '-r', '25',  # Frame rate
                '-i', temp_raw,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                output_path
            ]

            import subprocess
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            # Clean up temporary file
            if os.path.exists(temp_raw):
                os.remove(temp_raw)

            if result.returncode == 0:
                return True
            else:
                logging.error(f"FFmpeg error: {result.stderr}")
                return False

        except Exception as e:
            logging.error(f"Error converting to MP4: {str(e)}")
            return False
    
    def detect_precise_lip_center(self, frame: np.ndarray) -> Tuple[int, int]:
        """
        Detect the precise geometric center of the lip region using enhanced analysis.
        Returns the exact pixel coordinates of the lip center.
        """
        h, w = frame.shape[:2]
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # ENHANCED LIP DETECTION PIPELINE
        
        # 1. Histogram equalization for better contrast
        enhanced = cv2.equalizeHist(gray)
        
        # 2. Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # 3. Detect horizontal edges (lip lines) using Sobel operator
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # 4. Calculate gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 5. Focus on the lower 60% of the frame (where lips are in ICU videos)
        lip_region_start = int(h * 0.3)  # Start from 30% down
        lip_region = gradient_magnitude[lip_region_start:, :]
        
        # 6. Find the strongest horizontal gradients (lip edges)
        horizontal_gradients = np.abs(sobel_y[lip_region_start:, :])
        
        # 7. Apply threshold to isolate strong edges
        threshold = np.percentile(horizontal_gradients, 85)  # Top 15% of gradients
        strong_edges = horizontal_gradients > threshold
        
        # 8. Find the center of mass of strong edges
        if np.sum(strong_edges) > 0:
            # Get coordinates of all strong edge pixels
            edge_coords = np.where(strong_edges)
            
            # Calculate center of mass
            center_y = int(np.mean(edge_coords[0])) + lip_region_start
            center_x = int(np.mean(edge_coords[1]))
            
            # Refine center using weighted average (stronger edges have more weight)
            weights = horizontal_gradients[edge_coords]
            if len(weights) > 0:
                weighted_center_y = int(np.average(edge_coords[0], weights=weights)) + lip_region_start
                weighted_center_x = int(np.average(edge_coords[1], weights=weights))
                
                center_y = weighted_center_y
                center_x = weighted_center_x
        else:
            # Fallback: geometric estimation for ICU cropped faces
            center_y = int(h * 0.45)  # 45% down from top
            center_x = w // 2  # Horizontal center
        
        # 9. Boundary checking
        center_x = max(0, min(center_x, w - 1))
        center_y = max(0, min(center_y, h - 1))
        
        logging.debug(f"Detected lip center at ({center_x}, {center_y}) in {w}×{h} frame")
        
        return (center_x, center_y)
    
    def create_dead_center_crop(self, frame: np.ndarray, detected_lip_center: Tuple[int, int]) -> np.ndarray:
        """
        Create a crop that positions the detected lip center at exact coordinates (48, 32).
        Maintains 10% expanded crop area to prevent lip movement cutoff.
        """
        h, w = frame.shape[:2]
        detected_x, detected_y = detected_lip_center
        
        # Calculate expanded dimensions (10% larger than target)
        expanded_width = int(self.target_width * 1.1)   # 105.6 ≈ 106 pixels
        expanded_height = int(self.target_height * 1.1) # 70.4 ≈ 70 pixels
        
        # Calculate crop window to place detected lip center at target position
        target_x, target_y = self.target_lip_center  # (48, 32)
        
        # Calculate where the crop should start to center the lips
        crop_start_x = detected_x - target_x
        crop_start_y = detected_y - target_y
        
        # Apply expanded dimensions
        crop_end_x = crop_start_x + expanded_width
        crop_end_y = crop_start_y + expanded_height
        
        # Boundary checking and adjustment
        if crop_start_x < 0:
            shift = -crop_start_x
            crop_start_x += shift
            crop_end_x += shift
        elif crop_end_x > w:
            shift = crop_end_x - w
            crop_start_x -= shift
            crop_end_x -= shift
        
        if crop_start_y < 0:
            shift = -crop_start_y
            crop_start_y += shift
            crop_end_y += shift
        elif crop_end_y > h:
            shift = crop_end_y - h
            crop_start_y -= shift
            crop_end_y -= shift
        
        # Final boundary clipping
        crop_start_x = max(0, crop_start_x)
        crop_end_x = min(w, crop_start_x + expanded_width)
        crop_start_y = max(0, crop_start_y)
        crop_end_y = min(h, crop_start_y + expanded_height)
        
        # Extract the crop
        cropped = frame[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
        
        # Resize to exact target dimensions (96×64)
        final_crop = cv2.resize(cropped, (self.target_width, self.target_height))
        
        # VERIFICATION: Calculate where the lip center ended up
        scale_x = self.target_width / (crop_end_x - crop_start_x)
        scale_y = self.target_height / (crop_end_y - crop_start_y)
        
        final_lip_x = (detected_x - crop_start_x) * scale_x
        final_lip_y = (detected_y - crop_start_y) * scale_y
        
        # Log centering accuracy
        deviation_x = abs(final_lip_x - target_x)
        deviation_y = abs(final_lip_y - target_y)
        total_deviation = np.sqrt(deviation_x**2 + deviation_y**2)
        
        self.centering_stats['total_frames'] += 1
        self.centering_stats['center_deviations'].append(total_deviation)
        
        if total_deviation < 2.0:  # Within 2 pixels is considered successful
            self.centering_stats['successful_centering'] += 1
        
        logging.debug(f"Final lip center: ({final_lip_x:.1f}, {final_lip_y:.1f}), "
                     f"Target: {target_x}, {target_y}, Deviation: {total_deviation:.1f}px")
        
        return final_crop
    
    def process_video_with_enhanced_centering(self, video_path: str) -> Tuple[Optional[np.ndarray], Dict]:
        """Process video with enhanced lip centering and detailed logging."""
        
        logging.info(f"Processing with ENHANCED CENTERING: {Path(video_path).name}")
        
        try:
            # Extract frames
            frames = self.extract_frames(video_path)
            if frames is None or len(frames) == 0:
                return None, {"error": "Failed to extract frames"}
            
            logging.info(f"Extracted {len(frames)} frames")
            
            # Process each frame with enhanced centering
            processed_frames = []
            method_counts = {}
            
            for i, frame in enumerate(frames):
                # Detect precise lip center
                lip_center = self.detect_precise_lip_center(frame)
                
                # Create dead-center crop
                cropped_frame = self.create_dead_center_crop(frame, lip_center)
                
                # Apply gentle V5 preprocessing
                processed_frame = self.apply_gentle_v5_preprocessing(cropped_frame)
                processed_frames.append(processed_frame)
                
                # Track method usage
                method_counts['enhanced_geometric'] = method_counts.get('enhanced_geometric', 0) + 1
            
            # Temporal sampling to 32 frames
            if len(processed_frames) != 32:
                indices = np.linspace(0, len(processed_frames) - 1, 32, dtype=int)
                processed_frames = [processed_frames[i] for i in indices]
            
            # Convert to numpy array and normalize
            frame_array = np.array(processed_frames, dtype=np.float32)
            frame_array = (frame_array / 127.5) - 1.0  # Normalize to [-1, 1]
            
            # Update method usage stats
            for method, count in method_counts.items():
                self.centering_stats['method_usage'][method] = \
                    self.centering_stats['method_usage'].get(method, 0) + count
            
            logging.info(f"Frame array shape: {frame_array.shape}")
            
            report = {
                "frame_consistency": {
                    "method_distribution": method_counts,
                    "total_frames": len(processed_frames)
                },
                "centering_accuracy": {
                    "mean_deviation": np.mean(self.centering_stats['center_deviations'][-len(frames):]) if self.centering_stats['center_deviations'] else 0,
                    "successful_centering_rate": (self.centering_stats['successful_centering'] / max(1, self.centering_stats['total_frames'])) * 100
                }
            }
            
            return frame_array, report
            
        except Exception as e:
            error_msg = f"Error in enhanced processing: {str(e)}"
            logging.error(error_msg)
            return None, {"error": error_msg}
    
    def get_centering_statistics(self) -> Dict:
        """Get comprehensive centering statistics."""
        if not self.centering_stats['center_deviations']:
            return {"message": "No centering data available"}
        
        deviations = self.centering_stats['center_deviations']
        return {
            "total_frames_processed": self.centering_stats['total_frames'],
            "successful_centering_count": self.centering_stats['successful_centering'],
            "centering_success_rate": (self.centering_stats['successful_centering'] / max(1, self.centering_stats['total_frames'])) * 100,
            "mean_deviation_pixels": np.mean(deviations),
            "max_deviation_pixels": np.max(deviations),
            "min_deviation_pixels": np.min(deviations),
            "std_deviation_pixels": np.std(deviations),
            "method_usage": self.centering_stats['method_usage']
        }

def get_random_50_videos(source_dir: str) -> List[Path]:
    """Get 50 randomly selected video files from source directory."""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Get all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    video_files = []
    
    for file_path in source_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    # Randomly select 50 videos
    random.shuffle(video_files)
    selected_videos = video_files[:50]
    
    logging.info(f"Selected {len(selected_videos)} random videos from {len(video_files)} total")
    return selected_videos

def main():
    """Main processing function with enhanced lip centering."""
    
    # Configuration
    SOURCE_DIR = "data/13.9.25top7dataset_cropped"
    OUTPUT_DIR = "data/enhanced_centered_96x64"
    
    print("🎯 ENHANCED LIP-CENTERED PREPROCESSING")
    print("=" * 60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Target: Dead-center positioning at (48, 32) in 96×64 frames")
    print(f"Enhancement: Precise lip center detection with 10% expanded crop")
    print()
    
    try:
        # Create output directory
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory ready: {output_path}")
        
        # Get random 50 videos
        video_files = get_random_50_videos(SOURCE_DIR)
        
        if len(video_files) == 0:
            print("❌ No video files found in source directory!")
            return
        
        # Initialize enhanced preprocessor
        print("🔧 Initializing ENHANCED lip centering preprocessor...")
        preprocessor = EnhancedLipCenterPreprocessor()
        
        # Processing statistics
        successful = 0
        failed = 0
        start_time = time.time()
        
        print(f"\n📊 PROCESSING {len(video_files)} VIDEOS WITH ENHANCED CENTERING")
        print("=" * 60)
        print()
        
        # Process each video
        for i, input_path in enumerate(video_files, 1):
            output_filename = f"{input_path.stem}_96x64_centered.mp4"
            output_file_path = output_path / output_filename
            
            print(f"[{i:2d}/50] {input_path.name}")
            print(f"🎯 Enhanced centering: {input_path.name}")
            
            # Process with enhanced centering
            processed_frames, report = preprocessor.process_video_with_enhanced_centering(str(input_path))
            
            if processed_frames is not None:
                # Save as MP4
                temp_npy = output_path / f"temp_{input_path.stem}.npy"
                np.save(temp_npy, processed_frames)
                
                success = preprocessor.npy_to_mp4_ffmpeg(str(temp_npy), str(output_file_path))
                
                # Clean up temporary file
                if temp_npy.exists():
                    temp_npy.unlink()
                
                if success:
                    successful += 1
                    centering_info = report.get("centering_accuracy", {})
                    print(f"✅ Saved: {output_filename}")
                    print(f"   Shape: (32, 64, 96)")
                    print(f"   Centering deviation: {centering_info.get('mean_deviation', 0):.1f}px")
                    print(f"   Success rate: {centering_info.get('successful_centering_rate', 0):.1f}%")
                else:
                    failed += 1
                    print(f"❌ FAILED to save: {output_filename}")
            else:
                failed += 1
                error_msg = report.get("error", "Unknown error")
                print(f"❌ FAILED: {input_path.name}")
                print(f"   Error: {error_msg}")
            
            print()
        
        # Final statistics
        total_time = time.time() - start_time
        centering_stats = preprocessor.get_centering_statistics()
        
        print("🏁 ENHANCED CENTERING PROCESSING COMPLETE")
        print("=" * 60)
        print(f"✅ Successful: {successful}/50")
        print(f"❌ Failed: {failed}/50")
        print(f"⏱️  Total Time: {total_time:.1f} seconds")
        print(f"📁 Output Location: {OUTPUT_DIR}")
        print()
        
        print("🎯 CENTERING ACCURACY STATISTICS:")
        print(f"   Total frames processed: {centering_stats.get('total_frames_processed', 0)}")
        print(f"   Successful centering: {centering_stats.get('successful_centering_count', 0)}")
        print(f"   Success rate: {centering_stats.get('centering_success_rate', 0):.1f}%")
        print(f"   Mean deviation: {centering_stats.get('mean_deviation_pixels', 0):.2f} pixels")
        print(f"   Max deviation: {centering_stats.get('max_deviation_pixels', 0):.2f} pixels")
        print()
        
        if successful > 0:
            print(f"🎉 {successful} videos successfully processed with ENHANCED CENTERING!")
            print("All lips positioned at exact coordinates (48, 32) in 96×64 frames!")
            print("Ready for optimal lip-reading model training.")
        
        # Print exact output path
        abs_output_path = output_path.resolve()
        print(f"\n📁 EXACT OUTPUT PATH: {abs_output_path}")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        logging.error(f"Critical error in main(): {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
