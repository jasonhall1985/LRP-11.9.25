#!/usr/bin/env python3
"""
SIMPLE TEMPORAL ANALYSIS TEST
============================
Focused analysis of temporal sampling issues without JSON complications.

Author: Augment Agent
Date: 2025-09-17
"""

import os
import sys
import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Add current directory to Python path for imports
sys.path.append('/Users/client/Desktop/LRP classifier 11.9.25')

from ultra_precise_lip_centering import UltraPreciseLipCenterPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_single_video():
    """Analyze temporal sampling on a single representative video."""
    
    # Configuration
    SOURCE_DIR = "data/13.9.25top7dataset_cropped"
    OUTPUT_DIR = "data/temporal_analysis_simple"
    
    print("🔍 SIMPLE TEMPORAL ANALYSIS TEST")
    print("=" * 50)
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select the same test video
    test_video = Path(SOURCE_DIR) / "pillow__useruser01__65plus__female__caucasian__20250827T062536_topmid.mp4"
    
    if not test_video.exists():
        print(f"❌ Test video not found: {test_video}")
        return
    
    print(f"📹 Test Video: {test_video.name}")
    
    # Initialize preprocessor
    preprocessor = UltraPreciseLipCenterPreprocessor()
    
    # Process video
    print("\n🔧 PROCESSING VIDEO...")
    start_time = time.time()
    
    processed_frames, report = preprocessor.process_video_with_ultra_precision_and_temporal_analysis(str(test_video))
    
    processing_time = time.time() - start_time
    
    if processed_frames is not None:
        print("✅ PROCESSING SUCCESSFUL")
        print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
        
        # Extract key metrics
        temporal_metrics = report.get('temporal_metrics', {})
        temporal_transform = report.get('temporal_transformation', {})
        quality_control = report.get('quality_control', {})
        
        print("\n📊 TEMPORAL ANALYSIS RESULTS:")
        print("=" * 50)
        
        # Original video metrics
        print("ORIGINAL VIDEO:")
        print(f"  Frames: {temporal_metrics.get('original_frame_count', 'N/A')}")
        print(f"  Duration: {temporal_metrics.get('actual_duration_seconds', 0):.3f} seconds")
        print(f"  FPS: {temporal_metrics.get('original_fps', 0):.2f}")
        
        # Temporal transformation
        print("\nTEMPORAL TRANSFORMATION:")
        print(f"  Output Frames: 32 (fixed)")
        print(f"  Compression Ratio: {temporal_transform.get('compression_ratio', 0):.2f}x")
        print(f"  Frame Skip Interval: {temporal_transform.get('frame_skip_interval', 0):.2f}")
        print(f"  Recommended Dynamic FPS: {temporal_transform.get('dynamic_fps_recommended', 0):.1f}")
        
        # Quality assessment
        print("\nQUALITY CONTROL:")
        print(f"  Frames Checked: {quality_control.get('frames_checked', 0)}")
        print(f"  Perfect Centering: {quality_control.get('perfect_centering_count', 0)}")
        print(f"  Mean Deviation: {quality_control.get('mean_deviation', 0):.3f} pixels")
        print(f"  Quality Passed: {quality_control.get('quality_passed', False)}")
        
        # Assessment
        print("\n🎯 ASSESSMENT:")
        compression_ratio = temporal_transform.get('compression_ratio', 1.0)
        quality_passed = quality_control.get('quality_passed', False)
        
        if compression_ratio > 2.0:
            print("  ⚠️  HIGH TEMPORAL COMPRESSION - May cause artifacts")
        else:
            print("  ✅ ACCEPTABLE TEMPORAL COMPRESSION")
            
        if quality_passed:
            print("  ✅ CENTERING QUALITY PASSED")
        else:
            print("  ❌ CENTERING QUALITY FAILED")
        
        # Save processed video for inspection
        output_filename = f"{test_video.stem}_temporal_test.mp4"
        output_file_path = output_path / output_filename
        
        # Save as numpy array first
        temp_npy = output_path / "temp_test.npy"
        np.save(temp_npy, processed_frames)
        
        # Convert to MP4 with dynamic frame rate
        original_duration = temporal_metrics.get('actual_duration_seconds', 1.0)
        success = preprocessor.npy_to_mp4_ffmpeg_with_dynamic_fps(
            str(temp_npy), str(output_file_path), original_duration
        )
        
        # Clean up
        if temp_npy.exists():
            temp_npy.unlink()
        
        if success:
            print(f"\n📁 OUTPUT VIDEO: {output_file_path}")
            
            # Verify output video properties
            cap = cv2.VideoCapture(str(output_file_path))
            if cap.isOpened():
                output_fps = cap.get(cv2.CAP_PROP_FPS)
                output_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                output_duration = output_frames / output_fps if output_fps > 0 else 0
                cap.release()
                
                print(f"📊 OUTPUT VERIFICATION:")
                print(f"  Output FPS: {output_fps:.1f}")
                print(f"  Output Frames: {output_frames}")
                print(f"  Output Duration: {output_duration:.3f} seconds")
                print(f"  Original Duration: {original_duration:.3f} seconds")
                print(f"  Duration Preservation: {(output_duration/original_duration)*100:.1f}%")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if compression_ratio > 2.0:
            print("  • Implement adaptive frame sampling for long videos")
            print("  • Consider preserving more frames for natural motion")
        
        if not quality_passed:
            print("  • Improve centering algorithm accuracy")
            print("  • Add post-processing centering correction")
        
        dynamic_fps = temporal_transform.get('dynamic_fps_recommended', 25)
        if dynamic_fps < 15:
            print("  • Use dynamic frame rate to preserve natural timing")
        
        print("\n🎯 NEXT STEPS:")
        if quality_passed and compression_ratio <= 2.0:
            print("  ✅ Ready for 5-video diverse batch testing")
        else:
            print("  🔧 Algorithm improvements needed before batch testing")
            
    else:
        error_msg = report.get("error", "Unknown error")
        print(f"❌ PROCESSING FAILED: {error_msg}")

if __name__ == "__main__":
    analyze_single_video()
