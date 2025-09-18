#!/usr/bin/env python3
"""
SINGLE VIDEO TEMPORAL DIAGNOSTIC TEST
====================================
Comprehensive temporal analysis and quality control testing on one representative video
to validate ultra-precise lip centering with improved temporal sampling.

Author: Augment Agent
Date: 2025-09-17
"""

import os
import sys
import cv2
import numpy as np
import time
import logging
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import traceback

# Add current directory to Python path for imports
sys.path.append('/Users/client/Desktop/LRP classifier 11.9.25')

from ultra_precise_lip_centering import UltraPreciseLipCenterPreprocessor

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('single_video_temporal_diagnostic.log'),
        logging.StreamHandler()
    ]
)

def select_representative_video(source_dir: str) -> Optional[Path]:
    """Select a representative medium-duration video for testing."""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Get all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    video_files = []
    
    for file_path in source_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    if not video_files:
        return None
    
    # Analyze video durations to find medium-duration video (3-4 seconds)
    target_duration_range = (3.0, 4.0)
    best_candidate = None
    best_score = float('inf')
    
    for video_file in video_files[:50]:  # Check first 50 videos for efficiency
        try:
            cap = cv2.VideoCapture(str(video_file))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                # Score based on how close to target range
                if target_duration_range[0] <= duration <= target_duration_range[1]:
                    score = abs(duration - 3.5)  # Prefer 3.5 seconds
                    if score < best_score:
                        best_score = score
                        best_candidate = video_file
                        
        except Exception as e:
            logging.warning(f"Could not analyze {video_file.name}: {str(e)}")
            continue
    
    # If no medium-duration video found, pick first available
    if best_candidate is None:
        best_candidate = video_files[0]
    
    logging.info(f"Selected representative video: {best_candidate.name}")
    return best_candidate

def create_temporal_analysis_report(video_path: Path, report: Dict) -> str:
    """Create comprehensive temporal analysis report."""
    
    report_lines = [
        "=" * 80,
        "SINGLE VIDEO TEMPORAL DIAGNOSTIC REPORT",
        "=" * 80,
        f"Video: {video_path.name}",
        f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "ORIGINAL VIDEO METRICS:",
        f"  Frame Count: {report['temporal_metrics']['original_frame_count']} frames",
        f"  Duration: {report['temporal_metrics']['actual_duration_seconds']:.3f} seconds",
        f"  Frame Rate: {report['temporal_metrics']['original_fps']:.2f} fps",
        f"  Total Frames (property): {report['temporal_metrics'].get('total_frames_property', 'N/A')}",
        "",
        "TEMPORAL TRANSFORMATION ANALYSIS:",
        f"  Output Frame Count: 32 frames (fixed)",
        f"  Compression Ratio: {report['temporal_transformation']['compression_ratio']:.3f}x",
        f"  Frame Skip Interval: {report['temporal_transformation']['frame_skip_interval']:.2f}",
        f"  Target Output Duration (25fps): {report['temporal_transformation']['target_output_duration']:.3f}s",
        f"  Recommended Dynamic FPS: {report['temporal_transformation']['dynamic_fps_recommended']:.1f} fps",
        "",
        "CENTERING ACCURACY VALIDATION:",
        f"  Frames Checked: {report['quality_control']['frames_checked']}",
        f"  Perfect Centering Count: {report['quality_control']['perfect_centering_count']}",
        f"  Mean Deviation: {report['quality_control']['mean_deviation']:.3f} pixels",
        f"  Max Deviation: {report['quality_control']['max_deviation']:.3f} pixels",
        f"  Quality Passed (>90% frames): {report['quality_control']['quality_passed']}",
        f"  Within 10% Threshold: {report['quality_control']['deviation_threshold_10_percent']}",
        "",
        "PROCESSING STATISTICS:",
        f"  Method Used: {list(report['frame_consistency']['method_distribution'].keys())}",
        f"  Processing Success: {report.get('processing_success', False)}",
        "",
        "TEMPORAL SAMPLING ASSESSMENT:",
    ]
    
    # Assess temporal sampling quality
    compression_ratio = report['temporal_transformation']['compression_ratio']
    if compression_ratio > 2.0:
        report_lines.append(f"  ⚠️  HIGH COMPRESSION: {compression_ratio:.2f}x may cause temporal artifacts")
    elif compression_ratio < 0.5:
        report_lines.append(f"  ⚠️  HIGH EXPANSION: {compression_ratio:.2f}x may cause slow motion effect")
    else:
        report_lines.append(f"  ✅ ACCEPTABLE COMPRESSION: {compression_ratio:.2f}x within reasonable range")
    
    # Frame rate recommendations
    dynamic_fps = report['temporal_transformation']['dynamic_fps_recommended']
    if dynamic_fps < 15:
        report_lines.append(f"  ⚠️  LOW DYNAMIC FPS: {dynamic_fps:.1f} fps may appear slow")
    elif dynamic_fps > 50:
        report_lines.append(f"  ⚠️  HIGH DYNAMIC FPS: {dynamic_fps:.1f} fps may appear fast")
    else:
        report_lines.append(f"  ✅ OPTIMAL DYNAMIC FPS: {dynamic_fps:.1f} fps preserves natural timing")
    
    report_lines.extend([
        "",
        "QUALITY CONTROL ASSESSMENT:",
    ])
    
    if report['quality_control']['quality_passed']:
        report_lines.append("  ✅ CENTERING QUALITY: PASSED - Ready for main dataset")
    else:
        report_lines.append("  ❌ CENTERING QUALITY: FAILED - Requires quality review")
    
    report_lines.extend([
        "",
        "RECOMMENDATIONS:",
    ])
    
    if compression_ratio > 1.5:
        report_lines.append("  • Consider preserving more original frames for long videos")
    if dynamic_fps > 40:
        report_lines.append("  • Use dynamic frame rate to maintain natural speech timing")
    if not report['quality_control']['quality_passed']:
        report_lines.append("  • Video should be moved to quality review folder")
    
    report_lines.extend([
        "",
        "=" * 80
    ])
    
    return "\n".join(report_lines)

def main():
    """Main diagnostic function for single video temporal analysis."""
    
    # Configuration
    SOURCE_DIR = "data/13.9.25top7dataset_cropped"
    OUTPUT_DIR = "data/single_video_diagnostic"
    
    print("🔍 SINGLE VIDEO TEMPORAL DIAGNOSTIC TEST")
    print("=" * 60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Objective: Validate temporal sampling and centering accuracy")
    print()
    
    try:
        # Create output directory
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Select representative video
        print("🎯 Selecting representative video...")
        selected_video = select_representative_video(SOURCE_DIR)
        
        if selected_video is None:
            print("❌ No suitable video found for testing!")
            return
        
        print(f"✅ Selected: {selected_video.name}")
        print()
        
        # Initialize ultra-precise preprocessor
        print("🔧 Initializing ultra-precise preprocessor...")
        preprocessor = UltraPreciseLipCenterPreprocessor()
        
        # Process single video with comprehensive analysis
        print("📊 PROCESSING WITH COMPREHENSIVE TEMPORAL ANALYSIS")
        print("=" * 60)
        
        start_time = time.time()
        
        # Process the video
        processed_frames, report = preprocessor.process_video_with_ultra_precision_and_temporal_analysis(str(selected_video))
        
        processing_time = time.time() - start_time
        
        if processed_frames is not None:
            # Save processed video
            output_filename = f"{selected_video.stem}_diagnostic_96x64_ultra_centered.mp4"
            output_file_path = output_path / output_filename
            
            # Save as numpy array first
            temp_npy = output_path / f"temp_{selected_video.stem}.npy"
            np.save(temp_npy, processed_frames)
            
            # Convert to MP4 with dynamic frame rate
            original_duration = report['temporal_metrics']['actual_duration_seconds']
            success = preprocessor.npy_to_mp4_ffmpeg_with_dynamic_fps(
                str(temp_npy), str(output_file_path), original_duration
            )
            
            # Clean up temporary file
            if temp_npy.exists():
                temp_npy.unlink()
            
            if success:
                print(f"✅ DIAGNOSTIC VIDEO SAVED: {output_filename}")
                print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
                print()
                
                # Generate comprehensive report
                report_text = create_temporal_analysis_report(selected_video, report)
                
                # Save report to file
                report_file = output_path / f"{selected_video.stem}_temporal_analysis_report.txt"
                with open(report_file, 'w') as f:
                    f.write(report_text)
                
                # Save JSON report for programmatic analysis (convert numpy types)
                json_report_file = output_path / f"{selected_video.stem}_temporal_analysis_report.json"

                def convert_numpy_types(obj):
                    """Convert numpy types to native Python types for JSON serialization."""
                    if isinstance(obj, (np.integer, np.int32, np.int64)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    elif isinstance(obj, bool):
                        return obj
                    else:
                        return obj

                json_compatible_report = convert_numpy_types(report)

                with open(json_report_file, 'w') as f:
                    json.dump(json_compatible_report, f, indent=2)
                
                # Display report
                print(report_text)
                
                # Print file paths
                abs_output_path = output_path.resolve()
                print(f"\n📁 DIAGNOSTIC OUTPUT FILES:")
                print(f"   Video: {abs_output_path / output_filename}")
                print(f"   Report: {abs_output_path / report_file.name}")
                print(f"   JSON Data: {abs_output_path / json_report_file.name}")
                
                # Quality control assessment
                if report['quality_control']['quality_passed']:
                    print(f"\n🎉 QUALITY CONTROL: PASSED")
                    print("✅ Video meets centering accuracy standards")
                    print("✅ Ready to proceed with improved temporal sampling")
                else:
                    print(f"\n⚠️  QUALITY CONTROL: ATTENTION REQUIRED")
                    print("❌ Centering accuracy below threshold")
                    print("🔧 Algorithm refinement needed before full dataset processing")
                
            else:
                print(f"❌ FAILED to save diagnostic video")
        else:
            error_msg = report.get("error", "Unknown error")
            print(f"❌ PROCESSING FAILED: {error_msg}")
            
            # Save error report
            error_report_file = output_path / f"{selected_video.stem}_error_report.txt"
            with open(error_report_file, 'w') as f:
                f.write(f"Processing Error Report\n")
                f.write(f"Video: {selected_video.name}\n")
                f.write(f"Error: {error_msg}\n")
                f.write(f"Temporal Metrics: {report.get('temporal_metrics', {})}\n")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        logging.error(f"Critical error in diagnostic test: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
