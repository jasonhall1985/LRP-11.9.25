#!/usr/bin/env python3
"""
VISUAL CENTERING INSPECTION TOOL
===============================
Create visual inspection videos to understand centering issues and validate corrections.

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_visual_inspection_video():
    """Create a visual inspection video showing original vs processed frames with centering indicators."""
    
    # Configuration
    SOURCE_DIR = "data/13.9.25top7dataset_cropped"
    OUTPUT_DIR = "data/visual_inspection"
    
    print("👁️  VISUAL CENTERING INSPECTION")
    print("=" * 50)
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select test video
    test_video = Path(SOURCE_DIR) / "pillow__useruser01__65plus__female__caucasian__20250827T062536_topmid.mp4"
    
    if not test_video.exists():
        print(f"❌ Test video not found: {test_video}")
        return
    
    print(f"📹 Analyzing: {test_video.name}")
    
    # Load original video
    cap = cv2.VideoCapture(str(test_video))
    if not cap.isOpened():
        print("❌ Could not open video")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Original Video: {width}x{height}, {fps:.1f} fps")
    
    # Read all frames
    original_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original_frames.append(frame)
    cap.release()
    
    print(f"📊 Loaded {len(original_frames)} frames")
    
    # Load processed videos for comparison
    processed_videos = {
        "Original Ultra-Precise": "data/ultra_precise_centered_96x64/pillow__useruser01__65plus__female__caucasian__20250827T062536_topmid_96x64_ultra_centered.mp4",
        "Temporal Test": "data/temporal_analysis_simple/pillow__useruser01__65plus__female__caucasian__20250827T062536_topmid_temporal_test.mp4",
        "Corrected Test": "data/corrected_preprocessing_test/pillow__useruser01__65plus__female__caucasian__20250827T062536_topmid_corrected_test.mp4"
    }
    
    # Create comparison frames
    comparison_frames = []
    target_center = (48, 32)  # Target center for 96x64 frames
    
    # Sample every 3rd frame for manageable output
    sample_indices = range(0, min(len(original_frames), 90), 3)  # Max 30 frames
    
    for i in sample_indices:
        original_frame = original_frames[i]
        
        # Create comparison layout: Original + 3 processed versions
        # Layout: 2x2 grid
        grid_width = max(width, 96) * 2 + 20  # Padding
        grid_height = max(height, 64) * 2 + 60  # Padding + text space
        
        comparison_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Position 1: Original frame (top-left)
        orig_resized = cv2.resize(original_frame, (width//2, height//2))
        comparison_frame[30:30+height//2, 10:10+width//2] = orig_resized
        
        # Add title
        cv2.putText(comparison_frame, "Original", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Load and display processed versions
        positions = [
            (10 + width//2 + 10, 30, "Ultra-Precise"),
            (10, 30 + height//2 + 30, "Temporal Test"),
            (10 + width//2 + 10, 30 + height//2 + 30, "Corrected")
        ]
        
        processed_names = ["Original Ultra-Precise", "Temporal Test", "Corrected Test"]
        
        for idx, (x, y, title) in enumerate(positions):
            if idx < len(processed_names):
                video_path = processed_videos.get(processed_names[idx])
                
                if video_path and Path(video_path).exists():
                    # Load processed video frame
                    proc_cap = cv2.VideoCapture(video_path)
                    if proc_cap.isOpened():
                        # Seek to corresponding frame (accounting for different frame counts)
                        frame_ratio = i / len(original_frames)
                        proc_frame_count = int(proc_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        target_frame = int(frame_ratio * proc_frame_count)
                        
                        proc_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        ret, proc_frame = proc_cap.read()
                        
                        if ret:
                            # Resize processed frame for display
                            proc_resized = cv2.resize(proc_frame, (96*2, 64*2))  # 2x scale for visibility
                            
                            # Convert grayscale to BGR if needed
                            if len(proc_resized.shape) == 2:
                                proc_resized = cv2.cvtColor(proc_resized, cv2.COLOR_GRAY2BGR)
                            
                            # Place in comparison frame
                            comparison_frame[y:y+64*2, x:x+96*2] = proc_resized
                            
                            # Draw centering crosshairs on processed frame
                            center_x = x + target_center[0] * 2  # Scale target center
                            center_y = y + target_center[1] * 2
                            
                            # Green crosshairs for target center
                            cv2.line(comparison_frame, (center_x-10, center_y), (center_x+10, center_y), (0, 255, 0), 2)
                            cv2.line(comparison_frame, (center_x, center_y-10), (center_x, center_y+10), (0, 255, 0), 2)
                            cv2.circle(comparison_frame, (center_x, center_y), 3, (0, 255, 0), -1)
                        
                        proc_cap.release()
                
                # Add title
                cv2.putText(comparison_frame, title, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add frame number
        cv2.putText(comparison_frame, f"Frame {i+1}/{len(original_frames)}", 
                   (10, grid_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        comparison_frames.append(comparison_frame)
    
    # Save comparison video
    if comparison_frames:
        output_video_path = output_path / "centering_comparison_analysis.mp4"
        
        # Get dimensions from first frame
        h, w = comparison_frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, 2.0, (w, h))  # 2 fps for inspection
        
        for frame in comparison_frames:
            out.write(frame)
        
        out.release()
        
        print(f"✅ VISUAL INSPECTION VIDEO CREATED:")
        print(f"📁 {output_video_path}")
        print(f"📊 {len(comparison_frames)} comparison frames at 2 fps")
        print(f"🎯 Green crosshairs show target lip center (48, 32)")
        print()
        print("👁️  INSPECTION GUIDE:")
        print("  • Original: Shows source video frames")
        print("  • Ultra-Precise: Current ultra-precise algorithm result")
        print("  • Temporal Test: Dynamic FPS temporal test result")
        print("  • Corrected: Improved algorithm result")
        print("  • Green crosshairs: Target lip center position")
        print()
        print("🔍 LOOK FOR:")
        print("  • Are lips positioned at green crosshairs?")
        print("  • Is cropping too tight or too loose?")
        print("  • Are lips cut off during speech movements?")
        print("  • Is temporal motion smooth and natural?")
        
        # Create individual frame samples for detailed inspection
        sample_frames_dir = output_path / "sample_frames"
        sample_frames_dir.mkdir(exist_ok=True)
        
        # Save every 5th comparison frame as individual images
        for i, frame in enumerate(comparison_frames[::5]):
            frame_path = sample_frames_dir / f"comparison_frame_{i*5:03d}.jpg"
            cv2.imwrite(str(frame_path), frame)
        
        print(f"📸 Sample frames saved to: {sample_frames_dir}")
        
    else:
        print("❌ No comparison frames created")

def analyze_lip_positioning():
    """Analyze lip positioning in the original video to understand the challenge."""
    
    SOURCE_DIR = "data/13.9.25top7dataset_cropped"
    test_video = Path(SOURCE_DIR) / "pillow__useruser01__65plus__female__caucasian__20250827T062536_topmid.mp4"
    
    print("\n🔍 ANALYZING LIP POSITIONING IN ORIGINAL VIDEO")
    print("=" * 50)
    
    if not test_video.exists():
        print(f"❌ Test video not found")
        return
    
    cap = cv2.VideoCapture(str(test_video))
    if not cap.isOpened():
        print("❌ Could not open video")
        return
    
    # Analyze first few frames
    frame_count = 0
    lip_positions = []
    
    while frame_count < 10:  # Analyze first 10 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Simple lip detection: find darkest region in lower 40% of frame
        roi = gray[int(h*0.6):, :]  # Bottom 40%
        
        if roi.size > 0:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi)
            
            # Adjust coordinates for full frame
            lip_x = min_loc[0]
            lip_y = min_loc[1] + int(h*0.6)
            
            lip_positions.append((lip_x, lip_y))
            
            print(f"Frame {frame_count+1}: Estimated lip center at ({lip_x}, {lip_y}) in {w}x{h} frame")
        
        frame_count += 1
    
    cap.release()
    
    if lip_positions:
        # Calculate statistics
        avg_x = np.mean([pos[0] for pos in lip_positions])
        avg_y = np.mean([pos[1] for pos in lip_positions])
        
        print(f"\n📊 LIP POSITIONING ANALYSIS:")
        print(f"  Average lip center: ({avg_x:.1f}, {avg_y:.1f})")
        print(f"  Frame dimensions: {w}x{h}")
        print(f"  Lip position ratio: ({avg_x/w:.2f}, {avg_y/h:.2f})")
        
        # For ICU-style videos, lips should be in upper portion
        if avg_y/h > 0.7:
            print("  ⚠️  Lips appear to be in lower portion of frame")
            print("  💡 This suggests the video is NOT ICU-style cropped")
        else:
            print("  ✅ Lips appear to be in upper portion (ICU-style)")

if __name__ == "__main__":
    create_visual_inspection_video()
    analyze_lip_positioning()
