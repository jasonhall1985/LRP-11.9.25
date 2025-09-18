#!/usr/bin/env python3
"""
CREATE CENTERING PREVIEW VIDEO
==============================
Creates a preview video showing the enhanced centering results with visual indicators.

Author: Augment Agent
Date: 2025-09-17
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random

def create_centering_preview():
    """Create a preview video showing enhanced centering results."""
    
    # Configuration
    INPUT_DIR = "data/enhanced_centered_96x64"
    OUTPUT_PATH = "enhanced_centering_preview.mp4"
    
    print("🎬 CREATING ENHANCED CENTERING PREVIEW")
    print("=" * 50)
    
    # Get all processed videos
    input_path = Path(INPUT_DIR)
    video_files = list(input_path.glob("*.mp4"))
    
    if len(video_files) == 0:
        print("❌ No processed videos found!")
        return
    
    # Select 5 random videos for preview
    selected_videos = random.sample(video_files, min(5, len(video_files)))
    
    print(f"📹 Creating preview from {len(selected_videos)} videos:")
    for video in selected_videos:
        print(f"   - {video.name}")
    
    # Create preview frames
    preview_frames = []
    target_center = (48, 32)  # Dead center coordinates
    
    for video_path in selected_videos:
        # Load video
        cap = cv2.VideoCapture(str(video_path))
        
        # Read first frame
        ret, frame = cap.read()
        if ret:
            # Convert to RGB for display
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            # Draw centering indicators
            # Draw crosshair at target center (48, 32)
            cv2.line(frame_rgb, (target_center[0]-10, target_center[1]), 
                    (target_center[0]+10, target_center[1]), (255, 0, 0), 2)  # Red horizontal line
            cv2.line(frame_rgb, (target_center[0], target_center[1]-10), 
                    (target_center[0], target_center[1]+10), (255, 0, 0), 2)  # Red vertical line
            
            # Draw center point
            cv2.circle(frame_rgb, target_center, 3, (255, 0, 0), -1)  # Red dot
            
            # Add text label
            video_name = video_path.stem[:30]  # Truncate long names
            cv2.putText(frame_rgb, video_name, (5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add centering info
            cv2.putText(frame_rgb, f"Target: {target_center}", (5, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            
            preview_frames.append(frame_rgb)
        
        cap.release()
    
    if len(preview_frames) == 0:
        print("❌ No frames extracted for preview!")
        return
    
    # Create side-by-side grid
    rows = 1
    cols = len(preview_frames)
    
    # Resize frames for grid display
    frame_height, frame_width = 64, 96
    grid_width = frame_width * cols
    grid_height = frame_height * rows
    
    # Create grid
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    for i, frame in enumerate(preview_frames):
        row = i // cols
        col = i % cols
        
        y_start = row * frame_height
        y_end = y_start + frame_height
        x_start = col * frame_width
        x_end = x_start + frame_width
        
        grid[y_start:y_end, x_start:x_end] = frame
    
    # Scale up for better visibility
    scale_factor = 4
    grid_scaled = cv2.resize(grid, (grid_width * scale_factor, grid_height * scale_factor), 
                            interpolation=cv2.INTER_NEAREST)
    
    # Convert back to BGR for OpenCV
    grid_bgr = cv2.cvtColor(grid_scaled, cv2.COLOR_RGB2BGR)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 2.0, 
                         (grid_width * scale_factor, grid_height * scale_factor))
    
    # Write frames (hold each frame for 2 seconds at 2 fps = 4 frames each)
    for _ in range(8):  # 4 seconds total
        out.write(grid_bgr)
    
    out.release()
    
    print(f"✅ Preview created: {OUTPUT_PATH}")
    print(f"📐 Grid size: {grid_width * scale_factor}×{grid_height * scale_factor}")
    print(f"🎯 Red crosshairs show target center (48, 32)")
    print(f"📊 {len(preview_frames)} sample videos displayed")
    
    # Print absolute path
    abs_path = Path(OUTPUT_PATH).resolve()
    print(f"📁 EXACT PATH: {abs_path}")

if __name__ == "__main__":
    create_centering_preview()
