#!/usr/bin/env python3
"""
Inspect Video Frames - Visual Analysis
=====================================

Extract and save sample frames from the video to understand the content
and why MediaPipe might be failing to detect faces.

Author: Augment Agent
Date: 2025-09-14
"""

import cv2
import numpy as np
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_video_frames(video_path: str, output_dir: str = "frame_inspection", num_frames: int = 10):
    """
    Extract and save sample frames from video for visual inspection.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        num_frames: Number of frames to extract
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"Inspecting video: {video_path}")
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties:")
    logger.info(f"  Dimensions: {width}x{height}")
    logger.info(f"  FPS: {fps:.2f}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Duration: {total_frames/fps:.2f} seconds")
    
    # Extract frames at regular intervals
    frame_interval = max(1, total_frames // num_frames)
    
    for i in range(num_frames):
        frame_idx = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save original frame
        frame_path = output_path / f"frame_{frame_idx:04d}_original.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        # Create analysis overlay
        analysis_frame = frame.copy()
        
        # Add grid lines to show regions
        h, w = frame.shape[:2]
        
        # Vertical thirds
        cv2.line(analysis_frame, (w//3, 0), (w//3, h), (0, 255, 0), 1)
        cv2.line(analysis_frame, (2*w//3, 0), (2*w//3, h), (0, 255, 0), 1)
        
        # Horizontal thirds  
        cv2.line(analysis_frame, (0, h//3), (w, h//3), (0, 255, 0), 1)
        cv2.line(analysis_frame, (0, 2*h//3), (w, 2*h//3), (0, 255, 0), 1)
        
        # Add frame info
        cv2.putText(analysis_frame, f"Frame {frame_idx}", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(analysis_frame, f"{w}x{h}", (5, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Highlight expected lip region (top-middle based on ICU dataset description)
        lip_region_x = w // 3
        lip_region_y = 0
        lip_region_w = w // 3
        lip_region_h = h // 2
        
        cv2.rectangle(analysis_frame, 
                     (lip_region_x, lip_region_y), 
                     (lip_region_x + lip_region_w, lip_region_y + lip_region_h), 
                     (0, 0, 255), 2)
        cv2.putText(analysis_frame, "Expected Lip Region", 
                   (lip_region_x, lip_region_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Save analysis frame
        analysis_path = output_path / f"frame_{frame_idx:04d}_analysis.jpg"
        cv2.imwrite(str(analysis_path), analysis_frame)
        
        logger.info(f"Saved frame {frame_idx}: {frame_path}")
        
    cap.release()
    logger.info(f"Frame inspection complete. Check {output_dir}/ for results.")


def main():
    """Main function."""
    video_path = "/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped/doctor__useruser01__18to39__female__aboriginal__20250807T054104_topmid.mp4"
    
    inspect_video_frames(video_path, "frame_inspection", num_frames=15)


if __name__ == "__main__":
    main()
