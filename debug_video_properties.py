#!/usr/bin/env python3
"""
Debug Video Properties
======================
Analyze the properties of original videos to understand color channels and format.
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_video_properties(video_path):
    """Analyze video properties including color channels."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read first frame to check color channels
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'frame_shape': frame.shape,
        'frame_dtype': frame.dtype,
        'is_grayscale': len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1),
        'channels': frame.shape[2] if len(frame.shape) == 3 else 1
    }

def main():
    # Test a few sample videos
    source_dir = Path("data/the_best_videos_so_far")
    
    print("🔍 DEBUGGING VIDEO PROPERTIES")
    print("=" * 60)
    
    # Get sample videos from different classes
    sample_videos = []
    for video_file in source_dir.glob("*.mp4"):
        if len(sample_videos) < 5:
            sample_videos.append(video_file)
    
    for video_path in sample_videos:
        print(f"\n📹 {video_path.name}")
        print("-" * 40)
        
        props = analyze_video_properties(video_path)
        if props:
            print(f"Dimensions: {props['width']}×{props['height']}")
            print(f"FPS: {props['fps']:.1f}")
            print(f"Frame Count: {props['frame_count']}")
            print(f"Frame Shape: {props['frame_shape']}")
            print(f"Frame Type: {props['frame_dtype']}")
            print(f"Channels: {props['channels']}")
            print(f"Is Grayscale: {props['is_grayscale']}")
        else:
            print("❌ Could not analyze video")

if __name__ == "__main__":
    main()
