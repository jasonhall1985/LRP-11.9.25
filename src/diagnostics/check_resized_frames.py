#!/usr/bin/env python3
"""
Check Resized Frames
===================

Quick check to see what the resized frames actually look like.
"""

import cv2
import pathlib
import base64

def check_frame(video_path):
    """Check a single frame from resized video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    # Get middle frame
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None:
        print(f"Frame shape: {frame.shape}")
        print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
        
        # Save as image to check
        cv2.imwrite("sample_resized_frame.jpg", frame)
        print("Saved sample frame as 'sample_resized_frame.jpg'")
        
        return frame
    return None

# Check first resized video
resized_dir = pathlib.Path("resized_dataset")
if resized_dir.exists():
    videos = list(resized_dir.glob("*.mp4"))
    if videos:
        print(f"Checking: {videos[0].name}")
        frame = check_frame(videos[0])
    else:
        print("No videos found in resized_dataset")
else:
    print("resized_dataset directory not found")
