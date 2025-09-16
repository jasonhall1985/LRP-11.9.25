#!/usr/bin/env python3
"""
Video Loading Diagnostics Script
Analyzes video loading failures and dataset integrity issues.
"""

import pandas as pd
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import json
from collections import defaultdict
import traceback

def test_video_loading(video_path, max_frames=32):
    """Test if a video can be loaded and processed properly."""
    try:
        if not os.path.exists(video_path):
            return False, "File does not exist"
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if frame_count == 0:
            cap.release()
            return False, "Video has 0 frames"
        
        # Try to read first few frames
        frames_read = 0
        for i in range(min(5, frame_count)):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return False, f"Cannot read frame {i}"
            if frame is None:
                cap.release()
                return False, f"Frame {i} is None"
            frames_read += 1
        
        cap.release()
        
        info = {
            'frame_count': frame_count,
            'fps': fps,
            'width': width,
            'height': height,
            'frames_read': frames_read
        }
        
        return True, info
        
    except Exception as e:
        return False, f"Exception: {str(e)}"

def analyze_dataset_loading():
    """Analyze video loading issues in the dataset."""
    print("=== VIDEO LOADING DIAGNOSTICS ===")
    
    # Load manifest
    manifest_path = "clean_balanced_manifest.csv"
    if not os.path.exists(manifest_path):
        print(f"ERROR: Manifest file {manifest_path} not found!")
        return
    
    df = pd.read_csv(manifest_path)
    print(f"Total videos in manifest: {len(df)}")
    
    # Test video loading
    loading_results = []
    failed_videos = []
    successful_videos = []
    
    print("\nTesting video loading...")
    for idx, row in df.iterrows():
        video_path = row['path']
        success, result = test_video_loading(video_path)
        
        loading_results.append({
            'index': idx,
            'path': video_path,
            'class': row['class'],
            'success': success,
            'result': result
        })
        
        if success:
            successful_videos.append((idx, video_path, row['class'], result))
        else:
            failed_videos.append((idx, video_path, row['class'], result))
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} videos...")
    
    print(f"\nRESULTS:")
    print(f"  Successful: {len(successful_videos)}")
    print(f"  Failed: {len(failed_videos)}")
    print(f"  Success rate: {len(successful_videos)/len(df)*100:.1f}%")
    
    # Analyze failures
    if failed_videos:
        print(f"\nFAILURE ANALYSIS:")
        failure_reasons = defaultdict(int)
        failure_by_class = defaultdict(int)
        
        for idx, path, cls, reason in failed_videos:
            failure_reasons[reason] += 1
            failure_by_class[cls] += 1
        
        print("  Failure reasons:")
        for reason, count in failure_reasons.items():
            print(f"    {reason}: {count}")
        
        print("  Failures by class:")
        for cls, count in failure_by_class.items():
            print(f"    {cls}: {count}")
        
        print(f"\nFirst 10 failed videos:")
        for i, (idx, path, cls, reason) in enumerate(failed_videos[:10]):
            print(f"  {i+1}. Index {idx} ({cls}): {reason}")
            print(f"     Path: {path}")
    
    # Analyze successful videos
    if successful_videos:
        print(f"\nSUCCESSFUL VIDEO ANALYSIS:")
        frame_counts = []
        dimensions = []
        fps_values = []
        
        for idx, path, cls, info in successful_videos:
            frame_counts.append(info['frame_count'])
            dimensions.append((info['width'], info['height']))
            fps_values.append(info['fps'])
        
        print(f"  Frame count stats:")
        print(f"    Min: {min(frame_counts)}, Max: {max(frame_counts)}, Mean: {np.mean(frame_counts):.1f}")
        
        print(f"  Unique dimensions:")
        unique_dims = list(set(dimensions))
        for dim in unique_dims[:10]:  # Show first 10
            count = dimensions.count(dim)
            print(f"    {dim[0]}x{dim[1]}: {count} videos")
        
        print(f"  FPS stats:")
        print(f"    Min: {min(fps_values):.1f}, Max: {max(fps_values):.1f}, Mean: {np.mean(fps_values):.1f}")
    
    # Save detailed results
    results_file = "video_loading_diagnostics.json"
    with open(results_file, 'w') as f:
        json.dump(loading_results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    
    return loading_results, failed_videos, successful_videos

if __name__ == "__main__":
    analyze_dataset_loading()
