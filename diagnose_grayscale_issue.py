#!/usr/bin/env python3
"""
Diagnostic Script for Grayscale Conversion Issue
===============================================

Quickly diagnose and fix the grayscale conversion problem in visual validation.
Tests frame extraction and conversion on sample videos to identify the root cause.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os

def test_video_loading(video_path: str, max_frames: int = 5):
    """Test video loading and frame extraction."""
    print(f"\n🔍 Testing video: {Path(video_path).name}")
    print(f"📁 Full path: {video_path}")
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"❌ File does not exist!")
        return None
        
    # Get file size
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
    print(f"📊 File size: {file_size:.2f} MB")
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video with OpenCV")
            return None
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video properties:")
        print(f"   - Total frames: {total_frames}")
        print(f"   - FPS: {fps}")
        print(f"   - Dimensions: {width}x{height}")
        
        if total_frames <= 0:
            print(f"❌ No frames detected in video")
            cap.release()
            return None
            
        # Test frame extraction
        frames_tested = []
        frame_indices = [0, total_frames//4, total_frames//2, 3*total_frames//4, total_frames-1]
        frame_indices = [idx for idx in frame_indices if 0 <= idx < total_frames][:max_frames]
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print(f"❌ Could not read frame {frame_idx}")
                continue
                
            print(f"✅ Frame {frame_idx}: shape={frame.shape}, dtype={frame.dtype}")
            print(f"   - Min/Max values: {np.min(frame)} / {np.max(frame)}")
            print(f"   - Mean: {np.mean(frame):.1f}")
            print(f"   - Is color: {len(frame.shape) == 3}")
            
            # Test grayscale conversion
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print(f"   - Grayscale shape: {gray_frame.shape}")
                print(f"   - Grayscale min/max: {np.min(gray_frame)} / {np.max(gray_frame)}")
                print(f"   - Grayscale mean: {np.mean(gray_frame):.1f}")
                
                # Check if conversion actually changed anything
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Check if all channels are identical (already grayscale)
                    r, g, b = frame[:,:,0], frame[:,:,1], frame[:,:,2]
                    if np.array_equal(r, g) and np.array_equal(g, b):
                        print(f"   ⚠️  Original frame has identical RGB channels (already grayscale)")
                    else:
                        print(f"   ✅ Original frame has different RGB channels (true color)")
                        
                frames_tested.append({
                    'frame_idx': frame_idx,
                    'original': frame.copy(),
                    'grayscale': gray_frame.copy()
                })
            else:
                print(f"   ✅ Frame is already grayscale")
                frames_tested.append({
                    'frame_idx': frame_idx,
                    'original': frame.copy(),
                    'grayscale': frame.copy()
                })
                
        cap.release()
        return frames_tested
        
    except Exception as e:
        print(f"❌ Error testing video: {e}")
        return None

def save_test_frames(frames_data, video_name: str, output_dir: str = "./diagnostic_output"):
    """Save test frames for visual inspection."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for i, frame_data in enumerate(frames_data):
        frame_idx = frame_data['frame_idx']
        original = frame_data['original']
        grayscale = frame_data['grayscale']
        
        # Save original frame
        if len(original.shape) == 3:
            # Convert BGR to RGB for saving
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            plt.imsave(output_path / f"{video_name}_frame_{frame_idx}_original.png", original_rgb)
        else:
            plt.imsave(output_path / f"{video_name}_frame_{frame_idx}_original.png", original, cmap='gray')
            
        # Save grayscale frame
        plt.imsave(output_path / f"{video_name}_frame_{frame_idx}_grayscale.png", grayscale, cmap='gray')
        
    print(f"💾 Test frames saved to: {output_path}")

def test_clahe_processing(frame: np.ndarray):
    """Test CLAHE processing on a frame."""
    if frame is None:
        return None
        
    print(f"\n🔧 Testing CLAHE processing:")
    print(f"   - Input shape: {frame.shape}")
    print(f"   - Input range: {np.min(frame)} - {np.max(frame)}")
    print(f"   - Input mean: {np.mean(frame):.1f}")
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    if len(frame.shape) == 3:
        # Convert to grayscale first
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame.copy()
        
    clahe_frame = clahe.apply(gray_frame)
    
    print(f"   - CLAHE output shape: {clahe_frame.shape}")
    print(f"   - CLAHE output range: {np.min(clahe_frame)} - {np.max(clahe_frame)}")
    print(f"   - CLAHE output mean: {np.mean(clahe_frame):.1f}")
    
    # Check if CLAHE actually changed anything
    if np.array_equal(gray_frame, clahe_frame):
        print(f"   ⚠️  CLAHE did not change the frame")
    else:
        print(f"   ✅ CLAHE enhanced the frame")
        
    return clahe_frame

def find_sample_videos(manifest_path: str = "manifest.csv", num_samples: int = 5):
    """Find sample videos from manifest for testing."""
    if not os.path.exists(manifest_path):
        print(f"❌ Manifest not found: {manifest_path}")

        # Try to find videos directly from expanded_cropped_dataset
        expanded_dir = Path("./expanded_cropped_dataset")
        if expanded_dir.exists():
            print(f"🔍 Searching for videos in {expanded_dir}")
            video_files = list(expanded_dir.glob("*.mp4"))[:num_samples]

            sample_videos = []
            for video_path in video_files:
                # Extract class from filename
                filename = video_path.name
                if "doctor" in filename:
                    class_name = "doctor"
                elif "help" in filename:
                    class_name = "help"
                elif "glasses" in filename:
                    class_name = "glasses"
                elif "phone" in filename:
                    class_name = "phone"
                elif "pillow" in filename:
                    class_name = "pillow"
                else:
                    class_name = "unknown"

                sample_videos.append({
                    'path': str(video_path),
                    'class': class_name,
                    'source': 'expanded_cropped_dataset'
                })

            print(f"✅ Found {len(sample_videos)} videos in expanded_cropped_dataset")
            return sample_videos

        return []

    df = pd.read_csv(manifest_path)
    print(f"📋 Loaded manifest with {len(df)} videos")

    # Sample videos from different classes
    sample_videos = []
    classes = df['class'].unique()[:num_samples]

    for class_name in classes:
        class_videos = df[df['class'] == class_name]
        if len(class_videos) > 0:
            sample_video = class_videos.iloc[0]
            sample_videos.append({
                'path': sample_video['path'],
                'class': sample_video['class'],
                'source': sample_video.get('source', 'unknown')
            })

    return sample_videos

def main():
    """Main diagnostic function."""
    print("="*80)
    print("🔍 GRAYSCALE CONVERSION DIAGNOSTIC")
    print("="*80)
    
    # Find sample videos
    print("\n1️⃣ Finding sample videos...")
    sample_videos = find_sample_videos()
    
    if not sample_videos:
        print("❌ No sample videos found. Creating test video...")
        # Create a simple test video
        test_video_path = "./test_diagnostic_video.mp4"
        create_test_video(test_video_path)
        sample_videos = [{'path': test_video_path, 'class': 'test', 'source': 'synthetic'}]
    
    print(f"✅ Found {len(sample_videos)} sample videos")
    
    # Test each video
    all_results = []
    for i, video_info in enumerate(sample_videos):
        print(f"\n2️⃣ Testing video {i+1}/{len(sample_videos)}")
        
        frames_data = test_video_loading(video_info['path'])
        if frames_data:
            all_results.append({
                'video_info': video_info,
                'frames_data': frames_data
            })
            
            # Save test frames
            video_name = Path(video_info['path']).stem
            save_test_frames(frames_data, video_name)
            
            # Test CLAHE on middle frame
            if frames_data:
                middle_frame = frames_data[len(frames_data)//2]['grayscale']
                test_clahe_processing(middle_frame)
    
    # Summary
    print(f"\n3️⃣ DIAGNOSTIC SUMMARY")
    print("="*50)
    
    if all_results:
        print(f"✅ Successfully tested {len(all_results)} videos")
        print(f"💾 Test frames saved to: ./diagnostic_output/")
        print(f"🔍 Check the saved frames to see if they contain visible content")
        
        # Check for common issues
        issues_found = []
        for result in all_results:
            frames_data = result['frames_data']
            for frame_data in frames_data:
                grayscale = frame_data['grayscale']
                
                # Check if frame is all zeros or uniform
                if np.all(grayscale == 0):
                    issues_found.append("All-black frames detected")
                elif np.std(grayscale) < 1:
                    issues_found.append("Very low contrast frames (std < 1)")
                elif np.all(grayscale == grayscale[0,0]):
                    issues_found.append("Uniform gray frames detected")
                    
        if issues_found:
            print(f"\n⚠️  ISSUES DETECTED:")
            for issue in set(issues_found):
                print(f"   - {issue}")
        else:
            print(f"\n✅ No obvious issues detected in frame extraction")
            
    else:
        print(f"❌ Could not test any videos")
        
    print(f"\n🔧 Next steps:")
    print(f"   1. Check ./diagnostic_output/ for saved frame images")
    print(f"   2. Verify that frames show visible mouth/lip regions")
    print(f"   3. If frames are solid gray, check video file integrity")
    print(f"   4. Run fixed visual validation after diagnosis")

def create_test_video(output_path: str):
    """Create a simple test video with visible features."""
    print(f"🎬 Creating test video: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (96, 96))
    
    for frame_idx in range(30):
        # Create a frame with visible features
        frame = np.zeros((96, 96, 3), dtype=np.uint8)
        
        # Add background
        frame[:, :] = [50, 50, 50]
        
        # Add a "face" region
        cv2.rectangle(frame, (20, 20), (76, 76), (100, 100, 100), -1)
        
        # Add "mouth" region that changes
        mouth_y = 60 + int(5 * np.sin(frame_idx * 0.3))
        cv2.ellipse(frame, (48, mouth_y), (15, 8), 0, 0, 180, (150, 150, 150), -1)
        
        # Add some noise
        noise = np.random.randint(0, 30, (96, 96, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
        
    out.release()
    print(f"✅ Test video created")

if __name__ == "__main__":
    main()
