#!/usr/bin/env python3
"""
Test script to debug video format issues and convert WebM to MP4
"""

import cv2
import os
import subprocess
import tempfile

def test_video_file(video_path):
    """Test if OpenCV can properly read a video file"""
    print(f"\n🎬 Testing: {video_path}")
    print("=" * 50)
    
    if not os.path.exists(video_path):
        print("❌ File does not exist")
        return False
    
    # File size
    file_size = os.path.getsize(video_path)
    print(f"📊 File size: {file_size / 1024:.1f} KB")
    
    # Try to open with OpenCV
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("❌ OpenCV failed to open video")
        cap.release()
        return False
    
    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Properties: {width}x{height}, {fps:.2f} FPS, {frame_count} frames")
    
    # Try to read frames
    frames_read = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1
        
        # Safety limit
        if frames_read > 200:
            break
    
    cap.release()
    
    print(f"✅ Successfully read {frames_read} frames")
    
    if frames_read == 0:
        print("❌ No frames could be read")
        return False
    
    return True

def convert_webm_to_mp4(webm_path, mp4_path):
    """Convert WebM to MP4 using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', webm_path, 
            '-c:v', 'libx264', 
            '-c:a', 'aac', 
            '-y',  # Overwrite output
            mp4_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Converted {webm_path} → {mp4_path}")
            return True
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ FFmpeg not found. Install with: brew install ffmpeg")
        return False
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

def main():
    print("🧪 VIDEO FORMAT DEBUGGING")
    print("=" * 40)
    
    # Check debug uploads folder
    debug_folder = "debug_uploads"
    if not os.path.exists(debug_folder):
        print(f"❌ Debug folder not found: {debug_folder}")
        return
    
    # Find latest WebM file
    webm_files = [f for f in os.listdir(debug_folder) if f.endswith('.webm')]
    
    if not webm_files:
        print("❌ No WebM files found in debug_uploads/")
        print("💡 Record a video in the web demo first")
        return
    
    # Test the latest WebM file
    latest_webm = sorted(webm_files)[-1]
    webm_path = os.path.join(debug_folder, latest_webm)
    
    print(f"🎯 Testing latest WebM: {latest_webm}")
    webm_works = test_video_file(webm_path)
    
    if not webm_works:
        print("\n🔄 Attempting WebM → MP4 conversion...")
        mp4_path = webm_path.replace('.webm', '_converted.mp4')
        
        if convert_webm_to_mp4(webm_path, mp4_path):
            print(f"\n🎯 Testing converted MP4:")
            mp4_works = test_video_file(mp4_path)
            
            if mp4_works:
                print("\n✅ SOLUTION: MP4 format works!")
                print("💡 Need to modify backend to convert WebM → MP4")
            else:
                print("\n❌ Even MP4 conversion failed")
        else:
            print("\n❌ WebM → MP4 conversion failed")
    else:
        print("\n✅ WebM format works fine with OpenCV")
        print("💡 The issue might be elsewhere in the pipeline")

if __name__ == "__main__":
    main()
