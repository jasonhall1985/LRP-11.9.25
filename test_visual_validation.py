#!/usr/bin/env python3
"""
Test script for visual validation tool
=====================================

Quick test to verify the visual validation tool works correctly
without requiring the full dataset.
"""

import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil

def create_test_manifest_and_videos():
    """Create a small test manifest and sample videos for testing."""
    print("Creating test data...")
    
    # Create temporary directory
    test_dir = Path("./test_validation_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create test videos directory
    videos_dir = test_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    
    # Test data
    test_videos = []
    classes = ["help", "doctor", "glasses", "phone", "pillow"]
    genders = ["male", "female"]
    age_bands = ["18-39", "40-64"]
    ethnicities = ["caucasian", "asian", "african"]
    
    # Create 15 test videos
    for i in range(15):
        class_name = classes[i % len(classes)]
        gender = genders[i % len(genders)]
        age_band = age_bands[i % len(age_bands)]
        ethnicity = ethnicities[i % len(ethnicities)]
        
        # Create video filename
        video_name = f"test_video_{i:03d}_{class_name}_{gender}_{age_band}_{ethnicity}.mp4"
        video_path = videos_dir / video_name
        
        # Create a simple test video (96x96, grayscale, 30 frames)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (96, 96), False)
        
        # Generate 30 frames with some variation
        for frame_idx in range(30):
            # Create a simple pattern that varies by frame
            frame = np.zeros((96, 96), dtype=np.uint8)
            
            # Add some patterns to simulate mouth region
            center_y, center_x = 48, 48
            
            # Draw a simple "mouth" shape
            cv2.ellipse(frame, (center_x, center_y + 10), (20, 8), 0, 0, 180, 128, -1)
            
            # Add some noise and variation
            noise = np.random.randint(0, 50, (96, 96), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            
            # Vary brightness by frame to simulate talking
            brightness = int(20 * np.sin(frame_idx * 0.5) + 50)
            frame = cv2.add(frame, brightness)
            
            # Apply CLAHE to some videos (simulate processing)
            if i % 3 == 0:  # Every 3rd video gets CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                frame = clahe.apply(frame)
            
            out.write(frame)
            
        out.release()
        
        # Add to manifest data
        test_videos.append({
            'path': str(video_path),
            'class': class_name,
            'gender': gender,
            'age_band': age_band,
            'ethnicity': ethnicity,
            'source': 'test_data',
            'processed_version': 'original' if i % 2 == 0 else 'v3',
            'duration_frames': 30,
            'fps': 10.0,
            'width': 96,
            'height': 96
        })
    
    # Create manifest CSV
    manifest_df = pd.DataFrame(test_videos)
    manifest_path = test_dir / "test_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    
    print(f"Created {len(test_videos)} test videos")
    print(f"Test manifest: {manifest_path}")
    
    return str(manifest_path)

def run_test():
    """Run the visual validation test."""
    print("="*60)
    print("🧪 Testing Visual Validation Tool")
    print("="*60)
    
    try:
        # Create test data
        manifest_path = create_test_manifest_and_videos()
        
        # Import and run visual validation
        from visual_validation import VisualValidator
        
        print("\nRunning visual validation on test data...")
        validator = VisualValidator(manifest_path, "./test_validation_output")
        
        # Run with small sample size
        html_path = validator.run_validation(total_samples=10)
        
        print("\n" + "="*60)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 HTML Visualization: {html_path}")
        print(f"📋 Validation Report: {validator.output_dir}/validation_report.json")
        print(f"📝 Summary: {validator.output_dir}/VALIDATION_SUMMARY.md")
        
        # Check if files were created
        output_dir = Path("./test_validation_output")
        if (output_dir / "visual_validation.html").exists():
            print("✅ HTML visualization created")
        if (output_dir / "validation_report.json").exists():
            print("✅ JSON report created")
        if (output_dir / "VALIDATION_SUMMARY.md").exists():
            print("✅ Summary report created")
            
        print(f"\n🔍 Found {len(validator.quality_issues)} quality issues in test data")
        
        # Open in browser
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
        print("🌐 Opened visualization in browser")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup test data
        test_dir = Path("./test_validation_data")
        if test_dir.exists():
            print(f"\n🧹 Cleaning up test data: {test_dir}")
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    success = run_test()
    exit(0 if success else 1)
