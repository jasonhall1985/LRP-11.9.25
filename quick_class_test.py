#!/usr/bin/env python3
"""
Quick test to see if model can predict all 4 classes with direct API calls
"""

import requests
import json
import numpy as np
import tempfile
import cv2
import os

def create_test_video(pattern_name, output_path):
    """Create a test video with different visual patterns"""
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (96, 64))
    
    if pattern_name == "mouth_dry":
        # Pattern for "my mouth is dry" - static mouth, minimal movement
        for frame in range(90):  # 3 seconds at 30fps
            img = np.zeros((64, 96, 3), dtype=np.uint8)
            # Static mouth region
            cv2.rectangle(img, (35, 25), (60, 40), (128, 128, 128), -1)
            out.write(img)
            
    elif pattern_name == "need_move":
        # Pattern for "i need to move" - more dynamic movement
        for frame in range(90):
            img = np.zeros((64, 96, 3), dtype=np.uint8)
            # Moving mouth region
            y_offset = int(5 * np.sin(frame * 0.2))
            cv2.rectangle(img, (35, 25 + y_offset), (60, 40 + y_offset), (180, 180, 180), -1)
            out.write(img)
            
    elif pattern_name == "doctor":
        # Pattern for "doctor" - specific mouth shape
        for frame in range(90):
            img = np.zeros((64, 96, 3), dtype=np.uint8)
            # Oval mouth shape
            cv2.ellipse(img, (48, 32), (12, 8), 0, 0, 360, (200, 200, 200), -1)
            out.write(img)
            
    elif pattern_name == "pillow":
        # Pattern for "pillow" - different mouth shape
        for frame in range(90):
            img = np.zeros((64, 96, 3), dtype=np.uint8)
            # Rectangular mouth shape
            cv2.rectangle(img, (40, 28), (55, 36), (150, 150, 150), -1)
            out.write(img)
    
    out.release()
    return output_path

def test_api_prediction(video_path, expected_class):
    """Test API prediction for a video file"""
    API_URL = 'http://192.168.1.100:5000'
    
    try:
        with open(video_path, 'rb') as f:
            files = {'video': f}
            response = requests.post(f'{API_URL}/predict', files=files, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    top1 = data['top2'][0]
                    top2 = data['top2'][1]
                    
                    print(f"📊 {expected_class:15} → {top1['class']:15} ({top1['confidence']:.3f}), {top2['class']:15} ({top2['confidence']:.3f})")
                    
                    # Check if expected class is in top 2
                    predicted_classes = [pred['class'] for pred in data['top2']]
                    correct = expected_class in predicted_classes
                    
                    return correct, top1['class'], top1['confidence']
                else:
                    print(f"❌ {expected_class:15} → API Error: {data.get('error', 'Unknown')}")
                    return False, None, 0
            else:
                print(f"❌ {expected_class:15} → HTTP Error: {response.status_code}")
                return False, None, 0
                
    except Exception as e:
        print(f"❌ {expected_class:15} → Request failed: {e}")
        return False, None, 0

def main():
    print("🧪 QUICK CLASS PREDICTION TEST")
    print("=" * 50)
    print("Testing if model can predict all 4 classes with synthetic videos")
    print()
    
    # Test cases
    test_cases = [
        ("my_mouth_is_dry", "mouth_dry"),
        ("i_need_to_move", "need_move"), 
        ("doctor", "doctor"),
        ("pillow", "pillow")
    ]
    
    results = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        for expected_class, pattern in test_cases:
            # Create test video
            video_path = os.path.join(temp_dir, f"{pattern}_test.mp4")
            create_test_video(pattern, video_path)
            
            # Test prediction
            correct, predicted, confidence = test_api_prediction(video_path, expected_class)
            results.append((expected_class, predicted, confidence, correct))
            
            # Clean up
            if os.path.exists(video_path):
                os.remove(video_path)
    
    finally:
        # Clean up temp directory
        try:
            os.rmdir(temp_dir)
        except:
            pass
    
    # Analysis
    print(f"\n🎯 RESULTS ANALYSIS:")
    print("=" * 30)
    
    unique_predictions = set(result[1] for result in results if result[1])
    correct_predictions = sum(1 for result in results if result[3])
    
    print(f"✅ Correct predictions: {correct_predictions}/4")
    print(f"🎯 Unique classes predicted: {len(unique_predictions)}")
    print(f"📊 Classes predicted: {unique_predictions}")
    
    if len(unique_predictions) < 4:
        missing_classes = {'my_mouth_is_dry', 'i_need_to_move', 'doctor', 'pillow'} - unique_predictions
        print(f"❌ Never predicted: {missing_classes}")
        print(f"\n💡 DIAGNOSIS: Model has class collapse!")
        print(f"   The model is biased toward: {unique_predictions}")
        print(f"   This explains why you only see 'doctor' and 'pillow' predictions.")
        
        print(f"\n🔧 SOLUTIONS:")
        print(f"1. Retrain with balanced class weights")
        print(f"2. Use a different checkpoint with better class balance")
        print(f"3. Apply class-specific thresholds")
        print(f"4. Use ensemble methods")
    else:
        print(f"✅ Model can predict all 4 classes!")
        print(f"   The issue might be with real video preprocessing.")

if __name__ == "__main__":
    main()
