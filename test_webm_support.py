#!/usr/bin/env python3
"""
Test script to verify WebM support in OpenCV
"""

import cv2
import tempfile
import os
import requests

def test_webm_support():
    """Test if OpenCV can handle WebM files"""
    print("🧪 Testing WebM Support in OpenCV")
    print("=" * 40)
    
    # Check OpenCV build info
    print(f"📊 OpenCV Version: {cv2.__version__}")
    
    # Test WebM codec support
    fourcc_codes = [
        ('VP8', cv2.VideoWriter_fourcc(*'VP80')),
        ('VP9', cv2.VideoWriter_fourcc(*'VP90')),
        ('WebM', cv2.VideoWriter_fourcc(*'WEBM')),
    ]
    
    print("\n🎬 Codec Support:")
    for name, fourcc in fourcc_codes:
        try:
            # Try to create a temporary video writer
            temp_file = tempfile.mktemp(suffix='.webm')
            writer = cv2.VideoWriter(temp_file, fourcc, 30.0, (640, 480))
            if writer.isOpened():
                print(f"✅ {name}: Supported")
                writer.release()
                os.remove(temp_file) if os.path.exists(temp_file) else None
            else:
                print(f"❌ {name}: Not supported")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    print("\n🌐 Testing Backend Connection:")
    try:
        response = requests.get('http://192.168.1.100:5000/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend: Connected")
            print(f"📊 Model: {data['model_parameters']:,} parameters")
            print(f"🎯 Classes: {', '.join(data['classes'])}")
        else:
            print(f"❌ Backend: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Backend: Connection failed - {e}")
    
    print("\n💡 WebM Processing Tips:")
    print("• WebM files use VP8/VP9 codecs")
    print("• OpenCV should handle WebM files automatically")
    print("• If issues persist, try converting to MP4")
    print("• Browser MediaRecorder typically creates VP8 WebM")

if __name__ == "__main__":
    test_webm_support()
