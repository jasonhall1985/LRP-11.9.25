#!/usr/bin/env python3
"""
Test script to verify the model produces different outputs for different inputs
"""

import torch
import numpy as np
import requests
import json
import time

def test_model_variation():
    """Test if the model produces different outputs for different synthetic inputs"""
    print("🧪 MODEL VARIATION TEST")
    print("=" * 30)
    
    API_URL = 'http://192.168.1.100:5000'
    
    # Test 1: Create synthetic video tensors with different patterns
    print("\n🎯 Testing with synthetic data...")
    
    # Create different synthetic "videos" (32 frames, 64x96)
    test_cases = [
        ("zeros", np.zeros((32, 64, 96))),
        ("ones", np.ones((32, 64, 96))),
        ("random_1", np.random.rand(32, 64, 96)),
        ("random_2", np.random.rand(32, 64, 96)),
        ("gradient", np.linspace(0, 1, 32*64*96).reshape(32, 64, 96)),
    ]
    
    results = []
    
    for name, tensor_data in test_cases:
        # Convert to the expected format (1, 1, 32, 64, 96)
        tensor = torch.FloatTensor(tensor_data[np.newaxis, np.newaxis, ...])
        
        # Calculate stats
        mean = float(np.mean(tensor_data))
        std = float(np.std(tensor_data))
        checksum = float(np.sum(tensor_data))
        
        print(f"\n📊 {name}: mean={mean:.4f}, std={std:.4f}, checksum={checksum:.2f}")
        
        # We can't directly test the model without loading it, but we can test via API
        # For now, just show the tensor stats
        results.append({
            'name': name,
            'mean': mean,
            'std': std,
            'checksum': checksum
        })
    
    # Test 2: Check if recent API calls show variation
    print("\n🔍 RECENT API CALL ANALYSIS")
    print("=" * 30)
    
    # Check debug uploads folder for recent files
    import os
    debug_folder = "debug_uploads"
    
    if os.path.exists(debug_folder):
        webm_files = [f for f in os.listdir(debug_folder) if f.endswith('.webm')]
        webm_files.sort(key=lambda x: os.path.getmtime(os.path.join(debug_folder, x)))
        
        print(f"📁 Found {len(webm_files)} WebM files in debug folder")
        
        if len(webm_files) >= 2:
            # Test the two most recent files
            recent_files = webm_files[-2:]
            
            for i, filename in enumerate(recent_files):
                filepath = os.path.join(debug_folder, filename)
                file_size = os.path.getsize(filepath)
                mod_time = time.ctime(os.path.getmtime(filepath))
                
                print(f"\n📄 File {i+1}: {filename}")
                print(f"   Size: {file_size / 1024:.1f} KB")
                print(f"   Modified: {mod_time}")
                
                # Test this file via API
                try:
                    with open(filepath, 'rb') as f:
                        files = {'video': f}
                        response = requests.post(f'{API_URL}/predict', files=files, timeout=10)
                        
                        if response.status_code == 200:
                            data = response.json()
                            if data.get('success'):
                                top1 = data['top2'][0]
                                top2 = data['top2'][1]
                                print(f"   🎯 Predictions: {top1['class']} ({top1['confidence']:.3f}), {top2['class']} ({top2['confidence']:.3f})")
                            else:
                                print(f"   ❌ API error: {data.get('error', 'Unknown')}")
                        else:
                            print(f"   ❌ HTTP error: {response.status_code}")
                            
                except Exception as e:
                    print(f"   ❌ Request failed: {e}")
        else:
            print("💡 Need at least 2 video files to compare. Record more videos in the web demo.")
    else:
        print("❌ Debug folder not found")
    
    # Test 3: Health check to verify model is loaded
    print(f"\n🏥 BACKEND HEALTH CHECK")
    print("=" * 30)
    
    try:
        response = requests.get(f'{API_URL}/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend: {data.get('status', 'unknown')}")
            print(f"📊 Model: {data.get('model_parameters', 0):,} parameters")
            print(f"🎯 Classes: {', '.join(data.get('classes', []))}")
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("1. Record 3-4 different videos in the web demo")
    print("2. Say different words: 'doctor', 'pillow', 'my mouth is dry'")
    print("3. Include one silent/unclear video")
    print("4. Check backend logs for different tensor checksums")
    print("5. Verify predictions vary between recordings")

if __name__ == "__main__":
    test_model_variation()
