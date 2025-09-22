#!/usr/bin/env python3
"""
Quick test to check if model can predict all 4 classes
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path to import from demo_backend_server
sys.path.append('.')

def test_model_classes():
    """Test if the model can predict all 4 classes with synthetic data"""
    print("🧪 TESTING ALL 4 CLASSES")
    print("=" * 40)
    
    try:
        # Import the model loading function
        from demo_backend_server import load_model, model, classes, predict_video
        
        if model is None:
            print("❌ Model not loaded. Starting backend first...")
            if not load_model():
                print("❌ Failed to load model")
                return
        
        print(f"✅ Model loaded with classes: {classes}")
        
        # Test with different synthetic inputs to see if we can get all 4 classes
        test_patterns = [
            ("zeros", np.zeros((1, 1, 32, 64, 96))),
            ("ones", np.ones((1, 1, 32, 64, 96))),
            ("random_low", np.random.rand(1, 1, 32, 64, 96) * 0.3),
            ("random_high", np.random.rand(1, 1, 32, 64, 96) * 0.7 + 0.3),
            ("gradient_h", np.tile(np.linspace(0, 1, 96), (1, 1, 32, 64, 1))),
            ("gradient_v", np.tile(np.linspace(0, 1, 64).reshape(64, 1), (1, 1, 32, 1, 96))),
            ("checkerboard", np.tile(np.indices((64, 96)).sum(axis=0) % 2, (1, 1, 32, 1, 1))),
            ("center_bright", create_center_bright_pattern()),
            ("edges_bright", create_edge_pattern()),
            ("noise_pattern", np.random.normal(0.5, 0.2, (1, 1, 32, 64, 96)).clip(0, 1)),
        ]
        
        results = {}
        
        for name, pattern in test_patterns:
            tensor = torch.FloatTensor(pattern)
            
            # Get prediction
            result = predict_video(tensor)
            
            if result and result.get('success'):
                top_class = result['top2'][0]['class']
                top_conf = result['top2'][0]['confidence']
                
                print(f"📊 {name:15} → {top_class:15} ({top_conf:.3f})")
                
                if top_class not in results:
                    results[top_class] = []
                results[top_class].append((name, top_conf))
            else:
                print(f"❌ {name:15} → FAILED")
        
        print(f"\n🎯 CLASS DISTRIBUTION:")
        print("=" * 30)
        
        for class_name in classes:
            if class_name in results:
                count = len(results[class_name])
                best_conf = max(results[class_name], key=lambda x: x[1])
                print(f"✅ {class_name:20}: {count} predictions (best: {best_conf[1]:.3f} from {best_conf[0]})")
            else:
                print(f"❌ {class_name:20}: 0 predictions - MODEL NEVER PREDICTS THIS CLASS!")
        
        # Check if model is stuck on certain classes
        unique_classes = set(results.keys())
        if len(unique_classes) < len(classes):
            missing_classes = set(classes) - unique_classes
            print(f"\n⚠️  CRITICAL ISSUE: Model never predicts {missing_classes}")
            print("💡 This suggests class collapse - model is biased toward certain classes")
            
            # Test with extreme inputs to try to trigger other classes
            print(f"\n🔄 TESTING EXTREME INPUTS...")
            extreme_tests = [
                ("all_black", np.zeros((1, 1, 32, 64, 96))),
                ("all_white", np.ones((1, 1, 32, 64, 96))),
                ("very_noisy", np.random.rand(1, 1, 32, 64, 96)),
                ("mouth_region", create_mouth_focused_pattern()),
            ]
            
            for name, pattern in extreme_tests:
                tensor = torch.FloatTensor(pattern)
                result = predict_video(tensor)
                
                if result and result.get('success'):
                    all_preds = [(pred['class'], pred['confidence']) for pred in result['top2']]
                    print(f"🔍 {name:15} → {all_preds}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure demo_backend_server.py is running")
    except Exception as e:
        print(f"❌ Error: {e}")

def create_center_bright_pattern():
    """Create pattern with bright center (mouth region)"""
    pattern = np.zeros((1, 1, 32, 64, 96))
    # Bright rectangle in center where mouth would be
    pattern[:, :, :, 20:44, 30:66] = 1.0
    return pattern

def create_edge_pattern():
    """Create pattern with bright edges"""
    pattern = np.zeros((1, 1, 32, 64, 96))
    pattern[:, :, :, :5, :] = 1.0  # Top edge
    pattern[:, :, :, -5:, :] = 1.0  # Bottom edge
    pattern[:, :, :, :, :5] = 1.0  # Left edge
    pattern[:, :, :, :, -5:] = 1.0  # Right edge
    return pattern

def create_mouth_focused_pattern():
    """Create pattern focused on mouth region with movement"""
    pattern = np.random.rand(1, 1, 32, 64, 96) * 0.2  # Low background
    
    # Add mouth-like movement across frames
    for frame in range(32):
        mouth_open = 0.3 + 0.7 * np.sin(frame * np.pi / 16)  # Mouth opening/closing
        pattern[0, 0, frame, 25:35, 40:56] = mouth_open  # Mouth region
    
    return pattern

if __name__ == "__main__":
    test_model_classes()
