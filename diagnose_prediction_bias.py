#!/usr/bin/env python3
"""
Diagnostic script to test the model's prediction behavior with synthetic inputs.
This will help us understand if the class collapse is in the model or preprocessing.
"""

import torch
import numpy as np
from load_75_9_checkpoint import load_checkpoint

def create_synthetic_videos():
    """Create different synthetic video patterns to test model behavior."""
    batch_size = 1
    channels = 1
    frames = 32
    height = 64
    width = 96
    
    # Different synthetic patterns
    patterns = {
        "zeros": torch.zeros(batch_size, channels, frames, height, width),
        "ones": torch.ones(batch_size, channels, frames, height, width),
        "random_low": torch.rand(batch_size, channels, frames, height, width) * 0.3,
        "random_high": torch.rand(batch_size, channels, frames, height, width) * 0.7 + 0.3,
        "gradient_horizontal": torch.zeros(batch_size, channels, frames, height, width),
        "gradient_vertical": torch.zeros(batch_size, channels, frames, height, width),
        "center_bright": torch.zeros(batch_size, channels, frames, height, width),
        "edges_bright": torch.zeros(batch_size, channels, frames, height, width),
    }
    
    # Create gradient patterns
    for f in range(frames):
        for h in range(height):
            for w in range(width):
                patterns["gradient_horizontal"][0, 0, f, h, w] = w / width
                patterns["gradient_vertical"][0, 0, f, h, w] = h / height
    
    # Create center bright pattern (simulating mouth area)
    center_h, center_w = height // 2, width // 2
    for f in range(frames):
        for h in range(max(0, center_h-10), min(height, center_h+10)):
            for w in range(max(0, center_w-15), min(width, center_w+15)):
                patterns["center_bright"][0, 0, f, h, w] = 0.8
    
    # Create edges bright pattern
    patterns["edges_bright"][0, 0, :, 0, :] = 0.8  # Top edge
    patterns["edges_bright"][0, 0, :, -1, :] = 0.8  # Bottom edge
    patterns["edges_bright"][0, 0, :, :, 0] = 0.8  # Left edge
    patterns["edges_bright"][0, 0, :, :, -1] = 0.8  # Right edge
    
    return patterns

def test_model_predictions():
    """Test the model with synthetic inputs to diagnose prediction bias."""
    print("🔍 DIAGNOSING MODEL PREDICTION BIAS")
    print("=" * 50)
    
    try:
        # Load the model
        print("Loading model...")
        model, class_to_idx, idx_to_class, checkpoint = load_checkpoint()
        model.eval()
        
        print(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"📊 Classes: {list(class_to_idx.keys())}")
        print()
        
        # Create synthetic test patterns
        patterns = create_synthetic_videos()
        
        print("🧪 TESTING SYNTHETIC PATTERNS:")
        print("-" * 30)
        
        results = {}
        
        with torch.no_grad():
            for pattern_name, video_tensor in patterns.items():
                # Get model prediction
                outputs = model(video_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probabilities, k=4)
                
                print(f"\n🎯 Pattern: {pattern_name}")
                print(f"   Raw outputs: {outputs[0].numpy()}")
                print(f"   Predictions:")
                
                predictions = []
                for i in range(4):
                    class_name = idx_to_class[top_indices[0][i].item()]
                    confidence = top_probs[0][i].item() * 100
                    print(f"     {i+1}. {class_name}: {confidence:.1f}%")
                    predictions.append((class_name, confidence))
                
                results[pattern_name] = predictions
        
        print("\n" + "=" * 50)
        print("📊 ANALYSIS SUMMARY:")
        print("-" * 20)
        
        # Analyze results
        all_top_predictions = [results[pattern][0][0] for pattern in results]
        unique_predictions = set(all_top_predictions)
        
        print(f"🎯 Total patterns tested: {len(patterns)}")
        print(f"🎯 Unique top predictions: {len(unique_predictions)}")
        print(f"🎯 Top predictions: {unique_predictions}")
        
        # Count prediction frequency
        from collections import Counter
        prediction_counts = Counter(all_top_predictions)
        
        print(f"\n📈 Prediction frequency:")
        for pred, count in prediction_counts.most_common():
            percentage = (count / len(patterns)) * 100
            print(f"   {pred}: {count}/{len(patterns)} ({percentage:.1f}%)")
        
        # Check for class collapse
        if len(unique_predictions) <= 2:
            print(f"\n⚠️  CLASS COLLAPSE DETECTED!")
            print(f"   Model only predicts {len(unique_predictions)} out of 4 classes")
            print(f"   Dominant prediction: {prediction_counts.most_common(1)[0][0]}")
            
            # Check if it's always the same prediction
            if len(unique_predictions) == 1:
                print(f"   🚨 SEVERE: Model ALWAYS predicts '{list(unique_predictions)[0]}'")
            else:
                print(f"   🔶 MODERATE: Model alternates between {unique_predictions}")
        else:
            print(f"\n✅ Model shows diverse predictions across patterns")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main diagnostic function."""
    results = test_model_predictions()
    
    if results:
        print(f"\n🎯 DIAGNOSIS COMPLETE")
        print(f"   Results saved for analysis")
        
        # Save detailed results
        with open("prediction_bias_diagnosis.txt", "w") as f:
            f.write("PREDICTION BIAS DIAGNOSIS RESULTS\n")
            f.write("=" * 40 + "\n\n")
            
            for pattern_name, predictions in results.items():
                f.write(f"Pattern: {pattern_name}\n")
                for i, (class_name, confidence) in enumerate(predictions):
                    f.write(f"  {i+1}. {class_name}: {confidence:.1f}%\n")
                f.write("\n")
        
        print(f"   Detailed results saved to: prediction_bias_diagnosis.txt")
    else:
        print(f"\n❌ DIAGNOSIS FAILED")

if __name__ == "__main__":
    main()
