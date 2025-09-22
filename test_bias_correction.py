#!/usr/bin/env python3
"""
Test the bias correction system with synthetic inputs.
"""

import torch
import numpy as np
from load_75_9_checkpoint import load_checkpoint

# BIAS CORRECTION PARAMETERS (same as backend)
BIAS_CORRECTION = {
    'doctor': 0.01,     # Reduce doctor predictions by 99%
    'pillow': 5.0,      # Boost pillow dramatically
    'my_mouth_is_dry': 8.0,    # Boost extremely
    'i_need_to_move': 8.0      # Boost extremely
}

# DOCTOR SUPPRESSION THRESHOLD
DOCTOR_SUPPRESSION_THRESHOLD = 1.0
DOCTOR_SUPPRESSION_PENALTY = -3.0

def apply_bias_correction(raw_outputs, class_names):
    """Apply extreme bias correction with threshold-based doctor suppression."""
    corrected_outputs = raw_outputs.clone()

    # First pass: Apply standard correction factors
    for i, class_name in enumerate(class_names):
        if class_name in BIAS_CORRECTION:
            correction_factor = BIAS_CORRECTION[class_name]
            corrected_outputs[0, i] *= correction_factor

    # Second pass: Additional doctor suppression if above threshold
    doctor_idx = class_names.index('doctor') if 'doctor' in class_names else -1
    if doctor_idx >= 0 and raw_outputs[0, doctor_idx] > DOCTOR_SUPPRESSION_THRESHOLD:
        corrected_outputs[0, doctor_idx] += DOCTOR_SUPPRESSION_PENALTY
        print(f"⚡ DOCTOR SUPPRESSION: Additional {DOCTOR_SUPPRESSION_PENALTY} penalty applied")

    return corrected_outputs

def create_test_patterns():
    """Create test patterns."""
    batch_size = 1
    channels = 1
    frames = 32
    height = 64
    width = 96
    
    return {
        "zeros": torch.zeros(batch_size, channels, frames, height, width),
        "ones": torch.ones(batch_size, channels, frames, height, width),
        "random": torch.rand(batch_size, channels, frames, height, width),
        "center_bright": torch.zeros(batch_size, channels, frames, height, width),
    }

def test_bias_correction():
    """Test bias correction effectiveness."""
    print("🔧 TESTING BIAS CORRECTION SYSTEM")
    print("=" * 50)
    
    # Load model
    model, class_to_idx, idx_to_class, checkpoint = load_checkpoint()
    model.eval()
    class_names = list(class_to_idx.keys())
    
    print(f"✅ Model loaded: {class_names}")
    print(f"🔧 Correction factors: {BIAS_CORRECTION}")
    print()
    
    # Test patterns
    patterns = create_test_patterns()
    
    print("🧪 TESTING PATTERNS:")
    print("-" * 30)
    
    results_before = []
    results_after = []
    
    with torch.no_grad():
        for pattern_name, video_tensor in patterns.items():
            # Get raw model outputs
            raw_outputs = model(video_tensor)
            raw_probs = torch.softmax(raw_outputs, dim=1)
            
            # Apply bias correction
            corrected_outputs = apply_bias_correction(raw_outputs, class_names)
            corrected_probs = torch.softmax(corrected_outputs, dim=1)
            
            print(f"\n🎯 Pattern: {pattern_name}")
            print(f"   Raw outputs: {raw_outputs[0].numpy()}")
            print(f"   Corrected:   {corrected_outputs[0].numpy()}")
            
            # Get top predictions before and after
            raw_top = torch.argmax(raw_probs, dim=1).item()
            corrected_top = torch.argmax(corrected_probs, dim=1).item()
            
            raw_class = idx_to_class[raw_top]
            corrected_class = idx_to_class[corrected_top]
            
            raw_conf = raw_probs[0, raw_top].item() * 100
            corrected_conf = corrected_probs[0, corrected_top].item() * 100
            
            print(f"   BEFORE: {raw_class} ({raw_conf:.1f}%)")
            print(f"   AFTER:  {corrected_class} ({corrected_conf:.1f}%)")
            
            results_before.append(raw_class)
            results_after.append(corrected_class)
    
    print("\n" + "=" * 50)
    print("📊 BIAS CORRECTION ANALYSIS:")
    print("-" * 25)
    
    # Count unique predictions
    unique_before = set(results_before)
    unique_after = set(results_after)
    
    print(f"🎯 Unique predictions BEFORE: {len(unique_before)} - {unique_before}")
    print(f"🎯 Unique predictions AFTER:  {len(unique_after)} - {unique_after}")
    
    # Count doctor predictions
    doctor_before = results_before.count('doctor')
    doctor_after = results_after.count('doctor')
    
    print(f"🏥 Doctor predictions BEFORE: {doctor_before}/{len(patterns)} ({doctor_before/len(patterns)*100:.1f}%)")
    print(f"🏥 Doctor predictions AFTER:  {doctor_after}/{len(patterns)} ({doctor_after/len(patterns)*100:.1f}%)")
    
    # Success metrics
    if len(unique_after) > len(unique_before):
        print(f"✅ SUCCESS: Increased prediction diversity!")
    elif doctor_after < doctor_before:
        print(f"✅ PARTIAL SUCCESS: Reduced doctor bias!")
    else:
        print(f"⚠️  LIMITED SUCCESS: Bias correction needs tuning")
    
    return results_before, results_after

if __name__ == "__main__":
    test_bias_correction()
