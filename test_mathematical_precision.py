#!/usr/bin/env python3
"""
Test Mathematical Precision of Lip Centering
Verify that the mathematical calculations are correct.
"""

import cv2
import numpy as np
from pathlib import Path

def test_mathematical_centering():
    """Test the mathematical precision of centering calculations."""
    
    print("🧮 TESTING MATHEMATICAL CENTERING PRECISION")
    print("=" * 60)
    
    # Simulate a 132×100 frame (typical full-face video)
    frame_width = 132
    frame_height = 100
    
    # Simulate detected lip position (adjusted for better centering)
    detected_lip_x = 132 * 0.85  # 85% of width (adjusted from 99.2%)
    detected_lip_y = 100 * 0.80  # 80% of height (adjusted from 89%)
    
    print(f"Frame size: {frame_width}×{frame_height}")
    print(f"Detected lip center: ({detected_lip_x}, {detected_lip_y})")
    print()
    
    # Target dimensions
    target_width = 96
    target_height = 64
    expansion_factor = 1.1
    
    # Calculate expanded crop dimensions
    expanded_width = int(target_width * expansion_factor)  # 105
    expanded_height = int(target_height * expansion_factor)  # 70
    
    print(f"Target size: {target_width}×{target_height}")
    print(f"Expanded crop size: {expanded_width}×{expanded_height}")
    print()
    
    # STEP 1: Calculate ideal crop position
    ideal_crop_center_x = expanded_width / 2.0  # 52.5
    ideal_crop_center_y = expanded_height / 2.0  # 35.0
    
    print(f"Ideal crop center: ({ideal_crop_center_x}, {ideal_crop_center_y})")
    
    # Calculate crop start position to center the detected lip
    ideal_crop_start_x = detected_lip_x - ideal_crop_center_x  # 131 - 52.5 = 78.5
    ideal_crop_start_y = detected_lip_y - ideal_crop_center_y  # 89 - 35 = 54
    
    print(f"Ideal crop start: ({ideal_crop_start_x}, {ideal_crop_start_y})")
    
    # STEP 2: Handle boundary constraints
    actual_crop_start_x = max(0, min(ideal_crop_start_x, frame_width - expanded_width))
    actual_crop_start_y = max(0, min(ideal_crop_start_y, frame_height - expanded_height))
    
    print(f"Boundary-adjusted crop start: ({actual_crop_start_x}, {actual_crop_start_y})")
    
    # Convert to integer coordinates
    crop_x = int(round(actual_crop_start_x))
    crop_y = int(round(actual_crop_start_y))
    
    # Ensure we don't exceed boundaries
    crop_x = max(0, min(crop_x, frame_width - expanded_width))
    crop_y = max(0, min(crop_y, frame_height - expanded_height))
    
    print(f"Final crop position: ({crop_x}, {crop_y})")
    print(f"Crop region: ({crop_x}, {crop_y}) to ({crop_x + expanded_width}, {crop_y + expanded_height})")
    print()
    
    # STEP 3: Calculate actual lip position in crop
    actual_lip_in_crop_x = detected_lip_x - crop_x
    actual_lip_in_crop_y = detected_lip_y - crop_y
    
    print(f"Lip position in crop: ({actual_lip_in_crop_x}, {actual_lip_in_crop_y})")
    
    # STEP 4: Calculate final lip position after resize
    scale_x = target_width / expanded_width
    scale_y = target_height / expanded_height
    
    print(f"Scale factors: ({scale_x:.4f}, {scale_y:.4f})")
    
    final_lip_x = actual_lip_in_crop_x * scale_x
    final_lip_y = actual_lip_in_crop_y * scale_y
    
    print(f"Final lip position after resize: ({final_lip_x:.2f}, {final_lip_y:.2f})")
    
    # STEP 5: Calculate deviation from target
    target_x, target_y = 48, 32  # Target center for 96×64
    deviation_x = abs(final_lip_x - target_x)
    deviation_y = abs(final_lip_y - target_y)
    total_deviation = np.sqrt(deviation_x**2 + deviation_y**2)
    
    print(f"Target center: ({target_x}, {target_y})")
    print(f"Deviation: ({deviation_x:.2f}, {deviation_y:.2f})")
    print(f"Total deviation: {total_deviation:.3f} pixels")
    print()
    
    # STEP 6: Analysis
    if total_deviation <= 1.0:
        print("✅ PERFECT CENTERING (≤1px)")
    elif total_deviation <= 2.0:
        print("✅ EXCELLENT CENTERING (≤2px)")
    elif total_deviation <= 5.0:
        print("⚠️  GOOD CENTERING (≤5px)")
    else:
        print("❌ POOR CENTERING (>5px)")
    
    print()
    print("🔍 ANALYSIS:")
    
    # Check if the issue is with boundary constraints
    if crop_x == 0 or crop_y == 0:
        print("⚠️  Crop hit frame boundary - this may affect centering precision")
    
    if crop_x + expanded_width >= frame_width or crop_y + expanded_height >= frame_height:
        print("⚠️  Crop hit frame boundary - this may affect centering precision")
    
    # Check if the detected lip position is reasonable
    lip_x_percent = (detected_lip_x / frame_width) * 100
    lip_y_percent = (detected_lip_y / frame_height) * 100
    
    print(f"Detected lip position: {lip_x_percent:.1f}% right, {lip_y_percent:.1f}% down")
    
    if lip_x_percent > 95 or lip_y_percent > 95:
        print("⚠️  Lip detected very close to frame edge - may cause centering issues")
    
    return total_deviation

def test_with_different_positions():
    """Test centering with different lip positions."""
    
    print("\n🎯 TESTING DIFFERENT LIP POSITIONS")
    print("=" * 60)
    
    # Test positions: (x_percent, y_percent, description)
    test_positions = [
        (50, 50, "Center"),
        (75, 85, "Typical position"),
        (99, 89, "Empirical position"),
        (90, 80, "Slightly inward"),
        (95, 95, "Very edge")
    ]
    
    frame_width = 132
    frame_height = 100
    
    for x_percent, y_percent, description in test_positions:
        print(f"\n{description}: {x_percent}% right, {y_percent}% down")
        
        detected_lip_x = frame_width * (x_percent / 100.0)
        detected_lip_y = frame_height * (y_percent / 100.0)
        
        # Simulate the centering calculation
        target_width = 96
        target_height = 64
        expansion_factor = 1.1
        
        expanded_width = int(target_width * expansion_factor)
        expanded_height = int(target_height * expansion_factor)
        
        ideal_crop_center_x = expanded_width / 2.0
        ideal_crop_center_y = expanded_height / 2.0
        
        ideal_crop_start_x = detected_lip_x - ideal_crop_center_x
        ideal_crop_start_y = detected_lip_y - ideal_crop_center_y
        
        actual_crop_start_x = max(0, min(ideal_crop_start_x, frame_width - expanded_width))
        actual_crop_start_y = max(0, min(ideal_crop_start_y, frame_height - expanded_height))
        
        crop_x = int(round(actual_crop_start_x))
        crop_y = int(round(actual_crop_start_y))
        
        crop_x = max(0, min(crop_x, frame_width - expanded_width))
        crop_y = max(0, min(crop_y, frame_height - expanded_height))
        
        actual_lip_in_crop_x = detected_lip_x - crop_x
        actual_lip_in_crop_y = detected_lip_y - crop_y
        
        scale_x = target_width / expanded_width
        scale_y = target_height / expanded_height
        
        final_lip_x = actual_lip_in_crop_x * scale_x
        final_lip_y = actual_lip_in_crop_y * scale_y
        
        target_x, target_y = 48, 32
        deviation_x = abs(final_lip_x - target_x)
        deviation_y = abs(final_lip_y - target_y)
        total_deviation = np.sqrt(deviation_x**2 + deviation_y**2)
        
        print(f"  Final position: ({final_lip_x:.2f}, {final_lip_y:.2f})")
        print(f"  Deviation: {total_deviation:.3f}px")
        
        if total_deviation <= 1.0:
            print("  ✅ Perfect")
        elif total_deviation <= 2.0:
            print("  ✅ Excellent")
        elif total_deviation <= 5.0:
            print("  ⚠️  Good")
        else:
            print("  ❌ Poor")

if __name__ == "__main__":
    deviation = test_mathematical_centering()
    test_with_different_positions()
    
    print(f"\n🎯 SUMMARY:")
    print(f"Mathematical centering precision test completed.")
    print(f"Primary test deviation: {deviation:.3f} pixels")
