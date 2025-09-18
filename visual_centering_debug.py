#!/usr/bin/env python3
"""
Visual Centering Debug Tool
Creates visual inspection videos showing detected lip centers and final centering accuracy.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from final_corrected_preprocessing import FinalCorrectedPreprocessor

def create_visual_debug_video(video_path: str, output_path: str):
    """Create a visual debug video showing lip detection and centering."""

    # Initialize processor
    processor = FinalCorrectedPreprocessor()

    # Extract frames manually
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video")
        return

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame.copy()

        frames.append(gray_frame)

    cap.release()

    if not frames:
        print("❌ No frames extracted")
        return

    print(f"📹 Processing {len(frames)} frames for visual debug")

    # Sample frames to 32 for analysis
    if len(frames) > 32:
        indices = np.linspace(0, len(frames) - 1, 32, dtype=int)
        sampled_frames = [frames[i] for i in indices]
    else:
        sampled_frames = frames

    # Create debug frames
    debug_frames = []

    for i, frame in enumerate(sampled_frames):
        # Create a copy for visualization
        debug_frame = frame.copy()
        if len(debug_frame.shape) == 2:
            debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)

        h, w = frame.shape[:2]

        # Detect lip center
        detected_center = processor.detect_geometric_lip_center(frame)

        # Draw detected lip center (RED circle)
        cv2.circle(debug_frame, (int(detected_center[0]), int(detected_center[1])),
                  5, (0, 0, 255), -1)

        # Draw expected position based on analysis (GREEN circle)
        expected_x = int(w * 0.75)  # 75% to the right
        expected_y = int(h * 0.85)  # 85% down
        cv2.circle(debug_frame, (expected_x, expected_y), 5, (0, 255, 0), -1)

        # Draw frame center (BLUE cross)
        center_x, center_y = w // 2, h // 2
        cv2.line(debug_frame, (center_x - 10, center_y), (center_x + 10, center_y), (255, 0, 0), 2)
        cv2.line(debug_frame, (center_x, center_y - 10), (center_x, center_y + 10), (255, 0, 0), 2)

        # Add text annotations
        cv2.putText(debug_frame, f"Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Detected: ({detected_center[0]:.1f}, {detected_center[1]:.1f})",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(debug_frame, f"Expected: ({expected_x}, {expected_y})",
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Calculate deviation
        deviation = np.sqrt((detected_center[0] - expected_x)**2 + (detected_center[1] - expected_y)**2)
        cv2.putText(debug_frame, f"Deviation: {deviation:.1f}px",
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        debug_frames.append(debug_frame)
    
    # Now process the frames and create side-by-side comparison
    processed_frames = []

    for i, original_frame in enumerate(sampled_frames):
        if i < len(debug_frames):
            # Get the corresponding debug frame
            debug_frame = debug_frames[i]

            # Process the original frame
            detected_center = processor.detect_geometric_lip_center(original_frame)
            cropped = processor.create_precisely_centered_crop(original_frame, detected_center)
            processed = processor.apply_gentle_v5_preprocessing(cropped)

            # Convert processed frame for visualization
            if processed.dtype != np.uint8:
                processed_vis = ((processed + 1.0) * 127.5).astype(np.uint8)
            else:
                processed_vis = processed.copy()

            if len(processed_vis.shape) == 2:
                processed_vis = cv2.cvtColor(processed_vis, cv2.COLOR_GRAY2BGR)

            # Draw target center on processed frame (WHITE crosshairs)
            target_x, target_y = 48, 32  # Target center for 96x64
            cv2.line(processed_vis, (target_x - 10, target_y), (target_x + 10, target_y), (255, 255, 255), 2)
            cv2.line(processed_vis, (target_x, target_y - 10), (target_x, target_y + 10), (255, 255, 255), 2)

            # Re-detect lip center in processed frame for validation
            processed_gray = cv2.cvtColor(processed_vis, cv2.COLOR_BGR2GRAY) if len(processed_vis.shape) == 3 else processed_vis
            actual_center = processor._detect_lip_center_in_processed_frame(processed_gray)

            # Draw actual center in processed frame (RED circle)
            cv2.circle(processed_vis, (int(actual_center[0]), int(actual_center[1])), 3, (0, 0, 255), -1)

            # Calculate final centering accuracy
            final_deviation = np.sqrt((actual_center[0] - target_x)**2 + (actual_center[1] - target_y)**2)

            # Add text to processed frame
            cv2.putText(processed_vis, f"Target: ({target_x}, {target_y})",
                       (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(processed_vis, f"Actual: ({actual_center[0]:.1f}, {actual_center[1]:.1f})",
                       (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            cv2.putText(processed_vis, f"Dev: {final_deviation:.1f}px",
                       (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

            # Resize debug frame to match processed frame height
            debug_resized = cv2.resize(debug_frame, (int(debug_frame.shape[1] * 64 / debug_frame.shape[0]), 64))

            # Create side-by-side comparison
            combined = np.hstack([debug_resized, processed_vis])
            processed_frames.append(combined)
    
    # Save debug video
    if processed_frames:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10.0  # Slow playback for inspection
        
        h, w = processed_frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame in processed_frames:
            out.write(frame)
        
        out.release()
        print(f"✅ Visual debug video saved: {output_path}")
    else:
        print("❌ No frames to save")

if __name__ == "__main__":
    # Configuration
    test_video = "data/13.9.25top7dataset_cropped/pillow__useruser01__65plus__female__caucasian__20250827T062536_topmid.mp4"
    output_video = "data/final_corrected_test/visual_centering_debug.mp4"
    
    print("🔍 VISUAL CENTERING DEBUG")
    print("=" * 50)
    print("Creating visual inspection video...")
    print("RED circle = Detected lip center")
    print("GREEN circle = Expected lip position")
    print("BLUE cross = Frame center")
    print("WHITE cross = Target center (48, 32)")
    print()
    
    # Create output directory
    Path("data/final_corrected_test").mkdir(exist_ok=True)
    
    # Create visual debug video
    create_visual_debug_video(test_video, output_video)
