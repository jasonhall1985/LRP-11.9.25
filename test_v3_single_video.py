#!/usr/bin/env python3
"""
Test V3 Single Video - Aggressive Positioning Fix
=================================================

Test the V3 aggressive positioning algorithm on a single video to verify improvements.

Author: Augment Agent
Date: 2025-09-14
"""

import cv2
import numpy as np
import os
import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# Import V3 utilities
from improved_roi_utils_v3 import AdaptiveLipDetectorV3, create_debug_visualization_v3
from roi_utils import ROIGeometry, BBoxSmoother, RecropCalculator

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_v3_single_video():
    """
    Test V3 aggressive positioning on a single video.
    """
    # Test video (one that had poor positioning in V2)
    video_path = "/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped/help__useruser01__65plus__female__caucasian__20250730T033929_topmid.mp4"
    video_name = "help__useruser01__65plus__female__caucasian__20250730T033929_topmid.mp4"
    
    output_dir = Path("fixed_temporal_output")
    v3_debug_dir = output_dir / "v3_debug"
    v3_debug_dir.mkdir(exist_ok=True)
    
    # Processing parameters
    args = type('Args', (), {
        'min_area_ratio': 0.30,
        'min_h_ratio': 0.40,
        'min_w_ratio': 0.40,
        'target_h_ratio': 0.50,
        'target_w_ratio': 0.50,
        'out_size': 96,
        'fps_sample': 5,  # Only for analysis
        'pad': 0.12,
        'ema': 0.3
    })()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING V3 AGGRESSIVE POSITIONING: {video_name}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Initialize V3 detector
        detector = AdaptiveLipDetectorV3(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            auto_detect_mode=True
        )
        
        # Get original video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Original: {width}x{height}, {total_frames} frames, {duration:.2f}s")
        
        # Force detection mode detection with V3 algorithm
        logger.info("Detecting processing mode with V3 AGGRESSIVE algorithm...")
        for i in range(min(10, total_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                landmarks = detector.detect_lip_landmarks(frame)
                if landmarks is not None:
                    break
                    
        detection_mode = detector.get_detection_mode()
        logger.info(f"V3 Detection mode: {detection_mode}")
        
        # Process sample frames with V3 aggressive positioning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        # Create output filename with V3 prefix
        output_name = f"processed_v3_{video_name}"
        output_path = output_dir / "full_processed" / output_name
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (args.out_size, args.out_size))
        
        # Initialize smoother
        smoother = BBoxSmoother(alpha=args.ema)
        
        # Process frames with V3 aggressive debug tracking
        processed_frames = 0
        successful_detections = 0
        debug_frames_saved = 0
        
        # Save debug frames every 10 frames for detailed analysis
        for frame_idx in range(min(total_frames, 50)):  # Test first 50 frames
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frames += 1
            
            # Detect landmarks with V3 aggressive algorithm
            landmarks = detector.detect_lip_landmarks(frame)
            
            if landmarks is not None:
                successful_detections += 1
                
                # Calculate bbox with V3 aggressive positioning
                tight_bbox = ROIGeometry.calculate_tight_bbox(landmarks)
                padded_bbox = ROIGeometry.add_padding(
                    tight_bbox, args.pad, frame.shape[:2]
                )
                smoothed_bbox = smoother.smooth(padded_bbox)
                
                # Save debug visualization every 5 frames for detailed analysis
                if frame_idx % 5 == 0:
                    ratios = ROIGeometry.calculate_size_ratios(smoothed_bbox, frame.shape[:2])
                    debug_frame = create_debug_visualization_v3(
                        frame, landmarks, smoothed_bbox, detector, ratios
                    )
                    debug_path = v3_debug_dir / f"v3_debug_{video_name}_{frame_idx:04d}.jpg"
                    cv2.imwrite(str(debug_path), debug_frame)
                    debug_frames_saved += 1
                    logger.info(f"  V3 Debug frame saved: {frame_idx}")
                    
            else:
                # Use previous bbox if available
                smoothed_bbox = smoother.get_last_bbox()
                if smoothed_bbox is None:
                    # V3: Enhanced fallback to center crop
                    h, w = frame.shape[:2]
                    crop_size = min(h, w) // 2
                    center_y, center_x = h // 2, w // 2
                    smoothed_bbox = (
                        center_x - crop_size // 2,
                        center_y - crop_size // 2,
                        center_x + crop_size // 2,
                        center_y + crop_size // 2
                    )
                    
            # Crop and resize frame
            x1, y1, x2, y2 = smoothed_bbox
            cropped = frame[y1:y2, x1:x2]
            
            if cropped.size > 0:
                resized = cv2.resize(cropped, (args.out_size, args.out_size))
                out.write(resized)
                
            if frame_idx % 10 == 0:
                logger.info(f"  V3 Progress: {frame_idx}/{min(total_frames, 50)} ({frame_idx/min(total_frames, 50)*100:.1f}%)")
                
        cap.release()
        out.release()
        
        # Verify output
        verify_cap = cv2.VideoCapture(str(output_path))
        out_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = verify_cap.get(cv2.CAP_PROP_FPS)
        out_duration = out_frames / out_fps if out_fps > 0 else 0
        verify_cap.release()
        
        processing_time = time.time() - start_time
        detection_rate = successful_detections / processed_frames if processed_frames > 0 else 0
        
        logger.info(f"✅ V3 AGGRESSIVE TEST SUCCESS: {video_name}")
        logger.info(f"  Processed: {processed_frames} frames in {processing_time:.2f}s")
        logger.info(f"  Detection rate: {detection_rate:.1%}")
        logger.info(f"  Output: {out_frames} frames, {out_duration:.2f}s")
        logger.info(f"  Debug frames saved: {debug_frames_saved}")
        logger.info(f"  V3 Parameters used:")
        logger.info(f"    Lip region: {detector.expected_lip_region}")
        logger.info(f"    Area expansion: {detector.area_expansion_factor}x")
        logger.info(f"    Vertical offset: {detector.vertical_offset_ratio}")
        
        return {
            'success': True,
            'version': 'v3_aggressive',
            'video_name': video_name,
            'processing_time': processing_time,
            'detection_mode': detection_mode,
            'processing_stats': {
                'processed_frames': processed_frames,
                'successful_detections': successful_detections,
                'detection_rate': detection_rate,
                'debug_frames_saved': debug_frames_saved
            },
            'v3_parameters': {
                'lip_region_params': detector.expected_lip_region,
                'area_expansion_factor': detector.area_expansion_factor,
                'vertical_offset_ratio': detector.vertical_offset_ratio
            }
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ V3 AGGRESSIVE TEST FAILED: {video_name} - {str(e)}")
        
        return {
            'success': False,
            'version': 'v3_aggressive',
            'video_name': video_name,
            'processing_time': processing_time,
            'error': str(e)
        }


def main():
    """
    Main function to test V3 aggressive positioning.
    """
    result = test_v3_single_video()
    
    if result['success']:
        logger.info("\n✅ V3 aggressive positioning test completed successfully!")
        logger.info("Check the V3 debug frames to verify improved positioning.")
    else:
        logger.error(f"\n❌ V3 test failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
