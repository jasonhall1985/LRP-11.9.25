#!/usr/bin/env python3
"""
Reprocess Failed Video - Complete V2 Batch
==========================================

Reprocess the one failed video from the V2 batch to complete the validation.

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

# Import our enhanced utilities
from improved_roi_utils_v2 import AdaptiveLipDetectorV2, create_debug_visualization_v2
from roi_utils import ROIGeometry, BBoxSmoother, RecropCalculator

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def reprocess_failed_video():
    """
    Reprocess the failed video with the fixed BBoxSmoother.
    """
    # Failed video info
    video_path = "/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped/doctor__useruser01__40to64__female__caucasian__20250830T111114_topmid.mp4"
    video_name = "doctor__useruser01__40to64__female__caucasian__20250830T111114_topmid.mp4"
    
    output_dir = Path("fixed_temporal_output")
    
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
    logger.info(f"REPROCESSING FAILED VIDEO: {video_name}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Initialize enhanced detector
        detector = AdaptiveLipDetectorV2(
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
        
        # Force detection mode detection with enhanced algorithm
        logger.info("Detecting processing mode with V2 algorithm...")
        for i in range(min(10, total_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                landmarks = detector.detect_lip_landmarks(frame)
                if landmarks is not None:
                    break
                    
        detection_mode = detector.get_detection_mode()
        logger.info(f"V2 Detection mode: {detection_mode}")
        
        # Process all frames with enhanced positioning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        # Create output filename with V2 prefix
        output_name = f"processed_v2_{video_name}"
        output_path = output_dir / "full_processed" / output_name
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (args.out_size, args.out_size))
        
        # Initialize smoother with fixed method
        smoother = BBoxSmoother(alpha=args.ema)
        
        # Process all frames with enhanced debug tracking
        processed_frames = 0
        successful_detections = 0
        debug_frames_saved = 0
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frames += 1
            
            # Detect landmarks with V2 algorithm
            landmarks = detector.detect_lip_landmarks(frame)
            
            if landmarks is not None:
                successful_detections += 1
                
                # Calculate bbox with enhanced positioning
                tight_bbox = ROIGeometry.calculate_tight_bbox(landmarks)
                padded_bbox = ROIGeometry.add_padding(
                    tight_bbox, args.pad, frame.shape[:2]
                )
                smoothed_bbox = smoother.smooth(padded_bbox)
                
                # Save debug visualization every 20 frames
                if frame_idx % 20 == 0:
                    ratios = ROIGeometry.calculate_size_ratios(smoothed_bbox, frame.shape[:2])
                    debug_frame = create_debug_visualization_v2(
                        frame, landmarks, smoothed_bbox, detector, ratios
                    )
                    debug_path = output_dir / "v2_debug" / f"v2_debug_{video_name}_{frame_idx:04d}.jpg"
                    cv2.imwrite(str(debug_path), debug_frame)
                    debug_frames_saved += 1
                    
            else:
                # Use previous bbox if available (now with fixed method)
                smoothed_bbox = smoother.get_last_bbox()
                if smoothed_bbox is None:
                    # Enhanced fallback to center crop
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
                
            if frame_idx % 25 == 0:
                logger.info(f"  V2 Progress: {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
                
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
        
        logger.info(f"✅ V2 REPROCESS SUCCESS: {video_name}")
        logger.info(f"  Processed: {processed_frames} frames in {processing_time:.2f}s")
        logger.info(f"  Detection rate: {detection_rate:.1%}")
        logger.info(f"  Output: {out_frames} frames, {out_duration:.2f}s")
        logger.info(f"  Temporal preservation: {out_frames/total_frames:.1%}")
        logger.info(f"  Debug frames saved: {debug_frames_saved}")
        
        return {
            'success': True,
            'version': 'v2_reprocessed',
            'video_name': video_name,
            'processing_time': processing_time,
            'detection_mode': detection_mode,
            'original_properties': {
                'frames': total_frames,
                'fps': fps,
                'duration': duration,
                'dimensions': f"{width}x{height}"
            },
            'output_properties': {
                'frames': out_frames,
                'fps': out_fps,
                'duration': out_duration,
                'dimensions': f"{args.out_size}x{args.out_size}",
                'path': str(output_path)
            },
            'processing_stats': {
                'processed_frames': processed_frames,
                'successful_detections': successful_detections,
                'detection_rate': detection_rate,
                'debug_frames_saved': debug_frames_saved
            },
            'temporal_preservation': {
                'frame_preservation_rate': out_frames / total_frames if total_frames > 0 else 0,
                'duration_preservation_rate': out_duration / duration if duration > 0 else 0,
                'status': 'PRESERVED' if out_frames >= total_frames * 0.95 else 'PARTIAL'
            }
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ V2 REPROCESS FAILED: {video_name} - {str(e)}")
        
        return {
            'success': False,
            'version': 'v2_reprocessed',
            'video_name': video_name,
            'processing_time': processing_time,
            'error': str(e)
        }


def main():
    """
    Main function to reprocess the failed video.
    """
    result = reprocess_failed_video()
    
    if result['success']:
        logger.info("\n✅ Failed video reprocessed successfully!")
        logger.info("V2 batch is now complete with 10/10 videos processed.")
    else:
        logger.error(f"\n❌ Reprocessing failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
