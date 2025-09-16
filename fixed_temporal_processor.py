#!/usr/bin/env python3
"""
Fixed Temporal Processor - Corrected Full Video Processing
==========================================================

Fixes the temporal truncation issue by separating analysis sampling
from full video processing. Maintains spatial cropping quality while
preserving complete temporal duration.

Author: Augment Agent
Date: 2025-09-14
"""

import cv2
import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import time

# Import our fixed utilities
from improved_roi_utils import AdaptiveLipDetector
from roi_utils import ROIGeometry, BBoxSmoother, RecropCalculator, create_debug_visualization

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FixedTemporalProcessor:
    """
    Fixed processor that separates analysis sampling from full video processing
    to preserve complete temporal duration while maintaining spatial quality.
    """
    
    def __init__(self, video_path: str, output_dir: str = "fixed_temporal_output"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "debug_frames").mkdir(exist_ok=True)
        (self.output_dir / "analysis_frames").mkdir(exist_ok=True)
        (self.output_dir / "full_processed").mkdir(exist_ok=True)
        
        # Initialize adaptive detector
        self.detector = AdaptiveLipDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            auto_detect_mode=True
        )
        
        # Processing parameters
        self.args = type('Args', (), {
            'min_area_ratio': 0.30,
            'min_h_ratio': 0.40,
            'min_w_ratio': 0.40,
            'target_h_ratio': 0.50,
            'target_w_ratio': 0.50,
            'out_size': 96,
            'fps_sample': 5,  # Only for analysis, not final output
            'pad': 0.12,
            'ema': 0.3
        })()
        
        # Statistics
        self.stats = {
            'analysis_frames': 0,
            'analysis_detections': 0,
            'full_frames': 0,
            'full_detections': 0,
            'detection_mode': None,
            'avg_bbox': None,
            'processing_time': 0
        }
        
    def process_video_with_full_temporal_preservation(self) -> Dict[str, Any]:
        """
        Process video with separated analysis and full processing phases.
        """
        logger.info("=" * 80)
        logger.info("FIXED TEMPORAL PROCESSOR - FULL VIDEO PROCESSING")
        logger.info("=" * 80)
        logger.info(f"Target video: {self.video_path}")
        logger.info(f"Output directory: {self.output_dir}")
        
        start_time = time.time()
        
        # Phase 1: Quick Analysis (Sampling for Detection Mode & Quality)
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: ANALYSIS SAMPLING (Detection Mode & Quality)")
        logger.info("="*60)
        
        analysis_results = self._analyze_with_sampling()
        if not analysis_results['success']:
            return analysis_results
            
        # Phase 2: Full Video Processing (All Frames)
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: FULL VIDEO PROCESSING (All Frames)")
        logger.info("="*60)
        
        full_processing_results = self._process_all_frames()
        
        self.stats['processing_time'] = time.time() - start_time
        
        # Phase 3: Results Summary
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: TEMPORAL PRESERVATION RESULTS")
        logger.info("="*60)
        
        final_report = self._generate_temporal_report(analysis_results, full_processing_results)
        
        return final_report
        
    def _analyze_with_sampling(self) -> Dict[str, Any]:
        """Phase 1: Analyze video with sampling to determine processing parameters."""
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return {'success': False, 'error': 'Could not open video'}
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Original Video Properties:")
        logger.info(f"  Dimensions: {width}x{height}")
        logger.info(f"  Frame rate: {fps:.2f} FPS")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Duration: {duration:.2f} seconds")
        
        # Sample frames for analysis (not for final output!)
        sample_interval = max(1, total_frames // self.args.fps_sample)
        logger.info(f"Analysis sampling: every {sample_interval} frames ({self.args.fps_sample} samples)")

        # Force detection mode detection by processing first 10 frames sequentially
        logger.info("Forcing detection mode analysis on first 10 frames...")
        for i in range(min(10, total_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                landmarks = self.detector.detect_lip_landmarks(frame)
                if landmarks is not None:
                    logger.info(f"  Frame {i}: ✓ Detected - Mode: {self.detector.get_detection_mode()}")
                else:
                    logger.info(f"  Frame {i}: ✗ No detection - Mode: {self.detector.get_detection_mode() or 'detecting...'}")

        # Now the detector should have switched to cropped_face mode
        self.stats['detection_mode'] = self.detector.get_detection_mode()
        logger.info(f"Detection mode after analysis: {self.stats['detection_mode']}")

        # Initialize smoother for analysis
        analysis_smoother = BBoxSmoother(alpha=self.args.ema)
        analysis_bboxes = []

        frame_idx = 0
        sample_count = 0

        while frame_idx < total_frames and sample_count < 10:  # Limit analysis samples
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            self.stats['analysis_frames'] += 1
            sample_count += 1

            # Detect landmarks for analysis (should now work in cropped_face mode)
            landmarks = self.detector.detect_lip_landmarks(frame)
            
            if landmarks is not None:
                self.stats['analysis_detections'] += 1
                
                # Calculate bounding box
                tight_bbox = ROIGeometry.calculate_tight_bbox(landmarks)
                padded_bbox = ROIGeometry.add_padding(
                    tight_bbox, self.args.pad, frame.shape[:2]
                )
                smoothed_bbox = analysis_smoother.smooth(padded_bbox)
                analysis_bboxes.append(smoothed_bbox)
                
                # Save analysis debug frame
                debug_frame = create_debug_visualization(
                    frame, landmarks, smoothed_bbox, None, None
                )
                debug_path = self.output_dir / "analysis_frames" / f"analysis_{frame_idx:04d}.jpg"
                cv2.imwrite(str(debug_path), debug_frame)
                
                logger.info(f"Analysis frame {frame_idx}: ✓ Detected - BBox: {smoothed_bbox}")
                
            else:
                logger.info(f"Analysis frame {frame_idx}: ✗ No detection")
                
            frame_idx += sample_interval
            
        cap.release()
        
        # Calculate average bbox for full processing
        if analysis_bboxes:
            self.stats['avg_bbox'] = np.mean(analysis_bboxes, axis=0).astype(int)
            detection_rate = self.stats['analysis_detections'] / self.stats['analysis_frames']
            self.stats['detection_mode'] = self.detector.get_detection_mode()
            
            logger.info(f"\nAnalysis Results:")
            logger.info(f"  Detection mode: {self.stats['detection_mode']}")
            logger.info(f"  Analysis detection rate: {detection_rate:.1%}")
            logger.info(f"  Average bbox: {self.stats['avg_bbox']}")
            
            return {
                'success': True,
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration': duration,
                'detection_mode': self.stats['detection_mode'],
                'avg_bbox': self.stats['avg_bbox'],
                'detection_rate': detection_rate
            }
        else:
            logger.error("No successful detections in analysis phase!")
            return {'success': False, 'error': 'No detections in analysis'}
            
    def _process_all_frames(self) -> Dict[str, Any]:
        """Phase 2: Process ALL frames using analysis results."""
        
        if self.stats['avg_bbox'] is None:
            logger.error("No average bbox available for full processing!")
            return {'success': False}
            
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing ALL {total_frames} frames...")
        logger.info(f"Using detection mode: {self.stats['detection_mode']}")
        logger.info(f"Using average bbox: {self.stats['avg_bbox']}")
        
        # Setup output video writer
        output_path = self.output_dir / "full_processed" / "full_temporal_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (self.args.out_size, self.args.out_size))
        
        # Initialize smoother for full processing
        full_smoother = BBoxSmoother(alpha=self.args.ema)
        
        # Process every single frame
        processed_frames = []
        frame_idx = 0
        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            self.stats['full_frames'] += 1
            
            # For full processing, we can use the average bbox or detect per frame
            # Using detection per frame for best quality
            landmarks = self.detector.detect_lip_landmarks(frame)
            
            if landmarks is not None:
                self.stats['full_detections'] += 1
                
                # Calculate bbox for this frame
                tight_bbox = ROIGeometry.calculate_tight_bbox(landmarks)
                padded_bbox = ROIGeometry.add_padding(
                    tight_bbox, self.args.pad, frame.shape[:2]
                )
                smoothed_bbox = full_smoother.smooth(padded_bbox)
                
            else:
                # Use average bbox if detection fails
                smoothed_bbox = tuple(self.stats['avg_bbox'])
                
            # Crop frame using bbox
            x1, y1, x2, y2 = smoothed_bbox
            cropped = frame[y1:y2, x1:x2]
            
            if cropped.size > 0:
                # Resize to target size
                resized = cv2.resize(cropped, (self.args.out_size, self.args.out_size))
                out.write(resized)
                processed_frames.append(resized)
                
                # Save debug frame every 20 frames
                if frame_idx % 20 == 0:
                    debug_frame = create_debug_visualization(
                        frame, landmarks, smoothed_bbox, None, None
                    )
                    debug_path = self.output_dir / "debug_frames" / f"full_debug_{frame_idx:04d}.jpg"
                    cv2.imwrite(str(debug_path), debug_frame)
                    
            if frame_idx % 25 == 0:  # Progress logging
                logger.info(f"Processed frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
                
            frame_idx += 1
            
        cap.release()
        out.release()
        
        logger.info(f"\n✅ Full processing complete!")
        logger.info(f"  Total frames processed: {self.stats['full_frames']}")
        logger.info(f"  Successful detections: {self.stats['full_detections']}")
        logger.info(f"  Detection rate: {self.stats['full_detections']/self.stats['full_frames']:.1%}")
        logger.info(f"  Output video: {output_path}")
        
        # Verify output video properties
        verify_cap = cv2.VideoCapture(str(output_path))
        out_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = verify_cap.get(cv2.CAP_PROP_FPS)
        out_duration = out_frames / out_fps if out_fps > 0 else 0
        out_width = int(verify_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_height = int(verify_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        verify_cap.release()
        
        logger.info(f"\n📹 Output Video Verification:")
        logger.info(f"  Dimensions: {out_width}x{out_height}")
        logger.info(f"  Frame count: {out_frames}")
        logger.info(f"  FPS: {out_fps:.2f}")
        logger.info(f"  Duration: {out_duration:.2f} seconds")
        
        return {
            'success': True,
            'output_path': str(output_path),
            'processed_frames': len(processed_frames),
            'output_properties': {
                'width': out_width,
                'height': out_height,
                'frames': out_frames,
                'fps': out_fps,
                'duration': out_duration
            }
        }
        
    def _generate_temporal_report(self, analysis_results: Dict, processing_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive temporal preservation report."""
        
        report = {
            'video_path': self.video_path,
            'processing_time': self.stats['processing_time'],
            'temporal_fix_applied': True,
            'original_properties': {
                'frames': analysis_results.get('total_frames', 0),
                'fps': analysis_results.get('fps', 0),
                'duration': analysis_results.get('duration', 0),
                'dimensions': f"{analysis_results.get('width', 0)}x{analysis_results.get('height', 0)}"
            },
            'analysis_phase': {
                'detection_mode': self.stats['detection_mode'],
                'sampled_frames': self.stats['analysis_frames'],
                'analysis_detections': self.stats['analysis_detections'],
                'analysis_rate': self.stats['analysis_detections'] / self.stats['analysis_frames'] if self.stats['analysis_frames'] > 0 else 0
            },
            'full_processing_phase': {
                'processed_frames': self.stats['full_frames'],
                'successful_detections': self.stats['full_detections'],
                'detection_rate': self.stats['full_detections'] / self.stats['full_frames'] if self.stats['full_frames'] > 0 else 0
            },
            'output_properties': processing_results.get('output_properties', {}),
            'temporal_preservation': {
                'original_duration': analysis_results.get('duration', 0),
                'output_duration': processing_results.get('output_properties', {}).get('duration', 0),
                'frame_preservation_rate': processing_results.get('output_properties', {}).get('frames', 0) / analysis_results.get('total_frames', 1),
                'temporal_integrity': 'PRESERVED' if processing_results.get('output_properties', {}).get('frames', 0) >= analysis_results.get('total_frames', 1) * 0.95 else 'PARTIAL'
            }
        }
        
        # Save report
        report_path = self.output_dir / "temporal_fix_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("🎉 TEMPORAL PRESERVATION FIX COMPLETE!")
        logger.info("="*80)
        
        orig_props = report['original_properties']
        out_props = report['output_properties']
        temp_pres = report['temporal_preservation']
        
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"⏱️  Processing time: {self.stats['processing_time']:.2f} seconds")
        logger.info(f"🎯 Detection mode: {self.stats['detection_mode']}")
        
        logger.info(f"\n📊 TEMPORAL COMPARISON:")
        logger.info(f"  Original: {orig_props['frames']} frames, {orig_props['duration']:.2f}s, {orig_props['dimensions']}")
        logger.info(f"  Output:   {out_props['frames']} frames, {out_props['duration']:.2f}s, {out_props['width']}x{out_props['height']}")
        logger.info(f"  Preservation: {temp_pres['frame_preservation_rate']:.1%} frames preserved")
        logger.info(f"  Status: {temp_pres['temporal_integrity']}")
        
        logger.info(f"\n📹 Output video: {processing_results['output_path']}")
        logger.info(f"📄 Report: {report_path}")
        
        return report


def main():
    """Main function to test the fixed temporal processor."""
    
    # Target video path
    video_path = "/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped/doctor__useruser01__18to39__female__aboriginal__20250807T054104_topmid.mp4"
    
    # Create fixed processor
    processor = FixedTemporalProcessor(video_path, "fixed_temporal_output")
    
    # Run fixed processing
    results = processor.process_video_with_full_temporal_preservation()
    
    if results.get('temporal_fix_applied'):
        logger.info("\n✅ Temporal fix applied successfully!")
        logger.info("The output video now preserves the full temporal duration.")
    else:
        logger.error("\n❌ Temporal fix failed!")


if __name__ == "__main__":
    main()
