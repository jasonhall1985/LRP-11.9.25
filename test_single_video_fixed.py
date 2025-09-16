#!/usr/bin/env python3
"""
Test Single Video - Fixed Pipeline Demonstration
===============================================

Comprehensive test of the fixed mouth ROI pipeline on a single video
with detailed debugging, visualization, and analysis.

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


class SingleVideoTester:
    """
    Comprehensive tester for the fixed mouth ROI pipeline on a single video.
    """
    
    def __init__(self, video_path: str, output_dir: str = "single_video_test_output"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "debug_frames").mkdir(exist_ok=True)
        (self.output_dir / "cropped_frames").mkdir(exist_ok=True)
        (self.output_dir / "processed_videos").mkdir(exist_ok=True)
        
        # Initialize adaptive detector
        self.detector = AdaptiveLipDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            auto_detect_mode=True
        )
        
        # Initialize smoother
        self.smoother = BBoxSmoother(alpha=0.3)
        
        # Processing parameters (matching your requirements)
        self.args = type('Args', (), {
            'min_area_ratio': 0.30,
            'min_h_ratio': 0.40,
            'min_w_ratio': 0.40,
            'target_h_ratio': 0.50,
            'target_w_ratio': 0.50,
            'out_size': 96,
            'fps_sample': 5,
            'pad': 0.12,
            'ema': 0.3
        })()
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'sampled_frames': 0,
            'detected_frames': 0,
            'failed_frames': 0,
            'detection_mode': None,
            'bboxes': [],
            'ratios': [],
            'processing_time': 0
        }
        
    def process_video_comprehensive(self) -> Dict[str, Any]:
        """
        Process the video with comprehensive analysis and debugging.
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE SINGLE VIDEO TEST")
        logger.info("=" * 80)
        logger.info(f"Target video: {self.video_path}")
        logger.info(f"Output directory: {self.output_dir}")
        
        if not os.path.exists(self.video_path):
            logger.error(f"Video file not found: {self.video_path}")
            return {}
            
        start_time = time.time()
        
        # Phase 1: Video Analysis
        logger.info("\n" + "="*50)
        logger.info("PHASE 1: VIDEO ANALYSIS & DETECTION MODE")
        logger.info("="*50)
        
        analysis_results = self._analyze_video_properties()
        if not analysis_results:
            return {}
            
        # Phase 2: Frame Processing
        logger.info("\n" + "="*50)
        logger.info("PHASE 2: FRAME-BY-FRAME PROCESSING")
        logger.info("="*50)
        
        processing_results = self._process_frames_with_debug()
        
        # Phase 3: Decision Making
        logger.info("\n" + "="*50)
        logger.info("PHASE 3: QUALITY ANALYSIS & DECISION")
        logger.info("="*50)
        
        decision_results = self._make_processing_decision()
        
        # Phase 4: Output Generation
        logger.info("\n" + "="*50)
        logger.info("PHASE 4: OUTPUT GENERATION")
        logger.info("="*50)
        
        output_results = self._generate_outputs(decision_results)
        
        self.stats['processing_time'] = time.time() - start_time
        
        # Phase 5: Final Report
        logger.info("\n" + "="*50)
        logger.info("PHASE 5: COMPREHENSIVE RESULTS")
        logger.info("="*50)
        
        final_report = self._generate_final_report(analysis_results, processing_results, 
                                                 decision_results, output_results)
        
        return final_report
        
    def _analyze_video_properties(self) -> Dict[str, Any]:
        """Analyze video properties and detect processing mode."""
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error("Could not open video file")
            return {}
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        self.stats['total_frames'] = total_frames
        
        logger.info(f"Video Properties:")
        logger.info(f"  Dimensions: {width}x{height}")
        logger.info(f"  Frame rate: {fps:.2f} FPS")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Duration: {duration:.2f} seconds")
        
        # Test detection mode on first few frames
        logger.info(f"\nTesting detection modes on first 10 frames...")
        
        mode_test_results = []
        for i in range(min(10, total_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
                
            landmarks = self.detector.detect_lip_landmarks(frame)
            mode_test_results.append({
                'frame': i,
                'detected': landmarks is not None,
                'mode': self.detector.get_detection_mode()
            })
            
            if landmarks is not None:
                logger.info(f"  Frame {i}: ✓ Detected ({len(landmarks)} landmarks) - Mode: {self.detector.get_detection_mode()}")
            else:
                logger.info(f"  Frame {i}: ✗ No detection - Mode: {self.detector.get_detection_mode() or 'detecting...'}")
        
        cap.release()
        
        final_mode = self.detector.get_detection_mode()
        self.stats['detection_mode'] = final_mode
        
        logger.info(f"\n🎯 FINAL DETECTION MODE: {final_mode}")
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'detection_mode': final_mode,
            'mode_test_results': mode_test_results
        }
        
    def _process_frames_with_debug(self) -> Dict[str, Any]:
        """Process frames with detailed debugging and visualization."""
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames for processing
        sample_interval = max(1, total_frames // self.args.fps_sample)
        logger.info(f"Sampling every {sample_interval} frames ({self.args.fps_sample} samples total)")
        
        processed_frames = []
        debug_info = []
        
        frame_idx = 0
        sample_count = 0
        
        while frame_idx < total_frames and sample_count < 30:  # Limit for demo
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            self.stats['sampled_frames'] += 1
            sample_count += 1
            
            # Detect lip landmarks
            landmarks = self.detector.detect_lip_landmarks(frame)
            
            frame_info = {
                'frame_idx': frame_idx,
                'landmarks_detected': landmarks is not None,
                'landmark_count': len(landmarks) if landmarks is not None else 0
            }
            
            if landmarks is not None:
                self.stats['detected_frames'] += 1
                
                # Calculate bounding boxes
                tight_bbox = ROIGeometry.calculate_tight_bbox(landmarks)
                padded_bbox = ROIGeometry.add_padding(
                    tight_bbox, self.args.pad, frame.shape[:2]
                )
                smoothed_bbox = self.smoother.smooth(padded_bbox)
                
                # Calculate ratios
                ratios = ROIGeometry.calculate_size_ratios(smoothed_bbox, frame.shape[:2])
                
                # Store data
                self.stats['bboxes'].append(smoothed_bbox)
                self.stats['ratios'].append(ratios)
                
                frame_info.update({
                    'tight_bbox': tight_bbox,
                    'padded_bbox': padded_bbox,
                    'smoothed_bbox': smoothed_bbox,
                    'ratios': ratios
                })
                
                # Create debug visualization
                debug_frame = create_debug_visualization(
                    frame, landmarks, smoothed_bbox, None, ratios
                )
                
                # Add detection mode info
                mode_text = f"Mode: {self.detector.get_detection_mode()}"
                cv2.putText(debug_frame, mode_text, (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Save debug frame
                debug_path = self.output_dir / "debug_frames" / f"debug_{frame_idx:04d}.jpg"
                cv2.imwrite(str(debug_path), debug_frame)
                
                # Crop and resize frame
                x1, y1, x2, y2 = smoothed_bbox
                cropped = frame[y1:y2, x1:x2]
                if cropped.size > 0:
                    resized = cv2.resize(cropped, (self.args.out_size, self.args.out_size))
                    processed_frames.append(resized)
                    
                    # Save cropped frame
                    crop_path = self.output_dir / "cropped_frames" / f"cropped_{frame_idx:04d}.jpg"
                    cv2.imwrite(str(crop_path), resized)
                
                logger.info(f"Frame {frame_idx:4d}: ✓ Detected - BBox: {smoothed_bbox} - "
                          f"Ratios: area={ratios['area_ratio']:.3f}, h={ratios['h_ratio']:.3f}, w={ratios['w_ratio']:.3f}")
                
            else:
                self.stats['failed_frames'] += 1
                logger.info(f"Frame {frame_idx:4d}: ✗ No detection")
                
            debug_info.append(frame_info)
            frame_idx += sample_interval
            
        cap.release()
        
        # Create processed video
        if processed_frames:
            output_video_path = self.output_dir / "processed_videos" / "processed_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (self.args.out_size, self.args.out_size))
            
            for frame in processed_frames:
                out.write(frame)
                
            out.release()
            logger.info(f"✓ Processed video saved: {output_video_path}")
        
        detection_rate = self.stats['detected_frames'] / self.stats['sampled_frames'] if self.stats['sampled_frames'] > 0 else 0
        logger.info(f"\nProcessing Summary:")
        logger.info(f"  Sampled frames: {self.stats['sampled_frames']}")
        logger.info(f"  Successful detections: {self.stats['detected_frames']}")
        logger.info(f"  Failed detections: {self.stats['failed_frames']}")
        logger.info(f"  Detection rate: {detection_rate:.1%}")
        
        return {
            'processed_frames_count': len(processed_frames),
            'detection_rate': detection_rate,
            'debug_info': debug_info,
            'output_video_path': str(output_video_path) if processed_frames else None
        }
        
    def _make_processing_decision(self) -> Dict[str, Any]:
        """Analyze results and make processing decision."""
        
        if not self.stats['ratios']:
            logger.warning("No successful detections - cannot analyze quality")
            return {
                'decision': 'failed',
                'reason': 'No lip landmarks detected',
                'action': 'none'
            }
        
        # Calculate average ratios
        avg_ratios = {}
        for key in ['area_ratio', 'h_ratio', 'w_ratio']:
            values = [r[key] for r in self.stats['ratios']]
            avg_ratios[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        logger.info("Quality Analysis:")
        logger.info(f"  Average ratios:")
        for key, stats in avg_ratios.items():
            logger.info(f"    {key}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                      f"(range: {stats['min']:.3f} - {stats['max']:.3f})")
        
        # Check against thresholds
        mean_area = avg_ratios['area_ratio']['mean']
        mean_h = avg_ratios['h_ratio']['mean']
        mean_w = avg_ratios['w_ratio']['mean']
        
        logger.info(f"\nThreshold Comparison:")
        logger.info(f"  Area ratio: {mean_area:.3f} vs {self.args.min_area_ratio:.3f} (threshold) - {'✓' if mean_area >= self.args.min_area_ratio else '✗'}")
        logger.info(f"  Height ratio: {mean_h:.3f} vs {self.args.min_h_ratio:.3f} (threshold) - {'✓' if mean_h >= self.args.min_h_ratio else '✗'}")
        logger.info(f"  Width ratio: {mean_w:.3f} vs {self.args.min_w_ratio:.3f} (threshold) - {'✓' if mean_w >= self.args.min_w_ratio else '✗'}")
        
        # Make decision
        if (mean_area >= self.args.min_area_ratio and 
            mean_h >= self.args.min_h_ratio and 
            mean_w >= self.args.min_w_ratio):
            
            decision = 'keep'
            reason = 'ROI meets all size requirements'
            action = 'copy_original'
            logger.info(f"\n🎯 DECISION: KEEP - {reason}")
            
        else:
            decision = 'recrop'
            reason = 'ROI below size thresholds'
            action = 'recrop_video'
            logger.info(f"\n🎯 DECISION: RECROP - {reason}")
        
        return {
            'decision': decision,
            'reason': reason,
            'action': action,
            'avg_ratios': avg_ratios,
            'threshold_comparison': {
                'area_ratio': {'value': mean_area, 'threshold': self.args.min_area_ratio, 'passes': mean_area >= self.args.min_area_ratio},
                'h_ratio': {'value': mean_h, 'threshold': self.args.min_h_ratio, 'passes': mean_h >= self.args.min_h_ratio},
                'w_ratio': {'value': mean_w, 'threshold': self.args.min_w_ratio, 'passes': mean_w >= self.args.min_w_ratio}
            }
        }
        
    def _generate_outputs(self, decision_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final output files based on decision."""
        
        output_files = []
        
        if decision_results['action'] == 'copy_original':
            # Copy original file to keep folder
            keep_path = self.output_dir / "processed_videos" / "final_keep_original.mp4"
            import shutil
            shutil.copy2(self.video_path, keep_path)
            output_files.append(str(keep_path))
            logger.info(f"✓ Original video copied to: {keep_path}")
            
        elif decision_results['action'] == 'recrop_video':
            # Create recropped version
            recrop_path = self.output_dir / "processed_videos" / "final_recropped.mp4"
            success = self._create_recropped_video(recrop_path)
            if success:
                output_files.append(str(recrop_path))
                logger.info(f"✓ Recropped video saved to: {recrop_path}")
            else:
                logger.error("✗ Failed to create recropped video")
        
        return {
            'output_files': output_files,
            'action_taken': decision_results['action']
        }
        
    def _create_recropped_video(self, output_path: str) -> bool:
        """Create recropped video focusing on lip region."""
        try:
            if not self.stats['bboxes']:
                return False
                
            # Calculate average bbox for recropping
            avg_bbox = np.mean(self.stats['bboxes'], axis=0).astype(int)
            
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
                          int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            
            # Calculate recrop window
            crop_window = RecropCalculator.calculate_recrop_window(
                tuple(avg_bbox),
                self.args.target_h_ratio,
                self.args.target_w_ratio,
                frame_shape
            )
            
            logger.info(f"Recropping with window: {crop_window}")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (self.args.out_size, self.args.out_size))
            
            # Process all frames
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Crop frame
                x1, y1, x2, y2 = crop_window
                cropped = frame[y1:y2, x1:x2]
                
                if cropped.size > 0:
                    resized = cv2.resize(cropped, (self.args.out_size, self.args.out_size))
                    out.write(resized)
                    frame_count += 1
                    
            cap.release()
            out.release()
            
            logger.info(f"Recropped {frame_count} frames")
            return True
            
        except Exception as e:
            logger.error(f"Error creating recropped video: {str(e)}")
            return False
            
    def _generate_final_report(self, analysis_results: Dict, processing_results: Dict, 
                             decision_results: Dict, output_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        
        report = {
            'video_path': self.video_path,
            'processing_time': self.stats['processing_time'],
            'detection_mode': self.stats['detection_mode'],
            'video_properties': analysis_results,
            'processing_stats': {
                'total_frames': self.stats['total_frames'],
                'sampled_frames': self.stats['sampled_frames'],
                'detected_frames': self.stats['detected_frames'],
                'failed_frames': self.stats['failed_frames'],
                'detection_rate': processing_results.get('detection_rate', 0)
            },
            'quality_analysis': decision_results,
            'output_files': output_results,
            'debug_files': {
                'debug_frames': len(list((self.output_dir / "debug_frames").glob("*.jpg"))),
                'cropped_frames': len(list((self.output_dir / "cropped_frames").glob("*.jpg")))
            }
        }
        
        # Save report
        report_path = self.output_dir / "comprehensive_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print final summary
        logger.info("\n" + "="*80)
        logger.info("🎉 COMPREHENSIVE TEST COMPLETE!")
        logger.info("="*80)
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"⏱️  Processing time: {self.stats['processing_time']:.2f} seconds")
        logger.info(f"🎯 Detection mode: {self.stats['detection_mode']}")
        logger.info(f"📊 Detection rate: {processing_results.get('detection_rate', 0):.1%}")
        logger.info(f"🎬 Final decision: {decision_results['decision'].upper()}")
        logger.info(f"📝 Reason: {decision_results['reason']}")
        logger.info(f"📄 Report saved: {report_path}")
        
        if output_results['output_files']:
            logger.info("📹 Output videos:")
            for file in output_results['output_files']:
                logger.info(f"   {file}")
        
        logger.info(f"🖼️  Debug visualizations: {report['debug_files']['debug_frames']} frames")
        logger.info(f"✂️  Cropped frames: {report['debug_files']['cropped_frames']} frames")
        
        return report


def main():
    """Main function to test the fixed pipeline on a single video."""
    
    # Target video path
    video_path = "/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped/doctor__useruser01__18to39__female__aboriginal__20250807T054104_topmid.mp4"
    
    # Create tester
    tester = SingleVideoTester(video_path, "single_video_test_output")
    
    # Run comprehensive test
    results = tester.process_video_comprehensive()
    
    if results:
        logger.info("\n✅ Test completed successfully!")
        logger.info("Check the output directory for all generated files.")
    else:
        logger.error("\n❌ Test failed!")


if __name__ == "__main__":
    main()
