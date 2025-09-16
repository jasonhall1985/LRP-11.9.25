#!/usr/bin/env python3
"""
Batch Temporal Processor V2 - Enhanced Vertical Positioning
===========================================================

Enhanced batch processor with improved vertical positioning for mouth ROI crops.
Tests the fixed geometric detection algorithm on a new diverse sample.

Key improvements:
- Uses AdaptiveLipDetectorV2 with corrected vertical positioning
- Better lip region parameters: (0.15, 0.25, 0.7, 0.5) vs (0.2, 0.1, 0.6, 0.4)
- 25% larger crop area for better tolerance
- Enhanced debug visualizations
- Comparison reporting with previous version

Author: Augment Agent
Date: 2025-09-14
"""

import cv2
import numpy as np
import os
import json
import logging
import random
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import time
from collections import defaultdict

# Import our enhanced utilities - V3 for aggressive positioning
from improved_roi_utils_v3 import AdaptiveLipDetectorV3, create_debug_visualization_v3
from roi_utils import ROIGeometry, BBoxSmoother, RecropCalculator

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedBatchProcessor:
    """
    Enhanced batch processor with improved vertical positioning validation.
    """
    
    def __init__(self, dataset_dir: str, output_dir: str = "fixed_temporal_output"):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create enhanced subdirectories
        (self.output_dir / "full_processed").mkdir(exist_ok=True)
        (self.output_dir / "v2_debug").mkdir(exist_ok=True)
        (self.output_dir / "v2_reports").mkdir(exist_ok=True)
        (self.output_dir / "positioning_comparison").mkdir(exist_ok=True)
        
        # Get video files
        self.video_files = list(self.dataset_dir.glob("*.mp4"))
        logger.info(f"Found {len(self.video_files)} video files in dataset")
        
        # Processing parameters
        self.args = type('Args', (), {
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
        
        # Enhanced statistics
        self.batch_stats = {
            'total_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'correct_positioning': 0,
            'total_processing_time': 0,
            'results': [],
            'positioning_analysis': []
        }
        
    def select_new_diverse_sample(self, sample_size: int = 10, exclude_previous: List[str] = None) -> List[Dict[str, Any]]:
        """
        Select a new diverse sample, excluding previously processed videos.
        """
        logger.info(f"Selecting {sample_size} new demographically diverse videos...")
        
        if exclude_previous is None:
            exclude_previous = []
            
        # Parse all video demographics
        video_demographics = []
        for video_file in self.video_files:
            if video_file.name not in exclude_previous:
                demographics = self._parse_filename_demographics(video_file.name)
                demographics['filepath'] = video_file
                demographics['filename'] = video_file.name
                video_demographics.append(demographics)
                
        logger.info(f"Available videos after exclusion: {len(video_demographics)}")
        
        # Group by demographic combinations
        demographic_groups = defaultdict(list)
        for video in video_demographics:
            demo_key = f"{video['role']}_{video['age_group']}_{video['gender']}_{video['ethnicity']}"
            demographic_groups[demo_key].append(video)
            
        logger.info(f"Found {len(demographic_groups)} unique demographic combinations")
        
        # Select diverse sample
        selected_videos = []
        used_combinations = set()
        
        # First pass: select one video from each unique demographic combination
        for demo_key, videos in demographic_groups.items():
            if len(selected_videos) < sample_size:
                selected_video = random.choice(videos)
                selected_videos.append(selected_video)
                used_combinations.add(demo_key)
                logger.info(f"Selected: {selected_video['filename']} ({demo_key})")
                
        # Fill remaining slots if needed
        while len(selected_videos) < sample_size and len(video_demographics) > len(selected_videos):
            remaining_videos = [v for v in video_demographics if v not in selected_videos]
            if remaining_videos:
                selected_video = random.choice(remaining_videos)
                selected_videos.append(selected_video)
                demo_key = f"{selected_video['role']}_{selected_video['age_group']}_{selected_video['gender']}_{selected_video['ethnicity']}"
                logger.info(f"Selected (additional): {selected_video['filename']} ({demo_key})")
            else:
                break
                
        logger.info(f"Final V2 selection: {len(selected_videos)} videos")
        return selected_videos[:sample_size]
        
    def _parse_filename_demographics(self, filename: str) -> Dict[str, str]:
        """Parse demographic information from ICU dataset filename."""
        try:
            base_name = filename.replace('.mp4', '')
            parts = base_name.split('__')
            
            if len(parts) >= 5:
                return {
                    'role': parts[0],
                    'user': parts[1], 
                    'age_group': parts[2],
                    'gender': parts[3],
                    'ethnicity': parts[4],
                    'timestamp_location': '__'.join(parts[5:]) if len(parts) > 5 else 'unknown'
                }
            else:
                return {
                    'role': 'unknown',
                    'user': 'unknown',
                    'age_group': 'unknown', 
                    'gender': 'unknown',
                    'ethnicity': 'unknown',
                    'timestamp_location': base_name
                }
        except Exception as e:
            logger.warning(f"Could not parse demographics from {filename}: {e}")
            return {
                'role': 'unknown',
                'user': 'unknown',
                'age_group': 'unknown',
                'gender': 'unknown', 
                'ethnicity': 'unknown',
                'timestamp_location': filename
            }
            
    def process_single_video_v2(self, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single video using the enhanced V2 detector.
        """
        video_path = str(video_info['filepath'])
        video_name = video_info['filename']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING V2: {video_name}")
        logger.info(f"Demographics: {video_info['role']}, {video_info['age_group']}, {video_info['gender']}, {video_info['ethnicity']}")
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
            output_path = self.output_dir / "full_processed" / output_name
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (self.args.out_size, self.args.out_size))
            
            # Initialize smoother
            smoother = BBoxSmoother(alpha=self.args.ema)
            
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
                        tight_bbox, self.args.pad, frame.shape[:2]
                    )
                    smoothed_bbox = smoother.smooth(padded_bbox)
                    
                    # Save debug visualization every 20 frames
                    if frame_idx % 20 == 0:
                        ratios = ROIGeometry.calculate_size_ratios(smoothed_bbox, frame.shape[:2])
                        debug_frame = create_debug_visualization_v2(
                            frame, landmarks, smoothed_bbox, detector, ratios
                        )
                        debug_path = self.output_dir / "v2_debug" / f"v2_debug_{video_name}_{frame_idx:04d}.jpg"
                        cv2.imwrite(str(debug_path), debug_frame)
                        debug_frames_saved += 1
                        
                else:
                    # Use previous bbox if available
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
                    resized = cv2.resize(cropped, (self.args.out_size, self.args.out_size))
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
            
            logger.info(f"✅ V2 SUCCESS: {video_name}")
            logger.info(f"  Processed: {processed_frames} frames in {processing_time:.2f}s")
            logger.info(f"  Detection rate: {detection_rate:.1%}")
            logger.info(f"  Output: {out_frames} frames, {out_duration:.2f}s")
            logger.info(f"  Temporal preservation: {out_frames/total_frames:.1%}")
            logger.info(f"  Debug frames saved: {debug_frames_saved}")
            
            return {
                'success': True,
                'version': 'v2',
                'video_info': video_info,
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
                    'dimensions': f"{self.args.out_size}x{self.args.out_size}",
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
                },
                'positioning_improvements': {
                    'algorithm_version': 'v2',
                    'lip_region_params': detector.expected_lip_region,
                    'area_expansion_factor': detector.area_expansion_factor,
                    'vertical_offset_ratio': detector.vertical_offset_ratio
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ V2 FAILED: {video_name} - {str(e)}")
            
            return {
                'success': False,
                'version': 'v2',
                'video_info': video_info,
                'processing_time': processing_time,
                'error': str(e)
            }
            
    def process_v2_batch(self, selected_videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of videos with V2 enhanced positioning.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING V2 BATCH PROCESSING - {len(selected_videos)} VIDEOS")
        logger.info(f"Enhanced Vertical Positioning Algorithm")
        logger.info(f"{'='*80}")
        
        self.batch_stats['total_videos'] = len(selected_videos)
        batch_start_time = time.time()
        
        # Process each video with V2 algorithm
        for i, video_info in enumerate(selected_videos, 1):
            logger.info(f"\n[{i}/{len(selected_videos)}] Processing video with V2...")
            
            result = self.process_single_video_v2(video_info)
            self.batch_stats['results'].append(result)
            
            if result['success']:
                self.batch_stats['successful_videos'] += 1
            else:
                self.batch_stats['failed_videos'] += 1
                
        self.batch_stats['total_processing_time'] = time.time() - batch_start_time
        
        # Generate V2 summary report
        summary_report = self._generate_v2_summary()
        
        return summary_report
        
    def _generate_v2_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive V2 batch processing summary.
        """
        logger.info(f"\n{'='*80}")
        logger.info("V2 BATCH PROCESSING COMPLETE - GENERATING ENHANCED SUMMARY")
        logger.info(f"{'='*80}")
        
        # Calculate enhanced statistics
        successful_results = [r for r in self.batch_stats['results'] if r['success']]
        
        if successful_results:
            avg_detection_rate = np.mean([r['processing_stats']['detection_rate'] for r in successful_results])
            avg_temporal_preservation = np.mean([r['temporal_preservation']['frame_preservation_rate'] for r in successful_results])
            total_original_frames = sum([r['original_properties']['frames'] for r in successful_results])
            total_output_frames = sum([r['output_properties']['frames'] for r in successful_results])
            total_debug_frames = sum([r['processing_stats']['debug_frames_saved'] for r in successful_results])
        else:
            avg_detection_rate = 0
            avg_temporal_preservation = 0
            total_original_frames = 0
            total_output_frames = 0
            total_debug_frames = 0
            
        summary = {
            'v2_batch_processing_complete': True,
            'algorithm_version': 'v2_enhanced_positioning',
            'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'improvements': {
                'lip_region_params': '(0.15, 0.25, 0.7, 0.5) vs (0.2, 0.1, 0.6, 0.4)',
                'area_expansion': '25% larger crop area',
                'vertical_positioning': 'Enhanced centering algorithm',
                'debug_visualizations': 'Improved positioning indicators'
            },
            'batch_statistics': {
                'total_videos': self.batch_stats['total_videos'],
                'successful_videos': self.batch_stats['successful_videos'],
                'failed_videos': self.batch_stats['failed_videos'],
                'success_rate': self.batch_stats['successful_videos'] / self.batch_stats['total_videos'] if self.batch_stats['total_videos'] > 0 else 0,
                'total_processing_time': self.batch_stats['total_processing_time'],
                'average_processing_time': self.batch_stats['total_processing_time'] / self.batch_stats['total_videos'] if self.batch_stats['total_videos'] > 0 else 0
            },
            'quality_metrics': {
                'average_detection_rate': avg_detection_rate,
                'average_temporal_preservation': avg_temporal_preservation,
                'total_frames_processed': total_original_frames,
                'total_frames_output': total_output_frames,
                'overall_frame_preservation': total_output_frames / total_original_frames if total_original_frames > 0 else 0,
                'total_debug_frames_generated': total_debug_frames
            },
            'demographic_diversity': self._analyze_demographic_diversity(),
            'individual_results': self.batch_stats['results']
        }
        
        # Save V2 summary report
        summary_path = self.output_dir / "v2_reports" / "v2_batch_summary_report.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        # Print enhanced summary
        self._print_v2_summary(summary)
        
        return summary
        
    def _analyze_demographic_diversity(self) -> Dict[str, Any]:
        """Analyze demographic diversity of V2 processed videos."""
        demographics = {
            'roles': defaultdict(int),
            'age_groups': defaultdict(int),
            'genders': defaultdict(int),
            'ethnicities': defaultdict(int),
            'unique_combinations': set()
        }
        
        for result in self.batch_stats['results']:
            if result['success']:
                info = result['video_info']
                demographics['roles'][info['role']] += 1
                demographics['age_groups'][info['age_group']] += 1
                demographics['genders'][info['gender']] += 1
                demographics['ethnicities'][info['ethnicity']] += 1
                
                combo = f"{info['role']}_{info['age_group']}_{info['gender']}_{info['ethnicity']}"
                demographics['unique_combinations'].add(combo)
                
        return {
            'roles': dict(demographics['roles']),
            'age_groups': dict(demographics['age_groups']),
            'genders': dict(demographics['genders']),
            'ethnicities': dict(demographics['ethnicities']),
            'unique_combinations_count': len(demographics['unique_combinations']),
            'unique_combinations': list(demographics['unique_combinations'])
        }
        
    def _print_v2_summary(self, summary: Dict[str, Any]):
        """Print comprehensive V2 batch summary."""
        stats = summary['batch_statistics']
        quality = summary['quality_metrics']
        diversity = summary['demographic_diversity']
        improvements = summary['improvements']
        
        logger.info(f"\n🎉 V2 ENHANCED BATCH PROCESSING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"🔧 Algorithm Improvements:")
        logger.info(f"  Lip region params: {improvements['lip_region_params']}")
        logger.info(f"  Area expansion: {improvements['area_expansion']}")
        logger.info(f"  Vertical positioning: {improvements['vertical_positioning']}")
        logger.info(f"  Debug visualizations: {improvements['debug_visualizations']}")
        
        logger.info(f"\n📊 Processing Statistics:")
        logger.info(f"  Total videos: {stats['total_videos']}")
        logger.info(f"  Successful: {stats['successful_videos']} ({stats['success_rate']:.1%})")
        logger.info(f"  Failed: {stats['failed_videos']}")
        logger.info(f"  Total time: {stats['total_processing_time']:.2f} seconds")
        logger.info(f"  Average time per video: {stats['average_processing_time']:.2f} seconds")
        
        logger.info(f"\n📏 Quality Metrics:")
        logger.info(f"  Average detection rate: {quality['average_detection_rate']:.1%}")
        logger.info(f"  Average temporal preservation: {quality['average_temporal_preservation']:.1%}")
        logger.info(f"  Overall frame preservation: {quality['overall_frame_preservation']:.1%}")
        logger.info(f"  Total frames: {quality['total_frames_processed']} → {quality['total_frames_output']}")
        logger.info(f"  Debug frames generated: {quality['total_debug_frames_generated']}")
        
        logger.info(f"\n🌍 Demographic Diversity:")
        logger.info(f"  Unique combinations: {diversity['unique_combinations_count']}")
        logger.info(f"  Roles: {diversity['roles']}")
        logger.info(f"  Age groups: {diversity['age_groups']}")
        logger.info(f"  Genders: {diversity['genders']}")
        logger.info(f"  Ethnicities: {diversity['ethnicities']}")
        
        logger.info(f"\n📁 V2 Output Locations:")
        logger.info(f"  Processed videos: {self.output_dir}/full_processed/ (processed_v2_*.mp4)")
        logger.info(f"  Debug visualizations: {self.output_dir}/v2_debug/")
        logger.info(f"  Summary report: {self.output_dir}/v2_reports/v2_batch_summary_report.json")


def main():
    """
    Main function to process new diverse sample with V2 enhanced positioning.
    """
    # Set random seed for reproducible selection
    random.seed(123)  # Different seed for new selection
    
    # Dataset directory
    dataset_dir = "/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped"
    
    # Previous batch filenames to exclude
    previous_batch = [
        "help__useruser01__65plus__female__caucasian__20250731T053024_topmid.mp4",
        "doctor__useruser01__40to64__female__caucasian__20250820T085728_topmid.mp4",
        "i_need_to_move__useruser01__65plus__male__caucasian__20250723T071534_topmid.mp4",
        "help__useruser01__18to39__female__asian__20250911T033329_topmid.mp4",
        "phone__useruser01__18to39__male__not_specified__20250820T092039_topmid.mp4",
        "glasses__useruser01__65plus__female__caucasian__20250716T052347_topmid.mp4",
        "help__useruser01__18to39__male__caucasian__20250910T191901_topmid.mp4",
        "doctor__useruser01__65plus__female__caucasian__20250827T054201_topmid.mp4",
        "pillow__useruser01__65plus__female__caucasian__20250723T043054_topmid.mp4",
        "glasses__useruser01__40to64__female__caucasian__20250903T021404_topmid.mp4"
    ]
    
    # Create enhanced processor
    processor = EnhancedBatchProcessor(dataset_dir, "fixed_temporal_output")
    
    # Select new diverse sample
    selected_videos = processor.select_new_diverse_sample(sample_size=10, exclude_previous=previous_batch)
    
    # Process V2 batch
    summary = processor.process_v2_batch(selected_videos)
    
    if summary['v2_batch_processing_complete']:
        logger.info("\n✅ V2 Enhanced batch processing completed successfully!")
        logger.info("Check the output directory for processed videos with improved positioning.")
    else:
        logger.error("\n❌ V2 Enhanced batch processing encountered issues!")


if __name__ == "__main__":
    main()
