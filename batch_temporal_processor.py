#!/usr/bin/env python3
"""
Batch Temporal Processor - Diverse ICU Dataset Sample Processing
================================================================

Processes a demographically diverse sample of 10 videos from the ICU dataset
using the fixed temporal processor to validate temporal preservation across
different demographic groups.

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

# Import our fixed utilities
from improved_roi_utils import AdaptiveLipDetector
from roi_utils import ROIGeometry, BBoxSmoother, RecropCalculator, create_debug_visualization

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemographicVideoSelector:
    """
    Selects demographically diverse videos from the ICU dataset.
    """
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.video_files = list(self.dataset_dir.glob("*.mp4"))
        logger.info(f"Found {len(self.video_files)} video files in dataset")
        
    def parse_filename_demographics(self, filename: str) -> Dict[str, str]:
        """
        Parse demographic information from ICU dataset filename.
        Expected pattern: role__user__age__gender__ethnicity__timestamp_location.mp4
        """
        try:
            # Remove .mp4 extension and split by underscores
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
                # Fallback parsing
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
            
    def select_diverse_sample(self, sample_size: int = 10) -> List[Dict[str, Any]]:
        """
        Select a demographically diverse sample of videos.
        """
        logger.info(f"Selecting {sample_size} demographically diverse videos...")
        
        # Parse all video demographics
        video_demographics = []
        for video_file in self.video_files:
            demographics = self.parse_filename_demographics(video_file.name)
            demographics['filepath'] = video_file
            demographics['filename'] = video_file.name
            video_demographics.append(demographics)
            
        # Group by demographic combinations
        demographic_groups = defaultdict(list)
        for video in video_demographics:
            # Create demographic key (excluding user and timestamp)
            demo_key = f"{video['role']}_{video['age_group']}_{video['gender']}_{video['ethnicity']}"
            demographic_groups[demo_key].append(video)
            
        logger.info(f"Found {len(demographic_groups)} unique demographic combinations")
        
        # Log demographic distribution
        for demo_key, videos in demographic_groups.items():
            logger.info(f"  {demo_key}: {len(videos)} videos")
            
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
                
        # Second pass: fill remaining slots with random videos from unused combinations
        remaining_videos = [v for v in video_demographics 
                          if f"{v['role']}_{v['age_group']}_{v['gender']}_{v['ethnicity']}" not in used_combinations]
        
        while len(selected_videos) < sample_size and remaining_videos:
            selected_video = random.choice(remaining_videos)
            selected_videos.append(selected_video)
            demo_key = f"{selected_video['role']}_{selected_video['age_group']}_{selected_video['gender']}_{selected_video['ethnicity']}"
            used_combinations.add(demo_key)
            remaining_videos = [v for v in remaining_videos 
                              if f"{v['role']}_{v['age_group']}_{v['gender']}_{v['ethnicity']}" != demo_key]
            logger.info(f"Selected: {selected_video['filename']} ({demo_key})")
            
        # Third pass: fill any remaining slots randomly
        if len(selected_videos) < sample_size:
            remaining_count = sample_size - len(selected_videos)
            all_remaining = [v for v in video_demographics if v not in selected_videos]
            additional_videos = random.sample(all_remaining, min(remaining_count, len(all_remaining)))
            selected_videos.extend(additional_videos)
            
            for video in additional_videos:
                demo_key = f"{video['role']}_{video['age_group']}_{video['gender']}_{video['ethnicity']}"
                logger.info(f"Selected (additional): {video['filename']} ({demo_key})")
                
        logger.info(f"Final selection: {len(selected_videos)} videos")
        return selected_videos[:sample_size]


class BatchTemporalProcessor:
    """
    Batch processor for applying fixed temporal processing to multiple videos.
    """
    
    def __init__(self, output_dir: str = "fixed_temporal_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "full_processed").mkdir(exist_ok=True)
        (self.output_dir / "batch_reports").mkdir(exist_ok=True)
        (self.output_dir / "batch_debug").mkdir(exist_ok=True)
        
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
        
        # Batch statistics
        self.batch_stats = {
            'total_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'total_processing_time': 0,
            'results': []
        }
        
    def process_single_video(self, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single video using the fixed temporal processor.
        """
        video_path = str(video_info['filepath'])
        video_name = video_info['filename']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING: {video_name}")
        logger.info(f"Demographics: {video_info['role']}, {video_info['age_group']}, {video_info['gender']}, {video_info['ethnicity']}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Initialize detector for this video
            detector = AdaptiveLipDetector(
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
            
            # Force detection mode detection
            logger.info("Detecting processing mode...")
            for i in range(min(10, total_frames)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    landmarks = detector.detect_lip_landmarks(frame)
                    if landmarks is not None:
                        break
                        
            detection_mode = detector.get_detection_mode()
            logger.info(f"Detection mode: {detection_mode}")
            
            # Process all frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            
            # Create output filename
            output_name = f"processed_{video_name}"
            output_path = self.output_dir / "full_processed" / output_name
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (self.args.out_size, self.args.out_size))
            
            # Initialize smoother
            smoother = BBoxSmoother(alpha=self.args.ema)
            
            # Process all frames
            processed_frames = 0
            successful_detections = 0
            
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                processed_frames += 1
                
                # Detect landmarks
                landmarks = detector.detect_lip_landmarks(frame)
                
                if landmarks is not None:
                    successful_detections += 1
                    
                    # Calculate bbox
                    tight_bbox = ROIGeometry.calculate_tight_bbox(landmarks)
                    padded_bbox = ROIGeometry.add_padding(
                        tight_bbox, self.args.pad, frame.shape[:2]
                    )
                    smoothed_bbox = smoother.smooth(padded_bbox)
                else:
                    # Use previous bbox if available
                    smoothed_bbox = smoother.get_last_bbox()
                    if smoothed_bbox is None:
                        # Fallback to center crop
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
                    logger.info(f"  Progress: {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
                    
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
            
            logger.info(f"✅ SUCCESS: {video_name}")
            logger.info(f"  Processed: {processed_frames} frames in {processing_time:.2f}s")
            logger.info(f"  Detection rate: {detection_rate:.1%}")
            logger.info(f"  Output: {out_frames} frames, {out_duration:.2f}s")
            logger.info(f"  Temporal preservation: {out_frames/total_frames:.1%}")
            
            return {
                'success': True,
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
                    'detection_rate': detection_rate
                },
                'temporal_preservation': {
                    'frame_preservation_rate': out_frames / total_frames if total_frames > 0 else 0,
                    'duration_preservation_rate': out_duration / duration if duration > 0 else 0,
                    'status': 'PRESERVED' if out_frames >= total_frames * 0.95 else 'PARTIAL'
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ FAILED: {video_name} - {str(e)}")
            
            return {
                'success': False,
                'video_info': video_info,
                'processing_time': processing_time,
                'error': str(e)
            }
            
    def process_batch(self, selected_videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of selected videos.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING BATCH PROCESSING - {len(selected_videos)} VIDEOS")
        logger.info(f"{'='*80}")
        
        self.batch_stats['total_videos'] = len(selected_videos)
        batch_start_time = time.time()
        
        # Process each video
        for i, video_info in enumerate(selected_videos, 1):
            logger.info(f"\n[{i}/{len(selected_videos)}] Processing video...")
            
            result = self.process_single_video(video_info)
            self.batch_stats['results'].append(result)
            
            if result['success']:
                self.batch_stats['successful_videos'] += 1
            else:
                self.batch_stats['failed_videos'] += 1
                
        self.batch_stats['total_processing_time'] = time.time() - batch_start_time
        
        # Generate summary report
        summary_report = self._generate_batch_summary()
        
        return summary_report
        
    def _generate_batch_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive batch processing summary.
        """
        logger.info(f"\n{'='*80}")
        logger.info("BATCH PROCESSING COMPLETE - GENERATING SUMMARY")
        logger.info(f"{'='*80}")
        
        # Calculate statistics
        successful_results = [r for r in self.batch_stats['results'] if r['success']]
        
        if successful_results:
            avg_detection_rate = np.mean([r['processing_stats']['detection_rate'] for r in successful_results])
            avg_temporal_preservation = np.mean([r['temporal_preservation']['frame_preservation_rate'] for r in successful_results])
            total_original_frames = sum([r['original_properties']['frames'] for r in successful_results])
            total_output_frames = sum([r['output_properties']['frames'] for r in successful_results])
        else:
            avg_detection_rate = 0
            avg_temporal_preservation = 0
            total_original_frames = 0
            total_output_frames = 0
            
        summary = {
            'batch_processing_complete': True,
            'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
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
                'overall_frame_preservation': total_output_frames / total_original_frames if total_original_frames > 0 else 0
            },
            'demographic_diversity': self._analyze_demographic_diversity(),
            'individual_results': self.batch_stats['results']
        }
        
        # Save summary report
        summary_path = self.output_dir / "batch_reports" / "batch_summary_report.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        # Print summary
        self._print_batch_summary(summary)
        
        return summary
        
    def _analyze_demographic_diversity(self) -> Dict[str, Any]:
        """
        Analyze demographic diversity of processed videos.
        """
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
        
    def _print_batch_summary(self, summary: Dict[str, Any]):
        """
        Print comprehensive batch summary.
        """
        stats = summary['batch_statistics']
        quality = summary['quality_metrics']
        diversity = summary['demographic_diversity']
        
        logger.info(f"\n🎉 BATCH PROCESSING SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"📊 Processing Statistics:")
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
        
        logger.info(f"\n🌍 Demographic Diversity:")
        logger.info(f"  Unique combinations: {diversity['unique_combinations_count']}")
        logger.info(f"  Roles: {diversity['roles']}")
        logger.info(f"  Age groups: {diversity['age_groups']}")
        logger.info(f"  Genders: {diversity['genders']}")
        logger.info(f"  Ethnicities: {diversity['ethnicities']}")
        
        logger.info(f"\n📁 Output Location:")
        logger.info(f"  Processed videos: {self.output_dir}/full_processed/")
        logger.info(f"  Summary report: {self.output_dir}/batch_reports/batch_summary_report.json")


def main():
    """
    Main function to process diverse sample of ICU dataset videos.
    """
    # Set random seed for reproducible selection
    random.seed(42)
    
    # Dataset directory
    dataset_dir = "/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped"
    
    # Select diverse sample
    selector = DemographicVideoSelector(dataset_dir)
    selected_videos = selector.select_diverse_sample(sample_size=10)
    
    # Process batch
    processor = BatchTemporalProcessor("fixed_temporal_output")
    summary = processor.process_batch(selected_videos)
    
    if summary['batch_processing_complete']:
        logger.info("\n✅ Batch processing completed successfully!")
        logger.info("Check the output directory for processed videos and reports.")
    else:
        logger.error("\n❌ Batch processing encountered issues!")


if __name__ == "__main__":
    main()
