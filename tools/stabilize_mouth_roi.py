#!/usr/bin/env python3
"""
Mouth ROI Stabilization Tool
============================

Stabilizes mouth regions in ICU lip-reading videos using geometric cropping
optimized for the ICU dataset format (lower half of faces with lips in 
top-middle portion).

This tool addresses the primary generalization bottleneck by ensuring consistent
mouth region positioning across all videos and speakers.

Key Features:
- Geometric cropping (top 50% height, middle 33% width) 
- Preserves original pixel dimensions (no resizing)
- Handles ICU-style cropped face videos
- Batch processing with progress tracking
- Quality validation and reporting

Author: Augment Agent  
Date: 2025-09-27
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict
import shutil

class MouthROIStabilizer:
    """Stabilizes mouth regions in lip-reading videos using geometric cropping."""
    
    def __init__(self, crop_height_ratio: float = 0.5, crop_width_ratio: float = 0.33):
        """
        Initialize ROI stabilizer.
        
        Args:
            crop_height_ratio: Fraction of height to keep (from top)
            crop_width_ratio: Fraction of width to keep (from center)
        """
        self.crop_height_ratio = crop_height_ratio
        self.crop_width_ratio = crop_width_ratio
        self.processing_stats = []
    
    def calculate_crop_region(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """
        Calculate crop region coordinates for geometric cropping.
        
        Args:
            frame_width: Original frame width
            frame_height: Original frame height
            
        Returns:
            Tuple of (x1, y1, x2, y2) crop coordinates
        """
        # Calculate crop dimensions
        crop_height = int(frame_height * self.crop_height_ratio)
        crop_width = int(frame_width * self.crop_width_ratio)
        
        # Center the crop horizontally, start from top vertically
        x1 = (frame_width - crop_width) // 2
        y1 = 0
        x2 = x1 + crop_width
        y2 = y1 + crop_height
        
        return (x1, y1, x2, y2)
    
    def stabilize_video_roi(self, input_path: Path, output_path: Path) -> Dict:
        """
        Stabilize mouth ROI for a single video.
        
        Args:
            input_path: Path to input video
            output_path: Path to output stabilized video
            
        Returns:
            Processing statistics dictionary
        """
        if not input_path.exists():
            return self.create_error_result(input_path, "Input file not found")
        
        try:
            # Open input video
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                return self.create_error_result(input_path, "Cannot open input video")
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate crop region
            x1, y1, x2, y2 = self.calculate_crop_region(original_width, original_height)
            crop_width = x2 - x1
            crop_height = y2 - y1
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path), fourcc, fps, (crop_width, crop_height)
            )
            
            if not out.isOpened():
                cap.release()
                return self.create_error_result(input_path, "Cannot create output video")
            
            # Process frames
            frames_processed = 0
            frames_failed = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Apply geometric crop
                    cropped_frame = frame[y1:y2, x1:x2]
                    
                    # Validate crop
                    if cropped_frame.shape[0] != crop_height or cropped_frame.shape[1] != crop_width:
                        frames_failed += 1
                        continue
                    
                    # Write frame
                    out.write(cropped_frame)
                    frames_processed += 1
                    
                except Exception as e:
                    frames_failed += 1
                    continue
            
            # Cleanup
            cap.release()
            out.release()
            
            # Calculate statistics
            processing_stats = {
                'input_path': str(input_path),
                'output_path': str(output_path),
                'success': True,
                'original_resolution': (original_width, original_height),
                'cropped_resolution': (crop_width, crop_height),
                'crop_region': (x1, y1, x2, y2),
                'total_frames': frame_count,
                'frames_processed': frames_processed,
                'frames_failed': frames_failed,
                'processing_success_rate': frames_processed / frame_count if frame_count > 0 else 0.0,
                'fps': fps,
                'crop_ratio_applied': {
                    'height': self.crop_height_ratio,
                    'width': self.crop_width_ratio
                }
            }
            
            return processing_stats
            
        except Exception as e:
            return self.create_error_result(input_path, f"Processing error: {str(e)}")
    
    def create_error_result(self, input_path: Path, error_msg: str) -> Dict:
        """Create error result for failed processing."""
        return {
            'input_path': str(input_path),
            'output_path': None,
            'success': False,
            'error': error_msg,
            'frames_processed': 0,
            'processing_success_rate': 0.0
        }
    
    def validate_stabilized_video(self, video_path: Path) -> Dict:
        """
        Validate quality of stabilized video.
        
        Args:
            video_path: Path to stabilized video
            
        Returns:
            Validation metrics dictionary
        """
        if not video_path.exists():
            return {'valid': False, 'error': 'File not found'}
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {'valid': False, 'error': 'Cannot open video'}
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Sample a few frames for quality check
            sample_frames = min(5, frame_count)
            frame_qualities = []
            
            for i in range(sample_frames):
                frame_idx = i * (frame_count // sample_frames) if sample_frames > 1 else 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Calculate frame quality metrics
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Contrast (standard deviation)
                    contrast = np.std(gray) / 255.0
                    
                    # Sharpness (Laplacian variance)
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    sharpness = np.var(laplacian) / 10000.0  # Normalize
                    
                    # Brightness (mean intensity)
                    brightness = np.mean(gray) / 255.0
                    
                    frame_quality = {
                        'contrast': contrast,
                        'sharpness': min(1.0, sharpness),
                        'brightness': brightness,
                        'overall': (contrast + min(1.0, sharpness) + (1.0 - abs(brightness - 0.5) * 2)) / 3.0
                    }
                    frame_qualities.append(frame_quality)
            
            cap.release()
            
            if frame_qualities:
                avg_quality = np.mean([fq['overall'] for fq in frame_qualities])
                avg_contrast = np.mean([fq['contrast'] for fq in frame_qualities])
                avg_sharpness = np.mean([fq['sharpness'] for fq in frame_qualities])
                avg_brightness = np.mean([fq['brightness'] for fq in frame_qualities])
                
                return {
                    'valid': True,
                    'resolution': (width, height),
                    'frame_count': frame_count,
                    'fps': fps,
                    'quality_metrics': {
                        'overall_quality': avg_quality,
                        'contrast': avg_contrast,
                        'sharpness': avg_sharpness,
                        'brightness': avg_brightness
                    },
                    'quality_assessment': 'high' if avg_quality > 0.7 else 'medium' if avg_quality > 0.4 else 'low'
                }
            else:
                return {'valid': False, 'error': 'No frames could be analyzed'}
                
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}

def stabilize_speaker_sets(input_dir: Path, output_dir: Path, 
                          crop_height_ratio: float = 0.5, 
                          crop_width_ratio: float = 0.33,
                          max_videos_per_class: int = None) -> Dict:
    """
    Stabilize ROI for all videos in speaker sets directory.
    
    Args:
        input_dir: Input directory containing speaker sets
        output_dir: Output directory for stabilized videos
        crop_height_ratio: Height crop ratio (default: 0.5 for top 50%)
        crop_width_ratio: Width crop ratio (default: 0.33 for middle 33%)
        max_videos_per_class: Maximum videos to process per class (None for all)
        
    Returns:
        Processing results dictionary
    """
    
    stabilizer = MouthROIStabilizer(crop_height_ratio, crop_width_ratio)
    results = {
        'processing_summary': {},
        'speaker_results': {},
        'class_results': {},
        'overall_statistics': {},
        'failed_videos': []
    }
    
    print(f"🔧 Stabilizing mouth ROI in: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📏 Crop ratios - Height: {crop_height_ratio:.1%}, Width: {crop_width_ratio:.1%}")
    
    # Collect all videos to process
    all_videos = []
    for speaker_dir in input_dir.iterdir():
        if not speaker_dir.is_dir() or not speaker_dir.name.startswith('speaker'):
            continue
            
        for class_dir in speaker_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_videos = list(class_dir.glob('*.mp4'))
            if max_videos_per_class:
                class_videos = class_videos[:max_videos_per_class]
                
            for video_path in class_videos:
                # Create corresponding output path
                relative_path = video_path.relative_to(input_dir)
                output_path = output_dir / relative_path
                
                all_videos.append({
                    'input_path': video_path,
                    'output_path': output_path,
                    'speaker': speaker_dir.name,
                    'class': class_dir.name
                })
    
    print(f"📊 Found {len(all_videos)} videos to stabilize")
    
    # Process each video
    processing_results = []
    successful_count = 0
    failed_count = 0
    
    for video_info in tqdm(all_videos, desc="Stabilizing videos"):
        result = stabilizer.stabilize_video_roi(
            video_info['input_path'], 
            video_info['output_path']
        )
        
        result['speaker'] = video_info['speaker']
        result['class'] = video_info['class']
        processing_results.append(result)
        
        if result['success']:
            successful_count += 1
        else:
            failed_count += 1
            results['failed_videos'].append({
                'path': str(video_info['input_path']),
                'error': result.get('error', 'Unknown error')
            })
    
    # Aggregate results
    if processing_results:
        successful_results = [r for r in processing_results if r['success']]
        
        # Overall statistics
        results['overall_statistics'] = {
            'total_videos_processed': len(processing_results),
            'successful_stabilizations': successful_count,
            'failed_stabilizations': failed_count,
            'success_rate': successful_count / len(processing_results),
            'avg_processing_success_rate': np.mean([r['processing_success_rate'] for r in successful_results]) if successful_results else 0.0,
            'crop_settings': {
                'height_ratio': crop_height_ratio,
                'width_ratio': crop_width_ratio
            }
        }
        
        # Per-speaker statistics
        speaker_stats = defaultdict(list)
        for result in successful_results:
            speaker_stats[result['speaker']].append(result)
        
        results['speaker_results'] = {}
        for speaker, speaker_results in speaker_stats.items():
            results['speaker_results'][speaker] = {
                'video_count': len(speaker_results),
                'avg_processing_success_rate': np.mean([r['processing_success_rate'] for r in speaker_results]),
                'total_frames_processed': sum(r['frames_processed'] for r in speaker_results),
                'resolutions': list(set(str(r['cropped_resolution']) for r in speaker_results))
            }
        
        # Per-class statistics  
        class_stats = defaultdict(list)
        for result in successful_results:
            class_stats[result['class']].append(result)
        
        results['class_results'] = {}
        for class_name, class_results in class_stats.items():
            results['class_results'][class_name] = {
                'video_count': len(class_results),
                'avg_processing_success_rate': np.mean([r['processing_success_rate'] for r in class_results]),
                'total_frames_processed': sum(r['frames_processed'] for r in class_results)
            }
    
    # Save detailed results
    results['detailed_results'] = processing_results
    
    return results

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Mouth ROI Stabilization Tool')
    parser.add_argument('--input-dir', default='data/speaker sets',
                       help='Input directory containing speaker sets')
    parser.add_argument('--output-dir', default='data/stabilized_speaker_sets',
                       help='Output directory for stabilized videos')
    parser.add_argument('--crop-height-ratio', type=float, default=0.5,
                       help='Height crop ratio (0.5 = top 50%)')
    parser.add_argument('--crop-width-ratio', type=float, default=0.33,
                       help='Width crop ratio (0.33 = middle 33%)')
    parser.add_argument('--max-videos-per-class', type=int, default=None,
                       help='Maximum videos to process per class')
    parser.add_argument('--report-path', default='reports/roi_stabilization_report.json',
                       help='Path to save processing report')
    
    args = parser.parse_args()
    
    print("🔧 MOUTH ROI STABILIZATION")
    print("=" * 40)
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create report directory
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process videos
    results = stabilize_speaker_sets(
        input_dir, output_dir, 
        args.crop_height_ratio, args.crop_width_ratio,
        args.max_videos_per_class
    )
    
    # Save report
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    if results['overall_statistics']:
        stats = results['overall_statistics']
        print(f"\n📊 STABILIZATION SUMMARY")
        print(f"Total videos processed: {stats['total_videos_processed']}")
        print(f"Successful stabilizations: {stats['successful_stabilizations']}")
        print(f"Failed stabilizations: {stats['failed_stabilizations']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Average frame processing rate: {stats['avg_processing_success_rate']:.1%}")
        
        if results['failed_videos']:
            print(f"\n⚠️  Failed videos: {len(results['failed_videos'])}")
            for failed in results['failed_videos'][:5]:  # Show first 5
                print(f"  - {failed['path']}: {failed['error']}")
            if len(results['failed_videos']) > 5:
                print(f"  ... and {len(results['failed_videos']) - 5} more")
    
    print(f"\n💾 Saved processing report: {report_path}")
    print(f"✅ ROI stabilization complete")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
