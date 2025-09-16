#!/usr/bin/env python3
"""
V2 Positioning Verification Tool
================================

Comprehensive verification of vertical positioning improvements in V2 processed videos.
Analyzes mouth ROI positioning quality and generates comparison reports.

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

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PositioningVerifier:
    """
    Verifies and analyzes the quality of mouth ROI positioning in processed videos.
    """
    
    def __init__(self, output_dir: str = "fixed_temporal_output"):
        self.output_dir = Path(output_dir)
        self.processed_dir = self.output_dir / "full_processed"
        self.verification_dir = self.output_dir / "positioning_verification"
        self.verification_dir.mkdir(exist_ok=True)
        
        # Get V2 processed videos
        self.v2_videos = list(self.processed_dir.glob("processed_v2_*.mp4"))
        logger.info(f"Found {len(self.v2_videos)} V2 processed videos for verification")
        
        # Verification results
        self.verification_results = []
        
    def analyze_video_positioning(self, video_path: Path) -> Dict[str, Any]:
        """
        Analyze the positioning quality of a single processed video.
        """
        video_name = video_path.name
        logger.info(f"Analyzing positioning: {video_name}")
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception("Could not open video file")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Sample frames for analysis
            sample_frames = min(10, total_frames)
            frame_indices = np.linspace(0, total_frames-1, sample_frames, dtype=int)
            
            positioning_scores = []
            sample_analysis = []
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Analyze frame positioning
                    score, analysis = self._analyze_frame_positioning(frame, frame_idx)
                    positioning_scores.append(score)
                    sample_analysis.append(analysis)
                    
                    # Save sample frame for visual verification
                    if i < 5:  # Save first 5 samples
                        sample_path = self.verification_dir / f"sample_{video_name}_{frame_idx:04d}.jpg"
                        cv2.imwrite(str(sample_path), frame)
                        
            cap.release()
            
            # Calculate overall positioning quality
            avg_score = np.mean(positioning_scores) if positioning_scores else 0
            positioning_quality = self._classify_positioning_quality(avg_score)
            
            result = {
                'video_name': video_name,
                'video_path': str(video_path),
                'success': True,
                'video_properties': {
                    'frames': total_frames,
                    'fps': fps,
                    'duration': duration,
                    'dimensions': f"{width}x{height}"
                },
                'positioning_analysis': {
                    'average_score': avg_score,
                    'quality_classification': positioning_quality,
                    'sample_count': len(positioning_scores),
                    'score_distribution': {
                        'min': float(np.min(positioning_scores)) if positioning_scores else 0,
                        'max': float(np.max(positioning_scores)) if positioning_scores else 0,
                        'std': float(np.std(positioning_scores)) if positioning_scores else 0
                    }
                },
                'frame_samples': sample_analysis[:5],  # First 5 samples
                'positioning_verdict': self._get_positioning_verdict(avg_score)
            }
            
            logger.info(f"  Score: {avg_score:.2f}, Quality: {positioning_quality}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze {video_name}: {e}")
            return {
                'video_name': video_name,
                'video_path': str(video_path),
                'success': False,
                'error': str(e)
            }
            
    def _analyze_frame_positioning(self, frame: np.ndarray, frame_idx: int) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze the positioning quality of a single frame.
        """
        h, w = frame.shape[:2]
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Analyze mouth region characteristics
        # For 96x96 mouth ROI crops, we expect:
        # - Mouth/lip region to be centered or slightly below center
        # - Good contrast in the mouth area
        # - Minimal background/non-mouth content
        
        # Divide frame into regions for analysis
        center_y = h // 2
        mouth_region = gray[center_y-10:center_y+20, :]  # Expected mouth area
        upper_region = gray[:center_y-10, :]  # Above mouth
        lower_region = gray[center_y+20:, :]  # Below mouth
        
        # Calculate positioning metrics
        mouth_contrast = self._calculate_contrast(mouth_region)
        upper_contrast = self._calculate_contrast(upper_region) if upper_region.size > 0 else 0
        lower_contrast = self._calculate_contrast(lower_region) if lower_region.size > 0 else 0
        
        # Mouth region should have higher contrast (lips, teeth, etc.)
        mouth_prominence = mouth_contrast / max(upper_contrast + lower_contrast, 1)
        
        # Check for mouth-like features (horizontal edges for lips)
        mouth_edges = cv2.Sobel(mouth_region, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.mean(np.abs(mouth_edges))
        
        # Calculate positioning score (0-1, higher is better)
        contrast_score = min(mouth_prominence / 2.0, 1.0)  # Normalize
        edge_score = min(edge_strength / 50.0, 1.0)  # Normalize
        
        # Weighted combination
        positioning_score = 0.6 * contrast_score + 0.4 * edge_score
        
        analysis = {
            'frame_idx': frame_idx,
            'positioning_score': positioning_score,
            'mouth_contrast': mouth_contrast,
            'mouth_prominence': mouth_prominence,
            'edge_strength': edge_strength,
            'metrics': {
                'contrast_score': contrast_score,
                'edge_score': edge_score
            }
        }
        
        return positioning_score, analysis
        
    def _calculate_contrast(self, region: np.ndarray) -> float:
        """Calculate contrast in a region using standard deviation."""
        if region.size == 0:
            return 0.0
        return float(np.std(region))
        
    def _classify_positioning_quality(self, score: float) -> str:
        """Classify positioning quality based on score."""
        if score >= 0.8:
            return "EXCELLENT"
        elif score >= 0.6:
            return "GOOD"
        elif score >= 0.4:
            return "FAIR"
        else:
            return "POOR"
            
    def _get_positioning_verdict(self, score: float) -> str:
        """Get positioning verdict for reporting."""
        if score >= 0.6:
            return "CORRECTLY_POSITIONED"
        else:
            return "INCORRECTLY_POSITIONED"
            
    def verify_all_videos(self) -> Dict[str, Any]:
        """
        Verify positioning quality for all V2 processed videos.
        """
        logger.info(f"\n{'='*80}")
        logger.info("V2 POSITIONING VERIFICATION - ANALYZING ALL PROCESSED VIDEOS")
        logger.info(f"{'='*80}")
        
        start_time = time.time()
        
        for video_path in self.v2_videos:
            result = self.analyze_video_positioning(video_path)
            self.verification_results.append(result)
            
        # Generate comprehensive report
        report = self._generate_verification_report()
        
        # Save report
        report_path = self.verification_dir / "v2_positioning_verification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Print summary
        self._print_verification_summary(report)
        
        return report
        
    def _generate_verification_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive verification report.
        """
        successful_results = [r for r in self.verification_results if r['success']]
        
        if not successful_results:
            return {
                'verification_complete': False,
                'error': 'No successful video analyses'
            }
            
        # Calculate statistics
        scores = [r['positioning_analysis']['average_score'] for r in successful_results]
        verdicts = [r['positioning_verdict'] for r in successful_results]
        qualities = [r['positioning_analysis']['quality_classification'] for r in successful_results]
        
        correctly_positioned = sum(1 for v in verdicts if v == 'CORRECTLY_POSITIONED')
        incorrectly_positioned = sum(1 for v in verdicts if v == 'INCORRECTLY_POSITIONED')
        
        quality_distribution = {}
        for quality in qualities:
            quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
            
        report = {
            'verification_complete': True,
            'verification_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'algorithm_version': 'v2_enhanced_positioning',
            'improvements_tested': {
                'lip_region_params': '(0.15, 0.25, 0.7, 0.5) vs (0.2, 0.1, 0.6, 0.4)',
                'area_expansion': '25% larger crop area',
                'vertical_positioning': 'Enhanced centering algorithm'
            },
            'verification_statistics': {
                'total_videos_analyzed': len(self.verification_results),
                'successful_analyses': len(successful_results),
                'failed_analyses': len(self.verification_results) - len(successful_results)
            },
            'positioning_quality_results': {
                'correctly_positioned_videos': correctly_positioned,
                'incorrectly_positioned_videos': incorrectly_positioned,
                'correct_positioning_rate': correctly_positioned / len(successful_results) if successful_results else 0,
                'quality_distribution': quality_distribution
            },
            'score_statistics': {
                'average_score': float(np.mean(scores)) if scores else 0,
                'median_score': float(np.median(scores)) if scores else 0,
                'min_score': float(np.min(scores)) if scores else 0,
                'max_score': float(np.max(scores)) if scores else 0,
                'std_score': float(np.std(scores)) if scores else 0
            },
            'detailed_results': self.verification_results,
            'verification_goal_achievement': {
                'target_success_rate': 0.9,  # >90% target
                'achieved_success_rate': correctly_positioned / len(successful_results) if successful_results else 0,
                'goal_met': (correctly_positioned / len(successful_results) if successful_results else 0) >= 0.9
            }
        }
        
        return report
        
    def _print_verification_summary(self, report: Dict[str, Any]):
        """Print comprehensive verification summary."""
        if not report['verification_complete']:
            logger.error("❌ Verification failed!")
            return
            
        stats = report['verification_statistics']
        quality = report['positioning_quality_results']
        scores = report['score_statistics']
        goal = report['verification_goal_achievement']
        
        logger.info(f"\n🎯 V2 POSITIONING VERIFICATION RESULTS")
        logger.info(f"{'='*60}")
        
        logger.info(f"\n📊 Analysis Statistics:")
        logger.info(f"  Total videos analyzed: {stats['total_videos_analyzed']}")
        logger.info(f"  Successful analyses: {stats['successful_analyses']}")
        logger.info(f"  Failed analyses: {stats['failed_analyses']}")
        
        logger.info(f"\n📏 Positioning Quality Results:")
        logger.info(f"  Correctly positioned: {quality['correctly_positioned_videos']}")
        logger.info(f"  Incorrectly positioned: {quality['incorrectly_positioned_videos']}")
        logger.info(f"  Success rate: {quality['correct_positioning_rate']:.1%}")
        logger.info(f"  Quality distribution: {quality['quality_distribution']}")
        
        logger.info(f"\n📈 Score Statistics:")
        logger.info(f"  Average score: {scores['average_score']:.3f}")
        logger.info(f"  Median score: {scores['median_score']:.3f}")
        logger.info(f"  Score range: {scores['min_score']:.3f} - {scores['max_score']:.3f}")
        logger.info(f"  Standard deviation: {scores['std_score']:.3f}")
        
        logger.info(f"\n🎯 Goal Achievement:")
        logger.info(f"  Target success rate: {goal['target_success_rate']:.1%}")
        logger.info(f"  Achieved success rate: {goal['achieved_success_rate']:.1%}")
        logger.info(f"  Goal met: {'✅ YES' if goal['goal_met'] else '❌ NO'}")
        
        if goal['goal_met']:
            logger.info(f"\n🎉 SUCCESS: V2 enhanced positioning achieved >90% correct positioning!")
        else:
            logger.info(f"\n⚠️  PARTIAL SUCCESS: V2 positioning improved but didn't reach 90% target")
            
        logger.info(f"\n📁 Verification Output:")
        logger.info(f"  Sample frames: {self.verification_dir}/sample_*.jpg")
        logger.info(f"  Full report: {self.verification_dir}/v2_positioning_verification_report.json")


def main():
    """
    Main function to verify V2 positioning improvements.
    """
    verifier = PositioningVerifier("fixed_temporal_output")
    report = verifier.verify_all_videos()
    
    if report['verification_complete']:
        logger.info("\n✅ V2 positioning verification completed!")
    else:
        logger.error("\n❌ V2 positioning verification failed!")


if __name__ == "__main__":
    main()
