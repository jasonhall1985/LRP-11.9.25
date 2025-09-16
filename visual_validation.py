#!/usr/bin/env python3
"""
Comprehensive Visual Validation Tool for Lip Reading Dataset
===========================================================

Creates an interactive HTML visualization to validate video quality across:
- All data splits (train/val/test)
- All 7 classes with demographic diversity
- Frame quality, dimensions, and CLAHE processing
- Automatic browser display for immediate inspection

Features:
- Proportional sampling across splits and classes
- Representative middle frame extraction
- Quality checks and validation metrics
- Interactive HTML grid with detailed metadata
- Automatic browser opening for inspection

Author: Production Lip Reading System
Date: 2025-09-15
"""

import os
import cv2
import pandas as pd
import numpy as np
import random
import json
import webbrowser
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict, Counter
import argparse
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisualValidator:
    """
    Comprehensive visual validation tool for lip reading dataset.
    """
    
    def __init__(self, manifest_path: str, output_dir: str = "./validation_output"):
        """
        Initialize the visual validator.
        
        Args:
            manifest_path: Path to the dataset manifest CSV
            output_dir: Output directory for validation results
        """
        self.manifest_path = manifest_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load manifest
        self.manifest_df = pd.read_csv(manifest_path)
        logger.info(f"Loaded manifest with {len(self.manifest_df)} videos")
        
        # Class names
        self.class_names = [
            "help", "doctor", "glasses", "phone", "pillow",
            "i_need_to_move", "my_mouth_is_dry"
        ]

        # Expected dimensions - support both 96x96 and 132x100 (ICU dataset format)
        self.expected_dimensions = [(96, 96), (132, 100)]

        # Validation results
        self.validation_results = []
        self.quality_issues = []
        
        # CLAHE processor for comparison
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
    def create_demographic_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/val/test splits based on demographic criteria."""
        logger.info("Creating demographic splits...")
        
        # Validation: male 40-64
        val_mask = (self.manifest_df['gender'] == 'male') & (self.manifest_df['age_band'] == '40-64')
        val_df = self.manifest_df[val_mask].copy()
        
        # Test: female 18-39
        test_mask = (self.manifest_df['gender'] == 'female') & (self.manifest_df['age_band'] == '18-39')
        test_df = self.manifest_df[test_mask].copy()
        
        # Training: everything else
        train_mask = ~val_mask & ~test_mask
        train_df = self.manifest_df[train_mask].copy()
        
        logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
        
    def sample_videos_proportionally(self, total_samples: int = 40) -> List[Dict[str, Any]]:
        """
        Sample videos proportionally across splits and classes with demographic diversity.

        Args:
            total_samples: Total number of videos to sample

        Returns:
            List of sampled video information
        """
        logger.info(f"Sampling {total_samples} videos proportionally...")

        # If manifest is empty or doesn't exist, sample directly from expanded_cropped_dataset
        if len(self.manifest_df) == 0:
            logger.warning("Manifest is empty, sampling directly from expanded_cropped_dataset")
            return self._sample_from_directory(total_samples)

        # Create splits
        train_df, val_df, test_df = self.create_demographic_splits()

        # If no proper splits, sample directly from all videos
        if len(val_df) == 0 and len(test_df) == 0:
            logger.warning("No proper demographic splits found, sampling from all videos")
            return self._sample_from_all_videos(total_samples)

        # Calculate proportional samples per split
        total_videos = len(self.manifest_df)
        train_samples = max(1, int(total_samples * len(train_df) / total_videos))
        val_samples = max(1, int(total_samples * len(val_df) / total_videos))
        test_samples = total_samples - train_samples - val_samples

        logger.info(f"Target samples: train={train_samples}, val={val_samples}, test={test_samples}")

        sampled_videos = []

        # Sample from each split
        for split_name, split_df, n_samples in [
            ('train', train_df, train_samples),
            ('val', val_df, val_samples),
            ('test', test_df, test_samples)
        ]:
            if len(split_df) == 0:
                logger.warning(f"No videos in {split_name} split")
                continue

            split_samples = self._sample_from_split(split_df, n_samples, split_name)
            sampled_videos.extend(split_samples)

        logger.info(f"Successfully sampled {len(sampled_videos)} videos")
        return sampled_videos

    def _sample_from_directory(self, total_samples: int) -> List[Dict[str, Any]]:
        """Sample videos directly from expanded_cropped_dataset directory."""
        from pathlib import Path

        expanded_dir = Path("./expanded_cropped_dataset")
        if not expanded_dir.exists():
            logger.error("expanded_cropped_dataset directory not found")
            return []

        video_files = list(expanded_dir.glob("*.mp4"))
        if len(video_files) == 0:
            logger.error("No video files found in expanded_cropped_dataset")
            return []

        # Sample videos ensuring class diversity
        sampled_videos = []
        classes_found = set()

        # Shuffle for random sampling
        import random
        random.shuffle(video_files)

        for video_path in video_files:
            if len(sampled_videos) >= total_samples:
                break

            filename = video_path.name

            # Extract class from filename
            class_name = "unknown"
            for class_candidate in self.class_names:
                if class_candidate.replace("_", " ") in filename or class_candidate in filename:
                    class_name = class_candidate
                    break

            # Extract demographics from filename
            gender = "unknown"
            age_band = "unknown"
            ethnicity = "unknown"

            if "__male__" in filename:
                gender = "male"
            elif "__female__" in filename:
                gender = "female"

            if "__18to39__" in filename:
                age_band = "18-39"
            elif "__40to64__" in filename:
                age_band = "40-64"
            elif "__65plus__" in filename:
                age_band = "65+"

            if "__caucasian__" in filename:
                ethnicity = "caucasian"
            elif "__asian__" in filename:
                ethnicity = "asian"
            elif "__aboriginal__" in filename:
                ethnicity = "aboriginal"
            elif "__african__" in filename:
                ethnicity = "african"

            sampled_videos.append({
                'path': str(video_path),
                'class': class_name,
                'gender': gender,
                'age_band': age_band,
                'ethnicity': ethnicity,
                'source': 'expanded_cropped_dataset',
                'processed_version': 'cropped',
                'split': 'train'  # Default split
            })

            classes_found.add(class_name)

        logger.info(f"Sampled {len(sampled_videos)} videos from directory")
        logger.info(f"Classes found: {classes_found}")
        return sampled_videos

    def _sample_from_all_videos(self, total_samples: int) -> List[Dict[str, Any]]:
        """Sample from all videos in manifest when splits are not available."""
        if len(self.manifest_df) == 0:
            return self._sample_from_directory(total_samples)

        # Sample proportionally by class
        sampled_videos = []
        available_classes = self.manifest_df['class'].unique()
        samples_per_class = max(1, total_samples // len(available_classes))

        for class_name in available_classes:
            class_df = self.manifest_df[self.manifest_df['class'] == class_name]
            n_samples = min(samples_per_class, len(class_df))

            if len(sampled_videos) + n_samples > total_samples:
                n_samples = total_samples - len(sampled_videos)

            if n_samples <= 0:
                break

            class_samples = class_df.sample(n=n_samples)
            for _, sample in class_samples.iterrows():
                sample_dict = sample.to_dict()
                sample_dict['split'] = 'train'  # Default split
                sampled_videos.append(sample_dict)

        return sampled_videos
        
    def _sample_from_split(self, split_df: pd.DataFrame, n_samples: int, split_name: str) -> List[Dict[str, Any]]:
        """Sample videos from a specific split ensuring class and demographic diversity."""
        if len(split_df) == 0:
            return []
            
        # Calculate samples per class
        available_classes = split_df['class'].unique()
        samples_per_class = max(1, n_samples // len(available_classes))
        
        sampled_videos = []
        
        for class_name in self.class_names:
            if class_name not in available_classes:
                continue
                
            class_df = split_df[split_df['class'] == class_name]
            if len(class_df) == 0:
                continue
                
            # Sample with demographic diversity
            class_samples = min(samples_per_class, len(class_df))
            
            # Try to get diverse demographics
            if len(class_df) >= class_samples:
                # Group by demographics and sample from each group
                demo_groups = class_df.groupby(['gender', 'age_band', 'ethnicity'])
                diverse_samples = []
                
                for (gender, age, ethnicity), group in demo_groups:
                    if len(diverse_samples) >= class_samples:
                        break
                    sample = group.sample(n=1).iloc[0]
                    diverse_samples.append(sample)
                    
                # Fill remaining samples randomly if needed
                while len(diverse_samples) < class_samples:
                    remaining_df = class_df[~class_df.index.isin([s.name for s in diverse_samples])]
                    if len(remaining_df) == 0:
                        break
                    additional_sample = remaining_df.sample(n=1).iloc[0]
                    diverse_samples.append(additional_sample)
                    
                selected_samples = diverse_samples[:class_samples]
            else:
                selected_samples = class_df.sample(n=class_samples).to_dict('records')
                
            # Add split information
            for sample in selected_samples:
                if isinstance(sample, pd.Series):
                    sample_dict = sample.to_dict()
                else:
                    sample_dict = sample
                sample_dict['split'] = split_name
                sampled_videos.append(sample_dict)
                
        return sampled_videos
        
    def extract_representative_frame(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract representative middle frame from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Extracted frame as numpy array or None if failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return None
                
            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                logger.warning(f"No frames in video: {video_path}")
                cap.release()
                return None
                
            # Seek to middle frame
            middle_frame_idx = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                logger.warning(f"Could not read frame from: {video_path}")
                return None
                
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            return frame
            
        except Exception as e:
            logger.warning(f"Error extracting frame from {video_path}: {e}")
            return None
            
    def validate_frame_quality(self, frame: np.ndarray, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive quality checks on extracted frame.
        
        Args:
            frame: Extracted frame
            video_info: Video metadata
            
        Returns:
            Quality validation results
        """
        quality_results = {
            'video_path': video_info['path'],
            'class': video_info['class'],
            'split': video_info['split'],
            'dimensions_correct': False,
            'is_grayscale': False,
            'pixel_stats': {},
            'contrast_metrics': {},
            'clahe_applied': False,
            'quality_score': 0.0,
            'issues': []
        }
        
        if frame is None:
            quality_results['issues'].append("Could not extract frame")
            return quality_results
            
        # Check dimensions - support both 96x96 and 132x100 formats
        height, width = frame.shape[:2]
        quality_results['dimensions_correct'] = (width, height) in self.expected_dimensions
        quality_results['actual_dimensions'] = f"{width}x{height}"

        if not quality_results['dimensions_correct']:
            expected_dims_str = " or ".join([f"{w}x{h}" for w, h in self.expected_dimensions])
            quality_results['issues'].append(f"Wrong dimensions: {width}x{height}, expected {expected_dims_str}")
            
        # Check if grayscale
        quality_results['is_grayscale'] = len(frame.shape) == 2
        if not quality_results['is_grayscale']:
            quality_results['issues'].append("Frame is not grayscale")
            
        # Pixel statistics
        quality_results['pixel_stats'] = {
            'mean': float(np.mean(frame)),
            'std': float(np.std(frame)),
            'min': int(np.min(frame)),
            'max': int(np.max(frame)),
            'range': int(np.max(frame) - np.min(frame))
        }
        
        # Contrast metrics
        frame_max = float(np.max(frame))
        frame_min = float(np.min(frame))
        michelson_denom = frame_max + frame_min
        michelson_contrast = (frame_max - frame_min) / (michelson_denom + 1e-8) if michelson_denom > 0 else 0.0

        quality_results['contrast_metrics'] = {
            'rms_contrast': float(np.std(frame)),
            'michelson_contrast': float(michelson_contrast),
            'histogram_spread': float(np.percentile(frame, 95) - np.percentile(frame, 5))
        }
        
        # Check if CLAHE was likely applied (higher contrast, more uniform histogram)
        clahe_frame = self.clahe.apply(frame)
        clahe_contrast = np.std(clahe_frame)
        original_contrast = np.std(frame)
        
        # If current contrast is significantly higher than what CLAHE would produce,
        # it's likely CLAHE was already applied
        quality_results['clahe_applied'] = original_contrast > clahe_contrast * 0.9
        
        # Calculate quality score (0-100)
        score = 0
        if quality_results['dimensions_correct']:
            score += 30
        if quality_results['is_grayscale']:
            score += 20
        if quality_results['pixel_stats']['range'] > 50:  # Good dynamic range
            score += 20
        if quality_results['contrast_metrics']['rms_contrast'] > 20:  # Good contrast
            score += 20
        if 10 < quality_results['pixel_stats']['mean'] < 245:  # Not too dark/bright
            score += 10
            
        quality_results['quality_score'] = score
        
        # Add to issues if quality is poor
        if score < 70:
            quality_results['issues'].append(f"Low quality score: {score}/100")
            
        return quality_results
        
    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 string for HTML embedding."""
        if frame is None:
            return ""
            
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(frame)
            
            # Convert to base64
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logger.warning(f"Error converting frame to base64: {e}")
            return ""
            
    def create_pixel_histogram(self, frame: np.ndarray) -> str:
        """Create pixel histogram as base64 string."""
        if frame is None:
            return ""
            
        try:
            plt.figure(figsize=(3, 2))
            plt.hist(frame.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.title('Pixel Histogram')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logger.warning(f"Error creating histogram: {e}")
            return ""

    def run_validation(self, total_samples: int = 40) -> str:
        """
        Run complete visual validation pipeline.

        Args:
            total_samples: Number of videos to sample

        Returns:
            Path to generated HTML file
        """
        logger.info("Starting visual validation pipeline...")

        # Sample videos
        sampled_videos = self.sample_videos_proportionally(total_samples)

        # Process each video
        validation_results = []

        for i, video_info in enumerate(sampled_videos):
            logger.info(f"Processing video {i+1}/{len(sampled_videos)}: {Path(video_info['path']).name}")

            # Extract frame
            frame = self.extract_representative_frame(video_info['path'])

            # Validate quality
            quality_results = self.validate_frame_quality(frame, video_info)

            # Convert to base64
            frame_b64 = self.frame_to_base64(frame)
            histogram_b64 = self.create_pixel_histogram(frame)

            # Store results
            result = {
                'video_info': video_info,
                'quality': quality_results,
                'frame_b64': frame_b64,
                'histogram_b64': histogram_b64
            }

            validation_results.append(result)

            # Track quality issues
            if quality_results['issues']:
                self.quality_issues.append({
                    'path': video_info['path'],
                    'issues': quality_results['issues'],
                    'quality_score': quality_results['quality_score']
                })

        # Generate HTML visualization
        html_path = self.generate_html_visualization(validation_results)

        # Save validation report
        self.save_validation_report(validation_results)

        logger.info(f"Validation complete! Found {len(self.quality_issues)} videos with issues")
        return html_path

    def save_validation_report(self, validation_results: List[Dict[str, Any]]) -> None:
        """Save comprehensive validation report."""
        logger.info("Saving validation report...")

        # Summary statistics
        total_videos = len(validation_results)
        correct_dimensions = sum(1 for r in validation_results if r['quality']['dimensions_correct'])
        grayscale_count = sum(1 for r in validation_results if r['quality']['is_grayscale'])
        clahe_count = sum(1 for r in validation_results if r['quality']['clahe_applied'])
        avg_quality = np.mean([r['quality']['quality_score'] for r in validation_results])

        # Count by split and class
        split_counts = Counter(r['video_info']['split'] for r in validation_results)
        class_counts = Counter(r['video_info']['class'] for r in validation_results)

        # Quality distribution
        quality_scores = [r['quality']['quality_score'] for r in validation_results]
        excellent_count = sum(1 for score in quality_scores if score >= 90)
        good_count = sum(1 for score in quality_scores if 70 <= score < 90)
        poor_count = sum(1 for score in quality_scores if score < 70)

        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_videos': total_videos,
                'correct_dimensions': correct_dimensions,
                'correct_dimensions_pct': correct_dimensions / total_videos * 100,
                'grayscale_count': grayscale_count,
                'grayscale_pct': grayscale_count / total_videos * 100,
                'clahe_count': clahe_count,
                'clahe_pct': clahe_count / total_videos * 100,
                'avg_quality_score': avg_quality,
                'quality_distribution': {
                    'excellent': excellent_count,
                    'good': good_count,
                    'poor': poor_count
                }
            },
            'split_distribution': dict(split_counts),
            'class_distribution': dict(class_counts),
            'quality_issues': self.quality_issues,
            'detailed_results': [
                {
                    'path': r['video_info']['path'],
                    'class': r['video_info']['class'],
                    'split': r['video_info']['split'],
                    'demographics': {
                        'gender': r['video_info'].get('gender', 'unknown'),
                        'age_band': r['video_info'].get('age_band', 'unknown'),
                        'ethnicity': r['video_info'].get('ethnicity', 'unknown')
                    },
                    'quality_score': r['quality']['quality_score'],
                    'dimensions_correct': r['quality']['dimensions_correct'],
                    'is_grayscale': r['quality']['is_grayscale'],
                    'clahe_applied': r['quality']['clahe_applied'],
                    'pixel_stats': r['quality']['pixel_stats'],
                    'contrast_metrics': r['quality']['contrast_metrics'],
                    'issues': r['quality']['issues']
                }
                for r in validation_results
            ]
        }

        # Save JSON report (with numpy type conversion)
        report_path = self.output_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serializer)

        # Save markdown summary
        markdown_path = self.output_dir / "VALIDATION_SUMMARY.md"
        with open(markdown_path, 'w') as f:
            f.write(f"""# Visual Validation Summary

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Total Videos Sampled**: {total_videos}
- **Average Quality Score**: {avg_quality:.1f}/100
- **Videos with Issues**: {len(self.quality_issues)}/{total_videos} ({len(self.quality_issues)/total_videos*100:.1f}%)

## Quality Distribution
- **Excellent (90-100)**: {excellent_count} videos ({excellent_count/total_videos*100:.1f}%)
- **Good (70-89)**: {good_count} videos ({good_count/total_videos*100:.1f}%)
- **Poor (<70)**: {poor_count} videos ({poor_count/total_videos*100:.1f}%)

## Technical Validation
- **Correct Dimensions (96×96 or 132×100)**: {correct_dimensions}/{total_videos} ({correct_dimensions/total_videos*100:.1f}%)
- **Grayscale Format**: {grayscale_count}/{total_videos} ({grayscale_count/total_videos*100:.1f}%)
- **CLAHE Enhancement**: {clahe_count}/{total_videos} ({clahe_count/total_videos*100:.1f}%)

## Data Distribution

### By Split
""")
            for split, count in split_counts.items():
                f.write(f"- **{split.upper()}**: {count} videos ({count/total_videos*100:.1f}%)\n")

            f.write("\n### By Class\n")
            for class_name in self.class_names:
                count = class_counts.get(class_name, 0)
                f.write(f"- **{class_name}**: {count} videos ({count/total_videos*100:.1f}%)\n")

            if self.quality_issues:
                f.write(f"\n## Quality Issues Found\n\n")
                for issue in self.quality_issues:
                    f.write(f"### {Path(issue['path']).name}\n")
                    f.write(f"- **Quality Score**: {issue['quality_score']}/100\n")
                    f.write(f"- **Issues**:\n")
                    for problem in issue['issues']:
                        f.write(f"  - {problem}\n")
                    f.write("\n")
            else:
                f.write("\n## ✅ No Quality Issues Found!\n\nAll sampled videos meet the expected specifications.\n")

        logger.info(f"Validation report saved to: {report_path}")
        logger.info(f"Summary saved to: {markdown_path}")

    def generate_html_visualization(self, validation_results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive HTML visualization page."""
        logger.info("Generating HTML visualization...")

        # Calculate summary statistics
        total_videos = len(validation_results)
        correct_dimensions = sum(1 for r in validation_results if r['quality']['dimensions_correct'])
        grayscale_count = sum(1 for r in validation_results if r['quality']['is_grayscale'])
        clahe_count = sum(1 for r in validation_results if r['quality']['clahe_applied'])
        avg_quality = np.mean([r['quality']['quality_score'] for r in validation_results])

        # Count by split and class
        split_counts = Counter(r['video_info']['split'] for r in validation_results)
        class_counts = Counter(r['video_info']['class'] for r in validation_results)

        # Generate HTML (simplified version due to length constraints)
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lip Reading Dataset Visual Validation</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 2.5em; font-weight: 300; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }}
        .summary-card .value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .grid-container {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .video-card {{ background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .video-card.issue {{ border-left: 5px solid #e74c3c; }}
        .video-card.good {{ border-left: 5px solid #27ae60; }}
        .frame-container {{ position: relative; text-align: center; padding: 15px; background: #f8f9fa; }}
        .frame-image {{ max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 5px; image-rendering: pixelated; }}
        .quality-badge {{ position: absolute; top: 10px; right: 10px; padding: 5px 10px; border-radius: 15px; color: white; font-weight: bold; font-size: 0.8em; }}
        .quality-excellent {{ background-color: #27ae60; }}
        .quality-good {{ background-color: #f39c12; }}
        .quality-poor {{ background-color: #e74c3c; }}
        .metadata {{ padding: 15px; }}
        .metadata-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; padding-bottom: 5px; border-bottom: 1px solid #eee; }}
        .metadata-label {{ font-weight: 500; color: #555; }}
        .metadata-value {{ color: #333; font-family: monospace; }}
        .class-label {{ display: inline-block; padding: 4px 8px; background: #667eea; color: white; border-radius: 4px; font-size: 0.8em; font-weight: bold; }}
        .issues-list {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 10px; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Lip Reading Dataset Visual Validation</h1>
        <p>Comprehensive quality assessment of {total_videos} sampled videos</p>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="summary">
        <div class="summary-card">
            <h3>Total Videos</h3>
            <div class="value">{total_videos}</div>
        </div>
        <div class="summary-card">
            <h3>Correct Dimensions</h3>
            <div class="value">{correct_dimensions}/{total_videos}</div>
        </div>
        <div class="summary-card">
            <h3>Average Quality</h3>
            <div class="value">{avg_quality:.1f}/100</div>
        </div>
    </div>

    <div class="grid-container">"""

        # Add video cards (simplified)
        for result in validation_results:
            video_info = result['video_info']
            quality = result['quality']
            frame_b64 = result['frame_b64']

            quality_score = quality['quality_score']
            quality_class = "excellent" if quality_score >= 90 else ("good" if quality_score >= 70 else "poor")
            card_class = "good" if len(quality['issues']) == 0 else "issue"
            filename = Path(video_info['path']).name

            html_content += f"""
        <div class="video-card {card_class}">
            <div class="frame-container">
                <img src="{frame_b64}" alt="Frame" class="frame-image">
                <div class="quality-badge quality-{quality_class}">{quality_score:.0f}</div>
            </div>
            <div class="metadata">
                <div class="metadata-row">
                    <span class="metadata-label">File:</span>
                    <span class="metadata-value">{filename}</span>
                </div>
                <div class="metadata-row">
                    <span class="metadata-label">Class:</span>
                    <span class="class-label">{video_info['class']}</span>
                </div>
                <div class="metadata-row">
                    <span class="metadata-label">Dimensions:</span>
                    <span class="metadata-value">{quality['actual_dimensions']}</span>
                </div>"""

            if quality['issues']:
                html_content += f"""
                <div class="issues-list">
                    <h4>⚠️ Issues:</h4>
                    <ul>{''.join(f'<li>{issue}</li>' for issue in quality['issues'])}</ul>
                </div>"""

            html_content += "</div></div>"

        html_content += """
    </div>
</body>
</html>"""

        # Save HTML file
        html_path = self.output_dir / "visual_validation.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML visualization saved to: {html_path}")
        return str(html_path)

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Visual validation tool for lip reading dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation with 40 samples
  python visual_validation.py --manifest manifest.csv

  # Custom sample size and output directory
  python visual_validation.py --manifest manifest.csv --samples 60 --output ./validation_results

  # Auto-open in browser
  python visual_validation.py --manifest manifest.csv --open-browser
        """
    )

    parser.add_argument(
        '--manifest',
        type=str,
        required=True,
        help='Path to dataset manifest CSV file'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=40,
        help='Number of videos to sample for validation (default: 40)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./validation_output',
        help='Output directory for validation results (default: ./validation_output)'
    )

    parser.add_argument(
        '--open-browser',
        action='store_true',
        help='Automatically open HTML visualization in browser'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible sampling (default: 42)'
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Validate inputs
    if not os.path.exists(args.manifest):
        logger.error(f"Manifest file not found: {args.manifest}")
        return 1

    if args.samples < 1 or args.samples > 200:
        logger.error("Sample size must be between 1 and 200")
        return 1

    try:
        # Create validator
        logger.info("🚀 Starting Visual Validation Pipeline")
        logger.info(f"Manifest: {args.manifest}")
        logger.info(f"Samples: {args.samples}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Random seed: {args.seed}")

        validator = VisualValidator(args.manifest, args.output)

        # Run validation
        html_path = validator.run_validation(args.samples)

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("🎉 VALIDATION COMPLETE!")
        logger.info("="*60)
        logger.info(f"📊 HTML Visualization: {html_path}")
        logger.info(f"📋 Validation Report: {validator.output_dir}/validation_report.json")
        logger.info(f"📝 Summary: {validator.output_dir}/VALIDATION_SUMMARY.md")

        if validator.quality_issues:
            logger.warning(f"⚠️  Found {len(validator.quality_issues)} videos with quality issues")
            logger.info("Check the HTML visualization and validation report for details")
        else:
            logger.info("✅ All sampled videos meet quality specifications!")

        # Open browser if requested
        if args.open_browser:
            logger.info("🌐 Opening HTML visualization in browser...")
            webbrowser.open(f"file://{os.path.abspath(html_path)}")

        logger.info("="*60)
        return 0

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
