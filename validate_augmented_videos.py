#!/usr/bin/env python3
"""
Validation script for augmented videos.
Creates HTML visualization showing augmented video samples with quality metrics.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import random
from typing import Dict, List
import json
import base64
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AugmentedVideoValidator:
    """
    Validate augmented videos and create HTML visualization.
    """
    
    def __init__(self, balanced_manifest_path: str):
        """
        Initialize the validator.
        
        Args:
            balanced_manifest_path: Path to the balanced manifest CSV
        """
        self.manifest_path = balanced_manifest_path
        self.manifest_df = pd.read_csv(balanced_manifest_path)
        
        # Filter for augmented videos only
        self.augmented_df = self.manifest_df[
            self.manifest_df['source'].str.contains('augmented', na=False)
        ].copy()
        
        logger.info(f"Loaded {len(self.manifest_df)} total videos")
        logger.info(f"Found {len(self.augmented_df)} augmented videos")
        
    def extract_frame_from_video(self, video_path: str) -> np.ndarray:
        """Extract middle frame from video."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
                
            # Get total frames and seek to middle
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
                
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            return frame
            
        except Exception as e:
            logger.error(f"Error extracting frame from {video_path}: {e}")
            return None
            
    def validate_augmented_frame(self, frame: np.ndarray, video_info: Dict) -> Dict:
        """Validate an augmented video frame."""
        if frame is None:
            return {
                'quality_score': 0,
                'dimensions_correct': False,
                'is_grayscale': False,
                'issues': ['Could not extract frame']
            }
            
        # Check dimensions (132x100 or 96x96)
        height, width = frame.shape[:2]
        dimensions_correct = (width, height) in [(96, 96), (132, 100)]
        
        # Calculate quality metrics
        pixel_mean = float(np.mean(frame))
        pixel_std = float(np.std(frame))
        pixel_range = int(np.max(frame) - np.min(frame))
        
        # Quality scoring
        quality_score = 100
        issues = []
        
        if not dimensions_correct:
            quality_score -= 30
            issues.append(f"Wrong dimensions: {width}x{height}")
            
        if pixel_range < 50:
            quality_score -= 20
            issues.append("Low pixel range (poor contrast)")
            
        if pixel_std < 10:
            quality_score -= 15
            issues.append("Low pixel variation")
            
        return {
            'quality_score': max(0, quality_score),
            'dimensions_correct': dimensions_correct,
            'is_grayscale': True,
            'pixel_stats': {
                'mean': pixel_mean,
                'std': pixel_std,
                'range': pixel_range
            },
            'issues': issues,
            'actual_dimensions': f"{width}x{height}"
        }
        
    def get_augmented_samples(self, num_samples: int = 20) -> List[Dict]:
        """Get representative samples from augmented videos."""
        if len(self.augmented_df) == 0:
            logger.warning("No augmented videos found")
            return []
            
        samples = []
        
        # Get samples from each augmented class
        augmented_classes = self.augmented_df['class'].unique()
        samples_per_class = max(1, num_samples // len(augmented_classes))
        
        for class_name in augmented_classes:
            class_augmented = self.augmented_df[self.augmented_df['class'] == class_name]
            
            # Get different augmentation types for this class
            aug_types = class_augmented['processed_version'].unique()
            
            class_samples = []
            for aug_type in aug_types:
                type_videos = class_augmented[class_augmented['processed_version'] == aug_type]
                if len(type_videos) > 0:
                    sample = type_videos.sample(n=1).iloc[0].to_dict()
                    class_samples.append(sample)
                    
                if len(class_samples) >= samples_per_class:
                    break
                    
            samples.extend(class_samples[:samples_per_class])
            
        # Shuffle and limit to requested number
        random.shuffle(samples)
        return samples[:num_samples]
        
    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 for HTML embedding."""
        if frame is None:
            return ""
            
        # Encode frame as PNG
        _, buffer = cv2.imencode('.png', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{frame_base64}"
        
    def extract_augmentation_type(self, processed_version: str) -> str:
        """Extract augmentation type from processed_version field."""
        if 'aug_' in processed_version:
            parts = processed_version.split('aug_')
            if len(parts) > 1:
                return parts[1].split('_')[0]
        return "unknown"
        
    def create_html_visualization(self, samples: List[Dict], output_path: str):
        """Create HTML visualization for augmented video samples."""
        
        # Process samples and extract frames
        processed_samples = []
        
        for sample in samples:
            logger.info(f"Processing sample: {Path(sample['path']).name}")
            
            # Extract frame
            frame = self.extract_frame_from_video(sample['path'])
            
            # Validate frame
            validation_results = self.validate_augmented_frame(frame, sample)
            
            # Convert frame to base64
            frame_base64 = self.frame_to_base64(frame)
            
            # Extract augmentation type
            aug_type = self.extract_augmentation_type(sample['processed_version'])
            
            processed_sample = {
                'filename': Path(sample['path']).name,
                'class': sample['class'],
                'augmentation_type': aug_type,
                'quality_score': validation_results['quality_score'],
                'dimensions': validation_results['actual_dimensions'],
                'pixel_stats': validation_results['pixel_stats'],
                'issues': validation_results['issues'],
                'frame_base64': frame_base64,
                'demographics': {
                    'gender': sample.get('gender', 'unknown'),
                    'age_band': sample.get('age_band', 'unknown'),
                    'ethnicity': sample.get('ethnicity', 'unknown')
                }
            }
            
            processed_samples.append(processed_sample)
            
        # Generate HTML
        html_content = self.generate_html_content(processed_samples)
        
        # Save HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"HTML visualization saved to: {output_path}")
        
    def generate_html_content(self, samples: List[Dict]) -> str:
        """Generate HTML content for visualization."""
        
        # Calculate summary statistics
        total_samples = len(samples)
        avg_quality = sum(s['quality_score'] for s in samples) / total_samples if total_samples > 0 else 0
        issues_count = sum(1 for s in samples if s['issues'])
        
        # Group by augmentation type
        aug_type_counts = {}
        for sample in samples:
            aug_type = sample['augmentation_type']
            aug_type_counts[aug_type] = aug_type_counts.get(aug_type, 0) + 1
            
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Augmented Video Validation Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .summary {{ background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .video-card {{ background: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .video-frame {{ width: 100%; max-width: 300px; height: auto; border: 2px solid #ddd; border-radius: 5px; }}
        .quality-excellent {{ border-left: 5px solid #4CAF50; }}
        .quality-good {{ border-left: 5px solid #FF9800; }}
        .quality-poor {{ border-left: 5px solid #F44336; }}
        .aug-type {{ background: #e3f2fd; color: #1976d2; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 10px 0; }}
        .stat-item {{ background: #f8f9fa; padding: 8px; border-radius: 5px; text-align: center; }}
        .issues {{ color: #d32f2f; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 Augmented Video Validation Results</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>📊 Summary Statistics</h2>
        <div class="stats">
            <div class="stat-item">
                <strong>{total_samples}</strong><br>
                Total Samples
            </div>
            <div class="stat-item">
                <strong>{avg_quality:.1f}/100</strong><br>
                Avg Quality Score
            </div>
            <div class="stat-item">
                <strong>{total_samples - issues_count}</strong><br>
                No Issues
            </div>
            <div class="stat-item">
                <strong>{issues_count}</strong><br>
                With Issues
            </div>
        </div>
        
        <h3>Augmentation Types</h3>
        <div class="stats">
        """
        
        for aug_type, count in aug_type_counts.items():
            html_content += f"""
            <div class="stat-item">
                <strong>{count}</strong><br>
                {aug_type.replace('_', ' ').title()}
            </div>
            """
            
        html_content += """
        </div>
    </div>
    
    <div class="grid">
    """
    
        for sample in samples:
            quality_class = "quality-excellent" if sample['quality_score'] >= 90 else "quality-good" if sample['quality_score'] >= 70 else "quality-poor"
            
            issues_html = ""
            if sample['issues']:
                issues_html = f"<div class='issues'>⚠️ Issues: {', '.join(sample['issues'])}</div>"
                
            html_content += f"""
        <div class="video-card {quality_class}">
            <h3>{sample['class']} <span class="aug-type">{sample['augmentation_type'].replace('_', ' ').title()}</span></h3>
            <img src="{sample['frame_base64']}" alt="Video frame" class="video-frame">
            
            <div class="stats">
                <div class="stat-item">
                    <strong>{sample['quality_score']}/100</strong><br>
                    Quality Score
                </div>
                <div class="stat-item">
                    <strong>{sample['dimensions']}</strong><br>
                    Dimensions
                </div>
                <div class="stat-item">
                    <strong>{sample['pixel_stats']['mean']:.1f}</strong><br>
                    Pixel Mean
                </div>
                <div class="stat-item">
                    <strong>{sample['pixel_stats']['range']}</strong><br>
                    Pixel Range
                </div>
            </div>
            
            <p><strong>Demographics:</strong> {sample['demographics']['gender']}, {sample['demographics']['age_band']}, {sample['demographics']['ethnicity']}</p>
            <p><strong>File:</strong> {sample['filename']}</p>
            {issues_html}
        </div>
        """
        
        html_content += """
    </div>
</body>
</html>
"""
        
        return html_content

def main():
    """Main function to validate augmented videos."""
    logger.info("🚀 Starting Augmented Video Validation")
    
    # Initialize validator
    validator = AugmentedVideoValidator("balanced_comprehensive_manifest.csv")
    
    # Get samples
    samples = validator.get_augmented_samples(20)
    logger.info(f"Selected {len(samples)} augmented video samples")
    
    # Create HTML visualization
    output_path = "augmented_validation_results.html"
    validator.create_html_visualization(samples, output_path)
    
    logger.info("✅ Augmented video validation completed!")
    logger.info(f"📊 HTML visualization: {output_path}")
    
    return output_path

if __name__ == "__main__":
    main()
