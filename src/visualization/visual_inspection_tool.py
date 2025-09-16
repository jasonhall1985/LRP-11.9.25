#!/usr/bin/env python3
"""
Visual Inspection Tool for Mouth-Cropped Videos
===============================================

Generates an HTML page with sample frames from processed mouth-cropped videos
to visually verify preprocessing quality before classifier training.

Features:
- Random sampling across quality tiers and classes
- Frame extraction at key temporal points
- HTML grid layout with metadata
- Quality metrics and traceability information

Usage:
    python visual_inspection_tool.py MANIFEST_CSV VIDEOS_DIR [OUTPUT_HTML]

Example:
    python visual_inspection_tool.py training_manifest.csv mouth_crops_96x96_32f visual_inspection.html
"""

import sys
import csv
import cv2
import numpy as np
import pandas as pd
import pathlib
import random
import base64
from datetime import datetime
from typing import List, Dict, Tuple, Any
import logging

class VisualInspectionTool:
    """
    Tool for generating visual inspection reports of mouth-cropped videos.
    """
    
    def __init__(self, manifest_path: str, videos_dir: str, output_html: str = "visual_inspection.html"):
        """
        Initialize the visual inspection tool.
        
        Args:
            manifest_path: Path to training manifest CSV
            videos_dir: Directory containing cropped videos
            output_html: Output HTML file path
        """
        self.manifest_path = pathlib.Path(manifest_path)
        self.videos_dir = pathlib.Path(videos_dir)
        self.output_html = pathlib.Path(output_html)
        
        # Load manifest data
        self.df = pd.read_csv(manifest_path)
        
        # Frame extraction settings
        self.target_frames = [1, 8, 16, 24, 32]  # Evenly distributed frames
        self.sample_count = 8  # Number of videos to sample (reduced for geometric comparison)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def categorize_videos_by_quality(self) -> Dict[str, pd.DataFrame]:
        """
        Categorize videos by quality tiers.
        
        Returns:
            Dictionary with quality categories and corresponding DataFrames
        """
        categories = {
            'high_quality': self.df[self.df['detect_frac'] >= 0.5],
            'medium_quality': self.df[(self.df['detect_frac'] >= 0.3) & (self.df['detect_frac'] < 0.5)],
            'low_quality': self.df[self.df['detect_frac'] < 0.3]
        }
        
        self.logger.info(f"Quality distribution:")
        for category, data in categories.items():
            self.logger.info(f"  {category}: {len(data)} videos")
        
        return categories
    
    def select_representative_sample(self) -> List[Dict[str, Any]]:
        """
        Select a representative sample of videos across quality tiers and classes.
        
        Returns:
            List of selected video information dictionaries
        """
        categories = self.categorize_videos_by_quality()
        selected_videos = []
        
        # Sample from high quality (60% of samples)
        high_quality_count = max(1, int(self.sample_count * 0.6))
        if len(categories['high_quality']) > 0:
            high_sample = categories['high_quality'].sample(
                n=min(high_quality_count, len(categories['high_quality'])),
                random_state=42
            )
            for _, row in high_sample.iterrows():
                selected_videos.append({
                    'video_info': row.to_dict(),
                    'quality_tier': 'High Quality (≥50%)',
                    'tier_color': '#28a745'  # Green
                })
        
        # Sample from medium quality (30% of samples)
        medium_quality_count = max(1, int(self.sample_count * 0.3))
        if len(categories['medium_quality']) > 0:
            medium_sample = categories['medium_quality'].sample(
                n=min(medium_quality_count, len(categories['medium_quality'])),
                random_state=42
            )
            for _, row in medium_sample.iterrows():
                selected_videos.append({
                    'video_info': row.to_dict(),
                    'quality_tier': 'Medium Quality (30-50%)',
                    'tier_color': '#ffc107'  # Yellow
                })
        
        # Sample from low quality (10% of samples)
        low_quality_count = max(1, int(self.sample_count * 0.1))
        if len(categories['low_quality']) > 0:
            low_sample = categories['low_quality'].sample(
                n=min(low_quality_count, len(categories['low_quality'])),
                random_state=42
            )
            for _, row in low_sample.iterrows():
                selected_videos.append({
                    'video_info': row.to_dict(),
                    'quality_tier': 'Low Quality (<30%)',
                    'tier_color': '#dc3545'  # Red
                })
        
        # Shuffle to mix quality tiers
        random.shuffle(selected_videos)
        
        self.logger.info(f"Selected {len(selected_videos)} videos for inspection")
        return selected_videos
    
    def extract_frames_from_video(self, video_path: pathlib.Path) -> List[Tuple[int, np.ndarray]]:
        """
        Extract frames at specified positions from a video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of (frame_number, frame_image) tuples
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        extracted_frames = []
        
        for frame_num in self.target_frames:
            if frame_num <= total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # 0-indexed
                ret, frame = cap.read()
                if ret:
                    extracted_frames.append((frame_num, frame))
                else:
                    self.logger.warning(f"Failed to read frame {frame_num} from {video_path}")
        
        cap.release()
        return extracted_frames
    
    def frame_to_base64(self, frame: np.ndarray) -> str:
        """
        Convert OpenCV frame to base64 string for HTML embedding.
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            Base64 encoded image string
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Encode as PNG
        _, buffer = cv2.imencode('.png', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def generate_html_content(self, selected_videos: List[Dict[str, Any]]) -> str:
        """
        Generate HTML content for visual inspection.
        
        Args:
            selected_videos: List of selected video information
            
        Returns:
            Complete HTML content string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mouth-Cropped Videos - Visual Inspection Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .video-container {{
            background: white;
            margin-bottom: 30px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .video-header {{
            padding: 20px;
            border-left: 5px solid;
        }}
        .video-title {{
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .video-metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .metadata-item {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
        }}
        .metadata-label {{
            font-weight: bold;
            color: #666;
            font-size: 0.9em;
        }}
        .metadata-value {{
            font-size: 1.1em;
            margin-top: 5px;
        }}
        .frames-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            padding: 20px;
            background: #f8f9fa;
        }}
        .frame-item {{
            text-align: center;
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .frame-image {{
            width: 96px;
            height: 96px;
            border: 2px solid #ddd;
            border-radius: 5px;
            display: block;
            margin: 0 auto 10px auto;
        }}
        .frame-info {{
            font-size: 0.9em;
            color: #666;
        }}
        .quality-badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎥 Mouth-Cropped Videos</h1>
        <p>Visual Inspection Report - Generated on {timestamp}</p>
    </div>
    
    <div class="summary">
        <h2>📊 Dataset Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{len(self.df)}</div>
                <div class="stat-label">Total Videos</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(selected_videos)}</div>
                <div class="stat-label">Sampled for Inspection</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">96×96</div>
                <div class="stat-label">Resolution</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">32</div>
                <div class="stat-label">Frames per Video</div>
            </div>
        </div>
        <p><strong>Purpose:</strong> Visual verification of MediaPipe-based mouth cropping, temporal smoothing, and bridging quality.</p>
        <p><strong>Sample Strategy:</strong> Representative selection across quality tiers and class labels for comprehensive evaluation.</p>
    </div>
"""
        
        return html_content

    def process_selected_videos(self, selected_videos: List[Dict[str, Any]]) -> str:
        """
        Process selected videos and generate complete HTML content.

        Args:
            selected_videos: List of selected video information

        Returns:
            Complete HTML content with video frames
        """
        html_content = self.generate_html_content(selected_videos)

        # Process each selected video
        for i, video_data in enumerate(selected_videos):
            video_info = video_data['video_info']
            quality_tier = video_data['quality_tier']
            tier_color = video_data['tier_color']

            # Get video path
            video_path = pathlib.Path(video_info['path'])
            if not video_path.exists():
                self.logger.warning(f"Video not found: {video_path}")
                continue

            self.logger.info(f"Processing video {i+1}/{len(selected_videos)}: {video_path.name}")

            # Extract frames
            frames = self.extract_frames_from_video(video_path)
            if not frames:
                continue

            # Generate HTML for this video
            html_content += f"""
    <div class="video-container">
        <div class="video-header" style="border-left-color: {tier_color};">
            <div class="video-title">
                📁 {video_path.name}
                <span class="quality-badge" style="background-color: {tier_color};">{quality_tier}</span>
            </div>

            <div class="video-metadata">
                <div class="metadata-item">
                    <div class="metadata-label">Class Label</div>
                    <div class="metadata-value">{video_info['label'].title()}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Detection Rate</div>
                    <div class="metadata-value">{video_info['detect_frac']:.1%}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Frame Count</div>
                    <div class="metadata-value">{video_info['frames']} frames</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Frame Rate</div>
                    <div class="metadata-value">{video_info['fps']} fps</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Source Video</div>
                    <div class="metadata-value">{pathlib.Path(video_info['src']).name}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Processing Status</div>
                    <div class="metadata-value">{video_info.get('status', 'processed').title()}</div>
                </div>
            </div>
        </div>

        <div class="frames-grid">
"""

            # Add frames
            for frame_num, frame in frames:
                frame_base64 = self.frame_to_base64(frame)
                timestamp_sec = (frame_num - 1) / video_info['fps']

                html_content += f"""
            <div class="frame-item">
                <img src="{frame_base64}" alt="Frame {frame_num}" class="frame-image">
                <div class="frame-info">
                    <strong>Frame {frame_num}</strong><br>
                    {timestamp_sec:.2f}s
                </div>
            </div>
"""

            html_content += """
        </div>
    </div>
"""

        # Close HTML
        html_content += """
    <div class="summary">
        <h2>🔍 Inspection Guidelines</h2>
        <ul>
            <li><strong>ROI Accuracy:</strong> Check if mouth region is properly centered and includes lips, chin, and upper lip area</li>
            <li><strong>Temporal Consistency:</strong> Verify smooth transitions between frames without sudden jumps</li>
            <li><strong>Resolution Quality:</strong> Ensure 96×96 resolution maintains sufficient detail for lip reading</li>
            <li><strong>Bridging Effectiveness:</strong> Look for smooth interpolation during detection gaps</li>
            <li><strong>Class Representation:</strong> Verify diverse sampling across different classes and quality tiers</li>
        </ul>

        <h3>Quality Tier Interpretation:</h3>
        <ul>
            <li><span class="quality-badge" style="background-color: #28a745;">High Quality (≥50%)</span> - Recommended for initial training</li>
            <li><span class="quality-badge" style="background-color: #ffc107;">Medium Quality (30-50%)</span> - Add if training stable</li>
            <li><span class="quality-badge" style="background-color: #dc3545;">Low Quality (<30%)</span> - Consider excluding from training</li>
        </ul>
    </div>

    <footer style="text-align: center; margin-top: 40px; padding: 20px; color: #666;">
        <p>Generated by Visual Inspection Tool | MediaPipe-based Mouth Cropping Pipeline</p>
    </footer>
</body>
</html>
"""

        return html_content

    def generate_inspection_report(self) -> str:
        """
        Generate complete visual inspection report.

        Returns:
            Path to generated HTML file
        """
        self.logger.info("Starting visual inspection report generation")

        # Select representative sample
        selected_videos = self.select_representative_sample()

        if not selected_videos:
            raise ValueError("No videos selected for inspection")

        # Process videos and generate HTML
        html_content = self.process_selected_videos(selected_videos)

        # Write HTML file
        with open(self.output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"Visual inspection report generated: {self.output_html}")
        return str(self.output_html)


def main():
    """Main CLI interface for visual inspection tool."""
    if len(sys.argv) < 3:
        print("Usage: python visual_inspection_tool.py MANIFEST_CSV VIDEOS_DIR [OUTPUT_HTML]")
        print("\nExample:")
        print("  python visual_inspection_tool.py training_manifest.csv mouth_crops_96x96_32f visual_inspection.html")
        print("\nDescription:")
        print("  Generates HTML visual inspection report for mouth-cropped videos.")
        print("  Samples videos across quality tiers and classes for comprehensive evaluation.")
        sys.exit(1)

    manifest_path = sys.argv[1]
    videos_dir = sys.argv[2]
    output_html = sys.argv[3] if len(sys.argv) > 3 else "visual_inspection.html"

    # Validate inputs
    if not pathlib.Path(manifest_path).exists():
        print(f"Error: Manifest file '{manifest_path}' does not exist")
        sys.exit(1)

    if not pathlib.Path(videos_dir).exists():
        print(f"Error: Videos directory '{videos_dir}' does not exist")
        sys.exit(1)

    try:
        # Initialize and run inspection tool
        inspector = VisualInspectionTool(manifest_path, videos_dir, output_html)
        html_file = inspector.generate_inspection_report()

        print(f"\n✅ Visual inspection report generated successfully!")
        print(f"📄 HTML file: {html_file}")
        print(f"🔍 Open in browser to inspect preprocessing quality")

        return html_file

    except Exception as e:
        print(f"❌ Error generating inspection report: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
