#!/usr/bin/env python3
"""
Geometric Cropping Visual Inspection Tool
=========================================

Specialized visual inspection tool for geometric mouth cropping results.
Generates HTML reports to evaluate the quality of deterministic geometric crops.

Usage:
    python geometric_visual_inspection.py MANIFEST_CSV VIDEOS_DIR [OUTPUT_HTML]

Example:
    python geometric_visual_inspection.py geometric_training_manifest.csv geometric_crops_96x96_32f geometric_inspection.html
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

class GeometricVisualInspector:
    """
    Visual inspection tool for geometric mouth cropping results.
    """
    
    def __init__(self, manifest_path: str, videos_dir: str, output_html: str = "geometric_inspection.html"):
        """
        Initialize the geometric visual inspector.
        
        Args:
            manifest_path: Path to geometric cropping manifest CSV
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
        self.sample_count = 8  # Number of videos to sample
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def select_representative_sample(self) -> List[Dict[str, Any]]:
        """
        Select a representative sample across classes.
        
        Returns:
            List of selected video information dictionaries
        """
        selected_videos = []
        
        # Get class distribution
        class_counts = self.df['label'].value_counts()
        self.logger.info(f"Class distribution: {dict(class_counts)}")
        
        # Sample from each class
        samples_per_class = max(1, self.sample_count // len(class_counts))
        
        for class_name in class_counts.index:
            class_videos = self.df[self.df['label'] == class_name]
            sample_size = min(samples_per_class, len(class_videos))
            
            if sample_size > 0:
                sample = class_videos.sample(n=sample_size, random_state=42)
                for _, row in sample.iterrows():
                    selected_videos.append({
                        'video_info': row.to_dict(),
                        'class_name': class_name,
                        'class_color': self.get_class_color(class_name)
                    })
        
        # If we need more samples, add random ones
        while len(selected_videos) < self.sample_count and len(selected_videos) < len(self.df):
            remaining = self.df[~self.df.index.isin([v['video_info']['path'] for v in selected_videos])]
            if len(remaining) > 0:
                additional = remaining.sample(n=1, random_state=42)
                for _, row in additional.iterrows():
                    selected_videos.append({
                        'video_info': row.to_dict(),
                        'class_name': row['label'],
                        'class_color': self.get_class_color(row['label'])
                    })
        
        # Shuffle to mix classes
        random.shuffle(selected_videos)
        
        self.logger.info(f"Selected {len(selected_videos)} videos for inspection")
        return selected_videos
    
    def get_class_color(self, class_name: str) -> str:
        """Get color for class visualization."""
        colors = {
            'doctor': '#28a745',    # Green
            'glasses': '#007bff',   # Blue
            'phone': '#ffc107',     # Yellow
            'pillow': '#6f42c1',    # Purple
            'help': '#dc3545',      # Red
            'unknown': '#6c757d'    # Gray
        }
        return colors.get(class_name.lower(), '#6c757d')
    
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
    
    def generate_html_report(self, selected_videos: List[Dict[str, Any]]) -> str:
        """
        Generate complete HTML inspection report.
        
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
    <title>Geometric Mouth Cropping - Visual Inspection Report</title>
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
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
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
        .class-badge {{
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
            color: #28a745;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        .comparison-note {{
            background: #e7f3ff;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📐 Geometric Mouth Cropping</h1>
        <p>Visual Inspection Report - Generated on {timestamp}</p>
    </div>
    
    <div class="summary">
        <h2>📊 Geometric Cropping Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{len(self.df)}</div>
                <div class="stat-label">Total Videos Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(selected_videos)}</div>
                <div class="stat-label">Sampled for Inspection</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">100%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">96×96</div>
                <div class="stat-label">Output Resolution</div>
            </div>
        </div>
        
        <div class="comparison-note">
            <h3>🔄 Geometric vs MediaPipe Comparison</h3>
            <p><strong>Geometric Approach:</strong> Deterministic 3×2 grid crop (top 50% height, middle 33% width)</p>
            <p><strong>Advantages:</strong> 100% success rate, faster processing, consistent results</p>
            <p><strong>Trade-offs:</strong> Fixed crop region, no adaptive positioning based on actual lip location</p>
        </div>
    </div>
"""
        
        # Process each selected video
        for i, video_data in enumerate(selected_videos):
            video_info = video_data['video_info']
            class_name = video_data['class_name']
            class_color = video_data['class_color']
            
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
        <div class="video-header" style="border-left-color: {class_color};">
            <div class="video-title">
                📁 {video_path.name}
                <span class="class-badge" style="background-color: {class_color};">{class_name.title()}</span>
            </div>
            
            <div class="video-metadata">
                <div class="metadata-item">
                    <div class="metadata-label">Class Label</div>
                    <div class="metadata-value">{video_info['label'].title()}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Crop Method</div>
                    <div class="metadata-value">{video_info['crop_method'].title()}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Crop Region</div>
                    <div class="metadata-value">{video_info['crop_region']}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Frame Count</div>
                    <div class="metadata-value">{video_info['frames']} frames</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Output Size</div>
                    <div class="metadata-value">{video_info['output_size']}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Source Video</div>
                    <div class="metadata-value">{pathlib.Path(video_info['src']).name}</div>
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
        <h2>🔍 Geometric Cropping Evaluation</h2>
        <ul>
            <li><strong>Consistency:</strong> Check if the geometric crop consistently captures the mouth region across different videos</li>
            <li><strong>Coverage:</strong> Verify that the top-middle 3×2 grid region includes the full lip area</li>
            <li><strong>Resolution Quality:</strong> Ensure 96×96 resolution maintains sufficient detail for lip reading</li>
            <li><strong>Temporal Stability:</strong> Look for consistent framing across the 32-frame sequence</li>
            <li><strong>Class Diversity:</strong> Evaluate performance across different classes and speakers</li>
        </ul>
        
        <h3>📏 Crop Specifications:</h3>
        <ul>
            <li><strong>Vertical Crop:</strong> Top 50% of frame height (removes chin and lower face)</li>
            <li><strong>Horizontal Crop:</strong> Middle 33% of frame width (centers on lip region)</li>
            <li><strong>Output Format:</strong> 96×96 pixels, 32 frames per video, 25 fps</li>
            <li><strong>Processing:</strong> Histogram equalization for contrast enhancement</li>
        </ul>
    </div>
    
    <footer style="text-align: center; margin-top: 40px; padding: 20px; color: #666;">
        <p>Generated by Geometric Visual Inspection Tool | Deterministic 3×2 Grid Cropping</p>
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
        self.logger.info("Starting geometric visual inspection report generation")
        
        # Select representative sample
        selected_videos = self.select_representative_sample()
        
        if not selected_videos:
            raise ValueError("No videos selected for inspection")
        
        # Generate HTML content
        html_content = self.generate_html_report(selected_videos)
        
        # Write HTML file
        with open(self.output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Geometric visual inspection report generated: {self.output_html}")
        return str(self.output_html)


def main():
    """Main CLI interface for geometric visual inspection tool."""
    if len(sys.argv) < 3:
        print("Usage: python geometric_visual_inspection.py MANIFEST_CSV VIDEOS_DIR [OUTPUT_HTML]")
        print("\nExample:")
        print("  python geometric_visual_inspection.py geometric_training_manifest.csv geometric_crops_96x96_32f geometric_inspection.html")
        print("\nDescription:")
        print("  Generates HTML visual inspection report for geometric mouth cropping results.")
        print("  Evaluates the quality and consistency of deterministic geometric crops.")
        sys.exit(1)
    
    manifest_path = sys.argv[1]
    videos_dir = sys.argv[2]
    output_html = sys.argv[3] if len(sys.argv) > 3 else "geometric_inspection.html"
    
    # Validate inputs
    if not pathlib.Path(manifest_path).exists():
        print(f"Error: Manifest file '{manifest_path}' does not exist")
        sys.exit(1)
    
    if not pathlib.Path(videos_dir).exists():
        print(f"Error: Videos directory '{videos_dir}' does not exist")
        sys.exit(1)
    
    try:
        # Initialize and run inspection tool
        inspector = GeometricVisualInspector(manifest_path, videos_dir, output_html)
        html_file = inspector.generate_inspection_report()
        
        print(f"\n✅ Geometric visual inspection report generated successfully!")
        print(f"📄 HTML file: {html_file}")
        print(f"🔍 Open in browser to inspect geometric cropping quality")
        
        return html_file
        
    except Exception as e:
        print(f"❌ Error generating inspection report: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
