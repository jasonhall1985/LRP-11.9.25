#!/usr/bin/env python3
"""
Video Frame Analysis Tool
========================

Analyzes source videos to understand framing, cropping, and content distribution.
Helps diagnose why geometric cropping may not be working consistently.

Features:
- Samples frames from source videos
- Analyzes frame dimensions and aspect ratios
- Creates visual comparison between source and cropped frames
- Identifies videos with inconsistent framing
- Generates diagnostic report

Usage:
    python video_frame_analysis.py SOURCE_DIR CROPPED_DIR [NUM_SAMPLES]

Example:
    python video_frame_analysis.py "/Users/client/Desktop/13.9.25top7dataset" "expanded_cropped_dataset" 20
"""

import sys
import cv2
import numpy as np
import pandas as pd
import pathlib
import random
import base64
import logging
from datetime import datetime
from typing import List, Dict, Tuple

class VideoFrameAnalyzer:
    """
    Analyzes video frames to diagnose cropping issues.
    """
    
    def __init__(self, 
                 source_dir: str,
                 cropped_dir: str,
                 num_samples: int = 20):
        """
        Initialize video frame analyzer.
        
        Args:
            source_dir: Directory containing source videos
            cropped_dir: Directory containing cropped videos
            num_samples: Number of videos to analyze
        """
        self.source_dir = pathlib.Path(source_dir)
        self.cropped_dir = pathlib.Path(cropped_dir)
        self.num_samples = num_samples
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_video_info(self, video_path: pathlib.Path) -> Dict:
        """
        Get basic information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.release()
            
            return {
                'width': width,
                'height': height,
                'aspect_ratio': width / height if height > 0 else 0,
                'fps': fps,
                'frame_count': frame_count,
                'resolution': f'{width}x{height}'
            }
        except Exception as e:
            self.logger.error(f"Error analyzing {video_path}: {str(e)}")
            return None
    
    def extract_sample_frame(self, video_path: pathlib.Path, frame_position: float = 0.5) -> np.ndarray:
        """
        Extract a sample frame from video.
        
        Args:
            video_path: Path to video file
            frame_position: Position in video (0.0 to 1.0)
            
        Returns:
            Frame as numpy array or None if failed
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target_frame = int(frame_count * frame_position)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                return frame
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting frame from {video_path}: {str(e)}")
            return None
    
    def encode_frame_to_base64(self, frame: np.ndarray, max_size: int = 300) -> str:
        """
        Encode frame to base64 for HTML display.
        
        Args:
            frame: Frame as numpy array
            max_size: Maximum dimension for display
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Resize for display if needed
            height, width = frame.shape[:2]
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if buffer is None:
                return ""
            
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
            
        except Exception as e:
            self.logger.error(f"Error encoding frame: {str(e)}")
            return ""
    
    def analyze_sample_videos(self) -> List[Dict]:
        """
        Analyze a sample of videos to understand framing issues.
        
        Returns:
            List of analysis results
        """
        # Find source videos
        source_videos = list(self.source_dir.glob("*.mp4"))
        if len(source_videos) == 0:
            self.logger.error("No MP4 files found in source directory")
            return []
        
        # Sample videos
        sample_videos = random.sample(source_videos, min(self.num_samples, len(source_videos)))
        
        results = []
        
        for source_path in sample_videos:
            self.logger.info(f"Analyzing {source_path.name}")
            
            # Get source video info
            source_info = self.get_video_info(source_path)
            if not source_info:
                continue
            
            # Extract source frame
            source_frame = self.extract_sample_frame(source_path)
            if source_frame is None:
                continue
            
            # Find corresponding cropped video
            cropped_name = f"pure_cropped_{source_path.name}"
            cropped_path = self.cropped_dir / cropped_name
            
            cropped_info = None
            cropped_frame = None
            
            if cropped_path.exists():
                cropped_info = self.get_video_info(cropped_path)
                cropped_frame = self.extract_sample_frame(cropped_path)
            
            # Encode frames for display
            source_b64 = self.encode_frame_to_base64(source_frame)
            cropped_b64 = self.encode_frame_to_base64(cropped_frame) if cropped_frame is not None else ""
            
            # Extract label
            label = self.get_label_from_filename(source_path.name)
            
            results.append({
                'filename': source_path.name,
                'label': label,
                'source_info': source_info,
                'cropped_info': cropped_info,
                'source_frame_b64': source_b64,
                'cropped_frame_b64': cropped_b64,
                'has_cropped': cropped_path.exists()
            })
        
        return results
    
    def get_label_from_filename(self, filename: str) -> str:
        """Extract class label from filename."""
        filename_lower = filename.lower()
        
        if '__' in filename:
            parts = filename.split('__')
            if len(parts) > 0:
                return parts[0].lower()
        
        known_classes = {'doctor', 'glasses', 'phone', 'pillow', 'help', 'i_need_to_move', 'my_mouth_is_dry'}
        for class_name in known_classes:
            if filename_lower.startswith(class_name):
                return class_name
        
        return 'unknown'
    
    def generate_diagnostic_report(self, results: List[Dict]) -> None:
        """
        Generate HTML diagnostic report.
        
        Args:
            results: List of analysis results
        """
        # Analyze resolution patterns
        resolutions = {}
        aspect_ratios = {}
        
        for result in results:
            if result['source_info']:
                res = result['source_info']['resolution']
                resolutions[res] = resolutions.get(res, 0) + 1
                
                ar = round(result['source_info']['aspect_ratio'], 2)
                aspect_ratios[ar] = aspect_ratios.get(ar, 0) + 1
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Frame Analysis - Cropping Diagnostic</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 30px;
            border-radius: 20px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 2px solid #ff6b6b;
        }}
        
        .header h1 {{
            color: #ee5a24;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            color: #ff6b6b;
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 15px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stats-card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .stats-card h3 {{
            color: #ee5a24;
            margin-bottom: 15px;
            border-bottom: 2px solid #ff6b6b;
            padding-bottom: 5px;
        }}
        
        .stat-item {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .comparison-card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 2px solid transparent;
        }}
        
        .comparison-card:hover {{
            border-color: #ff6b6b;
        }}
        
        .card-header {{
            background: #ff6b6b;
            color: white;
            padding: 15px;
            text-align: center;
        }}
        
        .card-title {{
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 5px;
        }}
        
        .card-subtitle {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .frame-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            padding: 15px;
        }}
        
        .frame-section {{
            text-align: center;
        }}
        
        .frame-label {{
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            padding: 5px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        
        .frame-image {{
            width: 100%;
            max-width: 200px;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 8px;
            margin-bottom: 10px;
        }}
        
        .frame-info {{
            font-size: 0.85em;
            color: #666;
            line-height: 1.4;
        }}
        
        .missing-frame {{
            width: 100%;
            max-width: 200px;
            height: 150px;
            background: #f8f9fa;
            border: 2px dashed #ddd;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
            font-style: italic;
            margin-bottom: 10px;
        }}
        
        .issue-indicator {{
            background: #ff6b6b;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.7em;
            font-weight: 600;
            margin-left: 5px;
        }}
        
        .footer {{
            text-align: center;
            color: rgba(255, 255, 255, 0.9);
            margin-top: 40px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Video Frame Analysis</h1>
            <div class="subtitle">Cropping Diagnostic Report</div>
            <p>Analyzing source vs cropped frames to identify framing issues</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stats-card">
                <h3>Source Video Resolutions</h3>
"""
        
        for resolution, count in sorted(resolutions.items(), key=lambda x: x[1], reverse=True):
            html_content += f'<div class="stat-item"><span>{resolution}</span><span>{count} videos</span></div>'
        
        html_content += """
            </div>
            <div class="stats-card">
                <h3>Aspect Ratios</h3>
"""
        
        for ar, count in sorted(aspect_ratios.items(), key=lambda x: x[1], reverse=True):
            html_content += f'<div class="stat-item"><span>{ar}:1</span><span>{count} videos</span></div>'
        
        html_content += f"""
            </div>
            <div class="stats-card">
                <h3>Analysis Summary</h3>
                <div class="stat-item"><span>Videos Analyzed</span><span>{len(results)}</span></div>
                <div class="stat-item"><span>Cropped Videos Found</span><span>{sum(1 for r in results if r['has_cropped'])}</span></div>
                <div class="stat-item"><span>Unique Resolutions</span><span>{len(resolutions)}</span></div>
                <div class="stat-item"><span>Unique Aspect Ratios</span><span>{len(aspect_ratios)}</span></div>
            </div>
        </div>
        
        <div class="comparison-grid">
"""
        
        for i, result in enumerate(results, 1):
            source_info = result['source_info']
            cropped_info = result['cropped_info']
            
            html_content += f"""
            <div class="comparison-card">
                <div class="card-header">
                    <div class="card-title">{result['label'].title()} - Sample #{i}</div>
                    <div class="card-subtitle">{result['filename']}</div>
                </div>
                <div class="frame-comparison">
                    <div class="frame-section">
                        <div class="frame-label">Source Frame</div>
"""
            
            if result['source_frame_b64']:
                html_content += f'<img src="{result["source_frame_b64"]}" alt="Source frame" class="frame-image">'
            else:
                html_content += '<div class="missing-frame">No frame available</div>'
            
            if source_info:
                html_content += f"""
                        <div class="frame-info">
                            <strong>Resolution:</strong> {source_info['resolution']}<br>
                            <strong>Aspect Ratio:</strong> {source_info['aspect_ratio']:.2f}:1<br>
                            <strong>Frames:</strong> {source_info['frame_count']}<br>
                            <strong>FPS:</strong> {source_info['fps']:.1f}
                        </div>
"""
            
            html_content += """
                    </div>
                    <div class="frame-section">
                        <div class="frame-label">Cropped Frame</div>
"""
            
            if result['cropped_frame_b64']:
                html_content += f'<img src="{result["cropped_frame_b64"]}" alt="Cropped frame" class="frame-image">'
            else:
                html_content += '<div class="missing-frame">No cropped frame</div>'
            
            if cropped_info:
                html_content += f"""
                        <div class="frame-info">
                            <strong>Resolution:</strong> {cropped_info['resolution']}<br>
                            <strong>Aspect Ratio:</strong> {cropped_info['aspect_ratio']:.2f}:1<br>
                            <strong>Frames:</strong> {cropped_info['frame_count']}<br>
                            <strong>FPS:</strong> {cropped_info['fps']:.1f}
                        </div>
"""
            
            html_content += """
                    </div>
                </div>
            </div>
"""
        
        html_content += """
        </div>
        
        <div class="footer">
            <p><strong>Video Frame Analysis - Cropping Diagnostic</strong></p>
            <p>Compare source and cropped frames to identify framing inconsistencies</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Write HTML file
        output_path = pathlib.Path("frame_analysis_diagnostic.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Diagnostic report saved to: {output_path}")
    
    def run_analysis(self) -> None:
        """
        Run complete frame analysis.
        """
        self.logger.info(f"Starting video frame analysis with {self.num_samples} samples")
        
        # Analyze sample videos
        results = self.analyze_sample_videos()
        if not results:
            self.logger.error("No analysis results available")
            return
        
        # Generate diagnostic report
        self.logger.info("Generating diagnostic report...")
        self.generate_diagnostic_report(results)
        
        self.logger.info("Frame analysis complete!")


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python video_frame_analysis.py SOURCE_DIR CROPPED_DIR [NUM_SAMPLES]")
        print()
        print("Example:")
        print('  python video_frame_analysis.py "/Users/client/Desktop/13.9.25top7dataset" "expanded_cropped_dataset" 20')
        sys.exit(1)
    
    source_dir = sys.argv[1]
    cropped_dir = sys.argv[2]
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    # Run analysis
    analyzer = VideoFrameAnalyzer(source_dir, cropped_dir, num_samples)
    analyzer.run_analysis()
    
    print(f"\nFrame analysis complete!")
    print(f"Open 'frame_analysis_diagnostic.html' in your browser to view the diagnostic report.")


if __name__ == "__main__":
    main()
