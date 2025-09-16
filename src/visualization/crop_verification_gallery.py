#!/usr/bin/env python3
"""
Crop Verification Gallery Generator
==================================

Creates a visual gallery showing frames from cropped videos to verify
the quality and effectiveness of the geometric cropping pipeline.

Features:
- Samples 60 videos randomly across all classes
- Extracts middle frame from each video
- Creates responsive HTML gallery with thumbnails
- Shows class labels and video information
- Allows visual verification of crop quality

Usage:
    python crop_verification_gallery.py CROPPED_DIR MANIFEST_CSV [OUTPUT_HTML]

Example:
    python crop_verification_gallery.py grid_cropped_dataset grid_processing_manifest.csv crop_gallery.html
"""

import sys
import cv2
import pandas as pd
import pathlib
import random
import base64
import logging
from datetime import datetime
from typing import List, Dict, Tuple

class CropVerificationGallery:
    """
    Creates visual galleries for verifying crop quality across multiple videos.
    """
    
    def __init__(self, 
                 cropped_dir: str,
                 manifest_path: str,
                 output_html: str = "crop_gallery.html",
                 num_samples: int = 60):
        """
        Initialize crop verification gallery generator.
        
        Args:
            cropped_dir: Directory containing cropped videos
            manifest_path: Path to processing manifest CSV
            output_html: Output HTML file path
            num_samples: Number of videos to sample
        """
        self.cropped_dir = pathlib.Path(cropped_dir)
        self.manifest_path = pathlib.Path(manifest_path)
        self.output_html = pathlib.Path(output_html)
        self.num_samples = num_samples
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_manifest_and_sample(self) -> List[Dict]:
        """
        Load manifest and select random sample of videos.
        
        Returns:
            List of sampled video information
        """
        try:
            df = pd.read_csv(self.manifest_path)
            self.logger.info(f"Loaded manifest with {len(df)} entries")
            
            # Filter successful videos only
            successful_df = df[df['processing_status'] == 'success'].copy()
            self.logger.info(f"Found {len(successful_df)} successful videos")
            
            if len(successful_df) == 0:
                self.logger.error("No successful videos found")
                return []
            
            # Sample videos, ensuring representation across classes
            samples = []
            
            # Get class distribution
            class_counts = successful_df['label'].value_counts()
            self.logger.info(f"Class distribution: {dict(class_counts)}")
            
            # Calculate samples per class (roughly equal distribution)
            samples_per_class = max(1, self.num_samples // len(class_counts))
            remaining_samples = self.num_samples
            
            for label in class_counts.index:
                if remaining_samples <= 0:
                    break
                    
                label_df = successful_df[successful_df['label'] == label]
                sample_count = min(samples_per_class, len(label_df), remaining_samples)
                
                sampled = label_df.sample(n=sample_count, random_state=42)
                
                for _, row in sampled.iterrows():
                    samples.append({
                        'label': row['label'],
                        'output_path': row['output_path'],
                        'source_path': row['source_path'],
                        'original_frames': row['original_frames'],
                        'processed_frames': row['processed_frames'],
                        'original_resolution': row['original_resolution'],
                        'crop_coordinates': row['crop_coordinates'],
                        'crop_size': row['crop_size'],
                        'original_fps': row['original_fps']
                    })
                
                remaining_samples -= sample_count
            
            # If we still need more samples, fill randomly
            if remaining_samples > 0:
                remaining_df = successful_df[~successful_df.index.isin([s['output_path'] for s in samples])]
                if len(remaining_df) > 0:
                    additional_samples = remaining_df.sample(n=min(remaining_samples, len(remaining_df)), random_state=42)
                    for _, row in additional_samples.iterrows():
                        samples.append({
                            'label': row['label'],
                            'output_path': row['output_path'],
                            'source_path': row['source_path'],
                            'original_frames': row['original_frames'],
                            'processed_frames': row['processed_frames'],
                            'original_resolution': row['original_resolution'],
                            'crop_coordinates': row['crop_coordinates'],
                            'crop_size': row['crop_size'],
                            'original_fps': row['original_fps']
                        })
            
            # Shuffle final samples
            random.shuffle(samples)
            samples = samples[:self.num_samples]
            
            self.logger.info(f"Selected {len(samples)} videos for gallery")
            return samples
            
        except Exception as e:
            self.logger.error(f"Error loading manifest: {str(e)}")
            return []
    
    def extract_middle_frame(self, video_path: pathlib.Path) -> str:
        """
        Extract middle frame from video and encode as base64.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Base64 encoded image string or empty string if failed
        """
        try:
            if not video_path.exists():
                self.logger.warning(f"Video not found: {video_path}")
                return ""
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.warning(f"Cannot open video: {video_path}")
                return ""
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return ""
            
            # Get middle frame
            middle_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return ""
            
            # Encode as JPEG with good quality
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if buffer is None:
                return ""
            
            # Convert to base64
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
            
        except Exception as e:
            self.logger.error(f"Error extracting frame from {video_path}: {str(e)}")
            return ""
    
    def generate_gallery_html(self, samples: List[Dict]) -> None:
        """
        Generate HTML gallery showing all sampled frames.
        
        Args:
            samples: List of sample dictionaries with frame data
        """
        # Count samples by class
        class_counts = {}
        for sample in samples:
            label = sample['label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crop Verification Gallery - {len(samples)} Videos</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
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
        }}
        
        .header h1 {{
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header p {{
            color: #666;
            font-size: 1.1em;
            margin-bottom: 20px;
        }}
        
        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
        }}
        
        .stat {{
            text-align: center;
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .class-distribution {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .class-distribution h2 {{
            color: #333;
            margin-bottom: 15px;
            text-align: center;
        }}
        
        .class-tags {{
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }}
        
        .class-tag {{
            background: #667eea;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 500;
        }}
        
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .video-card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .video-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
        }}
        
        .video-image {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-bottom: 1px solid #eee;
        }}
        
        .video-info {{
            padding: 15px;
        }}
        
        .video-label {{
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
            display: inline-block;
            margin-bottom: 10px;
        }}
        
        .video-details {{
            font-size: 0.85em;
            color: #666;
            line-height: 1.4;
        }}
        
        .video-details strong {{
            color: #333;
        }}
        
        .no-image {{
            width: 100%;
            height: 200px;
            background: #f8f9fa;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-style: italic;
            border-bottom: 1px solid #eee;
        }}
        
        .footer {{
            text-align: center;
            color: rgba(255, 255, 255, 0.8);
            margin-top: 40px;
            padding: 20px;
        }}
        
        @media (max-width: 768px) {{
            .gallery {{
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 15px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .stats {{
                gap: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Crop Verification Gallery</h1>
            <p>Visual verification of 3×2 grid-based geometric cropping results</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-number">{len(samples)}</div>
                    <div class="stat-label">Videos Sampled</div>
                </div>
                <div class="stat">
                    <div class="stat-number">{len(class_counts)}</div>
                    <div class="stat-label">Classes</div>
                </div>
                <div class="stat">
                    <div class="stat-number">96×96</div>
                    <div class="stat-label">Output Size</div>
                </div>
                <div class="stat">
                    <div class="stat-number">3×2</div>
                    <div class="stat-label">Grid Layout</div>
                </div>
            </div>
        </div>
        
        <div class="class-distribution">
            <h2>Sample Distribution by Class</h2>
            <div class="class-tags">
"""
        
        for label, count in sorted(class_counts.items()):
            html_content += f'<div class="class-tag">{label.title()}: {count}</div>'
        
        html_content += """
            </div>
        </div>
        
        <div class="gallery">
"""
        
        # Add video cards
        for i, sample in enumerate(samples, 1):
            video_path = pathlib.Path(sample['output_path'])
            filename = video_path.name
            
            # Extract frame
            frame_data = self.extract_middle_frame(video_path)
            
            html_content += f"""
            <div class="video-card">
"""
            
            if frame_data:
                html_content += f'<img src="{frame_data}" alt="Frame from {filename}" class="video-image">'
            else:
                html_content += '<div class="no-image">Frame not available</div>'
            
            html_content += f"""
                <div class="video-info">
                    <div class="video-label">{sample['label']}</div>
                    <div class="video-details">
                        <strong>Video #{i}</strong><br>
                        <strong>Frames:</strong> {sample['processed_frames']}<br>
                        <strong>Original:</strong> {sample['original_resolution']}<br>
                        <strong>Crop:</strong> {sample['crop_size']}<br>
                        <strong>FPS:</strong> {sample['original_fps']:.1f}
                    </div>
                </div>
            </div>
"""
        
        html_content += """
        </div>
        
        <div class="footer">
            <p>Grid Geometric Cropping Pipeline - Crop Verification Gallery</p>
            <p>Each image shows the middle frame from a cropped video (96×96 pixels)</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Write HTML file
        with open(self.output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Gallery HTML saved to: {self.output_html}")
    
    def create_gallery(self) -> None:
        """
        Create complete verification gallery.
        """
        self.logger.info(f"Creating crop verification gallery with {self.num_samples} samples")
        
        # Load manifest and sample videos
        samples = self.load_manifest_and_sample()
        if not samples:
            self.logger.error("No samples available for gallery")
            return
        
        # Generate HTML gallery
        self.logger.info("Generating HTML gallery...")
        self.generate_gallery_html(samples)
        
        self.logger.info("Gallery creation complete!")


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python crop_verification_gallery.py CROPPED_DIR MANIFEST_CSV [OUTPUT_HTML] [NUM_SAMPLES]")
        print()
        print("Example:")
        print('  python crop_verification_gallery.py grid_cropped_dataset grid_processing_manifest.csv crop_gallery.html 60')
        sys.exit(1)
    
    cropped_dir = sys.argv[1]
    manifest_path = sys.argv[2]
    output_html = sys.argv[3] if len(sys.argv) > 3 else "crop_gallery.html"
    num_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    
    # Create gallery
    gallery = CropVerificationGallery(cropped_dir, manifest_path, output_html, num_samples)
    gallery.create_gallery()
    
    print(f"\nGallery created! Open '{output_html}' in your browser to view the results.")


if __name__ == "__main__":
    main()
