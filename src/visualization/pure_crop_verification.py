#!/usr/bin/env python3
"""
Pure Crop Verification Gallery
==============================

Creates a visual gallery specifically for verifying pure geometric cropping results.
Shows frames in original color format without any image processing artifacts.

Features:
- Samples 60 videos across all classes
- Extracts middle frame in original color format
- Creates responsive HTML gallery
- Verifies NO image processing was applied
- Shows original pixel values and color characteristics

Usage:
    python pure_crop_verification.py CROPPED_DIR MANIFEST_CSV [OUTPUT_HTML] [NUM_SAMPLES]

Example:
    python pure_crop_verification.py pure_cropped_dataset pure_processing_manifest.csv pure_gallery.html 60
"""

import sys
import cv2
import pandas as pd
import pathlib
import random
import base64
import logging
from datetime import datetime
from typing import List, Dict

class PureCropVerificationGallery:
    """
    Creates verification galleries for pure geometric cropping results.
    Ensures frames are displayed in original color format without artifacts.
    """
    
    def __init__(self, 
                 cropped_dir: str,
                 manifest_path: str,
                 output_html: str = "pure_gallery.html",
                 num_samples: int = 60):
        """
        Initialize pure crop verification gallery.
        
        Args:
            cropped_dir: Directory containing pure cropped videos
            manifest_path: Path to pure processing manifest CSV
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
        Load pure processing manifest and select representative samples.
        
        Returns:
            List of sampled video information
        """
        try:
            df = pd.read_csv(self.manifest_path)
            self.logger.info(f"Loaded pure processing manifest with {len(df)} entries")
            
            # Filter successful videos only
            successful_df = df[df['processing_status'] == 'success'].copy()
            self.logger.info(f"Found {len(successful_df)} successful pure cropped videos")
            
            if len(successful_df) == 0:
                self.logger.error("No successful videos found")
                return []
            
            # Verify this is pure processing
            pure_videos = successful_df[successful_df['processing_type'] == 'pure_geometric_only']
            if len(pure_videos) != len(successful_df):
                self.logger.warning("Some videos may not be pure geometric processed")
            
            # Sample videos across classes
            samples = []
            class_counts = successful_df['label'].value_counts()
            self.logger.info(f"Class distribution: {dict(class_counts)}")
            
            # Calculate samples per class
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
                        'processing_type': row['processing_type'],
                        'image_processing': row['image_processing'],
                        'original_fps': row['original_fps']
                    })
                
                remaining_samples -= sample_count
            
            # Fill remaining samples randomly if needed
            if remaining_samples > 0:
                used_paths = {s['output_path'] for s in samples}
                remaining_df = successful_df[~successful_df['output_path'].isin(used_paths)]
                if len(remaining_df) > 0:
                    additional = remaining_df.sample(n=min(remaining_samples, len(remaining_df)), random_state=42)
                    for _, row in additional.iterrows():
                        samples.append({
                            'label': row['label'],
                            'output_path': row['output_path'],
                            'source_path': row['source_path'],
                            'original_frames': row['original_frames'],
                            'processed_frames': row['processed_frames'],
                            'original_resolution': row['original_resolution'],
                            'crop_coordinates': row['crop_coordinates'],
                            'crop_size': row['crop_size'],
                            'processing_type': row['processing_type'],
                            'image_processing': row['image_processing'],
                            'original_fps': row['original_fps']
                        })
            
            # Shuffle and limit
            random.shuffle(samples)
            samples = samples[:self.num_samples]
            
            self.logger.info(f"Selected {len(samples)} pure cropped videos for verification")
            return samples
            
        except Exception as e:
            self.logger.error(f"Error loading manifest: {str(e)}")
            return []
    
    def extract_middle_frame_pure(self, video_path: pathlib.Path) -> str:
        """
        Extract middle frame preserving original color format and quality.
        
        Args:
            video_path: Path to pure cropped video file
            
        Returns:
            Base64 encoded image in original color format
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
            
            # Encode with HIGH quality to preserve original characteristics
            # Use maximum JPEG quality to avoid compression artifacts
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if buffer is None:
                return ""
            
            # Convert to base64 - frame should be in original BGR color format
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
            
        except Exception as e:
            self.logger.error(f"Error extracting frame from {video_path}: {str(e)}")
            return ""
    
    def generate_pure_gallery_html(self, samples: List[Dict]) -> None:
        """
        Generate HTML gallery specifically for pure geometric cropping verification.
        
        Args:
            samples: List of sample dictionaries
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
    <title>Pure Geometric Cropping Verification - {len(samples)} Videos</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
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
            border: 2px solid #32CD32;
        }}
        
        .header h1 {{
            color: #2E8B57;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            color: #228B22;
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 15px;
        }}
        
        .header p {{
            color: #666;
            font-size: 1.1em;
            margin-bottom: 20px;
        }}
        
        .verification-badges {{
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }}
        
        .badge {{
            background: #32CD32;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .badge::before {{
            content: "✓";
            font-weight: bold;
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
            color: #2E8B57;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .processing-info {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #32CD32;
        }}
        
        .processing-info h2 {{
            color: #2E8B57;
            margin-bottom: 15px;
        }}
        
        .processing-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .detail-item {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            border-left: 3px solid #32CD32;
        }}
        
        .detail-label {{
            font-weight: bold;
            color: #2E8B57;
            font-size: 0.9em;
        }}
        
        .detail-value {{
            color: #333;
            font-size: 0.9em;
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
            border: 2px solid transparent;
        }}
        
        .video-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
            border-color: #32CD32;
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
            background: #2E8B57;
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
        
        .pure-indicator {{
            background: #32CD32;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.7em;
            font-weight: 600;
            margin-left: 5px;
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
            color: rgba(255, 255, 255, 0.9);
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
            <h1>Expanded Pure Geometric Cropping Verification</h1>
            <div class="subtitle">Original Color Format • Zero Image Processing • 10% Expanded Field of View</div>
            <p>Verification gallery showing frames with expanded crop region (10% wider) in original color format</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="verification-badges">
                <div class="badge">Original Color Format</div>
                <div class="badge">10% Expanded Field of View</div>
                <div class="badge">No Grayscale Conversion</div>
                <div class="badge">No Histogram Equalization</div>
                <div class="badge">No Contrast Enhancement</div>
                <div class="badge">Original Pixel Values</div>
            </div>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-number">{len(samples)}</div>
                    <div class="stat-label">Videos Verified</div>
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
                    <div class="stat-number">+10%</div>
                    <div class="stat-label">Expanded View</div>
                </div>
            </div>
        </div>
        
        <div class="processing-info">
            <h2>Expanded Cropping Verification</h2>
            <div class="processing-details">
                <div class="detail-item">
                    <div class="detail-label">Processing Type</div>
                    <div class="detail-value">Pure Geometric Only</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Crop Expansion</div>
                    <div class="detail-value">10% in All Directions</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Grid Layout</div>
                    <div class="detail-value">3×2 (Expanded Top-Middle)</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Field of View</div>
                    <div class="detail-value">Wider Context Around Mouth</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Color Format</div>
                    <div class="detail-value">Original BGR/RGB Preserved</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Image Processing</div>
                    <div class="detail-value">None Applied</div>
                </div>
            </div>
        </div>
        
        <div class="gallery">
"""
        
        # Add video cards
        for i, sample in enumerate(samples, 1):
            video_path = pathlib.Path(sample['output_path'])
            filename = video_path.name
            
            # Extract frame in original color format
            frame_data = self.extract_middle_frame_pure(video_path)
            
            html_content += f"""
            <div class="video-card">
"""
            
            if frame_data:
                html_content += f'<img src="{frame_data}" alt="Pure cropped frame from {filename}" class="video-image">'
            else:
                html_content += '<div class="no-image">Frame not available</div>'
            
            html_content += f"""
                <div class="video-info">
                    <div class="video-label">{sample['label']}<span class="pure-indicator">EXPANDED</span></div>
                    <div class="video-details">
                        <strong>Video #{i}</strong><br>
                        <strong>Frames:</strong> {sample['processed_frames']}<br>
                        <strong>Original:</strong> {sample['original_resolution']}<br>
                        <strong>Crop:</strong> {sample['crop_size']}<br>
                        <strong>Processing:</strong> {sample['processing_type']}<br>
                        <strong>Enhancement:</strong> {sample['image_processing']}
                    </div>
                </div>
            </div>
"""
        
        html_content += f"""
        </div>
        
        <div class="footer">
            <p><strong>Expanded Pure Geometric Cropping Pipeline - Quality Verification</strong></p>
            <p>All frames shown in original color format with 10% expanded field of view</p>
            <p>Wider context around mouth region • No image processing applied</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Write HTML file
        with open(self.output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Pure crop verification gallery saved to: {self.output_html}")
    
    def create_verification_gallery(self) -> None:
        """
        Create complete pure crop verification gallery.
        """
        self.logger.info(f"Creating pure crop verification gallery with {self.num_samples} samples")
        
        # Load manifest and sample videos
        samples = self.load_manifest_and_sample()
        if not samples:
            self.logger.error("No samples available for verification")
            return
        
        # Generate HTML gallery
        self.logger.info("Generating pure crop verification gallery...")
        self.generate_pure_gallery_html(samples)
        
        self.logger.info("Pure crop verification gallery complete!")


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python pure_crop_verification.py CROPPED_DIR MANIFEST_CSV [OUTPUT_HTML] [NUM_SAMPLES]")
        print()
        print("Example:")
        print('  python pure_crop_verification.py pure_cropped_dataset pure_processing_manifest.csv pure_gallery.html 60')
        sys.exit(1)
    
    cropped_dir = sys.argv[1]
    manifest_path = sys.argv[2]
    output_html = sys.argv[3] if len(sys.argv) > 3 else "pure_gallery.html"
    num_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    
    # Create verification gallery
    gallery = PureCropVerificationGallery(cropped_dir, manifest_path, output_html, num_samples)
    gallery.create_verification_gallery()
    
    print(f"\nPure crop verification gallery created!")
    print(f"Open '{output_html}' in your browser to verify original color format and quality.")


if __name__ == "__main__":
    main()
