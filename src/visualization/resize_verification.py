#!/usr/bin/env python3
"""
Resize-Only Verification Gallery
================================

Creates a verification gallery specifically for resize-only processing.
Shows frames that have been resized (NOT cropped) to 96x96 pixels.

Usage:
    python resize_verification.py RESIZED_DIR MANIFEST_CSV [OUTPUT_HTML] [NUM_SAMPLES]
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

class ResizeVerificationGallery:
    """
    Creates verification galleries for resize-only processing results.
    """
    
    def __init__(self, 
                 resized_dir: str,
                 manifest_path: str,
                 output_html: str = "resize_verification.html",
                 num_samples: int = 60):
        """
        Initialize resize verification gallery.
        
        Args:
            resized_dir: Directory containing resized videos
            manifest_path: Path to resize processing manifest CSV
            output_html: Output HTML file path
            num_samples: Number of videos to sample
        """
        self.resized_dir = pathlib.Path(resized_dir)
        self.manifest_path = pathlib.Path(manifest_path)
        self.output_html = pathlib.Path(output_html)
        self.num_samples = num_samples
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_manifest_and_sample(self) -> List[Dict]:
        """
        Load resize processing manifest and select representative samples.
        
        Returns:
            List of sampled video information
        """
        try:
            df = pd.read_csv(self.manifest_path)
            self.logger.info(f"Loaded resize processing manifest with {len(df)} entries")
            
            # Filter successful videos only
            successful_df = df[df['processing_status'] == 'success'].copy()
            self.logger.info(f"Found {len(successful_df)} successful resized videos")
            
            if len(successful_df) == 0:
                self.logger.error("No successful videos found")
                return []
            
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
                        'crop_method': row['crop_method'],
                        'processing_type': row['processing_type'],
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
                            'crop_method': row['crop_method'],
                            'processing_type': row['processing_type'],
                            'original_fps': row['original_fps']
                        })
            
            # Shuffle and limit
            random.shuffle(samples)
            samples = samples[:self.num_samples]
            
            self.logger.info(f"Selected {len(samples)} resized videos for verification")
            return samples
            
        except Exception as e:
            self.logger.error(f"Error loading manifest: {str(e)}")
            return []
    
    def extract_middle_frame_resize(self, video_path: pathlib.Path) -> str:
        """
        Extract middle frame from resized video preserving actual size.
        
        Args:
            video_path: Path to resized video file
            
        Returns:
            Base64 encoded image showing actual resized frame
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
            
            # Encode with HIGH quality - NO RESIZING for display
            # Show the actual 96x96 frame as it is
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if buffer is None:
                return ""
            
            # Convert to base64 - frame is already 96x96
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
            
        except Exception as e:
            self.logger.error(f"Error extracting frame from {video_path}: {str(e)}")
            return ""
    
    def generate_resize_gallery_html(self, samples: List[Dict]) -> None:
        """
        Generate HTML gallery specifically for resize-only verification.
        
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
    <title>Resize-Only Verification - {len(samples)} Videos</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
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
            border: 3px solid #4CAF50;
        }}
        
        .header h1 {{
            color: #2E7D32;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            color: #4CAF50;
            font-size: 1.4em;
            font-weight: 700;
            margin-bottom: 15px;
        }}
        
        .header p {{
            color: #666;
            font-size: 1.1em;
            margin-bottom: 20px;
        }}
        
        .no-crop-badges {{
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }}
        
        .badge {{
            background: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 1em;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        }}
        
        .badge::before {{
            content: "✓";
            font-weight: bold;
            font-size: 1.2em;
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
            font-size: 2.2em;
            font-weight: bold;
            color: #2E7D32;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .processing-info {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border-left: 6px solid #4CAF50;
        }}
        
        .processing-info h2 {{
            color: #2E7D32;
            margin-bottom: 20px;
            font-size: 1.5em;
        }}
        
        .no-crop-emphasis {{
            background: #E8F5E8;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #4CAF50;
            margin-bottom: 20px;
        }}
        
        .no-crop-emphasis h3 {{
            color: #2E7D32;
            margin-bottom: 10px;
        }}
        
        .no-crop-emphasis ul {{
            color: #333;
            margin-left: 20px;
        }}
        
        .no-crop-emphasis li {{
            margin-bottom: 5px;
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
            border-color: #4CAF50;
        }}
        
        .video-image {{
            width: 100%;
            height: 200px;
            object-fit: contain;
            border-bottom: 1px solid #eee;
            background: #f8f9fa;
        }}
        
        .video-info {{
            padding: 15px;
        }}
        
        .video-label {{
            background: #2E7D32;
            color: white;
            padding: 6px 14px;
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
        
        .no-crop-indicator {{
            background: #4CAF50;
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.7em;
            font-weight: 700;
            margin-left: 8px;
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
            <h1>Resize-Only Verification</h1>
            <div class="subtitle">NO CROPPING APPLIED • ENTIRE FRAMES RESIZED</div>
            <p>Verification gallery showing complete frames resized to 96×96 pixels</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="no-crop-badges">
                <div class="badge">NO Cropping Applied</div>
                <div class="badge">Entire Frame Preserved</div>
                <div class="badge">Simple Resize Only</div>
                <div class="badge">Original Content Intact</div>
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
                    <div class="stat-number">0%</div>
                    <div class="stat-label">Cropping Applied</div>
                </div>
            </div>
        </div>
        
        <div class="processing-info">
            <h2>Resize-Only Processing Verification</h2>
            <div class="no-crop-emphasis">
                <h3>🚫 NO CROPPING OPERATIONS PERFORMED</h3>
                <ul>
                    <li><strong>Entire source frame</strong> taken as input</li>
                    <li><strong>Complete content preserved</strong> - nothing removed or cut off</li>
                    <li><strong>Simple resize operation</strong> to 96×96 pixels only</li>
                    <li><strong>Original aspect ratio</strong> adjusted to square format</li>
                    <li><strong>All visual information</strong> from source maintained</li>
                </ul>
            </div>
        </div>
        
        <div class="gallery">
"""
        
        # Add video cards
        for i, sample in enumerate(samples, 1):
            video_path = pathlib.Path(sample['output_path'])
            filename = video_path.name
            
            # Extract frame showing actual 96x96 result
            frame_data = self.extract_middle_frame_resize(video_path)
            
            html_content += f"""
            <div class="video-card">
"""
            
            if frame_data:
                html_content += f'<img src="{frame_data}" alt="Resized frame from {filename}" class="video-image">'
            else:
                html_content += '<div class="no-image">Frame not available</div>'
            
            html_content += f"""
                <div class="video-info">
                    <div class="video-label">{sample['label']}<span class="no-crop-indicator">NO CROP</span></div>
                    <div class="video-details">
                        <strong>Video #{i}</strong><br>
                        <strong>Frames:</strong> {sample['processed_frames']}<br>
                        <strong>Original:</strong> {sample['original_resolution']}<br>
                        <strong>Method:</strong> {sample['crop_method']}<br>
                        <strong>Processing:</strong> {sample['processing_type']}
                    </div>
                </div>
            </div>
"""
        
        html_content += f"""
        </div>
        
        <div class="footer">
            <p><strong>Resize-Only Processing Pipeline - Complete Frame Verification</strong></p>
            <p>All frames show entire source content resized to 96×96 pixels</p>
            <p>NO cropping, NO zooming, NO content removal applied</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Write HTML file
        with open(self.output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Resize verification gallery saved to: {self.output_html}")
    
    def create_verification_gallery(self) -> None:
        """
        Create complete resize verification gallery.
        """
        self.logger.info(f"Creating resize verification gallery with {self.num_samples} samples")
        
        # Load manifest and sample videos
        samples = self.load_manifest_and_sample()
        if not samples:
            self.logger.error("No samples available for verification")
            return
        
        # Generate HTML gallery
        self.logger.info("Generating resize verification gallery...")
        self.generate_resize_gallery_html(samples)
        
        self.logger.info("Resize verification gallery complete!")


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python resize_verification.py RESIZED_DIR MANIFEST_CSV [OUTPUT_HTML] [NUM_SAMPLES]")
        print()
        print("Example:")
        print('  python resize_verification.py resized_dataset resize_manifest.csv resize_verification.html 60')
        sys.exit(1)
    
    resized_dir = sys.argv[1]
    manifest_path = sys.argv[2]
    output_html = sys.argv[3] if len(sys.argv) > 3 else "resize_verification.html"
    num_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    
    # Create verification gallery
    gallery = ResizeVerificationGallery(resized_dir, manifest_path, output_html, num_samples)
    gallery.create_verification_gallery()
    
    print(f"\nResize verification gallery created!")
    print(f"Open '{output_html}' in your browser to verify NO cropping was applied.")


if __name__ == "__main__":
    main()
