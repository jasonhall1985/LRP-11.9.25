#!/usr/bin/env python3
"""
Grid Dataset Visual Inspection Tool
===================================

Creates visual inspection reports for grid-cropped videos to evaluate
the quality and consistency of the 3×2 grid-based geometric cropping.

Features:
- Samples representative videos across all classes
- Extracts frames at key positions for visual evaluation
- Generates HTML reports with embedded images
- Compares crop quality across different classes and demographics

Usage:
    python grid_visual_inspection.py MANIFEST_CSV OUTPUT_DIR [NUM_SAMPLES]

Example:
    python grid_visual_inspection.py grid_processing_manifest.csv grid_cropped_dataset grid_inspection.html
"""

import sys
import cv2
import pandas as pd
import pathlib
import random
from typing import List, Dict, Tuple
import base64
import logging
from datetime import datetime

class GridVisualInspector:
    """
    Visual inspection tool for grid-cropped video datasets.
    
    Generates comprehensive HTML reports showing frame samples
    from processed videos to evaluate cropping quality.
    """
    
    def __init__(self, 
                 manifest_path: str,
                 output_dir: str,
                 report_path: str = "grid_inspection.html"):
        """
        Initialize grid visual inspector.
        
        Args:
            manifest_path: Path to processing manifest CSV
            output_dir: Directory containing cropped videos
            report_path: Path for output HTML report
        """
        self.manifest_path = pathlib.Path(manifest_path)
        self.output_dir = pathlib.Path(output_dir)
        self.report_path = pathlib.Path(report_path)
        
        # Frame sampling configuration
        self.sample_positions = [1, 8, 16, 24, 32]  # Frame positions to sample
        self.samples_per_class = 2  # Number of videos to sample per class
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_manifest(self) -> pd.DataFrame:
        """
        Load processing manifest CSV.
        
        Returns:
            DataFrame with processing results
        """
        try:
            df = pd.read_csv(self.manifest_path)
            self.logger.info(f"Loaded manifest with {len(df)} entries")
            return df
        except Exception as e:
            self.logger.error(f"Error loading manifest: {str(e)}")
            return pd.DataFrame()
    
    def select_representative_samples(self, df: pd.DataFrame) -> List[Dict]:
        """
        Select representative video samples for inspection.
        
        Args:
            df: Manifest DataFrame
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        # Get successful videos only
        successful_df = df[df['processing_status'] == 'success'].copy()
        
        if successful_df.empty:
            self.logger.warning("No successful videos found in manifest")
            return samples
        
        # Group by label and sample
        for label in successful_df['label'].unique():
            label_df = successful_df[successful_df['label'] == label]
            
            # Sample videos for this label
            sample_count = min(self.samples_per_class, len(label_df))
            sampled_videos = label_df.sample(n=sample_count, random_state=42)
            
            for _, row in sampled_videos.iterrows():
                samples.append({
                    'label': row['label'],
                    'output_path': row['output_path'],
                    'source_path': row['source_path'],
                    'original_frames': row['original_frames'],
                    'processed_frames': row['processed_frames'],
                    'original_resolution': row['original_resolution'],
                    'crop_coordinates': row['crop_coordinates'],
                    'crop_size': row['crop_size']
                })
        
        self.logger.info(f"Selected {len(samples)} representative samples")
        return samples
    
    def extract_frame_at_position(self, video_path: pathlib.Path, position: int) -> str:
        """
        Extract frame at specific position and encode as base64.
        
        Args:
            video_path: Path to video file
            position: Frame position to extract
            
        Returns:
            Base64 encoded image string or empty string if failed
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return ""
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Adjust position if beyond video length
            actual_position = min(position, total_frames - 1) if total_frames > 0 else 0
            
            # Seek to position
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_position)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return ""
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if buffer is None:
                return ""
            
            # Convert to base64
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
            
        except Exception as e:
            self.logger.error(f"Error extracting frame from {video_path}: {str(e)}")
            return ""
    
    def generate_sample_frames(self, sample: Dict) -> Dict:
        """
        Generate frame samples for a video.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Dictionary with frame data
        """
        video_path = pathlib.Path(sample['output_path'])
        
        if not video_path.exists():
            self.logger.warning(f"Video not found: {video_path}")
            return sample
        
        # Extract frames at sample positions
        frames = {}
        for position in self.sample_positions:
            frame_data = self.extract_frame_at_position(video_path, position)
            if frame_data:
                frames[f"frame_{position}"] = frame_data
        
        sample['frames'] = frames
        return sample
    
    def generate_html_report(self, samples: List[Dict]) -> None:
        """
        Generate comprehensive HTML inspection report.
        
        Args:
            samples: List of sample dictionaries with frame data
        """
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grid Dataset Visual Inspection Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
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
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .summary h2 {{
            color: #333;
            margin-top: 0;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .sample {{
            background: white;
            margin-bottom: 30px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .sample-header {{
            background: #667eea;
            color: white;
            padding: 20px;
        }}
        .sample-header h3 {{
            margin: 0;
            font-size: 1.5em;
        }}
        .sample-info {{
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .info-item {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .info-label {{
            font-weight: bold;
            color: #333;
            font-size: 0.9em;
        }}
        .info-value {{
            color: #666;
            font-size: 0.9em;
        }}
        .frames-container {{
            padding: 20px;
        }}
        .frames-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .frame-item {{
            text-align: center;
        }}
        .frame-item img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }}
        .frame-item img:hover {{
            transform: scale(1.05);
        }}
        .frame-label {{
            margin-top: 8px;
            font-weight: bold;
            color: #333;
            font-size: 0.9em;
        }}
        .no-frames {{
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 40px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
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
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #e9ecef;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Grid Dataset Visual Inspection Report</h1>
        <p>3×2 Grid-Based Geometric Cropping Quality Assessment</p>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>Processing Summary</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{len(samples)}</div>
                <div class="stat-label">Sample Videos</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(set(s['label'] for s in samples))}</div>
                <div class="stat-label">Classes</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(self.sample_positions)}</div>
                <div class="stat-label">Frames per Video</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">96×96</div>
                <div class="stat-label">Output Resolution</div>
            </div>
        </div>
        
        <p><strong>Grid Configuration:</strong> 3×2 grid (3 columns × 2 rows), extracting top-middle cell</p>
        <p><strong>Crop Region:</strong> Top row, middle column of the grid division</p>
        <p><strong>Frame Positions:</strong> {', '.join(map(str, self.sample_positions))}</p>
    </div>
"""
        
        # Add samples
        for i, sample in enumerate(samples, 1):
            label = sample['label'].title()
            frames = sample.get('frames', {})
            
            html_content += f"""
    <div class="sample">
        <div class="sample-header">
            <h3>Sample {i}: {label}</h3>
        </div>
        
        <div class="sample-info">
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Class Label</div>
                    <div class="info-value">{sample['label']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Original Frames</div>
                    <div class="info-value">{sample['original_frames']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Processed Frames</div>
                    <div class="info-value">{sample['processed_frames']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Original Resolution</div>
                    <div class="info-value">{sample['original_resolution']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Crop Coordinates</div>
                    <div class="info-value">{sample['crop_coordinates']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Crop Size</div>
                    <div class="info-value">{sample['crop_size']}</div>
                </div>
            </div>
        </div>
        
        <div class="frames-container">
"""
            
            if frames:
                html_content += '<div class="frames-grid">'
                for position in self.sample_positions:
                    frame_key = f"frame_{position}"
                    if frame_key in frames:
                        html_content += f"""
                <div class="frame-item">
                    <img src="{frames[frame_key]}" alt="Frame {position}">
                    <div class="frame-label">Frame {position}</div>
                </div>
"""
                html_content += '</div>'
            else:
                html_content += '<div class="no-frames">No frames could be extracted from this video</div>'
            
            html_content += """
        </div>
    </div>
"""
        
        # Add footer
        html_content += f"""
    <div class="footer">
        <p>Grid Dataset Visual Inspection Report</p>
        <p>Generated by Grid Geometric Cropping Pipeline</p>
    </div>
</body>
</html>
"""
        
        # Write HTML file
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report saved to: {self.report_path}")
    
    def run_inspection(self) -> None:
        """
        Run complete visual inspection process.
        """
        self.logger.info("Starting grid dataset visual inspection")
        
        # Load manifest
        df = self.load_manifest()
        if df.empty:
            self.logger.error("Cannot proceed without valid manifest")
            return
        
        # Select samples
        samples = self.select_representative_samples(df)
        if not samples:
            self.logger.error("No samples selected for inspection")
            return
        
        # Generate frame samples
        self.logger.info("Extracting frame samples...")
        for i, sample in enumerate(samples):
            self.logger.info(f"Processing sample {i+1}/{len(samples)}: {sample['label']}")
            samples[i] = self.generate_sample_frames(sample)
        
        # Generate HTML report
        self.logger.info("Generating HTML report...")
        self.generate_html_report(samples)
        
        self.logger.info("Visual inspection complete!")


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python grid_visual_inspection.py MANIFEST_CSV OUTPUT_DIR [REPORT_HTML]")
        print()
        print("Example:")
        print('  python grid_visual_inspection.py grid_processing_manifest.csv grid_cropped_dataset grid_inspection.html')
        sys.exit(1)
    
    manifest_path = sys.argv[1]
    output_dir = sys.argv[2]
    report_path = sys.argv[3] if len(sys.argv) > 3 else "grid_inspection.html"
    
    # Initialize inspector
    inspector = GridVisualInspector(manifest_path, output_dir, report_path)
    
    # Run inspection
    inspector.run_inspection()
    
    print(f"\nInspection complete! Open '{report_path}' in your browser to view the results.")


if __name__ == "__main__":
    main()
