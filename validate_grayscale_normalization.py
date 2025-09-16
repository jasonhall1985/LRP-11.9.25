#!/usr/bin/env python3
"""
Grayscale Normalization Validation Script

This script validates the improved grayscale normalization implementation by:
1. Processing all 10 test videos with the new normalization
2. Generating before/after comparison images
3. Analyzing histogram uniformity across videos
4. Creating quality assessment reports
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import os
import sys

# Add the current directory to Python path for imports
sys.path.append('.')

from standardized_preprocessing_pipeline import StandardizedPreprocessingPipeline

class GrayscaleNormalizationValidator:
    def __init__(self, output_dir: str = "grayscale_validation_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "before_after_comparisons").mkdir(exist_ok=True)
        (self.output_dir / "histogram_analysis").mkdir(exist_ok=True)
        (self.output_dir / "sample_frames").mkdir(exist_ok=True)
        (self.output_dir / "processed_videos_new").mkdir(exist_ok=True)
        
        # Initialize pipeline with new normalization
        self.pipeline = StandardizedPreprocessingPipeline(
            output_dir=str(self.output_dir),
            target_frames=32,
            enable_visual_outputs=True
        )
        
        # Test videos from training sample
        self.test_videos = [
            "data/TRAINING SET 2.9.25/doctor 1.mp4",
            "data/TRAINING SET 2.9.25/doctor 5.mp4", 
            "data/TRAINING SET 2.9.25/glasses 1.mp4",
            "data/TRAINING SET 2.9.25/glasses 3.mp4",
            "data/TRAINING SET 2.9.25/help 1.mp4",
            "data/TRAINING SET 2.9.25/help 4.mp4",
            "data/TRAINING SET 2.9.25/phone 1.mp4",
            "data/TRAINING SET 2.9.25/phone 3.mp4",
            "data/TRAINING SET 2.9.25/pillow 1.mp4",
            "data/TRAINING SET 2.9.25/pillow 4.mp4"
        ]
        
        self.results = {}
        
    def analyze_old_videos(self):
        """Analyze the current processed videos (before normalization improvement)."""
        print("📊 Analyzing current processed videos (before normalization)...")
        
        old_video_dir = Path("training_sample_test_output/processed_videos")
        old_stats = {}
        
        for video_file in old_video_dir.glob("*.mp4"):
            video_name = video_file.stem.replace("_processed", "")
            
            cap = cv2.VideoCapture(str(video_file))
            frames_data = []
            
            # Sample frames from the video
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_indices = np.linspace(0, frame_count-1, min(10, frame_count), dtype=int)
            
            for i, frame_idx in enumerate(sample_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert to grayscale for analysis
                    if len(frame.shape) == 3:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = frame
                    
                    frames_data.append({
                        'mean': gray.mean(),
                        'std': gray.std(),
                        'min': gray.min(),
                        'max': gray.max(),
                        'frame': gray
                    })
            
            cap.release()
            
            if frames_data:
                # Calculate video statistics
                means = [f['mean'] for f in frames_data]
                stds = [f['std'] for f in frames_data]
                
                old_stats[video_name] = {
                    'mean_brightness': np.mean(means),
                    'std_brightness': np.std(means),
                    'mean_contrast': np.mean(stds),
                    'std_contrast': np.std(stds),
                    'min_intensity': min(f['min'] for f in frames_data),
                    'max_intensity': max(f['max'] for f in frames_data),
                    'sample_frame': frames_data[len(frames_data)//2]['frame']  # Middle frame
                }
        
        return old_stats
    
    def process_videos_with_new_normalization(self):
        """Process all test videos with the new grayscale normalization."""
        print("🔄 Processing videos with improved grayscale normalization...")
        
        new_stats = {}
        
        for video_path in self.test_videos:
            if not Path(video_path).exists():
                print(f"⚠️  Video not found: {video_path}")
                continue
                
            video_name = Path(video_path).stem
            print(f"Processing {video_name}...")
            
            try:
                # Process video with new pipeline
                results = self.pipeline.process_single_video(video_path, video_name)
                
                if results['status'] == 'success':
                    # Analyze the new processed video
                    new_video_path = self.output_dir / "processed_videos" / f"{video_name}_processed.mp4"
                    
                    if new_video_path.exists():
                        cap = cv2.VideoCapture(str(new_video_path))
                        frames_data = []
                        
                        # Sample frames
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        sample_indices = np.linspace(0, frame_count-1, min(10, frame_count), dtype=int)
                        
                        for frame_idx in sample_indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = cap.read()
                            if ret:
                                # Convert to grayscale for analysis
                                if len(frame.shape) == 3:
                                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                else:
                                    gray = frame
                                
                                frames_data.append({
                                    'mean': gray.mean(),
                                    'std': gray.std(),
                                    'min': gray.min(),
                                    'max': gray.max(),
                                    'frame': gray
                                })
                        
                        cap.release()
                        
                        if frames_data:
                            means = [f['mean'] for f in frames_data]
                            stds = [f['std'] for f in frames_data]
                            
                            new_stats[video_name] = {
                                'mean_brightness': np.mean(means),
                                'std_brightness': np.std(means),
                                'mean_contrast': np.mean(stds),
                                'std_contrast': np.std(stds),
                                'min_intensity': min(f['min'] for f in frames_data),
                                'max_intensity': max(f['max'] for f in frames_data),
                                'sample_frame': frames_data[len(frames_data)//2]['frame']
                            }
                
            except Exception as e:
                print(f"❌ Error processing {video_name}: {e}")
                continue
        
        return new_stats
    
    def generate_before_after_comparisons(self, old_stats, new_stats):
        """Generate before/after comparison images."""
        print("🖼️  Generating before/after comparison images...")
        
        for video_name in old_stats.keys():
            if video_name not in new_stats:
                continue
                
            # Create comparison figure
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Grayscale Normalization Comparison: {video_name}', fontsize=16)
            
            old_frame = old_stats[video_name]['sample_frame']
            new_frame = new_stats[video_name]['sample_frame']
            
            # Before image
            axes[0, 0].imshow(old_frame, cmap='gray', vmin=0, vmax=255)
            axes[0, 0].set_title('Before Normalization')
            axes[0, 0].axis('off')
            
            # After image  
            axes[1, 0].imshow(new_frame, cmap='gray', vmin=0, vmax=255)
            axes[1, 0].set_title('After Normalization')
            axes[1, 0].axis('off')
            
            # Before histogram
            axes[0, 1].hist(old_frame.flatten(), bins=50, alpha=0.7, color='red')
            axes[0, 1].set_title('Before Histogram')
            axes[0, 1].set_xlim(0, 255)
            
            # After histogram
            axes[1, 1].hist(new_frame.flatten(), bins=50, alpha=0.7, color='green')
            axes[1, 1].set_title('After Histogram')
            axes[1, 1].set_xlim(0, 255)
            
            # Statistics comparison
            old_stats_text = f"""Before:
Mean: {old_stats[video_name]['mean_brightness']:.1f}
Std: {old_stats[video_name]['mean_contrast']:.1f}
Range: {old_stats[video_name]['min_intensity']}-{old_stats[video_name]['max_intensity']}"""
            
            new_stats_text = f"""After:
Mean: {new_stats[video_name]['mean_brightness']:.1f}
Std: {new_stats[video_name]['mean_contrast']:.1f}
Range: {new_stats[video_name]['min_intensity']}-{new_stats[video_name]['max_intensity']}"""
            
            axes[0, 2].text(0.1, 0.5, old_stats_text, transform=axes[0, 2].transAxes, 
                           fontsize=12, verticalalignment='center')
            axes[0, 2].set_title('Before Stats')
            axes[0, 2].axis('off')
            
            axes[1, 2].text(0.1, 0.5, new_stats_text, transform=axes[1, 2].transAxes,
                           fontsize=12, verticalalignment='center')
            axes[1, 2].set_title('After Stats')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Save comparison
            comparison_path = self.output_dir / "before_after_comparisons" / f"{video_name}_comparison.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved comparison: {comparison_path}")

    def generate_histogram_uniformity_analysis(self, old_stats, new_stats):
        """Generate comprehensive histogram uniformity analysis."""
        print("📈 Generating histogram uniformity analysis...")

        # Create overall comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Grayscale Normalization: Overall Analysis', fontsize=16)

        # Collect data for analysis
        old_means = [stats['mean_brightness'] for stats in old_stats.values()]
        new_means = [stats['mean_brightness'] for stats in new_stats.values()]
        old_stds = [stats['mean_contrast'] for stats in old_stats.values()]
        new_stds = [stats['mean_contrast'] for stats in new_stats.values()]

        video_names = list(old_stats.keys())

        # Mean brightness comparison
        x = np.arange(len(video_names))
        width = 0.35

        axes[0, 0].bar(x - width/2, old_means, width, label='Before', alpha=0.7, color='red')
        axes[0, 0].bar(x + width/2, new_means, width, label='After', alpha=0.7, color='green')
        axes[0, 0].set_xlabel('Videos')
        axes[0, 0].set_ylabel('Mean Brightness')
        axes[0, 0].set_title('Mean Brightness Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([name[:10] for name in video_names], rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Contrast comparison
        axes[0, 1].bar(x - width/2, old_stds, width, label='Before', alpha=0.7, color='red')
        axes[0, 1].bar(x + width/2, new_stds, width, label='After', alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Videos')
        axes[0, 1].set_ylabel('Mean Contrast (Std Dev)')
        axes[0, 1].set_title('Contrast Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([name[:10] for name in video_names], rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Brightness distribution
        axes[1, 0].hist(old_means, bins=10, alpha=0.7, color='red', label='Before')
        axes[1, 0].hist(new_means, bins=10, alpha=0.7, color='green', label='After')
        axes[1, 0].set_xlabel('Mean Brightness')
        axes[1, 0].set_ylabel('Number of Videos')
        axes[1, 0].set_title('Brightness Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Contrast distribution
        axes[1, 1].hist(old_stds, bins=10, alpha=0.7, color='red', label='Before')
        axes[1, 1].hist(new_stds, bins=10, alpha=0.7, color='green', label='After')
        axes[1, 1].set_xlabel('Mean Contrast')
        axes[1, 1].set_ylabel('Number of Videos')
        axes[1, 1].set_title('Contrast Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save analysis
        analysis_path = self.output_dir / "histogram_analysis" / "overall_uniformity_analysis.png"
        plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved uniformity analysis: {analysis_path}")

        # Calculate and save uniformity metrics
        uniformity_metrics = {
            'before_normalization': {
                'brightness_std': np.std(old_means),
                'brightness_range': max(old_means) - min(old_means),
                'contrast_std': np.std(old_stds),
                'contrast_range': max(old_stds) - min(old_stds)
            },
            'after_normalization': {
                'brightness_std': np.std(new_means),
                'brightness_range': max(new_means) - min(new_means),
                'contrast_std': np.std(new_stds),
                'contrast_range': max(new_stds) - min(new_stds)
            }
        }

        # Calculate improvement percentages
        brightness_std_improvement = (uniformity_metrics['before_normalization']['brightness_std'] -
                                    uniformity_metrics['after_normalization']['brightness_std']) / uniformity_metrics['before_normalization']['brightness_std'] * 100

        contrast_std_improvement = (uniformity_metrics['before_normalization']['contrast_std'] -
                                  uniformity_metrics['after_normalization']['contrast_std']) / uniformity_metrics['before_normalization']['contrast_std'] * 100

        uniformity_metrics['improvements'] = {
            'brightness_uniformity_improvement_percent': brightness_std_improvement,
            'contrast_uniformity_improvement_percent': contrast_std_improvement
        }

        # Save metrics
        metrics_path = self.output_dir / "histogram_analysis" / "uniformity_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(uniformity_metrics, f, indent=2)

        print(f"✅ Saved uniformity metrics: {metrics_path}")
        return uniformity_metrics

    def save_sample_frames(self, new_stats):
        """Save high-quality sample frames demonstrating proper grayscale conversion."""
        print("💾 Saving sample frames with proper grayscale conversion...")

        for video_name, stats in new_stats.items():
            sample_frame = stats['sample_frame']

            # Save as high-quality PNG
            frame_path = self.output_dir / "sample_frames" / f"{video_name}_normalized_sample.png"
            cv2.imwrite(str(frame_path), sample_frame)

            print(f"✅ Saved sample frame: {frame_path}")

    def generate_quality_report(self, old_stats, new_stats, uniformity_metrics):
        """Generate comprehensive quality assessment report."""
        print("📋 Generating quality assessment report...")

        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_videos_analyzed': len(new_stats),
            'summary': {
                'brightness_uniformity_improved': bool(uniformity_metrics['improvements']['brightness_uniformity_improvement_percent'] > 0),
                'contrast_uniformity_improved': bool(uniformity_metrics['improvements']['contrast_uniformity_improvement_percent'] > 0),
                'brightness_improvement_percent': float(uniformity_metrics['improvements']['brightness_uniformity_improvement_percent']),
                'contrast_improvement_percent': float(uniformity_metrics['improvements']['contrast_uniformity_improvement_percent'])
            },
            'detailed_analysis': {}
        }

        # Detailed per-video analysis
        for video_name in old_stats.keys():
            if video_name in new_stats:
                old = old_stats[video_name]
                new = new_stats[video_name]

                brightness_change = new['mean_brightness'] - old['mean_brightness']
                contrast_change = new['mean_contrast'] - old['mean_contrast']

                report['detailed_analysis'][video_name] = {
                    'before': {
                        'mean_brightness': float(old['mean_brightness']),
                        'mean_contrast': float(old['mean_contrast']),
                        'intensity_range': f"{old['min_intensity']}-{old['max_intensity']}"
                    },
                    'after': {
                        'mean_brightness': float(new['mean_brightness']),
                        'mean_contrast': float(new['mean_contrast']),
                        'intensity_range': f"{new['min_intensity']}-{new['max_intensity']}"
                    },
                    'changes': {
                        'brightness_change': float(brightness_change),
                        'contrast_change': float(contrast_change),
                        'quality_improved': bool(abs(new['mean_brightness'] - 128) < abs(old['mean_brightness'] - 128))
                    }
                }

        # Save report
        report_path = self.output_dir / "grayscale_normalization_quality_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ Saved quality report: {report_path}")

        # Generate human-readable summary
        summary_path = self.output_dir / "GRAYSCALE_NORMALIZATION_SUMMARY.md"
        with open(summary_path, 'w') as f:
            f.write("# Grayscale Normalization Validation Summary\n\n")
            f.write(f"**Validation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Videos Analyzed:** {len(new_stats)}\n\n")

            f.write("## Key Improvements\n\n")
            f.write(f"- **Brightness Uniformity:** {uniformity_metrics['improvements']['brightness_uniformity_improvement_percent']:.1f}% improvement\n")
            f.write(f"- **Contrast Uniformity:** {uniformity_metrics['improvements']['contrast_uniformity_improvement_percent']:.1f}% improvement\n\n")

            f.write("## Technical Details\n\n")
            f.write("### Normalization Pipeline:\n")
            f.write("1. **Proper weighted RGB to grayscale conversion** (ITU-R BT.709 standard)\n")
            f.write("2. **CLAHE enhancement** (clipLimit=2.0, tileGridSize=8x8) for consistent contrast\n")
            f.write("3. **Robust percentile normalization** (2nd-98th percentile) to handle outliers\n")
            f.write("4. **Gamma correction** (γ=1.1) for better facial detail visibility\n")
            f.write("5. **Target brightness standardization** (mean ≈ 128) with controlled variance\n\n")

            f.write("### Quality Metrics:\n")
            f.write(f"- **Before normalization brightness std:** {uniformity_metrics['before_normalization']['brightness_std']:.2f}\n")
            f.write(f"- **After normalization brightness std:** {uniformity_metrics['after_normalization']['brightness_std']:.2f}\n")
            f.write(f"- **Before normalization contrast std:** {uniformity_metrics['before_normalization']['contrast_std']:.2f}\n")
            f.write(f"- **After normalization contrast std:** {uniformity_metrics['after_normalization']['contrast_std']:.2f}\n\n")

            f.write("## Files Generated\n\n")
            f.write("- `before_after_comparisons/`: Visual comparisons for each video\n")
            f.write("- `histogram_analysis/`: Uniformity analysis charts and metrics\n")
            f.write("- `sample_frames/`: High-quality normalized sample frames\n")
            f.write("- `processed_videos_new/`: Videos processed with improved normalization\n")

        print(f"✅ Saved summary: {summary_path}")

def main():
    """Main validation function."""
    print("🚀 Starting Grayscale Normalization Validation")
    print("=" * 60)

    validator = GrayscaleNormalizationValidator()

    # Step 1: Analyze old videos
    print("\n📊 Step 1: Analyzing existing videos (before normalization)...")
    old_stats = validator.analyze_old_videos()
    print(f"✅ Analyzed {len(old_stats)} existing videos")

    # Step 2: Process videos with new normalization
    print("\n🔄 Step 2: Processing videos with improved normalization...")
    new_stats = validator.process_videos_with_new_normalization()
    print(f"✅ Processed {len(new_stats)} videos with new normalization")

    # Step 3: Generate before/after comparisons
    print("\n🖼️  Step 3: Generating before/after comparisons...")
    validator.generate_before_after_comparisons(old_stats, new_stats)

    # Step 4: Generate histogram uniformity analysis
    print("\n📈 Step 4: Generating histogram uniformity analysis...")
    uniformity_metrics = validator.generate_histogram_uniformity_analysis(old_stats, new_stats)

    # Step 5: Save sample frames
    print("\n💾 Step 5: Saving sample frames...")
    validator.save_sample_frames(new_stats)

    # Step 6: Generate quality report
    print("\n📋 Step 6: Generating quality assessment report...")
    validator.generate_quality_report(old_stats, new_stats, uniformity_metrics)

    print("\n" + "=" * 60)
    print("✅ GRAYSCALE NORMALIZATION VALIDATION COMPLETED!")
    print("=" * 60)
    print(f"📁 All results saved in: {validator.output_dir}")
    print("\n📊 Key Improvements:")
    print(f"   • Brightness uniformity: {uniformity_metrics['improvements']['brightness_uniformity_improvement_percent']:.1f}% improvement")
    print(f"   • Contrast uniformity: {uniformity_metrics['improvements']['contrast_uniformity_improvement_percent']:.1f}% improvement")
    print("\n📋 Generated Files:")
    print("   • Before/after visual comparisons")
    print("   • Histogram uniformity analysis")
    print("   • High-quality sample frames")
    print("   • Comprehensive quality report")
    print("   • Updated processed videos with normalization")

if __name__ == "__main__":
    main()
