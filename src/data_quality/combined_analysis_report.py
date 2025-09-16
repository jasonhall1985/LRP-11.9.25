#!/usr/bin/env python3
"""
Combined Analysis Report Generator for ICU Lip-Reading Dataset
Integrates motion analysis and lip detection results for comprehensive insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json

class CombinedAnalysisReporter:
    """
    Generates combined analysis reports integrating motion and lip detection results.
    """
    
    def __init__(self, 
                 motion_reports_dir: str = "motion_analysis_reports",
                 lip_reports_dir: str = "lip_detection_reports",
                 output_dir: str = "combined_analysis_reports"):
        """
        Initialize combined analysis reporter.
        
        Args:
            motion_reports_dir: Directory containing motion analysis reports
            lip_reports_dir: Directory containing lip detection reports
            output_dir: Directory for combined analysis reports
        """
        self.motion_reports_dir = Path(motion_reports_dir)
        self.lip_reports_dir = Path(lip_reports_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load analysis results
        self.motion_df = self.load_latest_motion_results()
        self.lip_df = self.load_latest_lip_results()
        self.combined_df = None
        
    def load_latest_motion_results(self) -> pd.DataFrame:
        """Load the most recent motion analysis results."""
        try:
            csv_files = list(self.motion_reports_dir.glob("detailed_motion_analysis_*.csv"))
            if csv_files:
                latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                print(f"Loading motion analysis from: {latest_csv}")
                return pd.read_csv(latest_csv)
        except Exception as e:
            print(f"Error loading motion analysis: {e}")
        return pd.DataFrame()
    
    def load_latest_lip_results(self) -> pd.DataFrame:
        """Load the most recent lip detection results."""
        try:
            csv_files = list(self.lip_reports_dir.glob("detailed_lip_detection_*.csv"))
            if csv_files:
                latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                print(f"Loading lip detection from: {latest_csv}")
                return pd.read_csv(latest_csv)
        except Exception as e:
            print(f"Error loading lip detection: {e}")
        return pd.DataFrame()
    
    def merge_analysis_results(self) -> pd.DataFrame:
        """Merge motion and lip detection results."""
        if self.motion_df.empty or self.lip_df.empty:
            print("Warning: Missing motion or lip detection data")
            return pd.DataFrame()
        
        # Merge on filename
        combined = pd.merge(
            self.motion_df, 
            self.lip_df, 
            on='filename', 
            how='inner', 
            suffixes=('_motion', '_lip')
        )
        
        # Create quality categories
        combined['quality_category'] = combined.apply(self.categorize_quality, axis=1)
        
        # Calculate combined quality score
        combined['combined_quality_score'] = (
            combined['motion_score'] * 0.3 + 
            combined['lip_detection_rate'] * 0.7
        )
        
        self.combined_df = combined
        print(f"Successfully merged {len(combined)} videos")
        return combined
    
    def categorize_quality(self, row) -> str:
        """Categorize video quality based on motion and lip detection."""
        motion_good = row['motion_score'] >= 0.05
        lip_good = row['lip_detection_rate'] >= 0.80
        
        if motion_good and lip_good:
            return "excellent"
        elif motion_good or lip_good:
            return "good"
        elif row['motion_score'] >= 0.01 or row['lip_detection_rate'] >= 0.50:
            return "moderate"
        else:
            return "poor"
    
    def generate_quality_distribution_analysis(self) -> dict:
        """Generate quality distribution analysis."""
        if self.combined_df is None:
            return {}
        
        analysis = {}
        
        # Overall quality distribution
        quality_counts = self.combined_df['quality_category'].value_counts()
        analysis['overall_quality'] = quality_counts.to_dict()
        
        # Class-wise quality distribution
        class_quality = self.combined_df.groupby(['class_motion', 'quality_category']).size().unstack(fill_value=0)
        analysis['class_quality'] = class_quality.to_dict()
        
        # Top quality videos
        top_videos = self.combined_df.nlargest(10, 'combined_quality_score')[
            ['filename', 'class_motion', 'motion_score', 'lip_detection_rate', 'combined_quality_score']
        ]
        analysis['top_quality_videos'] = top_videos.to_dict('records')
        
        # Correlation analysis
        correlation = self.combined_df['motion_score'].corr(self.combined_df['lip_detection_rate'])
        analysis['motion_lip_correlation'] = correlation
        
        return analysis
    
    def generate_combined_visualizations(self) -> None:
        """Generate combined analysis visualizations."""
        if self.combined_df is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Combined Motion & Lip Detection Analysis - ICU Dataset', fontsize=16)
        
        # 1. Motion vs Lip Detection Scatter Plot
        scatter = axes[0, 0].scatter(
            self.combined_df['motion_score'], 
            self.combined_df['lip_detection_rate'],
            c=self.combined_df['combined_quality_score'],
            cmap='viridis',
            alpha=0.6
        )
        axes[0, 0].set_xlabel('Motion Score')
        axes[0, 0].set_ylabel('Lip Detection Rate')
        axes[0, 0].set_title('Motion vs Lip Detection')
        plt.colorbar(scatter, ax=axes[0, 0], label='Combined Quality Score')
        
        # 2. Quality Category Distribution
        quality_counts = self.combined_df['quality_category'].value_counts()
        axes[0, 1].pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Overall Quality Distribution')
        
        # 3. Class-wise Quality Distribution
        class_quality = pd.crosstab(self.combined_df['class_motion'], self.combined_df['quality_category'])
        class_quality.plot(kind='bar', stacked=True, ax=axes[0, 2])
        axes[0, 2].set_title('Quality Distribution by Class')
        axes[0, 2].set_xlabel('Class')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].legend(title='Quality')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Combined Quality Score Distribution
        axes[1, 0].hist(self.combined_df['combined_quality_score'], bins=30, alpha=0.7, color='skyblue')
        axes[1, 0].set_xlabel('Combined Quality Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Combined Quality Score Distribution')
        
        # 5. Motion Score by Class
        classes = self.combined_df['class_motion'].unique()
        motion_by_class = [self.combined_df[self.combined_df['class_motion'] == cls]['motion_score'] for cls in classes]
        axes[1, 1].boxplot(motion_by_class, labels=classes)
        axes[1, 1].set_title('Motion Score Distribution by Class')
        axes[1, 1].set_ylabel('Motion Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Lip Detection Rate by Class
        lip_by_class = [self.combined_df[self.combined_df['class_motion'] == cls]['lip_detection_rate'] for cls in classes]
        axes[1, 2].boxplot(lip_by_class, labels=classes)
        axes[1, 2].set_title('Lip Detection Rate by Class')
        axes[1, 2].set_ylabel('Lip Detection Rate')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"combined_analysis_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Combined visualizations saved to: {plot_path}")
    
    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive combined analysis report."""
        if self.combined_df is None:
            print("No combined data available for report generation")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate analysis
        analysis = self.generate_quality_distribution_analysis()
        
        # Save detailed CSV
        csv_path = self.output_dir / f"combined_analysis_{timestamp}.csv"
        self.combined_df.to_csv(csv_path, index=False)
        
        # Generate comprehensive report
        report_path = self.output_dir / f"combined_analysis_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("COMBINED MOTION & LIP DETECTION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("⚠️  NO FILES WERE MODIFIED - ANALYSIS ONLY\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total videos analyzed: {len(self.combined_df)}\n")
            f.write(f"Motion analysis coverage: {len(self.motion_df)} videos\n")
            f.write(f"Lip detection coverage: {len(self.lip_df)} videos\n")
            f.write(f"Combined analysis coverage: {len(self.combined_df)} videos\n\n")
            
            # Quality distribution
            f.write("QUALITY DISTRIBUTION:\n")
            f.write("-" * 50 + "\n")
            for quality, count in analysis['overall_quality'].items():
                percentage = count / len(self.combined_df) * 100
                f.write(f"{quality.capitalize()}: {count} videos ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Class-wise analysis
            f.write("CLASS-WISE QUALITY ANALYSIS:\n")
            f.write("=" * 50 + "\n")
            for class_name in ['doctor', 'glasses', 'phone', 'pillow', 'help']:
                class_data = self.combined_df[self.combined_df['class_motion'] == class_name]
                if len(class_data) > 0:
                    f.write(f"\n{class_name.upper()}:\n")
                    f.write(f"  Total videos: {len(class_data)}\n")
                    
                    # Quality breakdown
                    class_quality = class_data['quality_category'].value_counts()
                    for quality in ['excellent', 'good', 'moderate', 'poor']:
                        count = class_quality.get(quality, 0)
                        percentage = count / len(class_data) * 100
                        f.write(f"  {quality.capitalize()}: {count} ({percentage:.1f}%)\n")
                    
                    # Statistics
                    f.write(f"  Mean motion score: {class_data['motion_score'].mean():.4f}\n")
                    f.write(f"  Mean lip detection rate: {class_data['lip_detection_rate'].mean():.4f}\n")
                    f.write(f"  Mean combined quality: {class_data['combined_quality_score'].mean():.4f}\n")
            
            # Correlation analysis
            f.write(f"\nCORRELATION ANALYSIS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Motion-Lip Detection Correlation: {analysis['motion_lip_correlation']:.3f}\n")
            
            if abs(analysis['motion_lip_correlation']) > 0.3:
                strength = "moderate to strong"
            elif abs(analysis['motion_lip_correlation']) > 0.1:
                strength = "weak to moderate"
            else:
                strength = "very weak"
            f.write(f"Correlation strength: {strength}\n\n")
            
            # Top quality videos
            f.write("TOP 10 QUALITY VIDEOS:\n")
            f.write("-" * 50 + "\n")
            for i, video in enumerate(analysis['top_quality_videos'], 1):
                f.write(f"{i:2d}. {video['filename']}\n")
                f.write(f"    Class: {video['class_motion']}\n")
                f.write(f"    Motion Score: {video['motion_score']:.4f}\n")
                f.write(f"    Lip Detection Rate: {video['lip_detection_rate']:.4f}\n")
                f.write(f"    Combined Quality: {video['combined_quality_score']:.4f}\n\n")
            
            # Recommendations
            f.write("FILTERING RECOMMENDATIONS:\n")
            f.write("=" * 50 + "\n")
            
            excellent_count = analysis['overall_quality'].get('excellent', 0)
            good_count = analysis['overall_quality'].get('good', 0)
            moderate_count = analysis['overall_quality'].get('moderate', 0)
            
            f.write("Based on combined motion and lip detection analysis:\n\n")
            
            if excellent_count > 0:
                f.write(f"• TIER 1 (Excellent): {excellent_count} videos\n")
                f.write("  - High motion AND high lip detection\n")
                f.write("  - Recommended for primary training set\n\n")
            
            if good_count > 0:
                f.write(f"• TIER 2 (Good): {good_count} videos\n")
                f.write("  - Either high motion OR high lip detection\n")
                f.write("  - Suitable for training with careful review\n\n")
            
            if moderate_count > 0:
                f.write(f"• TIER 3 (Moderate): {moderate_count} videos\n")
                f.write("  - Some motion or lip detection present\n")
                f.write("  - Consider for augmentation or validation set\n\n")
            
            # Threshold recommendations
            total_usable = excellent_count + good_count + moderate_count
            f.write("THRESHOLD RECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            
            if total_usable < len(self.combined_df) * 0.1:
                f.write("• Current thresholds are too restrictive\n")
                f.write("• Consider lowering motion threshold to 0.01-0.03\n")
                f.write("• Consider lowering lip detection threshold to 0.50-0.60\n")
            elif total_usable < len(self.combined_df) * 0.3:
                f.write("• Thresholds may be slightly restrictive\n")
                f.write("• Consider moderate threshold adjustments\n")
            else:
                f.write("• Current thresholds appear reasonable\n")
                f.write("• Proceed with current filtering criteria\n")
        
        print(f"Combined analysis report saved to: {report_path}")
        
        # Save analysis configuration
        config_path = self.output_dir / f"combined_analysis_config_{timestamp}.json"
        config = {
            'timestamp': timestamp,
            'motion_videos': len(self.motion_df),
            'lip_videos': len(self.lip_df),
            'combined_videos': len(self.combined_df),
            'quality_distribution': analysis['overall_quality'],
            'motion_lip_correlation': analysis['motion_lip_correlation'],
            'analysis_only': True,
            'no_files_modified': True
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Analysis configuration saved to: {config_path}")
    
    def run_combined_analysis(self) -> dict:
        """Run the complete combined analysis pipeline."""
        print("Starting Combined Motion & Lip Detection Analysis")
        
        # Merge analysis results
        combined_df = self.merge_analysis_results()
        
        if combined_df.empty:
            print("Error: No combined data available")
            return {}
        
        # Generate comprehensive reports and visualizations
        self.generate_comprehensive_report()
        self.generate_combined_visualizations()
        
        # Generate summary statistics
        analysis = self.generate_quality_distribution_analysis()
        
        print("Combined Analysis completed successfully!")
        print(f"Quality distribution: {analysis['overall_quality']}")
        print("⚠️  NO FILES WERE MODIFIED - ALL ORIGINAL DATA PRESERVED")
        
        return analysis


def main():
    """Main execution function for Combined Analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Combined Motion & Lip Detection Analysis')
    parser.add_argument('--motion_dir', type=str, default='motion_analysis_reports',
                       help='Directory containing motion analysis reports')
    parser.add_argument('--lip_dir', type=str, default='lip_detection_reports',
                       help='Directory containing lip detection reports')
    parser.add_argument('--output_dir', type=str, default='combined_analysis_reports',
                       help='Directory for combined analysis reports')
    
    args = parser.parse_args()
    
    # Initialize and run combined analyzer
    analyzer = CombinedAnalysisReporter(
        motion_reports_dir=args.motion_dir,
        lip_reports_dir=args.lip_dir,
        output_dir=args.output_dir
    )
    
    # Run the combined analysis
    results = analyzer.run_combined_analysis()
    
    # Print final summary
    if results:
        print("\n" + "="*80)
        print("COMBINED ANALYSIS SUMMARY")
        print("="*80)
        print("Quality Distribution:")
        for quality, count in results['overall_quality'].items():
            print(f"  {quality.capitalize()}: {count} videos")
        print(f"\nMotion-Lip Correlation: {results['motion_lip_correlation']:.3f}")
        print(f"Detailed reports saved to: {args.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
