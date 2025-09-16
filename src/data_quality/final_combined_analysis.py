#!/usr/bin/env python3
"""
Final Combined Analysis: Motion + Simple Lip Motion Detection
Integrates overall motion analysis with focused lip region motion analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json

class FinalCombinedAnalyzer:
    """
    Final combined analyzer integrating overall motion and lip-specific motion.
    """
    
    def __init__(self, 
                 motion_reports_dir: str = "motion_analysis_reports",
                 lip_motion_reports_dir: str = "simple_lip_motion_reports",
                 output_dir: str = "final_combined_reports"):
        """
        Initialize final combined analyzer.
        """
        self.motion_reports_dir = Path(motion_reports_dir)
        self.lip_motion_reports_dir = Path(lip_motion_reports_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load analysis results
        self.motion_df = self.load_latest_motion_results()
        self.lip_motion_df = self.load_latest_lip_motion_results()
        self.combined_df = None
        
    def load_latest_motion_results(self) -> pd.DataFrame:
        """Load the most recent overall motion analysis results."""
        try:
            csv_files = list(self.motion_reports_dir.glob("detailed_motion_analysis_*.csv"))
            if csv_files:
                latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                print(f"Loading overall motion analysis from: {latest_csv}")
                return pd.read_csv(latest_csv)
        except Exception as e:
            print(f"Error loading motion analysis: {e}")
        return pd.DataFrame()
    
    def load_latest_lip_motion_results(self) -> pd.DataFrame:
        """Load the most recent lip motion analysis results."""
        try:
            csv_files = list(self.lip_motion_reports_dir.glob("simple_lip_motion_*.csv"))
            if csv_files:
                latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                print(f"Loading lip motion analysis from: {latest_csv}")
                return pd.read_csv(latest_csv)
        except Exception as e:
            print(f"Error loading lip motion analysis: {e}")
        return pd.DataFrame()
    
    def merge_analysis_results(self) -> pd.DataFrame:
        """Merge overall motion and lip motion results."""
        if self.motion_df.empty or self.lip_motion_df.empty:
            print("Warning: Missing motion or lip motion data")
            return pd.DataFrame()
        
        # Merge on filename
        combined = pd.merge(
            self.motion_df, 
            self.lip_motion_df, 
            on='filename', 
            how='inner', 
            suffixes=('_overall', '_lip')
        )
        
        # Create comprehensive quality categories
        combined['quality_tier'] = combined.apply(self.categorize_quality_tier, axis=1)
        
        # Calculate weighted quality score (lip motion more important for lip-reading)
        combined['final_quality_score'] = (
            combined['motion_score'] * 0.2 +  # Overall motion (20%)
            combined['lip_motion_rate'] * 0.8   # Lip-specific motion (80%)
        )
        
        self.combined_df = combined
        print(f"Successfully merged {len(combined)} videos")
        return combined
    
    def categorize_quality_tier(self, row) -> str:
        """Categorize video into quality tiers based on both motion types."""
        overall_motion_good = row['motion_score'] >= 0.03  # Relaxed overall motion
        lip_motion_excellent = row['lip_motion_rate'] >= 0.30  # Excellent lip motion
        lip_motion_good = row['lip_motion_rate'] >= 0.15      # Good lip motion
        lip_motion_moderate = row['lip_motion_rate'] >= 0.05  # Moderate lip motion
        
        if lip_motion_excellent:
            return "tier_1_excellent"
        elif lip_motion_good and overall_motion_good:
            return "tier_2_very_good"
        elif lip_motion_good:
            return "tier_3_good"
        elif lip_motion_moderate:
            return "tier_4_moderate"
        elif overall_motion_good:
            return "tier_5_motion_only"
        else:
            return "tier_6_poor"
    
    def generate_final_quality_analysis(self) -> dict:
        """Generate final quality analysis with actionable insights."""
        if self.combined_df is None:
            return {}
        
        analysis = {}
        
        # Quality tier distribution
        tier_counts = self.combined_df['quality_tier'].value_counts()
        analysis['quality_tiers'] = tier_counts.to_dict()
        
        # Class-wise tier distribution
        class_tiers = self.combined_df.groupby(['class_overall', 'quality_tier']).size().unstack(fill_value=0)
        analysis['class_tiers'] = class_tiers.to_dict()
        
        # Top videos by tier
        for tier in ['tier_1_excellent', 'tier_2_very_good', 'tier_3_good']:
            tier_videos = self.combined_df[self.combined_df['quality_tier'] == tier]
            if len(tier_videos) > 0:
                top_videos = tier_videos.nlargest(10, 'final_quality_score')[
                    ['filename', 'class_overall', 'motion_score', 'lip_motion_rate', 'final_quality_score']
                ]
                analysis[f'{tier}_videos'] = top_videos.to_dict('records')
        
        # Correlation analysis
        correlation = self.combined_df['motion_score'].corr(self.combined_df['lip_motion_rate'])
        analysis['motion_correlation'] = correlation
        
        # Filtering recommendations
        analysis['filtering_recommendations'] = self.generate_filtering_recommendations()
        
        return analysis
    
    def generate_filtering_recommendations(self) -> dict:
        """Generate specific filtering recommendations based on analysis."""
        recommendations = {}
        
        if self.combined_df is None:
            return recommendations
        
        # Count videos in each tier
        tier_counts = self.combined_df['quality_tier'].value_counts()
        total_videos = len(self.combined_df)
        
        # Tier 1: Excellent (lip motion ≥30%)
        tier1_count = tier_counts.get('tier_1_excellent', 0)
        tier1_rate = tier1_count / total_videos * 100
        
        # Tier 2: Very Good (lip motion ≥15% + overall motion ≥3%)
        tier2_count = tier_counts.get('tier_2_very_good', 0)
        tier2_rate = tier2_count / total_videos * 100
        
        # Tier 3: Good (lip motion ≥15%)
        tier3_count = tier_counts.get('tier_3_good', 0)
        tier3_rate = tier3_count / total_videos * 100
        
        # Combined top tiers
        top_tiers_count = tier1_count + tier2_count + tier3_count
        top_tiers_rate = top_tiers_count / total_videos * 100
        
        recommendations = {
            'conservative_filtering': {
                'description': 'Keep only Tier 1 (Excellent lip motion)',
                'criteria': 'lip_motion_rate >= 0.30',
                'videos_retained': tier1_count,
                'retention_rate': f"{tier1_rate:.1f}%",
                'pros': 'Highest quality videos with clear lip movement',
                'cons': 'Very small dataset, may lack diversity'
            },
            'balanced_filtering': {
                'description': 'Keep Tier 1-3 (Good+ lip motion)',
                'criteria': 'lip_motion_rate >= 0.15',
                'videos_retained': top_tiers_count,
                'retention_rate': f"{top_tiers_rate:.1f}%",
                'pros': 'Good balance of quality and quantity',
                'cons': 'Still relatively small dataset'
            },
            'inclusive_filtering': {
                'description': 'Keep Tier 1-4 (Moderate+ lip motion)',
                'criteria': 'lip_motion_rate >= 0.05',
                'videos_retained': tier_counts.get('tier_4_moderate', 0) + top_tiers_count,
                'retention_rate': f"{(tier_counts.get('tier_4_moderate', 0) + top_tiers_count) / total_videos * 100:.1f}%",
                'pros': 'Larger dataset for training',
                'cons': 'Includes videos with minimal lip movement'
            }
        }
        
        return recommendations
    
    def generate_final_visualizations(self) -> None:
        """Generate final comprehensive visualizations."""
        if self.combined_df is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Final Combined Analysis: Overall Motion + Lip Motion', fontsize=16)
        
        # 1. Overall vs Lip Motion Scatter
        scatter = axes[0, 0].scatter(
            self.combined_df['motion_score'], 
            self.combined_df['lip_motion_rate'],
            c=self.combined_df['final_quality_score'],
            cmap='viridis',
            alpha=0.7
        )
        axes[0, 0].set_xlabel('Overall Motion Score')
        axes[0, 0].set_ylabel('Lip Motion Rate')
        axes[0, 0].set_title('Overall vs Lip Motion')
        axes[0, 0].axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Lip Motion Threshold')
        axes[0, 0].axvline(x=0.03, color='orange', linestyle='--', alpha=0.5, label='Overall Motion Threshold')
        axes[0, 0].legend()
        plt.colorbar(scatter, ax=axes[0, 0], label='Final Quality Score')
        
        # 2. Quality Tier Distribution
        tier_counts = self.combined_df['quality_tier'].value_counts()
        tier_labels = [t.replace('tier_', 'T').replace('_', ' ').title() for t in tier_counts.index]
        colors = ['#2E8B57', '#4682B4', '#32CD32', '#FFD700', '#FF6347', '#8B0000']
        axes[0, 1].pie(tier_counts.values, labels=tier_labels, autopct='%1.1f%%', colors=colors[:len(tier_counts)])
        axes[0, 1].set_title('Quality Tier Distribution')
        
        # 3. Class-wise Quality Distribution
        class_quality = pd.crosstab(self.combined_df['class_overall'], self.combined_df['quality_tier'])
        class_quality.plot(kind='bar', stacked=True, ax=axes[0, 2], color=colors)
        axes[0, 2].set_title('Quality Tiers by Class')
        axes[0, 2].set_xlabel('Class')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].legend(title='Quality Tier', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Final Quality Score Distribution
        axes[1, 0].hist(self.combined_df['final_quality_score'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1, 0].set_xlabel('Final Quality Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Final Quality Score Distribution')
        
        # 5. Lip Motion Rate by Class
        classes = self.combined_df['class_overall'].unique()
        lip_motion_by_class = [self.combined_df[self.combined_df['class_overall'] == cls]['lip_motion_rate'] for cls in classes]
        box_plot = axes[1, 1].boxplot(lip_motion_by_class, labels=classes, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        axes[1, 1].set_title('Lip Motion Rate by Class')
        axes[1, 1].set_ylabel('Lip Motion Rate')
        axes[1, 1].axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='Good Threshold')
        axes[1, 1].axhline(y=0.30, color='green', linestyle='--', alpha=0.7, label='Excellent Threshold')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Filtering Impact Analysis
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        retention_rates = []
        for thresh in thresholds:
            retained = len(self.combined_df[self.combined_df['lip_motion_rate'] >= thresh])
            retention_rates.append(retained / len(self.combined_df) * 100)
        
        axes[1, 2].plot(thresholds, retention_rates, 'o-', linewidth=2, markersize=8, color='darkblue')
        axes[1, 2].set_xlabel('Lip Motion Threshold')
        axes[1, 2].set_ylabel('Retention Rate (%)')
        axes[1, 2].set_title('Filtering Impact Analysis')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axvline(x=0.15, color='red', linestyle='--', alpha=0.7, label='Recommended Threshold')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"final_combined_analysis_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Final visualizations saved to: {plot_path}")
    
    def generate_final_report(self) -> None:
        """Generate final comprehensive report."""
        if self.combined_df is None:
            print("No combined data available for final report generation")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate analysis
        analysis = self.generate_final_quality_analysis()
        
        # Save detailed CSV
        csv_path = self.output_dir / f"final_combined_analysis_{timestamp}.csv"
        self.combined_df.to_csv(csv_path, index=False)
        
        # Generate final report
        report_path = self.output_dir / f"final_combined_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("FINAL COMBINED ANALYSIS REPORT\n")
            f.write("OVERALL MOTION + LIP-SPECIFIC MOTION ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("⚠️  NO FILES WERE MODIFIED - ANALYSIS ONLY\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY:\n")
            f.write("=" * 50 + "\n")
            f.write("This analysis combines overall frame motion detection with focused\n")
            f.write("lip region motion detection to identify the highest quality videos\n")
            f.write("for lip-reading model training.\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total videos analyzed: {len(self.combined_df)}\n")
            f.write(f"Overall motion analysis coverage: {len(self.motion_df)} videos\n")
            f.write(f"Lip motion analysis coverage: {len(self.lip_motion_df)} videos\n")
            f.write(f"Combined analysis coverage: {len(self.combined_df)} videos\n\n")
            
            # Quality tier distribution
            f.write("QUALITY TIER DISTRIBUTION:\n")
            f.write("=" * 50 + "\n")
            tier_descriptions = {
                'tier_1_excellent': 'Tier 1 - Excellent (Lip motion ≥30%)',
                'tier_2_very_good': 'Tier 2 - Very Good (Lip ≥15% + Overall ≥3%)',
                'tier_3_good': 'Tier 3 - Good (Lip motion ≥15%)',
                'tier_4_moderate': 'Tier 4 - Moderate (Lip motion 5-15%)',
                'tier_5_motion_only': 'Tier 5 - Motion Only (Overall ≥3%, Lip <5%)',
                'tier_6_poor': 'Tier 6 - Poor (Low motion overall)'
            }
            
            for tier, count in analysis['quality_tiers'].items():
                percentage = count / len(self.combined_df) * 100
                description = tier_descriptions.get(tier, tier)
                f.write(f"{description}: {count} videos ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Class-wise analysis
            f.write("CLASS-WISE QUALITY ANALYSIS:\n")
            f.write("=" * 50 + "\n")
            for class_name in ['doctor', 'glasses', 'phone', 'pillow', 'help']:
                class_data = self.combined_df[self.combined_df['class_overall'] == class_name]
                if len(class_data) > 0:
                    f.write(f"\n{class_name.upper()}:\n")
                    f.write(f"  Total videos: {len(class_data)}\n")
                    
                    # Tier breakdown
                    class_tiers = class_data['quality_tier'].value_counts()
                    tier1_count = class_tiers.get('tier_1_excellent', 0)
                    tier2_count = class_tiers.get('tier_2_very_good', 0)
                    tier3_count = class_tiers.get('tier_3_good', 0)
                    top_tiers = tier1_count + tier2_count + tier3_count
                    
                    f.write(f"  Tier 1 (Excellent): {tier1_count}\n")
                    f.write(f"  Tier 2 (Very Good): {tier2_count}\n")
                    f.write(f"  Tier 3 (Good): {tier3_count}\n")
                    f.write(f"  Top 3 Tiers Total: {top_tiers} ({top_tiers/len(class_data)*100:.1f}%)\n")
                    
                    # Statistics
                    f.write(f"  Mean overall motion: {class_data['motion_score'].mean():.4f}\n")
                    f.write(f"  Mean lip motion: {class_data['lip_motion_rate'].mean():.4f}\n")
                    f.write(f"  Mean final quality: {class_data['final_quality_score'].mean():.4f}\n")
            
            # Filtering recommendations
            f.write(f"\nFILTERING RECOMMENDATIONS:\n")
            f.write("=" * 50 + "\n")
            
            for strategy, details in analysis['filtering_recommendations'].items():
                f.write(f"\n{strategy.replace('_', ' ').upper()}:\n")
                f.write(f"  Description: {details['description']}\n")
                f.write(f"  Criteria: {details['criteria']}\n")
                f.write(f"  Videos retained: {details['videos_retained']}\n")
                f.write(f"  Retention rate: {details['retention_rate']}\n")
                f.write(f"  Pros: {details['pros']}\n")
                f.write(f"  Cons: {details['cons']}\n")
            
            # Top quality videos
            f.write(f"\nTOP QUALITY VIDEOS BY TIER:\n")
            f.write("=" * 50 + "\n")
            
            for tier in ['tier_1_excellent', 'tier_2_very_good', 'tier_3_good']:
                if f'{tier}_videos' in analysis:
                    tier_name = tier.replace('tier_', 'Tier ').replace('_', ' ').title()
                    f.write(f"\n{tier_name}:\n")
                    for i, video in enumerate(analysis[f'{tier}_videos'][:5], 1):  # Top 5 per tier
                        f.write(f"  {i}. {video['filename']}\n")
                        f.write(f"     Class: {video['class_overall']}\n")
                        f.write(f"     Overall Motion: {video['motion_score']:.4f}\n")
                        f.write(f"     Lip Motion: {video['lip_motion_rate']:.4f}\n")
                        f.write(f"     Final Quality: {video['final_quality_score']:.4f}\n\n")
            
            # Final recommendations
            f.write("FINAL RECOMMENDATIONS:\n")
            f.write("=" * 50 + "\n")
            f.write("Based on the comprehensive analysis:\n\n")
            
            tier1_count = analysis['quality_tiers'].get('tier_1_excellent', 0)
            top3_count = sum(analysis['quality_tiers'].get(t, 0) for t in ['tier_1_excellent', 'tier_2_very_good', 'tier_3_good'])
            
            if tier1_count >= 20:
                f.write("• RECOMMENDED: Conservative filtering (Tier 1 only)\n")
                f.write("  - Sufficient high-quality videos available\n")
                f.write("  - Focus on videos with excellent lip movement\n")
            elif top3_count >= 50:
                f.write("• RECOMMENDED: Balanced filtering (Tiers 1-3)\n")
                f.write("  - Good balance of quality and quantity\n")
                f.write("  - Includes all videos with good lip movement\n")
            else:
                f.write("• RECOMMENDED: Inclusive filtering (Tiers 1-4)\n")
                f.write("  - Dataset has limited high-quality videos\n")
                f.write("  - Include moderate quality to ensure sufficient training data\n")
            
            f.write(f"\n• Lip motion is more predictive than overall motion for lip-reading\n")
            f.write(f"• Focus filtering criteria on lip_motion_rate >= 0.15\n")
            f.write(f"• Consider manual review of borderline cases\n")
            f.write(f"• Correlation between overall and lip motion: {analysis['motion_correlation']:.3f}\n")
        
        print(f"Final comprehensive report saved to: {report_path}")
        
        # Save configuration (convert numpy types to native Python types)
        config_path = self.output_dir / f"final_analysis_config_{timestamp}.json"
        config = {
            'timestamp': timestamp,
            'total_videos': int(len(self.combined_df)),
            'quality_tiers': {k: int(v) for k, v in analysis['quality_tiers'].items()},
            'filtering_recommendations': {
                k: {
                    'description': v['description'],
                    'criteria': v['criteria'],
                    'videos_retained': int(v['videos_retained']),
                    'retention_rate': v['retention_rate'],
                    'pros': v['pros'],
                    'cons': v['cons']
                } for k, v in analysis['filtering_recommendations'].items()
            },
            'motion_correlation': float(analysis['motion_correlation']),
            'analysis_only': True,
            'no_files_modified': True
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Final analysis configuration saved to: {config_path}")
    
    def run_final_analysis(self) -> dict:
        """Run the complete final combined analysis pipeline."""
        print("Starting Final Combined Analysis: Overall Motion + Lip Motion")
        
        # Merge analysis results
        combined_df = self.merge_analysis_results()
        
        if combined_df.empty:
            print("Error: No combined data available")
            return {}
        
        # Generate comprehensive reports and visualizations
        self.generate_final_report()
        self.generate_final_visualizations()
        
        # Generate summary statistics
        analysis = self.generate_final_quality_analysis()
        
        print("Final Combined Analysis completed successfully!")
        print(f"Quality tier distribution: {analysis['quality_tiers']}")
        print("⚠️  NO FILES WERE MODIFIED - ALL ORIGINAL DATA PRESERVED")
        
        return analysis


def main():
    """Main execution function for Final Combined Analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Final Combined Motion Analysis')
    parser.add_argument('--motion_dir', type=str, default='motion_analysis_reports',
                       help='Directory containing overall motion analysis reports')
    parser.add_argument('--lip_motion_dir', type=str, default='simple_lip_motion_reports',
                       help='Directory containing lip motion analysis reports')
    parser.add_argument('--output_dir', type=str, default='final_combined_reports',
                       help='Directory for final combined analysis reports')
    
    args = parser.parse_args()
    
    # Initialize and run final analyzer
    analyzer = FinalCombinedAnalyzer(
        motion_reports_dir=args.motion_dir,
        lip_motion_reports_dir=args.lip_motion_dir,
        output_dir=args.output_dir
    )
    
    # Run the final analysis
    results = analyzer.run_final_analysis()
    
    # Print final summary
    if results:
        print("\n" + "="*80)
        print("FINAL COMBINED ANALYSIS SUMMARY")
        print("="*80)
        print("Quality Tier Distribution:")
        for tier, count in results['quality_tiers'].items():
            tier_name = tier.replace('tier_', 'Tier ').replace('_', ' ').title()
            print(f"  {tier_name}: {count} videos")
        
        print(f"\nFiltering Recommendations:")
        for strategy, details in results['filtering_recommendations'].items():
            print(f"  {strategy.replace('_', ' ').title()}: {details['videos_retained']} videos ({details['retention_rate']})")
        
        print(f"\nDetailed reports saved to: {args.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
