#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis for 4-Class Cross-Demographic Training
Phase 1: Analyze ALL demographics and classes in the full dataset
"""

import os
import csv
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd

class ComprehensiveDatasetAnalyzer:
    def __init__(self):
        self.data_dir = Path("data/the_best_videos_so_far")
        self.output_dir = Path("4class_analysis_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Known demographics from previous work
        self.known_training_demographics = {
            '65plus_female_caucasian',
            '18to39_male_not_specified'
        }
        
        print("🔍 COMPREHENSIVE DATASET ANALYSIS FOR 4-CLASS TRAINING")
        print("=" * 80)
        print("📊 Phase 1: Analyzing ALL demographics and classes in full dataset")
        print(f"📁 Data directory: {self.data_dir}")
        
    def analyze_full_dataset_structure(self):
        """Analyze the complete dataset structure from video filenames."""
        print("\n📋 ANALYZING FULL DATASET STRUCTURE")
        print("=" * 60)
        
        # Collect all video files
        video_files = []
        
        # Check main directory
        if self.data_dir.exists():
            for video_file in self.data_dir.glob("*.mp4"):
                video_files.append(video_file)
            
            # Check augmented videos subdirectory
            augmented_dir = self.data_dir / "augmented_videos"
            if augmented_dir.exists():
                for video_file in augmented_dir.glob("*.mp4"):
                    video_files.append(video_file)
        
        print(f"📊 Total video files found: {len(video_files)}")
        
        # Parse video information from filenames
        video_data = []
        demographic_stats = defaultdict(lambda: defaultdict(int))
        class_stats = defaultdict(int)
        
        for video_file in video_files:
            video_info = self.parse_video_filename(video_file)
            if video_info:
                video_data.append(video_info)
                
                # Update statistics
                demographic = video_info['demographic_group']
                class_name = video_info['class']
                demographic_stats[demographic][class_name] += 1
                class_stats[class_name] += 1
        
        print(f"📊 Successfully parsed: {len(video_data)} videos")
        
        # Display comprehensive statistics
        self.display_demographic_analysis(demographic_stats, class_stats)
        
        # Save detailed analysis
        self.save_comprehensive_analysis(video_data, demographic_stats, class_stats)
        
        return video_data, demographic_stats, class_stats
    
    def parse_video_filename(self, video_path):
        """Parse video filename to extract demographic and class information."""
        filename = video_path.name
        
        # Pattern: class__user__age__gender__ethnicity__timestamp_processing.mp4
        # Example: doctor__useruser01__65plus__female__caucasian__20250731T051928_topmid_96x64_processed.mp4
        
        parts = filename.split('__')
        if len(parts) >= 5:
            try:
                class_name = parts[0]
                age_group = parts[2]
                gender = parts[3]
                ethnicity = parts[4].split('__')[0]  # Remove timestamp part
                
                # Create demographic group identifier
                demographic_group = f"{age_group}_{gender}_{ethnicity}"
                
                # Determine if augmented
                is_augmented = 'augmented' in filename
                
                return {
                    'video_path': str(video_path),
                    'class': class_name,
                    'age_group': age_group,
                    'gender': gender,
                    'ethnicity': ethnicity,
                    'demographic_group': demographic_group,
                    'original_or_augmented': 'augmented' if is_augmented else 'original',
                    'filename': filename
                }
            except Exception as e:
                print(f"⚠️  Failed to parse {filename}: {e}")
                return None
        else:
            print(f"⚠️  Unexpected filename format: {filename}")
            return None
    
    def display_demographic_analysis(self, demographic_stats, class_stats):
        """Display comprehensive demographic and class analysis."""
        print(f"\n📊 COMPREHENSIVE DEMOGRAPHIC ANALYSIS")
        print("=" * 60)
        
        print(f"🎯 Total Demographics Found: {len(demographic_stats)}")
        print(f"🎯 Total Classes Found: {len(class_stats)}")
        
        print(f"\n📈 CLASS DISTRIBUTION (Total Videos):")
        sorted_classes = sorted(class_stats.items(), key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_classes:
            print(f"   {class_name}: {count} videos")
        
        print(f"\n📈 DEMOGRAPHIC DISTRIBUTION:")
        sorted_demographics = sorted(demographic_stats.items(), key=lambda x: sum(x[1].values()), reverse=True)
        
        for demographic, class_counts in sorted_demographics:
            total_videos = sum(class_counts.values())
            print(f"\n   📍 {demographic}: {total_videos} videos")
            
            # Show class breakdown for this demographic
            sorted_demo_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_demo_classes:
                percentage = (count / total_videos) * 100
                print(f"      {class_name}: {count} videos ({percentage:.1f}%)")
        
        # Identify potential validation demographics
        print(f"\n🎯 VALIDATION DEMOGRAPHIC CANDIDATES:")
        print("   (Excluding known training demographics)")
        
        validation_candidates = []
        for demographic, class_counts in demographic_stats.items():
            if demographic not in self.known_training_demographics:
                total_videos = sum(class_counts.values())
                num_classes = len(class_counts)
                validation_candidates.append((demographic, total_videos, num_classes, class_counts))
        
        # Sort by total videos and class diversity
        validation_candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        for demographic, total_videos, num_classes, class_counts in validation_candidates:
            print(f"   📍 {demographic}: {total_videos} videos across {num_classes} classes")
            class_list = [f"{cls}({cnt})" for cls, cnt in sorted(class_counts.items())]
            print(f"      Classes: {', '.join(class_list)}")
    
    def save_comprehensive_analysis(self, video_data, demographic_stats, class_stats):
        """Save detailed analysis to files."""
        
        # Save full video inventory
        video_df = pd.DataFrame(video_data)
        video_inventory_path = self.output_dir / "full_video_inventory.csv"
        video_df.to_csv(video_inventory_path, index=False)
        print(f"\n📄 Full video inventory saved: {video_inventory_path}")
        
        # Save demographic summary
        demographic_summary_path = self.output_dir / "demographic_summary.csv"
        demographic_rows = []
        
        for demographic, class_counts in demographic_stats.items():
            total_videos = sum(class_counts.values())
            num_classes = len(class_counts)
            
            row = {
                'demographic_group': demographic,
                'total_videos': total_videos,
                'num_classes': num_classes,
                'is_training_demographic': demographic in self.known_training_demographics
            }
            
            # Add individual class counts
            for class_name in sorted(class_stats.keys()):
                row[f'{class_name}_count'] = class_counts.get(class_name, 0)
            
            demographic_rows.append(row)
        
        demographic_df = pd.DataFrame(demographic_rows)
        demographic_df.to_csv(demographic_summary_path, index=False)
        print(f"📄 Demographic summary saved: {demographic_summary_path}")
        
        # Save class summary
        class_summary_path = self.output_dir / "class_summary.csv"
        class_rows = []
        
        for class_name, total_count in class_stats.items():
            # Count demographics that have this class
            demographics_with_class = sum(1 for demo_classes in demographic_stats.values() 
                                        if class_name in demo_classes)
            
            # Count training demographic representation
            training_count = sum(demographic_stats[demo].get(class_name, 0) 
                               for demo in self.known_training_demographics 
                               if demo in demographic_stats)
            
            # Count validation demographic representation
            validation_count = sum(demographic_stats[demo].get(class_name, 0) 
                                 for demo in demographic_stats 
                                 if demo not in self.known_training_demographics)
            
            class_rows.append({
                'class_name': class_name,
                'total_videos': total_count,
                'demographics_with_class': demographics_with_class,
                'training_demographic_count': training_count,
                'validation_demographic_count': validation_count,
                'training_validation_ratio': training_count / max(validation_count, 1)
            })
        
        class_df = pd.DataFrame(class_rows)
        class_df.to_csv(class_summary_path, index=False)
        print(f"📄 Class summary saved: {class_summary_path}")
    
    def recommend_4class_configuration(self, demographic_stats, class_stats):
        """Recommend optimal 4-class configuration based on analysis."""
        print(f"\n🎯 4-CLASS CONFIGURATION RECOMMENDATION")
        print("=" * 60)
        
        # Analyze class suitability for 4-class training
        class_suitability = []
        
        for class_name, total_count in class_stats.items():
            # Calculate training demographic representation
            training_count = sum(demographic_stats[demo].get(class_name, 0) 
                               for demo in self.known_training_demographics 
                               if demo in demographic_stats)
            
            # Calculate validation demographic representation
            validation_count = sum(demographic_stats[demo].get(class_name, 0) 
                                 for demo in demographic_stats 
                                 if demo not in self.known_training_demographics)
            
            # Calculate suitability score
            # Factors: training count (weight 0.4), validation count (weight 0.3), total count (weight 0.3)
            suitability_score = (training_count * 0.4 + validation_count * 0.3 + total_count * 0.3)
            
            class_suitability.append({
                'class_name': class_name,
                'total_count': total_count,
                'training_count': training_count,
                'validation_count': validation_count,
                'suitability_score': suitability_score
            })
        
        # Sort by suitability score
        class_suitability.sort(key=lambda x: x['suitability_score'], reverse=True)
        
        print("📊 CLASS SUITABILITY RANKING:")
        print("   (Based on training count, validation count, and total representation)")
        
        for i, class_info in enumerate(class_suitability, 1):
            print(f"   {i}. {class_info['class_name']}: "
                  f"Score={class_info['suitability_score']:.1f} "
                  f"(Train:{class_info['training_count']}, "
                  f"Val:{class_info['validation_count']}, "
                  f"Total:{class_info['total_count']})")
        
        # Recommend top 4 classes
        recommended_classes = [item['class_name'] for item in class_suitability[:4]]
        
        print(f"\n✅ RECOMMENDED 4-CLASS CONFIGURATION:")
        print(f"   Classes: {', '.join(recommended_classes)}")
        
        # Calculate expected dataset sizes
        total_training_videos = sum(
            sum(demographic_stats[demo].get(class_name, 0) for class_name in recommended_classes)
            for demo in self.known_training_demographics if demo in demographic_stats
        )
        
        total_validation_videos = sum(
            sum(demographic_stats[demo].get(class_name, 0) for class_name in recommended_classes)
            for demo in demographic_stats if demo not in self.known_training_demographics
        )
        
        print(f"   Expected Training Videos: {total_training_videos}")
        print(f"   Expected Validation Videos: {total_validation_videos}")
        
        return recommended_classes, class_suitability
    
    def run_comprehensive_analysis(self):
        """Execute complete dataset analysis."""
        try:
            # Phase 1: Analyze full dataset structure
            video_data, demographic_stats, class_stats = self.analyze_full_dataset_structure()
            
            # Phase 2: Recommend 4-class configuration
            recommended_classes, class_suitability = self.recommend_4class_configuration(
                demographic_stats, class_stats
            )
            
            print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETED")
            print(f"📊 Total Videos Analyzed: {len(video_data)}")
            print(f"📊 Demographics Found: {len(demographic_stats)}")
            print(f"📊 Classes Found: {len(class_stats)}")
            print(f"🎯 Recommended 4-Class Configuration: {', '.join(recommended_classes)}")
            
            return {
                'video_data': video_data,
                'demographic_stats': demographic_stats,
                'class_stats': class_stats,
                'recommended_classes': recommended_classes,
                'class_suitability': class_suitability
            }
            
        except Exception as e:
            print(f"\n❌ ANALYSIS FAILED: {e}")
            raise

def main():
    """Execute comprehensive dataset analysis."""
    print("🚀 STARTING COMPREHENSIVE DATASET ANALYSIS")
    print("🎯 Goal: Identify optimal 4-class cross-demographic configuration")
    
    analyzer = ComprehensiveDatasetAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("📄 Results saved to 4class_analysis_results/ directory")
    print("🔄 Ready for Phase 2: 4-Class Training Pipeline Implementation")

if __name__ == "__main__":
    main()
