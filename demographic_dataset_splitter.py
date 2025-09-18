#!/usr/bin/env python3
"""
Demographic Dataset Splitter
============================
Create training, validation, and test splits for the balanced lip-reading dataset
with demographic constraints and class balance preservation.

Key Requirements:
- Split by demographics only (not individual videos)
- Male 18-39 demographic MUST be exclusively in training set
- Maintain class balance across splits
- Generate comprehensive manifest CSV

Author: Augment Agent
Date: 2025-09-18
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import re
import random

class DemographicDatasetSplitter:
    """Dataset splitter with demographic constraints."""
    
    def __init__(self, original_dir: str, augmented_dir: str, output_dir: str = "dataset_splits"):
        self.original_dir = Path(original_dir)
        self.augmented_dir = Path(augmented_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Target split ratios
        self.target_train_ratio = 0.70
        self.target_val_ratio = 0.20
        self.target_test_ratio = 0.10
        
        # Dataset storage
        self.videos_data = []
        self.demographic_groups = defaultdict(list)
        self.class_counts = defaultdict(int)
        
    def extract_demographics_from_filename(self, filename: str) -> dict:
        """Extract demographic information from filename."""
        # Initialize with defaults
        demographics = {
            'age_group': 'unknown',
            'gender': 'unknown', 
            'ethnicity': 'unknown',
            'format_type': 'unknown'
        }
        
        # Check for structured filename format
        if '__' in filename:
            parts = filename.split('__')
            if len(parts) >= 5:
                # Format: class__useruser01__age__gender__ethnicity__timestamp_topmid_96x64_processed.mp4
                demographics['age_group'] = parts[2]
                demographics['gender'] = parts[3]
                demographics['ethnicity'] = parts[4].split('_')[0]  # Remove timestamp part
                demographics['format_type'] = 'structured'
                return demographics
        
        # Check for numbered format
        if re.match(r'^[a-z_]+\s+\d+_processed', filename):
            demographics['format_type'] = 'numbered'
            return demographics
        
        return demographics
    
    def extract_class_from_filename(self, filename: str) -> str:
        """Extract class name from filename."""
        filename_lower = filename.lower()
        
        # Direct class name matching
        classes = ['doctor', 'glasses', 'help', 'phone', 'pillow', 'i_need_to_move', 'my_mouth_is_dry']
        
        for class_name in classes:
            if filename.startswith(class_name):
                return class_name
        
        # Try structured filename
        if '__' in filename:
            parts = filename.split('__')
            if len(parts) > 0:
                return parts[0]
        
        return 'unknown'
    
    def create_demographic_key(self, demographics: dict) -> str:
        """Create unique demographic group key."""
        age = demographics['age_group']
        gender = demographics['gender']
        ethnicity = demographics['ethnicity']
        
        # Normalize age groups
        if age in ['18to39', '18-39']:
            age = '18-39'
        elif age in ['40to64', '40-64']:
            age = '40-64'
        elif age in ['65plus', '65+']:
            age = '65+'
        
        return f"{gender}_{age}_{ethnicity}"
    
    def analyze_dataset(self):
        """Analyze the complete dataset and group by demographics."""
        print("📊 ANALYZING DATASET STRUCTURE")
        print("=" * 70)
        
        # Process original videos
        print(f"📁 Processing original videos from: {self.original_dir}")
        original_count = 0
        for video_file in self.original_dir.glob("*.mp4"):
            if video_file.is_file():
                demographics = self.extract_demographics_from_filename(video_file.name)
                class_name = self.extract_class_from_filename(video_file.name)
                
                video_info = {
                    'filename': video_file.name,
                    'full_path': str(video_file),
                    'class': class_name,
                    'age_group': demographics['age_group'],
                    'gender': demographics['gender'],
                    'ethnicity': demographics['ethnicity'],
                    'format_type': demographics['format_type'],
                    'video_type': 'original',
                    'demographic_key': self.create_demographic_key(demographics)
                }
                
                self.videos_data.append(video_info)
                self.demographic_groups[video_info['demographic_key']].append(video_info)
                self.class_counts[class_name] += 1
                original_count += 1
        
        # Process augmented videos
        print(f"📁 Processing augmented videos from: {self.augmented_dir}")
        augmented_count = 0
        for video_file in self.augmented_dir.glob("*.mp4"):
            if video_file.is_file():
                # Extract original filename pattern from augmented name
                original_pattern = video_file.name.replace('_augmented_', '_TEMP_').split('_TEMP_')[0]
                demographics = self.extract_demographics_from_filename(original_pattern)
                class_name = self.extract_class_from_filename(video_file.name)
                
                video_info = {
                    'filename': video_file.name,
                    'full_path': str(video_file),
                    'class': class_name,
                    'age_group': demographics['age_group'],
                    'gender': demographics['gender'],
                    'ethnicity': demographics['ethnicity'],
                    'format_type': demographics['format_type'],
                    'video_type': 'augmented',
                    'demographic_key': self.create_demographic_key(demographics)
                }
                
                self.videos_data.append(video_info)
                self.demographic_groups[video_info['demographic_key']].append(video_info)
                self.class_counts[class_name] += 1
                augmented_count += 1
        
        print(f"✅ Total videos processed: {len(self.videos_data)}")
        print(f"   - Original videos: {original_count}")
        print(f"   - Augmented videos: {augmented_count}")
        print(f"   - Unique demographic groups: {len(self.demographic_groups)}")
        
        return len(self.videos_data)
    
    def print_demographic_analysis(self):
        """Print detailed demographic analysis."""
        print("\n📈 DEMOGRAPHIC ANALYSIS")
        print("=" * 70)
        
        # Count by demographic groups
        print("👥 Videos by Demographic Group:")
        print("-" * 50)
        
        male_18_39_count = 0
        for demo_key, videos in sorted(self.demographic_groups.items()):
            count = len(videos)
            print(f"{demo_key:<30} | {count:>4} videos")
            
            # Track male 18-39 specifically
            if 'male_18-39' in demo_key or 'male_18to39' in demo_key:
                male_18_39_count += count
        
        print("-" * 50)
        print(f"{'TOTAL':<30} | {len(self.videos_data):>4} videos")
        
        # Highlight critical constraint
        print(f"\n🚨 CRITICAL CONSTRAINT:")
        print(f"   Male 18-39 demographic: {male_18_39_count} videos")
        print(f"   These MUST be in training set only!")
        
        # Class distribution
        print(f"\n📊 Class Distribution:")
        print("-" * 30)
        for class_name, count in sorted(self.class_counts.items()):
            print(f"{class_name:<20} | {count:>3} videos")
    
    def create_demographic_splits(self):
        """Create dataset splits respecting demographic constraints."""
        print(f"\n🎯 CREATING DEMOGRAPHIC SPLITS")
        print("=" * 70)
        
        # Initialize splits
        train_demographics = set()
        val_demographics = set()
        test_demographics = set()
        
        # Step 1: Force male 18-39 demographics into training
        male_18_39_demos = []
        other_demos = []
        
        for demo_key in self.demographic_groups.keys():
            if 'male_18-39' in demo_key or 'male_18to39' in demo_key:
                male_18_39_demos.append(demo_key)
                train_demographics.add(demo_key)
            else:
                other_demos.append(demo_key)
        
        print(f"🔒 Forced to TRAINING: {len(male_18_39_demos)} male 18-39 demographic groups")
        for demo in male_18_39_demos:
            print(f"   - {demo}")
        
        # Step 2: Calculate remaining split requirements
        total_videos = len(self.videos_data)
        male_18_39_videos = sum(len(self.demographic_groups[demo]) for demo in male_18_39_demos)
        remaining_videos = total_videos - male_18_39_videos
        
        target_train_total = int(total_videos * self.target_train_ratio)
        target_val_total = int(total_videos * self.target_val_ratio)
        target_test_total = total_videos - target_train_total - target_val_total
        
        remaining_train_needed = max(0, target_train_total - male_18_39_videos)
        
        print(f"\n📊 Split Calculations:")
        print(f"   Total videos: {total_videos}")
        print(f"   Male 18-39 videos (forced to train): {male_18_39_videos}")
        print(f"   Remaining videos to split: {remaining_videos}")
        print(f"   Target train total: {target_train_total}")
        print(f"   Additional train needed: {remaining_train_needed}")
        print(f"   Target validation: {target_val_total}")
        print(f"   Target test: {target_test_total}")
        
        # Step 3: Split remaining demographics
        random.shuffle(other_demos)
        
        # Calculate videos per demographic group
        demo_sizes = [(demo, len(self.demographic_groups[demo])) for demo in other_demos]
        demo_sizes.sort(key=lambda x: x[1], reverse=True)  # Sort by size, largest first
        
        current_train = male_18_39_videos
        current_val = 0
        current_test = 0
        
        for demo, size in demo_sizes:
            # Decide which split this demographic should go to
            if current_val < target_val_total and (current_val + size <= target_val_total * 1.2):
                val_demographics.add(demo)
                current_val += size
            elif current_test < target_test_total and (current_test + size <= target_test_total * 1.2):
                test_demographics.add(demo)
                current_test += size
            else:
                train_demographics.add(demo)
                current_train += size
        
        # Final assignment
        splits = {
            'train': train_demographics,
            'validation': val_demographics,
            'test': test_demographics
        }
        
        return splits
    
    def assign_videos_to_splits(self, splits: dict):
        """Assign videos to splits based on demographic groups."""
        print(f"\n📋 ASSIGNING VIDEOS TO SPLITS")
        print("=" * 70)
        
        # Assign split labels to videos
        for video in self.videos_data:
            demo_key = video['demographic_key']
            
            if demo_key in splits['train']:
                video['dataset_split'] = 'train'
            elif demo_key in splits['validation']:
                video['dataset_split'] = 'validation'
            elif demo_key in splits['test']:
                video['dataset_split'] = 'test'
            else:
                video['dataset_split'] = 'train'  # Default fallback
        
        # Count final splits
        split_counts = Counter(video['dataset_split'] for video in self.videos_data)
        total = len(self.videos_data)
        
        print("📊 Final Split Distribution:")
        print("-" * 40)
        for split_name in ['train', 'validation', 'test']:
            count = split_counts[split_name]
            percentage = (count / total) * 100
            print(f"{split_name.upper():<12} | {count:>4} videos ({percentage:>5.1f}%)")
        
        return split_counts
    
    def verify_constraints(self):
        """Verify that all constraints are met."""
        print(f"\n🔍 VERIFYING CONSTRAINTS")
        print("=" * 70)
        
        # Check male 18-39 constraint
        male_18_39_in_val_test = []
        
        for video in self.videos_data:
            demo_key = video['demographic_key']
            split = video['dataset_split']
            
            if ('male_18-39' in demo_key or 'male_18to39' in demo_key) and split != 'train':
                male_18_39_in_val_test.append(video)
        
        if male_18_39_in_val_test:
            print("❌ CONSTRAINT VIOLATION!")
            print(f"   Found {len(male_18_39_in_val_test)} male 18-39 videos in val/test:")
            for video in male_18_39_in_val_test[:5]:  # Show first 5
                print(f"   - {video['filename']} -> {video['dataset_split']}")
            return False
        else:
            print("✅ CONSTRAINT SATISFIED!")
            print("   All male 18-39 videos are in training set")
        
        # Check class balance
        print(f"\n📊 Class Balance Verification:")
        print("-" * 50)
        
        class_split_counts = defaultdict(lambda: defaultdict(int))
        for video in self.videos_data:
            class_split_counts[video['class']][video['dataset_split']] += 1
        
        for class_name in sorted(class_split_counts.keys()):
            splits = class_split_counts[class_name]
            total_class = sum(splits.values())
            print(f"{class_name:<20} | Train: {splits['train']:>3} | Val: {splits['validation']:>3} | Test: {splits['test']:>3} | Total: {total_class:>3}")
        
        return True
    
    def create_manifest_csv(self):
        """Create comprehensive manifest CSV file."""
        print(f"\n📄 CREATING MANIFEST CSV")
        print("=" * 70)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.videos_data)
        
        # Reorder columns for better readability
        column_order = [
            'filename', 'full_path', 'class', 'dataset_split',
            'age_group', 'gender', 'ethnicity', 'demographic_key',
            'video_type', 'format_type'
        ]
        
        df = df[column_order]
        
        # Sort by split, then class, then filename
        df = df.sort_values(['dataset_split', 'class', 'filename'])
        
        # Save manifest
        manifest_path = self.output_dir / 'dataset_manifest.csv'
        df.to_csv(manifest_path, index=False)
        
        print(f"✅ Manifest saved: {manifest_path}")
        print(f"   Total records: {len(df)}")
        print(f"   Columns: {', '.join(df.columns)}")
        
        return df
    
    def generate_summary_statistics(self, df: pd.DataFrame):
        """Generate comprehensive summary statistics."""
        print(f"\n📈 GENERATING SUMMARY STATISTICS")
        print("=" * 70)
        
        # Create summary statistics
        summary_stats = {
            'total_videos': len(df),
            'split_distribution': df['dataset_split'].value_counts().to_dict(),
            'class_distribution': df['class'].value_counts().to_dict(),
            'video_type_distribution': df['video_type'].value_counts().to_dict(),
            'demographic_distribution': df['demographic_key'].value_counts().to_dict()
        }
        
        # Save detailed statistics
        stats_path = self.output_dir / 'dataset_statistics.txt'
        with open(stats_path, 'w') as f:
            f.write("BALANCED LIP-READING DATASET STATISTICS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Videos: {summary_stats['total_videos']}\n\n")
            
            f.write("SPLIT DISTRIBUTION:\n")
            f.write("-" * 30 + "\n")
            for split, count in summary_stats['split_distribution'].items():
                percentage = (count / summary_stats['total_videos']) * 100
                f.write(f"{split.upper():<12} | {count:>4} videos ({percentage:>5.1f}%)\n")
            
            f.write(f"\nCLASS DISTRIBUTION:\n")
            f.write("-" * 30 + "\n")
            for class_name, count in sorted(summary_stats['class_distribution'].items()):
                f.write(f"{class_name:<20} | {count:>3} videos\n")
            
            f.write(f"\nVIDEO TYPE DISTRIBUTION:\n")
            f.write("-" * 30 + "\n")
            for video_type, count in summary_stats['video_type_distribution'].items():
                f.write(f"{video_type.upper():<12} | {count:>4} videos\n")
            
            f.write(f"\nDEMOGRAPHIC DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            for demo_key, count in sorted(summary_stats['demographic_distribution'].items()):
                f.write(f"{demo_key:<30} | {count:>4} videos\n")
        
        print(f"✅ Statistics saved: {stats_path}")
        
        return summary_stats
    
    def run_complete_splitting(self):
        """Run the complete dataset splitting process."""
        print("🎯 DEMOGRAPHIC DATASET SPLITTING PIPELINE")
        print("=" * 70)
        print(f"📁 Original Directory: {self.original_dir}")
        print(f"📁 Augmented Directory: {self.augmented_dir}")
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"🎯 Target Ratios: {self.target_train_ratio:.0%} train, {self.target_val_ratio:.0%} val, {self.target_test_ratio:.0%} test")
        print(f"🚨 Constraint: Male 18-39 → Training only")
        print()
        
        # Step 1: Analyze dataset
        total_videos = self.analyze_dataset()
        if total_videos == 0:
            print("❌ No videos found!")
            return None
        
        # Step 2: Print demographic analysis
        self.print_demographic_analysis()
        
        # Step 3: Create demographic splits
        splits = self.create_demographic_splits()
        
        # Step 4: Assign videos to splits
        split_counts = self.assign_videos_to_splits(splits)
        
        # Step 5: Verify constraints
        constraints_satisfied = self.verify_constraints()
        
        if not constraints_satisfied:
            print("❌ Constraints not satisfied! Aborting.")
            return None
        
        # Step 6: Create manifest CSV
        df = self.create_manifest_csv()
        
        # Step 7: Generate summary statistics
        summary_stats = self.generate_summary_statistics(df)
        
        print(f"\n🎯 DATASET SPLITTING COMPLETE!")
        print("=" * 70)
        print(f"✅ {total_videos} videos successfully split")
        print(f"✅ All constraints satisfied")
        print(f"✅ Manifest and statistics generated")
        print(f"📁 Output directory: {self.output_dir}")
        
        return {
            'manifest_df': df,
            'summary_stats': summary_stats,
            'split_counts': split_counts
        }

def main():
    """Main execution function."""
    original_dir = "data/the_best_videos_so_far"
    augmented_dir = "data/the_best_videos_so_far/augmented_videos"
    output_dir = "dataset_splits"
    
    # Initialize splitter
    splitter = DemographicDatasetSplitter(original_dir, augmented_dir, output_dir)
    
    # Run complete splitting process
    results = splitter.run_complete_splitting()
    
    return results

if __name__ == "__main__":
    main()
