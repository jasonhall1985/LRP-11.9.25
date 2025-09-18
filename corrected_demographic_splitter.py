#!/usr/bin/env python3
"""
Corrected Demographic Dataset Splitter
======================================
Create splits with corrected constraint: ONLY male 18-39 must be in training.
Female 18-39 can be distributed across all splits for better balance.

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

class CorrectedDemographicSplitter:
    """Corrected splitter with proper male-only constraint."""
    
    def __init__(self, original_dir: str, augmented_dir: str, output_dir: str = "corrected_dataset_splits"):
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
        
    def extract_demographics_from_filename(self, filename: str) -> dict:
        """Extract demographic information from filename."""
        demographics = {
            'age_group': 'unknown',
            'gender': 'unknown', 
            'ethnicity': 'unknown',
            'format_type': 'unknown'
        }
        
        if '__' in filename:
            parts = filename.split('__')
            if len(parts) >= 5:
                demographics['age_group'] = parts[2]
                demographics['gender'] = parts[3]
                demographics['ethnicity'] = parts[4].split('_')[0]
                demographics['format_type'] = 'structured'
                return demographics
        
        if re.match(r'^[a-z_]+\s+\d+_processed', filename):
            demographics['format_type'] = 'numbered'
            return demographics
        
        return demographics
    
    def extract_class_from_filename(self, filename: str) -> str:
        """Extract class name from filename."""
        classes = ['doctor', 'glasses', 'help', 'phone', 'pillow', 'i_need_to_move', 'my_mouth_is_dry']
        
        for class_name in classes:
            if filename.startswith(class_name):
                return class_name
        
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
        """Analyze the complete dataset."""
        print("📊 ANALYZING DATASET FOR CORRECTED SPLITTING")
        print("=" * 70)
        
        # Process all videos
        for video_dir, video_type in [(self.original_dir, 'original'), (self.augmented_dir, 'augmented')]:
            for video_file in video_dir.glob("*.mp4"):
                if video_file.is_file():
                    # For augmented videos, extract original pattern
                    if video_type == 'augmented':
                        original_pattern = video_file.name.replace('_augmented_', '_TEMP_').split('_TEMP_')[0]
                        demographics = self.extract_demographics_from_filename(original_pattern)
                    else:
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
                        'video_type': video_type,
                        'demographic_key': self.create_demographic_key(demographics)
                    }
                    
                    self.videos_data.append(video_info)
                    self.demographic_groups[video_info['demographic_key']].append(video_info)
        
        print(f"✅ Total videos processed: {len(self.videos_data)}")
        return len(self.videos_data)
    
    def create_corrected_splits(self):
        """Create splits with corrected constraint: only male 18-39 in training."""
        print(f"\n🎯 CREATING CORRECTED DEMOGRAPHIC SPLITS")
        print("=" * 70)
        print("🚨 Constraint: ONLY male 18-39 must be in training set")
        print("✅ Female 18-39 can be in any split for better balance")
        
        # Step 1: Separate male 18-39 demographics (must be in training)
        male_18_39_demos = []
        other_demos = []
        
        for demo_key in self.demographic_groups.keys():
            if demo_key.startswith('male_18-39') or demo_key.startswith('male_18to39'):
                male_18_39_demos.append(demo_key)
            else:
                other_demos.append(demo_key)
        
        print(f"\n🔒 FORCED TO TRAINING (Male 18-39 only):")
        male_18_39_count = 0
        for demo in male_18_39_demos:
            count = len(self.demographic_groups[demo])
            male_18_39_count += count
            print(f"   - {demo}: {count} videos")
        
        print(f"\n📊 Split Calculations:")
        total_videos = len(self.videos_data)
        remaining_videos = total_videos - male_18_39_count
        
        target_train_total = int(total_videos * self.target_train_ratio)
        target_val_total = int(total_videos * self.target_val_ratio)
        target_test_total = total_videos - target_train_total - target_val_total
        
        remaining_train_needed = max(0, target_train_total - male_18_39_count)
        
        print(f"   Total videos: {total_videos}")
        print(f"   Male 18-39 videos (forced to train): {male_18_39_count}")
        print(f"   Remaining videos to distribute: {remaining_videos}")
        print(f"   Additional train needed: {remaining_train_needed}")
        print(f"   Target validation: {target_val_total}")
        print(f"   Target test: {target_test_total}")
        
        # Step 2: Distribute remaining demographics
        random.shuffle(other_demos)
        
        # Calculate sizes and sort by size for better distribution
        demo_sizes = [(demo, len(self.demographic_groups[demo])) for demo in other_demos]
        demo_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize splits
        train_demographics = set(male_18_39_demos)  # Start with male 18-39
        val_demographics = set()
        test_demographics = set()
        
        current_train = male_18_39_count
        current_val = 0
        current_test = 0
        
        # Distribute remaining demographics to achieve target ratios
        for demo, size in demo_sizes:
            # Calculate which split needs this demographic most
            train_deficit = max(0, target_train_total - current_train)
            val_deficit = max(0, target_val_total - current_val)
            test_deficit = max(0, target_test_total - current_test)
            
            # Assign to split with highest deficit (proportionally)
            if test_deficit > 0 and (test_deficit >= val_deficit or current_test == 0):
                test_demographics.add(demo)
                current_test += size
                print(f"   → {demo} ({size} videos) → TEST")
            elif val_deficit > 0 and (val_deficit >= train_deficit or current_val == 0):
                val_demographics.add(demo)
                current_val += size
                print(f"   → {demo} ({size} videos) → VALIDATION")
            else:
                train_demographics.add(demo)
                current_train += size
                print(f"   → {demo} ({size} videos) → TRAINING")
        
        splits = {
            'train': train_demographics,
            'validation': val_demographics,
            'test': test_demographics
        }
        
        print(f"\n📊 Final demographic distribution:")
        print(f"   Training: {len(train_demographics)} demographic groups ({current_train} videos)")
        print(f"   Validation: {len(val_demographics)} demographic groups ({current_val} videos)")
        print(f"   Test: {len(test_demographics)} demographic groups ({current_test} videos)")
        
        return splits
    
    def assign_videos_to_splits(self, splits: dict):
        """Assign videos to splits based on demographic groups."""
        print(f"\n📋 ASSIGNING VIDEOS TO CORRECTED SPLITS")
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
    
    def verify_corrected_constraints(self):
        """Verify that corrected constraints are met."""
        print(f"\n🔍 VERIFYING CORRECTED CONSTRAINTS")
        print("=" * 70)
        
        # Check male 18-39 constraint (must be training only)
        male_18_39_violations = []
        female_18_39_distribution = {'train': 0, 'validation': 0, 'test': 0}
        
        for video in self.videos_data:
            gender = video['gender']
            age_group = video['age_group']
            split = video['dataset_split']
            
            # Check male 18-39 constraint
            if gender == 'male' and age_group in ['18-39', '18to39'] and split != 'train':
                male_18_39_violations.append(video)
            
            # Track female 18-39 distribution
            if gender == 'female' and age_group in ['18-39', '18to39']:
                female_18_39_distribution[split] += 1
        
        # Report male constraint
        if male_18_39_violations:
            print("❌ MALE 18-39 CONSTRAINT VIOLATION!")
            print(f"   Found {len(male_18_39_violations)} male 18-39 videos in val/test")
            return False
        else:
            male_18_39_total = sum(1 for v in self.videos_data 
                                 if v['gender'] == 'male' and v['age_group'] in ['18-39', '18to39'])
            print("✅ MALE 18-39 CONSTRAINT SATISFIED!")
            print(f"   All {male_18_39_total} male 18-39 videos are in training set only")
        
        # Report female distribution
        female_total = sum(female_18_39_distribution.values())
        print(f"\n👩 FEMALE 18-39 DISTRIBUTION:")
        print(f"   Total: {female_total} videos")
        for split, count in female_18_39_distribution.items():
            if female_total > 0:
                pct = (count / female_total) * 100
                print(f"   {split.capitalize()}: {count} videos ({pct:.1f}%)")
        
        return True
    
    def create_corrected_manifest(self):
        """Create corrected manifest CSV file."""
        print(f"\n📄 CREATING CORRECTED MANIFEST")
        print("=" * 70)
        
        df = pd.DataFrame(self.videos_data)
        
        column_order = [
            'filename', 'full_path', 'class', 'dataset_split',
            'age_group', 'gender', 'ethnicity', 'demographic_key',
            'video_type', 'format_type'
        ]
        
        df = df[column_order]
        df = df.sort_values(['dataset_split', 'class', 'filename'])
        
        # Save corrected manifest
        manifest_path = self.output_dir / 'corrected_dataset_manifest.csv'
        df.to_csv(manifest_path, index=False)
        
        print(f"✅ Corrected manifest saved: {manifest_path}")
        
        # Create split-specific manifests
        for split in ['train', 'validation', 'test']:
            split_df = df[df['dataset_split'] == split].copy()
            split_manifest_path = self.output_dir / f"corrected_{split}_manifest.csv"
            split_df.to_csv(split_manifest_path, index=False)
            print(f"✅ {split.capitalize()} manifest: {len(split_df)} videos")
        
        return df
    
    def run_corrected_splitting(self):
        """Run the corrected dataset splitting process."""
        print("🎯 CORRECTED DEMOGRAPHIC DATASET SPLITTING")
        print("=" * 70)
        print("🚨 Constraint: ONLY male 18-39 → training set")
        print("✅ Female 18-39 → can be in any split")
        print()
        
        # Analyze dataset
        total_videos = self.analyze_dataset()
        if total_videos == 0:
            print("❌ No videos found!")
            return None
        
        # Create corrected splits
        splits = self.create_corrected_splits()
        
        # Assign videos to splits
        split_counts = self.assign_videos_to_splits(splits)
        
        # Verify constraints
        constraints_satisfied = self.verify_corrected_constraints()
        
        if not constraints_satisfied:
            print("❌ Constraints not satisfied!")
            return None
        
        # Create manifest
        df = self.create_corrected_manifest()
        
        print(f"\n🎯 CORRECTED SPLITTING COMPLETE!")
        print("=" * 70)
        print(f"✅ {total_videos} videos successfully split with corrected constraints")
        print(f"✅ Male 18-39 constraint satisfied")
        print(f"✅ Female 18-39 distributed for better balance")
        
        return {
            'manifest_df': df,
            'split_counts': split_counts
        }

def main():
    """Main execution function."""
    original_dir = "data/the_best_videos_so_far"
    augmented_dir = "data/the_best_videos_so_far/augmented_videos"
    output_dir = "corrected_dataset_splits"
    
    splitter = CorrectedDemographicSplitter(original_dir, augmented_dir, output_dir)
    results = splitter.run_corrected_splitting()
    
    return results

if __name__ == "__main__":
    main()
