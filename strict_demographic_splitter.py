#!/usr/bin/env python3
"""
Strict Demographic Dataset Splitter
===================================
Create dataset splits with ZERO demographic overlap to prevent data leakage.
Each demographic group (age+gender+ethnicity) assigned to ONLY ONE split.

CRITICAL REQUIREMENTS:
- Zero demographic overlap between splits
- 65+ age groups → Training only
- Male 18-39 demographics → Training only
- Complete demographic separation

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

class StrictDemographicSplitter:
    """Strict demographic splitter with zero overlap guarantee."""
    
    def __init__(self, original_dir: str, augmented_dir: str, output_dir: str = "strict_demographic_splits"):
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
        self.demographic_assignments = {}  # Track which split each demographic goes to
        
    def extract_demographics_from_filename(self, filename: str) -> dict:
        """Extract demographic information from filename."""
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
        """Create unique demographic group key (age+gender+ethnicity)."""
        age = demographics['age_group']
        gender = demographics['gender']
        ethnicity = demographics['ethnicity']
        
        # Normalize age groups for consistency
        if age in ['18to39']:
            age = '18-39'
        elif age in ['40to64']:
            age = '40-64'
        elif age in ['65plus']:
            age = '65+'
        
        return f"{gender}_{age}_{ethnicity}"
    
    def analyze_dataset(self):
        """Analyze the complete dataset and group by demographics."""
        print("📊 ANALYZING DATASET FOR STRICT DEMOGRAPHIC SEPARATION")
        print("=" * 80)
        
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
                augmented_count += 1
        
        print(f"✅ Total videos processed: {len(self.videos_data)}")
        print(f"   - Original videos: {original_count}")
        print(f"   - Augmented videos: {augmented_count}")
        print(f"   - Unique demographic groups: {len(self.demographic_groups)}")
        
        return len(self.videos_data)
    
    def print_demographic_analysis(self):
        """Print detailed demographic analysis."""
        print("\n📈 DEMOGRAPHIC GROUP ANALYSIS")
        print("=" * 80)
        
        print("👥 Videos by Demographic Group:")
        print("-" * 60)
        
        # Sort demographic groups by size for better visibility
        sorted_demos = sorted(self.demographic_groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        mandatory_train_count = 0
        for demo_key, videos in sorted_demos:
            count = len(videos)
            
            # Check if this demographic must go to training
            is_mandatory_train = False
            if '65+' in demo_key or 'male_18-39' in demo_key:
                is_mandatory_train = True
                mandatory_train_count += count
            
            status = "🔒 MANDATORY TRAIN" if is_mandatory_train else "📊 DISTRIBUTABLE"
            print(f"{demo_key:<35} | {count:>4} videos | {status}")
        
        print("-" * 60)
        print(f"{'TOTAL':<35} | {len(self.videos_data):>4} videos")
        
        print(f"\n🚨 MANDATORY TRAINING ASSIGNMENTS:")
        print(f"   - 65+ age groups: Must be in training")
        print(f"   - Male 18-39 groups: Must be in training")
        print(f"   - Total mandatory training videos: {mandatory_train_count}")
        print(f"   - Remaining distributable videos: {len(self.videos_data) - mandatory_train_count}")
    
    def assign_demographic_groups_to_splits(self):
        """Assign entire demographic groups to splits with zero overlap."""
        print(f"\n🎯 ASSIGNING DEMOGRAPHIC GROUPS TO SPLITS")
        print("=" * 80)
        print("🚨 ZERO OVERLAP GUARANTEE: Each demographic group → ONE split only")
        
        # Step 1: Mandatory training assignments
        mandatory_train_demos = []
        distributable_demos = []
        
        for demo_key in self.demographic_groups.keys():
            # Check mandatory training conditions
            if '65+' in demo_key or 'male_18-39' in demo_key:
                mandatory_train_demos.append(demo_key)
                self.demographic_assignments[demo_key] = 'train'
            else:
                distributable_demos.append(demo_key)
        
        print(f"\n🔒 MANDATORY TRAINING ASSIGNMENTS:")
        mandatory_train_videos = 0
        for demo in mandatory_train_demos:
            count = len(self.demographic_groups[demo])
            mandatory_train_videos += count
            print(f"   - {demo}: {count} videos → TRAINING")
        
        # Step 2: Calculate remaining distribution needs
        total_videos = len(self.videos_data)
        remaining_videos = total_videos - mandatory_train_videos
        
        target_train_total = int(total_videos * self.target_train_ratio)
        target_val_total = int(total_videos * self.target_val_ratio)
        target_test_total = total_videos - target_train_total - target_val_total
        
        additional_train_needed = max(0, target_train_total - mandatory_train_videos)
        
        print(f"\n📊 DISTRIBUTION CALCULATIONS:")
        print(f"   Total videos: {total_videos}")
        print(f"   Mandatory training videos: {mandatory_train_videos}")
        print(f"   Remaining distributable videos: {remaining_videos}")
        print(f"   Target training total: {target_train_total}")
        print(f"   Additional training needed: {additional_train_needed}")
        print(f"   Target validation: {target_val_total}")
        print(f"   Target test: {target_test_total}")
        
        # Step 3: Distribute remaining demographic groups
        random.shuffle(distributable_demos)
        
        # Sort by size for better distribution
        demo_sizes = [(demo, len(self.demographic_groups[demo])) for demo in distributable_demos]
        demo_sizes.sort(key=lambda x: x[1], reverse=True)
        
        current_train = mandatory_train_videos
        current_val = 0
        current_test = 0
        
        print(f"\n📋 DISTRIBUTING REMAINING DEMOGRAPHIC GROUPS:")
        
        for demo, size in demo_sizes:
            # Calculate deficits for each split
            train_deficit = max(0, target_train_total - current_train)
            val_deficit = max(0, target_val_total - current_val)
            test_deficit = max(0, target_test_total - current_test)
            
            # Assign to split with highest need
            if test_deficit > 0 and (test_deficit >= val_deficit or current_test == 0):
                self.demographic_assignments[demo] = 'test'
                current_test += size
                print(f"   - {demo}: {size} videos → TEST")
            elif val_deficit > 0 and (val_deficit >= train_deficit or current_val == 0):
                self.demographic_assignments[demo] = 'validation'
                current_val += size
                print(f"   - {demo}: {size} videos → VALIDATION")
            else:
                self.demographic_assignments[demo] = 'train'
                current_train += size
                print(f"   - {demo}: {size} videos → TRAINING")
        
        print(f"\n📊 FINAL DEMOGRAPHIC GROUP DISTRIBUTION:")
        print(f"   Training: {len([d for d, s in self.demographic_assignments.items() if s == 'train'])} groups ({current_train} videos)")
        print(f"   Validation: {len([d for d, s in self.demographic_assignments.items() if s == 'validation'])} groups ({current_val} videos)")
        print(f"   Test: {len([d for d, s in self.demographic_assignments.items() if s == 'test'])} groups ({current_test} videos)")
        
        return {
            'train_videos': current_train,
            'val_videos': current_val,
            'test_videos': current_test
        }
    
    def assign_videos_to_splits(self):
        """Assign individual videos based on their demographic group assignments."""
        print(f"\n📋 ASSIGNING VIDEOS BASED ON DEMOGRAPHIC GROUPS")
        print("=" * 80)
        
        # Assign each video to the split determined by its demographic group
        for video in self.videos_data:
            demo_key = video['demographic_key']
            assigned_split = self.demographic_assignments.get(demo_key, 'train')  # Default to train
            video['dataset_split'] = assigned_split
        
        # Count final splits
        split_counts = Counter(video['dataset_split'] for video in self.videos_data)
        total = len(self.videos_data)
        
        print("📊 Final Video Distribution:")
        print("-" * 50)
        for split_name in ['train', 'validation', 'test']:
            count = split_counts[split_name]
            percentage = (count / total) * 100
            print(f"{split_name.upper():<12} | {count:>4} videos ({percentage:>5.1f}%)")
        
        return split_counts
    
    def verify_zero_demographic_overlap(self):
        """Verify that no demographic group appears in multiple splits."""
        print(f"\n🔍 VERIFYING ZERO DEMOGRAPHIC OVERLAP")
        print("=" * 80)
        
        # Check for demographic overlap
        demographic_split_map = defaultdict(set)
        
        for video in self.videos_data:
            demo_key = video['demographic_key']
            split = video['dataset_split']
            demographic_split_map[demo_key].add(split)
        
        # Find violations
        overlap_violations = []
        for demo_key, splits in demographic_split_map.items():
            if len(splits) > 1:
                overlap_violations.append((demo_key, splits))
        
        if overlap_violations:
            print("❌ DEMOGRAPHIC OVERLAP DETECTED!")
            print("   The following demographic groups appear in multiple splits:")
            for demo_key, splits in overlap_violations:
                print(f"   - {demo_key}: {', '.join(splits)}")
            return False
        else:
            print("✅ ZERO DEMOGRAPHIC OVERLAP CONFIRMED!")
            print(f"   All {len(demographic_split_map)} demographic groups assigned to single splits only")
        
        # Verify mandatory assignments
        print(f"\n🚨 VERIFYING MANDATORY ASSIGNMENTS:")
        
        # Check 65+ and male 18-39 constraints
        violations = []
        
        for video in self.videos_data:
            demo_key = video['demographic_key']
            split = video['dataset_split']
            
            # Check 65+ constraint
            if '65+' in demo_key and split != 'train':
                violations.append(f"65+ demographic {demo_key} in {split}")
            
            # Check male 18-39 constraint
            if 'male_18-39' in demo_key and split != 'train':
                violations.append(f"Male 18-39 demographic {demo_key} in {split}")
        
        if violations:
            print("❌ MANDATORY ASSIGNMENT VIOLATIONS:")
            for violation in violations:
                print(f"   - {violation}")
            return False
        else:
            print("✅ ALL MANDATORY ASSIGNMENTS SATISFIED!")
            print("   - All 65+ demographics in training only")
            print("   - All male 18-39 demographics in training only")
        
        return True
    
    def analyze_class_balance(self):
        """Analyze class balance across splits."""
        print(f"\n📊 CLASS BALANCE ANALYSIS")
        print("=" * 80)
        
        # Create class-split crosstab
        class_split_counts = defaultdict(lambda: defaultdict(int))
        for video in self.videos_data:
            class_split_counts[video['class']][video['dataset_split']] += 1
        
        print("Class distribution across splits:")
        print("-" * 70)
        print(f"{'CLASS':<20} | {'TRAIN':<6} | {'VAL':<6} | {'TEST':<6} | {'TOTAL':<6}")
        print("-" * 70)
        
        for class_name in sorted(class_split_counts.keys()):
            if class_name == 'unknown':
                continue
            splits = class_split_counts[class_name]
            total_class = sum(splits.values())
            print(f"{class_name:<20} | {splits['train']:>6} | {splits['validation']:>6} | {splits['test']:>6} | {total_class:>6}")
        
        # Check for classes missing from splits
        missing_classes = []
        for class_name in ['doctor', 'glasses', 'help', 'phone', 'pillow', 'i_need_to_move', 'my_mouth_is_dry']:
            for split in ['train', 'validation', 'test']:
                if class_split_counts[class_name][split] == 0:
                    missing_classes.append(f"{class_name} missing from {split}")
        
        if missing_classes:
            print(f"\n⚠️  CLASS BALANCE WARNINGS:")
            for warning in missing_classes:
                print(f"   - {warning}")
        else:
            print(f"\n✅ All classes represented in all splits")
    
    def create_strict_manifest(self):
        """Create comprehensive manifest with demographic assignments."""
        print(f"\n📄 CREATING STRICT DEMOGRAPHIC MANIFEST")
        print("=" * 80)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.videos_data)
        
        # Add demographic assignment info
        df['demographic_assignment'] = df['demographic_key'].map(self.demographic_assignments)
        
        # Reorder columns
        column_order = [
            'filename', 'full_path', 'class', 'dataset_split',
            'demographic_key', 'demographic_assignment',
            'age_group', 'gender', 'ethnicity', 
            'video_type', 'format_type'
        ]
        
        df = df[column_order]
        df = df.sort_values(['dataset_split', 'demographic_key', 'class', 'filename'])
        
        # Save main manifest
        manifest_path = self.output_dir / 'strict_demographic_manifest.csv'
        df.to_csv(manifest_path, index=False)
        
        print(f"✅ Strict manifest saved: {manifest_path}")
        print(f"   Total records: {len(df)}")
        
        # Create split-specific manifests
        for split in ['train', 'validation', 'test']:
            split_df = df[df['dataset_split'] == split].copy()
            split_manifest_path = self.output_dir / f"strict_{split}_manifest.csv"
            split_df.to_csv(split_manifest_path, index=False)
            print(f"✅ {split.capitalize()} manifest: {len(split_df)} videos")
        
        return df
    
    def create_demographic_assignment_summary(self):
        """Create summary of demographic group assignments."""
        summary_path = self.output_dir / 'demographic_assignments.txt'
        
        with open(summary_path, 'w') as f:
            f.write("STRICT DEMOGRAPHIC GROUP ASSIGNMENTS\n")
            f.write("=" * 50 + "\n\n")
            f.write("ZERO OVERLAP GUARANTEE: Each demographic group assigned to ONE split only\n\n")
            
            # Group assignments by split
            split_assignments = defaultdict(list)
            for demo_key, split in self.demographic_assignments.items():
                video_count = len(self.demographic_groups[demo_key])
                split_assignments[split].append((demo_key, video_count))
            
            for split in ['train', 'validation', 'test']:
                f.write(f"{split.upper()} SPLIT DEMOGRAPHICS:\n")
                f.write("-" * 40 + "\n")
                
                assignments = sorted(split_assignments[split], key=lambda x: x[1], reverse=True)
                total_videos = sum(count for _, count in assignments)
                
                for demo_key, count in assignments:
                    f.write(f"{demo_key:<35} | {count:>4} videos\n")
                
                f.write("-" * 40 + "\n")
                f.write(f"{'TOTAL':<35} | {total_videos:>4} videos\n\n")
        
        print(f"✅ Demographic assignments summary: {summary_path}")
    
    def run_strict_splitting(self):
        """Run the complete strict demographic splitting process."""
        print("🎯 STRICT DEMOGRAPHIC DATASET SPLITTING")
        print("=" * 80)
        print("🚨 ZERO DEMOGRAPHIC OVERLAP GUARANTEE")
        print("🔒 Mandatory: 65+ and male 18-39 → Training only")
        print("📊 Target: ~70% train, ~20% val, ~10% test")
        print()
        
        # Step 1: Analyze dataset
        total_videos = self.analyze_dataset()
        if total_videos == 0:
            print("❌ No videos found!")
            return None
        
        # Step 2: Print demographic analysis
        self.print_demographic_analysis()
        
        # Step 3: Assign demographic groups to splits
        split_distribution = self.assign_demographic_groups_to_splits()
        
        # Step 4: Assign individual videos based on demographic groups
        split_counts = self.assign_videos_to_splits()
        
        # Step 5: Verify zero demographic overlap
        overlap_verified = self.verify_zero_demographic_overlap()
        
        if not overlap_verified:
            print("❌ Demographic overlap detected! Aborting.")
            return None
        
        # Step 6: Analyze class balance
        self.analyze_class_balance()
        
        # Step 7: Create manifests and summaries
        df = self.create_strict_manifest()
        self.create_demographic_assignment_summary()
        
        print(f"\n🎯 STRICT DEMOGRAPHIC SPLITTING COMPLETE!")
        print("=" * 80)
        print(f"✅ {total_videos} videos split with ZERO demographic overlap")
        print(f"✅ All mandatory assignments satisfied")
        print(f"✅ Comprehensive manifests and summaries generated")
        print(f"📁 Output directory: {self.output_dir}")
        
        return {
            'manifest_df': df,
            'split_counts': split_counts,
            'demographic_assignments': self.demographic_assignments
        }

def main():
    """Main execution function."""
    original_dir = "data/the_best_videos_so_far"
    augmented_dir = "data/the_best_videos_so_far/augmented_videos"
    output_dir = "strict_demographic_splits"
    
    # Initialize strict splitter
    splitter = StrictDemographicSplitter(original_dir, augmented_dir, output_dir)
    
    # Run strict splitting process
    results = splitter.run_strict_splitting()
    
    return results

if __name__ == "__main__":
    main()
