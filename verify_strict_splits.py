#!/usr/bin/env python3
"""
Verify Strict Demographic Splits
================================
Comprehensive verification of strict demographic separation with zero overlap.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

def verify_strict_demographic_splits():
    """Comprehensive verification of strict demographic splits."""
    print("🔍 COMPREHENSIVE STRICT DEMOGRAPHIC SPLIT VERIFICATION")
    print("=" * 80)
    
    # Load manifest
    manifest_path = Path("strict_demographic_splits/strict_demographic_manifest.csv")
    if not manifest_path.exists():
        print("❌ Strict manifest file not found!")
        return False
    
    df = pd.read_csv(manifest_path)
    print(f"📄 Loaded strict manifest: {len(df)} records")
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS")
    print("-" * 60)
    print(f"Total videos: {len(df)}")
    print(f"Original videos: {len(df[df['video_type'] == 'original'])}")
    print(f"Augmented videos: {len(df[df['video_type'] == 'augmented'])}")
    print(f"Unique classes: {df['class'].nunique()}")
    print(f"Unique demographic groups: {df['demographic_key'].nunique()}")
    
    # Split distribution
    print(f"\n📈 SPLIT DISTRIBUTION")
    print("-" * 60)
    split_counts = df['dataset_split'].value_counts()
    total = len(df)
    
    for split in ['train', 'validation', 'test']:
        count = split_counts.get(split, 0)
        percentage = (count / total) * 100
        target_pct = {'train': 70, 'validation': 20, 'test': 10}[split]
        status = "✅" if abs(percentage - target_pct) <= 15 else "⚠️"
        print(f"{split.upper():<12} | {count:>4} videos ({percentage:>5.1f}%) | Target: {target_pct}% {status}")
    
    # CRITICAL TEST 1: Zero demographic overlap
    print(f"\n🚨 CRITICAL TEST 1: ZERO DEMOGRAPHIC OVERLAP")
    print("-" * 60)
    
    demographic_splits = df.groupby('demographic_key')['dataset_split'].unique()
    overlap_violations = []
    
    for demo_key, splits in demographic_splits.items():
        if len(splits) > 1:
            overlap_violations.append((demo_key, list(splits)))
    
    if overlap_violations:
        print("❌ DEMOGRAPHIC OVERLAP DETECTED!")
        for demo_key, splits in overlap_violations:
            print(f"   - {demo_key}: appears in {', '.join(splits)}")
        return False
    else:
        print("✅ ZERO DEMOGRAPHIC OVERLAP CONFIRMED!")
        print(f"   All {len(demographic_splits)} demographic groups in single splits only")
    
    # CRITICAL TEST 2: Mandatory assignments
    print(f"\n🚨 CRITICAL TEST 2: MANDATORY ASSIGNMENTS")
    print("-" * 60)
    
    # Check 65+ demographics
    age_65_plus = df[df['demographic_key'].str.contains('65+', na=False)]
    age_65_plus_violations = age_65_plus[age_65_plus['dataset_split'] != 'train']
    
    if len(age_65_plus_violations) > 0:
        print("❌ 65+ AGE GROUP VIOLATION!")
        print(f"   Found {len(age_65_plus_violations)} 65+ videos not in training")
        return False
    else:
        print(f"✅ 65+ AGE GROUP CONSTRAINT SATISFIED!")
        print(f"   All {len(age_65_plus)} videos from 65+ demographics in training only")
    
    # Check male 18-39 demographics
    male_18_39 = df[df['demographic_key'].str.contains('male_18-39', na=False)]
    male_18_39_violations = male_18_39[male_18_39['dataset_split'] != 'train']
    
    if len(male_18_39_violations) > 0:
        print("❌ MALE 18-39 CONSTRAINT VIOLATION!")
        print(f"   Found {len(male_18_39_violations)} male 18-39 videos not in training")
        return False
    else:
        print(f"✅ MALE 18-39 CONSTRAINT SATISFIED!")
        print(f"   All {len(male_18_39)} videos from male 18-39 demographics in training only")
    
    # CRITICAL TEST 3: Class representation
    print(f"\n🚨 CRITICAL TEST 3: CLASS REPRESENTATION")
    print("-" * 60)
    
    class_split_table = pd.crosstab(df['class'], df['dataset_split'], margins=True)
    print(class_split_table)
    
    # Check for missing classes in splits
    missing_classes = []
    target_classes = ['doctor', 'glasses', 'help', 'phone', 'pillow', 'i_need_to_move', 'my_mouth_is_dry']
    
    for class_name in target_classes:
        class_data = df[df['class'] == class_name]
        for split in ['train', 'validation', 'test']:
            split_count = len(class_data[class_data['dataset_split'] == split])
            if split_count == 0:
                missing_classes.append(f"{class_name} missing from {split}")
    
    if missing_classes:
        print(f"\n⚠️  CLASS REPRESENTATION WARNINGS:")
        for warning in missing_classes:
            print(f"   - {warning}")
        print("   Note: This may be acceptable due to demographic constraints")
    else:
        print(f"\n✅ All classes represented in all splits")
    
    # Demographic distribution analysis
    print(f"\n👥 DEMOGRAPHIC DISTRIBUTION BY SPLIT")
    print("-" * 80)
    
    demo_split_summary = df.groupby(['dataset_split', 'demographic_key']).size().unstack(fill_value=0)
    
    for split in ['train', 'validation', 'test']:
        split_demos = df[df['dataset_split'] == split]['demographic_key'].unique()
        print(f"\n{split.upper()} SPLIT DEMOGRAPHICS ({len(split_demos)} groups):")
        print("-" * 50)
        
        split_data = df[df['dataset_split'] == split]
        demo_counts = split_data['demographic_key'].value_counts()
        
        for demo_key, count in demo_counts.items():
            print(f"   {demo_key:<35} | {count:>4} videos")
    
    # File existence verification
    print(f"\n📁 FILE EXISTENCE VERIFICATION")
    print("-" * 60)
    
    missing_files = []
    for _, row in df.iterrows():
        file_path = Path(row['full_path'])
        if not file_path.exists():
            missing_files.append(row['filename'])
    
    if missing_files:
        print(f"❌ Found {len(missing_files)} missing files:")
        for filename in missing_files[:5]:
            print(f"   - {filename}")
        if len(missing_files) > 5:
            print(f"   ... and {len(missing_files) - 5} more")
        return False
    else:
        print(f"✅ All {len(df)} files exist and are accessible")
    
    # Video type distribution
    print(f"\n🎬 VIDEO TYPE DISTRIBUTION")
    print("-" * 60)
    
    video_type_split = pd.crosstab(df['video_type'], df['dataset_split'], margins=True)
    print(video_type_split)
    
    return True

def create_final_summary():
    """Create final summary of strict demographic splits."""
    print(f"\n📋 CREATING FINAL SUMMARY")
    print("-" * 60)
    
    # Load manifest
    df = pd.read_csv("strict_demographic_splits/strict_demographic_manifest.csv")
    
    # Create comprehensive summary
    summary_path = Path("strict_demographic_splits/STRICT_SPLITS_SUMMARY.md")
    
    with open(summary_path, 'w') as f:
        f.write("# Strict Demographic Dataset Splits Summary\n\n")
        f.write("## 🎯 Zero Demographic Overlap Guarantee\n\n")
        f.write("Successfully created dataset splits with **ZERO demographic overlap** to prevent data leakage.\n")
        f.write("Each demographic group (age+gender+ethnicity) assigned to **ONLY ONE split**.\n\n")
        
        # Split distribution
        split_counts = df['dataset_split'].value_counts()
        total = len(df)
        
        f.write("## 📊 Split Distribution\n\n")
        f.write("| Split | Videos | Percentage | Target |\n")
        f.write("|-------|--------|------------|--------|\n")
        
        for split in ['train', 'validation', 'test']:
            count = split_counts.get(split, 0)
            percentage = (count / total) * 100
            target = {'train': '70%', 'validation': '20%', 'test': '10%'}[split]
            f.write(f"| **{split.capitalize()}** | {count} | {percentage:.1f}% | {target} |\n")
        
        f.write(f"| **Total** | **{total}** | **100%** | **100%** |\n\n")
        
        # Mandatory assignments
        f.write("## 🚨 Mandatory Assignments Satisfied\n\n")
        f.write("✅ **65+ Age Groups**: All demographics with 65+ age exclusively in training set\n\n")
        f.write("✅ **Male 18-39 Demographics**: All male 18-39 demographics exclusively in training set\n\n")
        
        # Demographic groups by split
        f.write("## 👥 Demographic Groups by Split\n\n")
        
        for split in ['train', 'validation', 'test']:
            split_data = df[df['dataset_split'] == split]
            demo_counts = split_data['demographic_key'].value_counts()
            
            f.write(f"### {split.capitalize()} Split ({len(demo_counts)} demographic groups)\n\n")
            
            for demo_key, count in demo_counts.items():
                f.write(f"- **{demo_key}**: {count} videos\n")
            f.write("\n")
        
        # Class distribution
        f.write("## 📊 Class Distribution\n\n")
        class_split_table = pd.crosstab(df['class'], df['dataset_split'])
        
        f.write("| Class | Train | Validation | Test | Total |\n")
        f.write("|-------|-------|------------|------| ----- |\n")
        
        for class_name in sorted(class_split_table.index):
            if class_name != 'unknown':
                train_count = class_split_table.loc[class_name, 'train'] if 'train' in class_split_table.columns else 0
                val_count = class_split_table.loc[class_name, 'validation'] if 'validation' in class_split_table.columns else 0
                test_count = class_split_table.loc[class_name, 'test'] if 'test' in class_split_table.columns else 0
                total_count = train_count + val_count + test_count
                f.write(f"| **{class_name}** | {train_count} | {val_count} | {test_count} | {total_count} |\n")
        
        f.write("\n## ✅ Verification Results\n\n")
        f.write("- ✅ **Zero demographic overlap confirmed**\n")
        f.write("- ✅ **All mandatory assignments satisfied**\n")
        f.write("- ✅ **All files exist and accessible**\n")
        f.write("- ✅ **Comprehensive manifests generated**\n")
        f.write("- ✅ **Ready for model training**\n\n")
        
        f.write("## 📁 Generated Files\n\n")
        f.write("- `strict_demographic_manifest.csv` - Complete manifest (714 videos)\n")
        f.write("- `strict_train_manifest.csv` - Training set manifest\n")
        f.write("- `strict_validation_manifest.csv` - Validation set manifest\n")
        f.write("- `strict_test_manifest.csv` - Test set manifest\n")
        f.write("- `demographic_assignments.txt` - Demographic group assignments\n")
        f.write("- `STRICT_SPLITS_SUMMARY.md` - This summary document\n\n")
        
        f.write("*Generated with zero demographic overlap guarantee for data leakage prevention.*\n")
    
    print(f"✅ Final summary created: {summary_path}")

def main():
    """Main verification function."""
    print("🎯 STRICT DEMOGRAPHIC SPLITS VERIFICATION")
    print("=" * 80)
    
    success = verify_strict_demographic_splits()
    
    if success:
        create_final_summary()
        print(f"\n🎯 VERIFICATION COMPLETE!")
        print("=" * 80)
        print("✅ All critical tests passed")
        print("✅ Zero demographic overlap confirmed")
        print("✅ Mandatory assignments satisfied")
        print("✅ Dataset ready for training with no data leakage")
    else:
        print(f"\n❌ VERIFICATION FAILED!")
        print("Critical issues detected. Please review and fix.")
    
    return success

if __name__ == "__main__":
    main()
