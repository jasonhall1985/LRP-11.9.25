#!/usr/bin/env python3
"""
Verify Male 18-39 Constraint
============================
Check that ONLY male 18-39 videos are restricted to training set.
Female 18-39 videos can be in any split.
"""

import pandas as pd
from pathlib import Path

def verify_male_constraint():
    """Verify that only male 18-39 videos are in training set exclusively."""
    print("🔍 VERIFYING MALE 18-39 CONSTRAINT")
    print("=" * 60)
    
    # Load manifest
    manifest_path = Path("dataset_splits/dataset_manifest.csv")
    df = pd.read_csv(manifest_path)
    
    print(f"📄 Loaded manifest: {len(df)} records")
    
    # Check male 18-39 constraint
    print(f"\n🚨 MALE 18-39 CONSTRAINT CHECK:")
    print("-" * 50)
    
    # Filter for male 18-39 videos
    male_18_39_mask = (
        (df['gender'] == 'male') & 
        (df['age_group'].isin(['18-39', '18to39']))
    )
    
    male_18_39_videos = df[male_18_39_mask]
    male_18_39_in_val_test = male_18_39_videos[male_18_39_videos['dataset_split'] != 'train']
    
    print(f"Total male 18-39 videos: {len(male_18_39_videos)}")
    print(f"Male 18-39 in training: {len(male_18_39_videos[male_18_39_videos['dataset_split'] == 'train'])}")
    print(f"Male 18-39 in validation: {len(male_18_39_videos[male_18_39_videos['dataset_split'] == 'validation'])}")
    print(f"Male 18-39 in test: {len(male_18_39_videos[male_18_39_videos['dataset_split'] == 'test'])}")
    
    if len(male_18_39_in_val_test) > 0:
        print(f"\n❌ CONSTRAINT VIOLATION!")
        print(f"Found {len(male_18_39_in_val_test)} male 18-39 videos in val/test:")
        for _, video in male_18_39_in_val_test.iterrows():
            print(f"   - {video['filename']} -> {video['dataset_split']}")
        male_constraint_satisfied = False
    else:
        print(f"\n✅ MALE CONSTRAINT SATISFIED!")
        print(f"All {len(male_18_39_videos)} male 18-39 videos are in training set only")
        male_constraint_satisfied = True
    
    # Check female 18-39 distribution (should be allowed in all splits)
    print(f"\n👩 FEMALE 18-39 DISTRIBUTION CHECK:")
    print("-" * 50)
    
    female_18_39_mask = (
        (df['gender'] == 'female') & 
        (df['age_group'].isin(['18-39', '18to39']))
    )
    
    female_18_39_videos = df[female_18_39_mask]
    
    print(f"Total female 18-39 videos: {len(female_18_39_videos)}")
    print(f"Female 18-39 in training: {len(female_18_39_videos[female_18_39_videos['dataset_split'] == 'train'])}")
    print(f"Female 18-39 in validation: {len(female_18_39_videos[female_18_39_videos['dataset_split'] == 'validation'])}")
    print(f"Female 18-39 in test: {len(female_18_39_videos[female_18_39_videos['dataset_split'] == 'test'])}")
    
    # Show detailed breakdown
    print(f"\n📊 DETAILED 18-39 AGE GROUP BREAKDOWN:")
    print("-" * 60)
    
    age_18_39_videos = df[df['age_group'].isin(['18-39', '18to39'])]
    
    breakdown = age_18_39_videos.groupby(['gender', 'dataset_split']).size().unstack(fill_value=0)
    print(breakdown)
    
    # Show demographic keys for 18-39 age group
    print(f"\n🔑 18-39 DEMOGRAPHIC GROUPS:")
    print("-" * 50)
    
    demo_18_39 = age_18_39_videos.groupby(['demographic_key', 'dataset_split']).size().unstack(fill_value=0)
    print(demo_18_39)
    
    return male_constraint_satisfied

def main():
    """Main verification function."""
    constraint_satisfied = verify_male_constraint()
    
    print(f"\n🎯 CONSTRAINT VERIFICATION RESULT:")
    print("=" * 60)
    
    if constraint_satisfied:
        print("✅ MALE 18-39 CONSTRAINT SATISFIED")
        print("   All male 18-39 videos are exclusively in training set")
        print("   Female 18-39 videos can be in any split (as intended)")
    else:
        print("❌ MALE 18-39 CONSTRAINT VIOLATED")
        print("   Some male 18-39 videos found in validation or test sets")
        print("   Dataset needs to be re-split")
    
    return constraint_satisfied

if __name__ == "__main__":
    main()
