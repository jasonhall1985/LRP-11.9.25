#!/usr/bin/env python3
"""
Fix VAL set size - expand to ≥10-15% of data and ≥10 clips per class
Current: VAL (male 40-64) = 41 videos (2.0%)
Target: VAL ≥200-300 videos (10-15%) with ≥10 per class

Strategy: Add female 40-64 to VAL holdout to expand the validation set
"""

import pandas as pd
import numpy as np
from collections import Counter

def analyze_current_splits():
    """Analyze current demographic splits"""
    df = pd.read_csv('clean_balanced_manifest.csv')
    
    print("=== CURRENT DATASET ANALYSIS ===")
    print(f"Total videos: {len(df)}")
    print(f"Classes: {sorted(df['class'].unique())}")
    print(f"Class distribution: {dict(df['class'].value_counts().sort_index())}")
    
    # Current split logic from config
    val_condition = (df['gender'] == 'male') & (df['age_band'] == '40-64')
    test_condition = (df['gender'] == 'female') & (df['age_band'] == '18-39')
    
    val_df = df[val_condition]
    test_df = df[test_condition]
    train_df = df[~(val_condition | test_condition)]
    
    print(f"\n=== CURRENT SPLITS ===")
    print(f"TRAIN: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"VAL:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"TEST:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    print(f"\nVAL class distribution: {dict(val_df['class'].value_counts().sort_index())}")
    print(f"TEST class distribution: {dict(test_df['class'].value_counts().sort_index())}")
    
    return df, val_df, test_df, train_df

def create_expanded_val_set():
    """Create expanded VAL set by adding female 40-64"""
    df, _, _, _ = analyze_current_splits()
    
    # New expanded VAL: male 40-64 + female 40-64
    val_condition = (df['age_band'] == '40-64')  # Both male and female 40-64
    test_condition = (df['gender'] == 'female') & (df['age_band'] == '18-39')  # Keep TEST same
    
    val_df = df[val_condition]
    test_df = df[test_condition]
    train_df = df[~(val_condition | test_condition)]
    
    print(f"\n=== PROPOSED NEW SPLITS ===")
    print(f"TRAIN: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"VAL:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"TEST:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    print(f"\nNew VAL class distribution: {dict(val_df['class'].value_counts().sort_index())}")
    val_per_class = val_df['class'].value_counts().sort_index()
    min_val_per_class = val_per_class.min()
    print(f"Min clips per class in VAL: {min_val_per_class}")
    
    if len(val_df) >= len(df) * 0.10 and min_val_per_class >= 10:
        print("✅ VAL set meets requirements: ≥10% total data and ≥10 clips per class")
        return True, val_condition
    else:
        print("❌ VAL set still too small, need more demographics")
        return False, None

def update_config_yaml():
    """Update config.yaml with new VAL holdout condition"""
    
    # Read current config
    with open('config.yaml', 'r') as f:
        content = f.read()
    
    # Replace val_holdout condition
    old_val = 'val_holdout: "gender=male,age_band=40-64"'
    new_val = 'val_holdout: "age_band=40-64"'  # Both male and female 40-64
    
    if old_val in content:
        content = content.replace(old_val, new_val)
        
        with open('config.yaml', 'w') as f:
            f.write(content)
        
        print(f"✅ Updated config.yaml: {old_val} → {new_val}")
        return True
    else:
        print("❌ Could not find val_holdout in config.yaml")
        return False

if __name__ == "__main__":
    print("Analyzing and fixing VAL set size...")
    
    # Analyze current
    analyze_current_splits()
    
    # Check if expansion works
    success, val_condition = create_expanded_val_set()
    
    if success:
        # Update config
        if update_config_yaml():
            print("\n🎯 VAL set expansion complete!")
            print("New VAL holdout: age_band=40-64 (both male and female)")
            print("This should give ~10-15% VAL data with ≥10 clips per class")
        else:
            print("\n❌ Failed to update config.yaml")
    else:
        print("\n❌ Need to add more demographics to VAL set")
        print("Consider adding: male 65+ or female 65+ to reach target size")
