#!/usr/bin/env python3
"""
Sanity checks for the fixed pipeline:
1. Speaker overlap validation
2. Label map consistency validation
"""
import os
import json
import csv
import sys
from collections import defaultdict

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.id_norm import norm_speaker_id, validate_no_speaker_overlap, validate_label_consistency

def check_speaker_overlap(splits_dir):
    """Check for speaker overlap across all LOSO folds."""
    print("🔍 CHECKING SPEAKER OVERLAP...")
    
    splits_file = os.path.join(splits_dir, "loso_splits_info.json")
    with open(splits_file, 'r') as f:
        splits_info = json.load(f)
    
    all_overlaps = []
    
    for held_out_speaker, split_info in splits_info.items():
        train_speakers = split_info['train_speakers']
        val_speakers = split_info['val_speakers']
        
        print(f"\nFold: {held_out_speaker}")
        print(f"  Train speakers: {train_speakers}")
        print(f"  Val speakers: {val_speakers}")
        
        try:
            validate_no_speaker_overlap(train_speakers, val_speakers)
            print(f"  ✅ No overlap detected")
        except ValueError as e:
            print(f"  ❌ OVERLAP DETECTED: {e}")
            all_overlaps.append((held_out_speaker, str(e)))
    
    if all_overlaps:
        print(f"\n❌ SPEAKER OVERLAP ISSUES FOUND:")
        for fold, error in all_overlaps:
            print(f"  {fold}: {error}")
        return False
    else:
        print(f"\n✅ ALL FOLDS PASS: No speaker overlap detected")
        return True

def check_label_consistency(splits_dir, global_label_path="checkpoints/label2idx.json"):
    """Check label consistency across train/val and global label map."""
    print("\n🔍 CHECKING LABEL CONSISTENCY...")
    
    # Load global label map
    with open(global_label_path, 'r') as f:
        global_label_map = json.load(f)
    global_labels = list(global_label_map.keys())
    
    print(f"Global labels: {global_labels}")
    
    splits_file = os.path.join(splits_dir, "loso_splits_info.json")
    with open(splits_file, 'r') as f:
        splits_info = json.load(f)
    
    all_issues = []
    
    for held_out_speaker, split_info in splits_info.items():
        print(f"\nFold: {held_out_speaker}")
        
        # Load train and val CSVs to check labels
        train_labels = set()
        val_labels = set()
        
        # Read train CSV
        with open(split_info['train_csv'], 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_labels.add(row['class_label'])
        
        # Read val CSV
        with open(split_info['val_csv'], 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                val_labels.add(row['class_label'])
        
        print(f"  Train labels: {sorted(train_labels)}")
        print(f"  Val labels: {sorted(val_labels)}")
        
        try:
            validate_label_consistency(train_labels, val_labels, global_labels)
            print(f"  ✅ Labels consistent")
        except ValueError as e:
            print(f"  ❌ LABEL ISSUE: {e}")
            all_issues.append((held_out_speaker, str(e)))
    
    if all_issues:
        print(f"\n❌ LABEL CONSISTENCY ISSUES FOUND:")
        for fold, error in all_issues:
            print(f"  {fold}: {error}")
        return False
    else:
        print(f"\n✅ ALL FOLDS PASS: Label consistency verified")
        return True

def check_head_parameters():
    """Check that SmallFCHead has <100k parameters."""
    print("\n🔍 CHECKING HEAD PARAMETERS...")
    
    try:
        from models.heads.small_fc import SmallFCHead
        
        # Test with typical encoder output size
        head = SmallFCHead(in_ch=256, num_classes=4)
        total_params = sum(p.numel() for p in head.parameters())
        
        print(f"SmallFCHead parameters: {total_params:,}")
        
        if total_params < 100000:
            print(f"✅ HEAD PASSES: {total_params:,} < 100k parameters")
            return True
        else:
            print(f"❌ HEAD FAILS: {total_params:,} >= 100k parameters")
            return False
            
    except Exception as e:
        print(f"❌ HEAD CHECK FAILED: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run sanity checks")
    parser.add_argument("--splits-dir", required=True, help="Directory containing LOSO splits")
    parser.add_argument("--global-labels", default="checkpoints/label2idx.json", help="Global label map file")
    
    args = parser.parse_args()
    
    print("🧪 RUNNING SANITY CHECKS FOR FIXED PIPELINE")
    print("=" * 60)
    
    # Run all checks
    checks = [
        ("Speaker Overlap", lambda: check_speaker_overlap(args.splits_dir)),
        ("Label Consistency", lambda: check_label_consistency(args.splits_dir, args.global_labels)),
        ("Head Parameters", check_head_parameters)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} CHECK FAILED: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🧪 SANITY CHECK SUMMARY:")
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL SANITY CHECKS PASSED! Ready for training.")
    else:
        print("\n⚠️  SOME CHECKS FAILED! Fix issues before training.")
    
    return all_passed

if __name__ == "__main__":
    main()
