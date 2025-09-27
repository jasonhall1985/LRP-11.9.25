#!/usr/bin/env python3
"""
Create LOSO splits from manifest with validation of no speaker overlap.
"""
import os
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.id_norm import norm_speaker_id, validate_no_speaker_overlap

def load_manifest(manifest_path):
    """Load manifest CSV and group by speaker."""
    videos_by_speaker = defaultdict(list)
    
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker_id = row['speaker_id']  # Already normalized
            videos_by_speaker[speaker_id].append(row)
    
    return videos_by_speaker

def create_loso_splits(videos_by_speaker, output_dir):
    """Create Leave-One-Speaker-Out splits."""
    os.makedirs(output_dir, exist_ok=True)
    
    speakers = sorted(videos_by_speaker.keys())
    print(f"Creating LOSO splits for {len(speakers)} speakers: {speakers}")
    
    splits_info = {}
    
    for held_out_speaker in speakers:
        print(f"\nCreating split for held-out speaker: {held_out_speaker}")
        
        # Split videos
        train_videos = []
        val_videos = videos_by_speaker[held_out_speaker]
        
        for speaker in speakers:
            if speaker != held_out_speaker:
                train_videos.extend(videos_by_speaker[speaker])
        
        # Validate no overlap
        train_speakers = [v['speaker_id'] for v in train_videos]
        val_speakers = [v['speaker_id'] for v in val_videos]
        validate_no_speaker_overlap(train_speakers, val_speakers)
        
        # Write CSV files
        train_csv = os.path.join(output_dir, f"loso_train_holdout_{held_out_speaker.replace(' ', '_')}.csv")
        val_csv = os.path.join(output_dir, f"loso_val_holdout_{held_out_speaker.replace(' ', '_')}.csv")
        
        write_split_csv(train_videos, train_csv)
        write_split_csv(val_videos, val_csv)
        
        # Store split info
        splits_info[held_out_speaker] = {
            'train_csv': train_csv,
            'val_csv': val_csv,
            'train_count': len(train_videos),
            'val_count': len(val_videos),
            'train_speakers': sorted(set(train_speakers)),
            'val_speakers': sorted(set(val_speakers))
        }
        
        print(f"  Train: {len(train_videos)} videos from {len(set(train_speakers))} speakers")
        print(f"  Val: {len(val_videos)} videos from {len(set(val_speakers))} speakers")
    
    # Save splits metadata
    splits_json = os.path.join(output_dir, "loso_splits_info.json")
    with open(splits_json, 'w') as f:
        json.dump(splits_info, f, indent=2)
    
    print(f"\nSplits metadata saved to: {splits_json}")
    return splits_info

def write_split_csv(videos, output_path):
    """Write split to CSV file."""
    with open(output_path, 'w', newline='') as f:
        if videos:
            writer = csv.DictWriter(f, fieldnames=videos[0].keys())
            writer.writeheader()
            writer.writerows(videos)

def main():
    parser = argparse.ArgumentParser(description="Create LOSO splits with validation")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV")
    parser.add_argument("--mode", default="loso", choices=["loso"], help="Split mode")
    parser.add_argument("--outdir", required=True, help="Output directory for splits")
    
    args = parser.parse_args()
    
    # Load manifest
    videos_by_speaker = load_manifest(args.manifest)
    
    # Create splits
    if args.mode == "loso":
        splits_info = create_loso_splits(videos_by_speaker, args.outdir)
        
        # Print summary
        print(f"\n=== LOSO SPLITS SUMMARY ===")
        for speaker, info in splits_info.items():
            print(f"Held-out: {speaker}")
            print(f"  Train speakers: {info['train_speakers']}")
            print(f"  Val speakers: {info['val_speakers']}")
            print(f"  Videos: {info['train_count']} train, {info['val_count']} val")

if __name__ == "__main__":
    main()
