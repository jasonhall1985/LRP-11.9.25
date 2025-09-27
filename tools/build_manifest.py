#!/usr/bin/env python3
"""
Build manifest CSV from video directory structure with ID normalization.
"""
import os
import csv
import argparse
from pathlib import Path
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.id_norm import norm_speaker_id, norm_label

def scan_videos(root_dir):
    """
    Scan directory structure and build manifest with normalized IDs.
    Expected structure: root/<speaker>/<class>/*.mp4
    """
    root_path = Path(root_dir)
    videos = []
    
    print(f"Scanning {root_dir}...")
    
    for speaker_dir in root_path.iterdir():
        if not speaker_dir.is_dir():
            continue
            
        speaker_raw = speaker_dir.name
        speaker_norm = norm_speaker_id(speaker_raw)
        
        print(f"Processing speaker: '{speaker_raw}' -> '{speaker_norm}'")
        
        for class_dir in speaker_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_raw = class_dir.name
            class_norm = norm_label(class_raw)
            
            print(f"  Class: '{class_raw}' -> '{class_norm}'")
            
            # Find all video files
            video_files = list(class_dir.glob("*.mp4"))
            print(f"    Found {len(video_files)} videos")
            
            for video_file in video_files:
                videos.append({
                    'video_path': str(video_file.relative_to(root_path)),
                    'speaker_id': speaker_norm,
                    'class_label': class_norm,
                    'speaker_raw': speaker_raw,
                    'class_raw': class_raw
                })
    
    return videos

def write_manifest(videos, output_path):
    """Write manifest to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'video_path', 'speaker_id', 'class_label', 'speaker_raw', 'class_raw'
        ])
        writer.writeheader()
        writer.writerows(videos)
    
    print(f"Manifest written to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Build manifest with normalized IDs")
    parser.add_argument("--root", required=True, help="Root directory containing speaker folders")
    parser.add_argument("--out", required=True, help="Output manifest CSV path")
    
    args = parser.parse_args()
    
    # Scan videos and normalize IDs
    videos = scan_videos(args.root)
    
    # Print summary
    speakers = set(v['speaker_id'] for v in videos)
    classes = set(v['class_label'] for v in videos)
    
    print(f"\nSummary:")
    print(f"  Total videos: {len(videos)}")
    print(f"  Speakers: {len(speakers)} - {sorted(speakers)}")
    print(f"  Classes: {len(classes)} - {sorted(classes)}")
    
    # Write manifest
    write_manifest(videos, args.out)

if __name__ == "__main__":
    main()
