#!/usr/bin/env python3
"""
Organize Speaker Data for Three-Stage Pipeline
==============================================

Organizes existing video data into the required speaker sets structure for the
three-stage training pipeline (GRID pretraining → ICU fine-tuning → personalization).

This script analyzes the existing data structure and creates the required
`data/speaker sets/<speaker>/<class>/` organization.

Author: Augment Agent
Date: 2025-09-27
"""

import os
import sys
import shutil
import re
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
import argparse

def extract_speaker_info_from_filename(filename: str) -> Dict[str, str]:
    """
    Extract speaker demographic information from filename.
    
    Expected format: class__useruser01__age__gender__ethnicity__timestamp_topmid.ext
    """
    # Remove extension
    name_without_ext = Path(filename).stem
    
    # Split by double underscores
    parts = name_without_ext.split('__')
    
    if len(parts) >= 5:
        return {
            'class': parts[0],
            'user_id': parts[1],
            'age_group': parts[2],
            'gender': parts[3],
            'ethnicity': parts[4],
            'timestamp': parts[5] if len(parts) > 5 else 'unknown'
        }
    else:
        # Fallback for non-standard filenames
        return {
            'class': 'unknown',
            'user_id': 'unknown',
            'age_group': 'unknown',
            'gender': 'unknown',
            'ethnicity': 'unknown',
            'timestamp': 'unknown'
        }

def create_speaker_id(demo_info: Dict[str, str]) -> str:
    """Create a consistent speaker ID from demographic information."""
    age = demo_info['age_group']
    gender = demo_info['gender']
    ethnicity = demo_info['ethnicity']
    
    # Create speaker ID based on demographics
    speaker_id = f"{age}_{gender}_{ethnicity}"
    
    # Clean up the ID
    speaker_id = speaker_id.replace('not_specified', 'unknown')
    speaker_id = re.sub(r'[^a-zA-Z0-9_]', '_', speaker_id)
    
    return speaker_id

def normalize_class_name(class_name: str) -> str:
    """Normalize class names to ICU standard format."""
    class_mappings = {
        'doctor': 'doctor',
        'glasses': 'i_need_to_move',  # Map glasses to i_need_to_move for ICU
        'help': 'i_need_to_move',     # Map help to i_need_to_move for ICU  
        'phone': 'my_mouth_is_dry',   # Map phone to my_mouth_is_dry for ICU
        'pillow': 'pillow',
        'i_need_to_move': 'i_need_to_move',
        'my_mouth_is_dry': 'my_mouth_is_dry'
    }
    
    normalized = class_name.lower().strip()
    return class_mappings.get(normalized, normalized)

def scan_video_directory(video_dir: Path) -> List[Dict]:
    """Scan directory for video files and extract metadata."""
    videos = []
    
    # Supported video extensions
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
    
    for video_file in video_dir.rglob('*'):
        if video_file.is_file() and video_file.suffix.lower() in video_extensions:
            # Extract speaker info from filename
            demo_info = extract_speaker_info_from_filename(video_file.name)
            speaker_id = create_speaker_id(demo_info)
            normalized_class = normalize_class_name(demo_info['class'])
            
            # Skip unknown classes
            if normalized_class not in ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']:
                continue
            
            videos.append({
                'original_path': str(video_file),
                'filename': video_file.name,
                'speaker_id': speaker_id,
                'class': normalized_class,
                'original_class': demo_info['class'],
                'age_group': demo_info['age_group'],
                'gender': demo_info['gender'],
                'ethnicity': demo_info['ethnicity'],
                'timestamp': demo_info['timestamp']
            })
    
    return videos

def organize_speaker_sets(videos: List[Dict], output_dir: Path, copy_files: bool = True):
    """Organize videos into speaker sets structure."""
    
    # Group videos by speaker
    speaker_groups = {}
    for video in videos:
        speaker_id = video['speaker_id']
        if speaker_id not in speaker_groups:
            speaker_groups[speaker_id] = []
        speaker_groups[speaker_id].append(video)
    
    # Sort speakers by video count (descending) and take top 6
    sorted_speakers = sorted(speaker_groups.items(), key=lambda x: len(x[1]), reverse=True)
    top_speakers = sorted_speakers[:6]
    
    print(f"📊 Found {len(speaker_groups)} unique speakers")
    print(f"🎯 Selecting top 6 speakers with most videos:")
    
    for i, (speaker_id, speaker_videos) in enumerate(top_speakers, 1):
        class_counts = {}
        for video in speaker_videos:
            class_name = video['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"  Speaker {i}: {speaker_id} ({len(speaker_videos)} videos)")
        for class_name, count in sorted(class_counts.items()):
            print(f"    {class_name}: {count} videos")
    
    # Create speaker sets directory structure
    speaker_sets_dir = output_dir / "speaker sets"
    speaker_sets_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize top 6 speakers
    for i, (speaker_id, speaker_videos) in enumerate(top_speakers, 1):
        speaker_dir = speaker_sets_dir / f"speaker {i}"
        speaker_dir.mkdir(exist_ok=True)
        
        # Group videos by class
        class_groups = {}
        for video in speaker_videos:
            class_name = video['class']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(video)
        
        # Create class directories and copy/link files
        for class_name, class_videos in class_groups.items():
            class_dir = speaker_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            for video in class_videos:
                source_path = Path(video['original_path'])
                dest_path = class_dir / source_path.name
                
                if copy_files:
                    if not dest_path.exists():
                        shutil.copy2(source_path, dest_path)
                        print(f"📁 Copied: {source_path.name} → speaker {i}/{class_name}/")
                else:
                    if not dest_path.exists():
                        dest_path.symlink_to(source_path.resolve())
                        print(f"🔗 Linked: {source_path.name} → speaker {i}/{class_name}/")
    
    # Generate summary report
    summary_path = speaker_sets_dir / "organization_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("SPEAKER SETS ORGANIZATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total speakers found: {len(speaker_groups)}\n")
        f.write(f"Top 6 speakers selected for training\n\n")
        
        for i, (speaker_id, speaker_videos) in enumerate(top_speakers, 1):
            f.write(f"Speaker {i}: {speaker_id}\n")
            f.write(f"  Total videos: {len(speaker_videos)}\n")
            
            class_counts = {}
            for video in speaker_videos:
                class_name = video['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in sorted(class_counts.items()):
                f.write(f"  {class_name}: {count} videos\n")
            f.write("\n")
    
    print(f"\n✅ Speaker sets organized in: {speaker_sets_dir}")
    print(f"📋 Summary saved to: {summary_path}")
    
    return speaker_sets_dir

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Organize Speaker Data for Three-Stage Pipeline')
    parser.add_argument('--input-dir', default='data/13.9.25top7dataset_cropped',
                       help='Input directory containing video files')
    parser.add_argument('--output-dir', default='data',
                       help='Output directory for organized speaker sets')
    parser.add_argument('--copy-files', action='store_true',
                       help='Copy files instead of creating symlinks')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    print("🎯 SPEAKER DATA ORGANIZATION")
    print("=" * 40)
    print("Organizing data for three-stage training pipeline:")
    print("• Stage 1: GRID Pretraining")
    print("• Stage 2: ICU Fine-tuning (LOSO)")
    print("• Stage 3: Few-shot Personalization")
    print("=" * 40)
    
    # Check input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1
    
    print(f"📂 Scanning input directory: {input_dir}")
    
    # Scan for videos
    videos = scan_video_directory(input_dir)
    
    if not videos:
        print("❌ No videos found in input directory")
        return 1
    
    print(f"📊 Found {len(videos)} total videos")
    
    # Show class distribution
    class_counts = {}
    for video in videos:
        class_name = video['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("📈 Class distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count} videos")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No changes will be made")
        return 0
    
    # Organize speaker sets
    output_dir = Path(args.output_dir)
    speaker_sets_dir = organize_speaker_sets(videos, output_dir, args.copy_files)
    
    print(f"\n🎯 ORGANIZATION COMPLETE")
    print(f"Speaker sets ready for three-stage training pipeline")
    print(f"Next steps:")
    print(f"1. Implement GRID pretraining components")
    print(f"2. Run LOSO cross-validation with organized speakers")
    print(f"3. Execute few-shot personalization")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
