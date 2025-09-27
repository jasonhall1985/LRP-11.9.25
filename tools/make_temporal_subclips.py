#!/usr/bin/env python3
"""
Create temporal subclips from training videos only to augment dataset.
Respects LOSO splits to avoid contaminating validation data.
"""
import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.id_norm import norm_speaker_id, norm_label

def load_splits_info(splits_dir):
    """Load LOSO splits information."""
    splits_file = os.path.join(splits_dir, "loso_splits_info.json")
    with open(splits_file, 'r') as f:
        return json.load(f)

def get_train_only_speakers(splits_info):
    """Get set of all speakers that appear in training across all folds."""
    all_train_speakers = set()
    for fold_info in splits_info.values():
        all_train_speakers.update(fold_info['train_speakers'])
    return all_train_speakers

def extract_subclips(video_path, frames_per_clip=32, stride=16, max_clips=3):
    """
    Extract temporal subclips from a video.
    
    Args:
        video_path: Path to input video
        frames_per_clip: Number of frames per subclip
        stride: Stride between subclips
        max_clips: Maximum number of subclips to extract
    
    Returns:
        List of frame arrays for each subclip
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open {video_path}")
        return []
    
    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if len(frames) < frames_per_clip:
        print(f"Warning: Video {video_path} has only {len(frames)} frames, need {frames_per_clip}")
        return []
    
    # Extract subclips
    subclips = []
    start_idx = 0
    clip_count = 0
    
    while start_idx + frames_per_clip <= len(frames) and clip_count < max_clips:
        end_idx = start_idx + frames_per_clip
        subclip = frames[start_idx:end_idx]
        subclips.append(subclip)
        
        start_idx += stride
        clip_count += 1
    
    return subclips

def save_subclip(frames, output_path, fps=25):
    """Save frames as video file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not frames:
        return False
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()
    return True

def process_video_file(video_path, output_root, relative_path, frames_per_clip, stride):
    """Process a single video file and create subclips."""
    subclips = extract_subclips(video_path, frames_per_clip, stride)
    
    saved_count = 0
    for i, subclip in enumerate(subclips):
        # Create output path with subclip index
        base_name = Path(relative_path).stem
        ext = Path(relative_path).suffix
        subclip_name = f"{base_name}_subclip_{i:02d}{ext}"
        
        output_path = output_root / Path(relative_path).parent / subclip_name
        
        if save_subclip(subclip, output_path):
            saved_count += 1
    
    return saved_count

def main():
    parser = argparse.ArgumentParser(description="Create temporal subclips for training augmentation")
    parser.add_argument("--in-root", required=True, help="Input root directory")
    parser.add_argument("--out-root", required=True, help="Output root directory")
    parser.add_argument("--frames", type=int, default=32, help="Frames per subclip")
    parser.add_argument("--stride", type=int, default=16, help="Stride between subclips")
    parser.add_argument("--train-only", action="store_true", help="Only process training speakers")
    parser.add_argument("--splits-dir", help="Directory containing LOSO splits info")
    
    args = parser.parse_args()
    
    input_root = Path(args.in_root)
    output_root = Path(args.out_root)
    
    # Load splits info if train-only mode
    train_speakers = None
    if args.train_only and args.splits_dir:
        splits_info = load_splits_info(args.splits_dir)
        train_speakers = get_train_only_speakers(splits_info)
        print(f"Train-only mode: processing speakers {sorted(train_speakers)}")
    
    # Process all videos
    total_videos = 0
    total_subclips = 0
    
    for speaker_dir in input_root.iterdir():
        if not speaker_dir.is_dir():
            continue
        
        speaker_norm = norm_speaker_id(speaker_dir.name)
        
        # Skip if not in training speakers (train-only mode)
        if train_speakers and speaker_norm not in train_speakers:
            print(f"Skipping validation speaker: {speaker_norm}")
            continue
        
        print(f"Processing speaker: {speaker_norm}")
        
        for class_dir in speaker_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_norm = norm_label(class_dir.name)
            print(f"  Processing class: {class_norm}")
            
            video_files = list(class_dir.glob("*.mp4"))
            
            for video_file in tqdm(video_files, desc=f"  {class_norm}"):
                relative_path = video_file.relative_to(input_root)
                
                # Update relative path with normalized names
                norm_relative = Path(speaker_norm) / class_norm / video_file.name
                
                subclip_count = process_video_file(
                    video_file, output_root, norm_relative, 
                    args.frames, args.stride
                )
                
                total_videos += 1
                total_subclips += subclip_count
    
    print(f"\nProcessing complete:")
    print(f"  Input videos: {total_videos}")
    print(f"  Output subclips: {total_subclips}")
    print(f"  Augmentation factor: {total_subclips/total_videos:.1f}x")

if __name__ == "__main__":
    main()
