#!/usr/bin/env python3
"""
Dataset Analysis and Balancing for Lip-Reading Training

This script analyzes the complete preprocessed dataset and implements intelligent
balancing to ensure equal representation across all classes.
"""

import os
import glob
import shutil
import random
import csv
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np

class DatasetAnalyzer:
    def __init__(self, dataset_path="grayscale_validation_output/processed_videos"):
        self.dataset_path = dataset_path
        self.classes = ["doctor", "glasses", "help", "phone", "pillow"]
        self.class_counts = defaultdict(list)
        self.total_videos = 0
        
    def analyze_dataset(self):
        """Analyze the complete dataset and generate class distribution report."""
        print("🔍 DATASET ANALYSIS & BALANCING")
        print("=" * 80)
        
        # Get all processed videos
        video_files = glob.glob(os.path.join(self.dataset_path, "*_processed.mp4"))
        self.total_videos = len(video_files)
        
        print(f"📊 Total processed videos found: {self.total_videos}")
        print(f"📁 Dataset location: {self.dataset_path}")
        print()
        
        # Analyze class distribution
        for video_file in video_files:
            video_name = Path(video_file).stem.replace("_processed", "")
            
            # Extract class from video name (first word)
            class_name = video_name.split()[0].lower()
            if class_name in self.classes:
                self.class_counts[class_name].append({
                    'original_name': video_name,
                    'file_path': video_file,
                    'processed_name': Path(video_file).stem
                })
            else:
                print(f"⚠️  Warning: Unknown class '{class_name}' for video: {video_name}")
        
        # Generate class distribution report
        print("📋 CLASS DISTRIBUTION ANALYSIS:")
        print("-" * 50)
        
        class_stats = {}
        for class_name in self.classes:
            count = len(self.class_counts[class_name])
            percentage = (count / self.total_videos) * 100 if self.total_videos > 0 else 0
            class_stats[class_name] = count
            print(f"   • {class_name:<12}: {count:>3} videos ({percentage:>5.1f}%)")
        
        # Identify target sample size (highest class count)
        target_size = max(class_stats.values())
        print(f"\n🎯 Target sample size (highest class): {target_size} videos")
        
        # Calculate balancing requirements
        print(f"\n📈 BALANCING REQUIREMENTS:")
        print("-" * 50)
        
        total_needed = 0
        for class_name in self.classes:
            current_count = class_stats[class_name]
            needed = target_size - current_count
            total_needed += needed
            
            if needed > 0:
                print(f"   • {class_name:<12}: needs {needed:>2} more videos (current: {current_count})")
            else:
                print(f"   • {class_name:<12}: balanced ✅ (current: {current_count})")
        
        print(f"\n📊 SUMMARY:")
        print(f"   • Current total videos: {self.total_videos}")
        print(f"   • Target balanced total: {target_size * len(self.classes)}")
        print(f"   • Additional videos needed: {total_needed}")
        
        return target_size, class_stats
    
    def verify_video_quality(self, video_path):
        """Verify that a video maintains preprocessing standards."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Cannot open video"
            
            # Check frame count
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            # Verify standards
            if frame_count != 32:
                return False, f"Frame count mismatch: {frame_count} (expected 32)"
            
            # Check dimensions (should be 640x432 or smaller cropped dimensions)
            if width > 640 or height > 432:
                return False, f"Dimensions too large: {width}x{height}"
            
            return True, f"Valid: {frame_count} frames, {width}x{height}px"
            
        except Exception as e:
            return False, f"Error: {str(e)}"

class DatasetBalancer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.balanced_dataset_path = "balanced_training_dataset"
        self.manifest_path = "balanced_dataset_manifest.csv"
        
    def create_balanced_dataset(self, target_size):
        """Create a balanced dataset using intelligent duplication."""
        print(f"\n🔄 CREATING BALANCED DATASET")
        print("=" * 80)
        
        # Create output directory
        os.makedirs(self.balanced_dataset_path, exist_ok=True)
        
        # Initialize manifest
        manifest_entries = []
        duplication_stats = defaultdict(int)
        
        print(f"📁 Creating balanced dataset in: {self.balanced_dataset_path}")
        print()
        
        # Process each class
        for class_name in self.analyzer.classes:
            print(f"🔄 Processing class: {class_name}")
            
            original_videos = self.analyzer.class_counts[class_name]
            current_count = len(original_videos)
            needed = target_size - current_count
            
            print(f"   • Original videos: {current_count}")
            print(f"   • Target videos: {target_size}")
            print(f"   • Duplicates needed: {needed}")
            
            # Copy all original videos
            for i, video_info in enumerate(original_videos):
                src_path = video_info['file_path']
                dst_name = f"{class_name}_{i+1:02d}_original.mp4"
                dst_path = os.path.join(self.balanced_dataset_path, dst_name)
                
                shutil.copy2(src_path, dst_path)
                
                # Verify quality
                is_valid, quality_msg = self.analyzer.verify_video_quality(dst_path)
                
                manifest_entries.append({
                    'class': class_name,
                    'filename': dst_name,
                    'original_video': video_info['original_name'],
                    'is_duplicate': False,
                    'duplicate_of': '',
                    'quality_check': quality_msg,
                    'valid': is_valid
                })
            
            # Create intelligent duplicates if needed
            if needed > 0:
                print(f"   • Creating {needed} intelligent duplicates...")
                
                # Create duplication sequence avoiding consecutive duplicates
                duplication_sequence = self._create_duplication_sequence(original_videos, needed)
                
                for dup_idx, source_video in enumerate(duplication_sequence):
                    src_path = source_video['file_path']
                    dst_name = f"{class_name}_{current_count + dup_idx + 1:02d}_dup{dup_idx+1:02d}.mp4"
                    dst_path = os.path.join(self.balanced_dataset_path, dst_name)
                    
                    shutil.copy2(src_path, dst_path)
                    duplication_stats[source_video['original_name']] += 1
                    
                    # Verify quality
                    is_valid, quality_msg = self.analyzer.verify_video_quality(dst_path)
                    
                    manifest_entries.append({
                        'class': class_name,
                        'filename': dst_name,
                        'original_video': source_video['original_name'],
                        'is_duplicate': True,
                        'duplicate_of': source_video['original_name'],
                        'quality_check': quality_msg,
                        'valid': is_valid
                    })
            
            print(f"   ✅ Completed: {target_size} videos for {class_name}")
        
        # Save manifest
        self._save_manifest(manifest_entries)
        
        # Generate duplication report
        self._generate_duplication_report(duplication_stats, target_size)
        
        return len(manifest_entries)
    
    def _create_duplication_sequence(self, original_videos, needed):
        """Create intelligent duplication sequence avoiding consecutive duplicates."""
        if len(original_videos) == 0:
            return []
        
        sequence = []
        available_videos = original_videos.copy()
        last_selected = None
        
        for _ in range(needed):
            # Filter out the last selected video to avoid consecutive duplicates
            if last_selected and len(available_videos) > 1:
                candidates = [v for v in available_videos if v['original_name'] != last_selected]
                if not candidates:
                    candidates = available_videos
            else:
                candidates = available_videos
            
            # Randomly select from candidates
            selected = random.choice(candidates)
            sequence.append(selected)
            last_selected = selected['original_name']
            
            # If we've used all videos, reset the available pool
            if len(set(v['original_name'] for v in sequence[-len(original_videos):])) == len(original_videos):
                last_selected = None
        
        return sequence
    
    def _save_manifest(self, manifest_entries):
        """Save the balanced dataset manifest."""
        with open(self.manifest_path, 'w', newline='') as csvfile:
            fieldnames = ['class', 'filename', 'original_video', 'is_duplicate', 
                         'duplicate_of', 'quality_check', 'valid']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for entry in manifest_entries:
                writer.writerow(entry)
        
        print(f"\n📄 Manifest saved: {self.manifest_path}")
    
    def _generate_duplication_report(self, duplication_stats, target_size):
        """Generate comprehensive duplication report."""
        print(f"\n📊 DUPLICATION REPORT:")
        print("-" * 50)
        
        if duplication_stats:
            print("Videos used for duplication:")
            for video_name, count in sorted(duplication_stats.items()):
                print(f"   • {video_name}: duplicated {count} times")
        else:
            print("   • No duplications were needed - dataset was already balanced!")
        
        # Final verification
        balanced_videos = glob.glob(os.path.join(self.balanced_dataset_path, "*.mp4"))
        final_class_counts = defaultdict(int)
        
        for video_file in balanced_videos:
            filename = Path(video_file).stem
            class_name = filename.split('_')[0]
            final_class_counts[class_name] += 1
        
        print(f"\n✅ BALANCED DATASET VERIFICATION:")
        print("-" * 50)
        
        all_balanced = True
        for class_name in self.analyzer.classes:
            count = final_class_counts[class_name]
            is_balanced = count == target_size
            status = "✅" if is_balanced else "❌"
            print(f"   • {class_name:<12}: {count:>3} videos {status}")
            if not is_balanced:
                all_balanced = False
        
        total_balanced = sum(final_class_counts.values())
        expected_total = target_size * len(self.analyzer.classes)
        
        print(f"\n📈 FINAL STATISTICS:")
        print(f"   • Total balanced videos: {total_balanced}")
        print(f"   • Expected total: {expected_total}")
        print(f"   • Balance status: {'✅ PERFECT' if all_balanced else '❌ NEEDS ATTENTION'}")
        
        return all_balanced

def main():
    """Main execution function."""
    print("🚀 DATASET ANALYSIS & BALANCING FOR LIP-READING TRAINING")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer()
    
    # Phase 1: Analyze dataset
    target_size, class_stats = analyzer.analyze_dataset()
    
    # Phase 2: Create balanced dataset
    balancer = DatasetBalancer(analyzer)
    total_videos = balancer.create_balanced_dataset(target_size)
    
    print(f"\n🎉 DATASET BALANCING COMPLETED!")
    print("=" * 80)
    print(f"✅ Balanced dataset created with {total_videos} videos")
    print(f"📁 Location: {balancer.balanced_dataset_path}")
    print(f"📄 Manifest: {balancer.manifest_path}")
    print()
    print("🔄 NEXT STEPS:")
    print("   1. Review the balancing report above")
    print("   2. Verify the balanced dataset quality")
    print("   3. Confirm to proceed with training configuration")
    
    return True

if __name__ == "__main__":
    # Set random seed for reproducible duplication
    random.seed(42)
    main()
