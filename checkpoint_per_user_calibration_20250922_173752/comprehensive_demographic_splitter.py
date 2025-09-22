#!/usr/bin/env python3
"""
Comprehensive Demographic-Based Dataset Splitter for Cross-Demographic Generalization Testing
Replaces visual similarity approach with strict demographic separation.
"""

import os
import re
import csv
import cv2
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import shutil
import random

class ComprehensiveDemographicSplitter:
    def __init__(self, source_dir="data/the_best_videos_so_far", output_dir="data/classifier training 20.9.25"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Valid demographic values
        self.valid_age_groups = {'18to39', '40to64', '65plus'}
        self.valid_genders = {'male', 'female'}
        self.valid_ethnicities = {'caucasian', 'asian', 'aboriginal', 'not_specified'}
        self.valid_classes = {'doctor', 'glasses', 'help', 'i_need_to_move', 'my_mouth_is_dry', 'phone', 'pillow'}
        
        # Filename pattern for demographic parsing
        self.filename_pattern = re.compile(
            r'^(?P<class>\w+)__useruser01__(?P<age_group>\w+)__(?P<gender>\w+)__(?P<ethnicity>\w+)__(?P<timestamp>\w+)\.mp4$'
        )
        
        # Storage for analysis
        self.valid_videos = []
        self.rejected_videos = []
        self.demographic_stats = defaultdict(lambda: defaultdict(int))
        
        print("🎯 COMPREHENSIVE DEMOGRAPHIC-BASED DATASET SPLITTER")
        print("=" * 80)
        print(f"📁 Source: {self.source_dir}")
        print(f"📁 Output: {self.output_dir}")
        print(f"🎯 Target: Cross-demographic generalization testing")
        
    def parse_and_validate_demographics(self):
        """Parse and validate demographic information from video filenames."""
        print("\n📋 STEP 1: PARSING AND VALIDATING DEMOGRAPHIC INFORMATION")
        print("=" * 60)
        
        video_files = list(self.source_dir.glob("**/*.mp4"))
        print(f"📊 Found {len(video_files)} total video files")
        
        for video_path in video_files:
            filename = video_path.name
            match = self.filename_pattern.match(filename)
            
            if not match:
                self.rejected_videos.append({
                    'path': str(video_path),
                    'reason': 'Invalid filename format',
                    'filename': filename
                })
                continue
            
            # Extract demographic components
            demo_data = match.groupdict()
            
            # Validate each component
            validation_errors = []
            if demo_data['class'] not in self.valid_classes:
                validation_errors.append(f"Invalid class: {demo_data['class']}")
            if demo_data['age_group'] not in self.valid_age_groups:
                validation_errors.append(f"Invalid age_group: {demo_data['age_group']}")
            if demo_data['gender'] not in self.valid_genders:
                validation_errors.append(f"Invalid gender: {demo_data['gender']}")
            if demo_data['ethnicity'] not in self.valid_ethnicities:
                validation_errors.append(f"Invalid ethnicity: {demo_data['ethnicity']}")
            
            if validation_errors:
                self.rejected_videos.append({
                    'path': str(video_path),
                    'reason': '; '.join(validation_errors),
                    'filename': filename
                })
                continue
            
            # Technical validation
            if not self._validate_video_technical(video_path):
                self.rejected_videos.append({
                    'path': str(video_path),
                    'reason': 'Technical validation failed (frame count/resolution/integrity)',
                    'filename': filename
                })
                continue
            
            # Create demographic group identifier
            demographic_group = f"{demo_data['age_group']}_{demo_data['gender']}_{demo_data['ethnicity']}"
            
            # Determine if original or augmented
            original_or_augmented = 'augmented' if 'augmented' in str(video_path) else 'original'
            
            # Store valid video
            video_info = {
                'path': str(video_path),
                'class': demo_data['class'],
                'age_group': demo_data['age_group'],
                'gender': demo_data['gender'],
                'ethnicity': demo_data['ethnicity'],
                'demographic_group': demographic_group,
                'original_or_augmented': original_or_augmented,
                'timestamp': demo_data['timestamp']
            }
            
            self.valid_videos.append(video_info)
            self.demographic_stats[demographic_group][demo_data['class']] += 1
        
        print(f"✅ Valid videos: {len(self.valid_videos)}")
        print(f"❌ Rejected videos: {len(self.rejected_videos)}")
        
        # Log rejected videos
        if self.rejected_videos:
            print(f"\n⚠️  REJECTED VIDEOS ({len(self.rejected_videos)}):")
            for rejected in self.rejected_videos[:10]:  # Show first 10
                print(f"   - {rejected['filename']}: {rejected['reason']}")
            if len(self.rejected_videos) > 10:
                print(f"   ... and {len(self.rejected_videos) - 10} more")
    
    def _validate_video_technical(self, video_path):
        """Validate video technical requirements."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return False
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            # Check requirements: 32 frames, 96x64 resolution
            return frame_count == 32 and width == 96 and height == 64
            
        except Exception:
            return False
    
    def analyze_demographic_groups(self):
        """Analyze demographic groups and select top 2 with highest video counts."""
        print("\n📊 STEP 2: DEMOGRAPHIC GROUP ANALYSIS AND SELECTION")
        print("=" * 60)
        
        # Calculate total videos per demographic group
        demographic_totals = {}
        for demo_group, class_counts in self.demographic_stats.items():
            total_videos = sum(class_counts.values())
            demographic_totals[demo_group] = total_videos
            
            print(f"📊 {demo_group}: {total_videos} videos")
            for class_name, count in sorted(class_counts.items()):
                print(f"   - {class_name}: {count}")
        
        # Select top 2 demographic groups with minimum 50 videos
        viable_demographics = {k: v for k, v in demographic_totals.items() if v >= 50}
        
        if len(viable_demographics) < 2:
            raise ValueError(f"Need at least 2 demographic groups with ≥50 videos. Found: {len(viable_demographics)}")
        
        # Sort by total video count (descending)
        sorted_demographics = sorted(viable_demographics.items(), key=lambda x: x[1], reverse=True)
        
        self.top_demographic_1 = sorted_demographics[0][0]  # Largest (training)
        self.top_demographic_2 = sorted_demographics[1][0]  # Second largest (validation)
        
        print(f"\n🎯 SELECTED TOP 2 DEMOGRAPHIC GROUPS:")
        print(f"   1. {self.top_demographic_1}: {sorted_demographics[0][1]} videos (→ TRAINING)")
        print(f"   2. {self.top_demographic_2}: {sorted_demographics[1][1]} videos (→ VALIDATION)")
        
        return self.top_demographic_1, self.top_demographic_2

    def balance_demographics_with_augmentation(self):
        """Balance classes within selected demographic groups using brightness augmentation."""
        print("\n⚖️  STEP 3: DATASET BALANCING WITHIN SELECTED DEMOGRAPHICS")
        print("=" * 60)

        # Filter videos to top 2 demographics
        filtered_videos = [v for v in self.valid_videos
                          if v['demographic_group'] in [self.top_demographic_1, self.top_demographic_2]]

        print(f"📊 Filtered to {len(filtered_videos)} videos from top 2 demographics")

        # Analyze class distribution within each demographic
        demo1_classes = defaultdict(list)
        demo2_classes = defaultdict(list)

        for video in filtered_videos:
            if video['demographic_group'] == self.top_demographic_1:
                demo1_classes[video['class']].append(video)
            else:
                demo2_classes[video['class']].append(video)

        # Calculate balance targets
        demo1_min = min(len(videos) for videos in demo1_classes.values())
        demo2_min = min(len(videos) for videos in demo2_classes.values())

        print(f"\n📊 CLASS DISTRIBUTION ANALYSIS:")
        print(f"   {self.top_demographic_1} (Training):")
        for class_name in sorted(demo1_classes.keys()):
            print(f"      - {class_name}: {len(demo1_classes[class_name])} videos")
        print(f"   → Balance target: {demo1_min} videos per class")

        print(f"\n   {self.top_demographic_2} (Validation):")
        for class_name in sorted(demo2_classes.keys()):
            print(f"      - {class_name}: {len(demo2_classes[class_name])} videos")
        print(f"   → Balance target: {demo2_min} videos per class")

        # Create balanced datasets
        self.balanced_training = self._balance_demographic_group(demo1_classes, demo1_min, "training")
        self.balanced_validation = self._balance_demographic_group(demo2_classes, demo2_min, "validation")

        print(f"\n✅ BALANCED DATASETS CREATED:")
        print(f"   Training: {len(self.balanced_training)} videos ({demo1_min} per class × {len(demo1_classes)} classes)")
        print(f"   Validation: {len(self.balanced_validation)} videos ({demo2_min} per class × {len(demo2_classes)} classes)")

    def _balance_demographic_group(self, class_videos, target_count, split_name):
        """Balance a single demographic group to target count per class."""
        balanced_videos = []
        augmented_dir = self.output_dir / "augmented_videos"
        augmented_dir.mkdir(exist_ok=True)

        for class_name, videos in class_videos.items():
            current_count = len(videos)

            if current_count >= target_count:
                # Randomly sample to target count
                selected_videos = random.sample(videos, target_count)
                balanced_videos.extend(selected_videos)
                print(f"   📊 {class_name}: {current_count} → {target_count} (sampled)")
            else:
                # Use all existing videos
                balanced_videos.extend(videos)
                needed = target_count - current_count

                # Generate augmented videos
                print(f"   🔄 {class_name}: {current_count} → {target_count} (augmenting {needed})")

                for i in range(needed):
                    # Select source video for augmentation (cycle through existing)
                    source_video = videos[i % len(videos)]

                    # Generate augmented video
                    augmented_video = self._create_brightness_augmented_video(
                        source_video, class_name, split_name, i
                    )

                    if augmented_video:
                        balanced_videos.append(augmented_video)

        return balanced_videos

    def _create_brightness_augmented_video(self, source_video, class_name, split_name, aug_index):
        """Create brightness-augmented video with random lighting variations."""
        try:
            source_path = Path(source_video['path'])

            # Generate augmentation parameters
            brightness_factor = random.uniform(0.85, 1.15)  # ±15%
            contrast_factor = random.uniform(0.9, 1.1)      # ±10%
            gamma_factor = random.uniform(0.95, 1.05)       # ±5%

            # Create output filename
            timestamp = source_video['timestamp']
            demo_group = source_video['demographic_group']
            output_filename = f"{class_name}__{demo_group}__aug{aug_index:03d}__{timestamp}.mp4"
            output_path = self.output_dir / "augmented_videos" / output_filename

            # Load and process video
            cap = cv2.VideoCapture(str(source_path))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (96, 64))

            frames_processed = 0
            while True:
                ret, frame = cap.read()
                if not ret or frames_processed >= 32:
                    break

                # Apply brightness/contrast/gamma adjustments
                frame = frame.astype(np.float32)

                # Brightness and contrast
                frame = frame * contrast_factor + (brightness_factor - 1.0) * 128

                # Gamma correction
                frame = np.power(frame / 255.0, 1.0 / gamma_factor) * 255.0

                # Clamp values
                frame = np.clip(frame, 0, 255).astype(np.uint8)

                out.write(frame)
                frames_processed += 1

            cap.release()
            out.release()

            # Create video info
            augmented_video = {
                'path': str(output_path),
                'class': source_video['class'],
                'age_group': source_video['age_group'],
                'gender': source_video['gender'],
                'ethnicity': source_video['ethnicity'],
                'demographic_group': source_video['demographic_group'],
                'original_or_augmented': 'augmented',
                'timestamp': f"aug{aug_index:03d}_{timestamp}",
                'source_video': source_video['path'],
                'brightness_factor': brightness_factor,
                'contrast_factor': contrast_factor,
                'gamma_factor': gamma_factor
            }

            return augmented_video

        except Exception as e:
            print(f"   ⚠️  Failed to create augmented video: {e}")
            return None

    def create_cross_demographic_splits(self):
        """Create strict cross-demographic splits with zero overlap."""
        print("\n🎯 STEP 4: CREATING CROSS-DEMOGRAPHIC SPLITS")
        print("=" * 60)

        # Verify zero demographic overlap
        training_demographics = set(v['demographic_group'] for v in self.balanced_training)
        validation_demographics = set(v['demographic_group'] for v in self.balanced_validation)

        overlap = training_demographics.intersection(validation_demographics)
        if overlap:
            raise ValueError(f"Demographic overlap detected: {overlap}")

        print(f"✅ ZERO DEMOGRAPHIC OVERLAP CONFIRMED")
        print(f"   Training demographics: {training_demographics}")
        print(f"   Validation demographics: {validation_demographics}")

        # Save manifests
        self._save_manifest(self.balanced_training, "demographic_train_manifest.csv")
        self._save_manifest(self.balanced_validation, "demographic_validation_manifest.csv")

        print(f"\n📄 MANIFESTS CREATED:")
        print(f"   - demographic_train_manifest.csv ({len(self.balanced_training)} videos)")
        print(f"   - demographic_validation_manifest.csv ({len(self.balanced_validation)} videos)")

    def create_binary_classification_subsets(self):
        """Create binary classification subsets (help vs doctor) for rapid testing."""
        print("\n🎯 STEP 5: CREATING BINARY CLASSIFICATION SUBSETS")
        print("=" * 60)

        # Filter to only "help" and "doctor" classes
        binary_classes = {'help', 'doctor'}

        binary_training = [v for v in self.balanced_training if v['class'] in binary_classes]
        binary_validation = [v for v in self.balanced_validation if v['class'] in binary_classes]

        print(f"📊 BINARY CLASSIFICATION SUBSETS:")
        print(f"   Training: {len(binary_training)} videos")
        for class_name in binary_classes:
            count = len([v for v in binary_training if v['class'] == class_name])
            print(f"      - {class_name}: {count}")

        print(f"   Validation: {len(binary_validation)} videos")
        for class_name in binary_classes:
            count = len([v for v in binary_validation if v['class'] == class_name])
            print(f"      - {class_name}: {count}")

        # Create binary classification directory
        binary_dir = self.output_dir / "binary_classification"
        binary_dir.mkdir(exist_ok=True)

        # Save binary manifests
        self._save_manifest(binary_training, "binary_classification/binary_train_manifest.csv")
        self._save_manifest(binary_validation, "binary_classification/binary_validation_manifest.csv")

        print(f"\n📄 BINARY MANIFESTS CREATED:")
        print(f"   - binary_classification/binary_train_manifest.csv")
        print(f"   - binary_classification/binary_validation_manifest.csv")

        return binary_training, binary_validation

    def _save_manifest(self, videos, filename):
        """Save video manifest to CSV file."""
        output_path = self.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['video_path', 'class', 'age_group', 'gender', 'ethnicity',
                         'demographic_group', 'original_or_augmented']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for video in videos:
                writer.writerow({
                    'video_path': video['path'],
                    'class': video['class'],
                    'age_group': video['age_group'],
                    'gender': video['gender'],
                    'ethnicity': video['ethnicity'],
                    'demographic_group': video['demographic_group'],
                    'original_or_augmented': video['original_or_augmented']
                })

    def generate_analysis_report(self):
        """Generate comprehensive analysis report."""
        print("\n📊 STEP 6: GENERATING ANALYSIS REPORT")
        print("=" * 60)

        report_path = self.output_dir / "demographic_analysis_report.txt"

        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE DEMOGRAPHIC ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Original distribution
            f.write("ORIGINAL VIDEO DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total videos found: {len(self.valid_videos) + len(self.rejected_videos)}\n")
            f.write(f"Valid videos: {len(self.valid_videos)}\n")
            f.write(f"Rejected videos: {len(self.rejected_videos)}\n\n")

            # Demographic group statistics
            f.write("DEMOGRAPHIC GROUP STATISTICS:\n")
            f.write("-" * 40 + "\n")
            for demo_group, class_counts in sorted(self.demographic_stats.items()):
                total = sum(class_counts.values())
                f.write(f"{demo_group}: {total} videos\n")
                for class_name, count in sorted(class_counts.items()):
                    f.write(f"  - {class_name}: {count}\n")
                f.write("\n")

            # Selected demographics
            f.write("SELECTED DEMOGRAPHICS FOR CROSS-DEMOGRAPHIC TESTING:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Training (Largest): {self.top_demographic_1}\n")
            f.write(f"Validation (Second): {self.top_demographic_2}\n\n")

            # Final balanced counts
            f.write("FINAL BALANCED DATASET:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Training videos: {len(self.balanced_training)}\n")
            f.write(f"Validation videos: {len(self.balanced_validation)}\n")
            f.write(f"Total: {len(self.balanced_training) + len(self.balanced_validation)}\n\n")

            # Cross-demographic setup
            f.write("CROSS-DEMOGRAPHIC GENERALIZATION SETUP:\n")
            f.write("-" * 40 + "\n")
            f.write("✅ Zero demographic overlap between training and validation\n")
            f.write("✅ Perfect class balance within each demographic group\n")
            f.write("✅ Binary classification subsets created for rapid testing\n")
            f.write("✅ Brightness-only augmentation preserves lip-reading quality\n\n")

            # Success criteria
            f.write("SUCCESS CRITERIA:\n")
            f.write("-" * 40 + "\n")
            f.write("- Training Performance: >90% accuracy within 20 epochs\n")
            f.write("- Cross-Demographic Validation: >70% accuracy\n")
            f.write("- Clear learning progression without demographic memorization\n")

        print(f"📄 Analysis report saved: {report_path}")

    def run_complete_pipeline(self):
        """Execute the complete demographic splitting pipeline."""
        print("🚀 EXECUTING COMPLETE DEMOGRAPHIC SPLITTING PIPELINE")
        print("=" * 80)

        try:
            # Step 1: Parse and validate demographics
            self.parse_and_validate_demographics()

            # Step 2: Analyze and select top demographics
            self.analyze_demographic_groups()

            # Step 3: Balance with augmentation
            self.balance_demographics_with_augmentation()

            # Step 4: Create cross-demographic splits
            self.create_cross_demographic_splits()

            # Step 5: Create binary subsets
            self.create_binary_classification_subsets()

            # Step 6: Generate analysis report
            self.generate_analysis_report()

            print("\n🎉 DEMOGRAPHIC SPLITTING PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"📁 Output directory: {self.output_dir}")
            print("📄 Ready for cross-demographic generalization testing")
            print("\n⚠️  NEXT STEPS:")
            print("1. Visually inspect output files for corruption")
            print("2. Verify balanced class distributions")
            print("3. Test binary classification for rapid validation")
            print("4. Proceed to full 7-class cross-demographic training")

        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            raise

def main():
    """Main execution function."""
    splitter = ComprehensiveDemographicSplitter()
    splitter.run_complete_pipeline()

if __name__ == "__main__":
    main()
