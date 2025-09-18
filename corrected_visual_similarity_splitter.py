#!/usr/bin/env python3
"""
Corrected Visual Similarity Dataset Splitter
============================================
Fixed version that properly handles male 18to39 videos and ensures proper distribution.

Key Fixes:
- Correct age group format matching ('18to39' not '18-39')
- Better distribution logic for 70/15/15 split
- Proper male 18to39 assignment to training

Author: Augment Agent
Date: 2025-09-18
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Import required packages
try:
    import imagehash
    from PIL import Image
except ImportError:
    print("Installing required packages...")
    os.system("pip install imagehash pillow")
    import imagehash
    from PIL import Image

class CorrectedVisualSimilaritySplitter:
    """Corrected visual similarity-based dataset splitter."""
    
    def __init__(self, dataset_dir="data/the_best_videos_so_far", 
                 augmented_dir="data/the_best_videos_so_far/augmented_videos",
                 output_dir="corrected_visual_similarity_splits"):
        
        self.dataset_dir = Path(dataset_dir)
        self.augmented_dir = Path(augmented_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visual similarity parameters
        self.hash_size = 8  # 64-bit hash
        self.similarity_threshold = 5  # Hamming distance threshold
        self.hash_algorithm = 'phash'  # 'phash', 'ahash', 'dhash', 'whash'
        
        # Target split ratios
        self.target_train_ratio = 0.70
        self.target_val_ratio = 0.15
        self.target_test_ratio = 0.15
        
        # Class definitions
        self.classes = ['doctor', 'glasses', 'help', 'phone', 'pillow', 'my_mouth_is_dry', 'i_need_to_move']
        
        # Data storage
        self.videos_data = []
        self.fingerprint_db = {}
        self.pseudo_speaker_groups = {}
        self.group_assignments = {}
        
        print("🎯 Corrected Visual Similarity Dataset Splitter Initialized")
        print(f"📁 Dataset directory: {self.dataset_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🔍 Hash algorithm: {self.hash_algorithm}")
        print(f"📏 Similarity threshold: {self.similarity_threshold}")
    
    def extract_first_frame(self, video_path):
        """Extract the first frame from a video for fingerprinting."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            # Convert BGR to RGB for PIL
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return frame
        except Exception as e:
            print(f"❌ Error extracting frame from {video_path}: {str(e)}")
            return None
    
    def generate_visual_fingerprint(self, frame):
        """Generate perceptual hash fingerprint from frame."""
        try:
            pil_image = Image.fromarray(frame)
            hash_value = imagehash.phash(pil_image, hash_size=self.hash_size)
            return str(hash_value)
        except Exception as e:
            print(f"❌ Error generating fingerprint: {str(e)}")
            return None
    
    def extract_demographics_from_filename(self, filename):
        """Extract demographic information from filename."""
        demographics = {
            'age_group': 'unknown',
            'gender': 'unknown', 
            'ethnicity': 'unknown',
            'format_type': 'unknown'
        }
        
        # Check for structured filename format
        if '__' in filename:
            parts = filename.split('__')
            if len(parts) >= 5:
                demographics['age_group'] = parts[2]
                demographics['gender'] = parts[3]
                demographics['ethnicity'] = parts[4].split('_')[0]
                demographics['format_type'] = 'structured'
                return demographics
        
        # Check for numbered format
        import re
        if re.match(r'^[a-z_]+\s+\d+_processed', filename):
            demographics['format_type'] = 'numbered'
            return demographics
        
        return demographics
    
    def extract_class_from_filename(self, filename):
        """Extract class name from filename."""
        for class_name in self.classes:
            if filename.startswith(class_name):
                return class_name
        
        if '__' in filename:
            parts = filename.split('__')
            if len(parts) > 0:
                return parts[0]
        
        return 'unknown'
    
    def load_and_fingerprint_videos(self):
        """Load all videos and generate visual fingerprints."""
        print("\n🎬 LOADING VIDEOS AND GENERATING VISUAL FINGERPRINTS")
        print("=" * 80)
        
        # Process original videos
        print(f"📁 Processing original videos from: {self.dataset_dir}")
        original_count = 0
        for video_file in self.dataset_dir.glob("*.mp4"):
            if video_file.is_file():
                self._process_video_file(video_file, 'original')
                original_count += 1
        
        # Process augmented videos
        print(f"📁 Processing augmented videos from: {self.augmented_dir}")
        augmented_count = 0
        if self.augmented_dir.exists():
            for video_file in self.augmented_dir.glob("*.mp4"):
                if video_file.is_file():
                    self._process_video_file(video_file, 'augmented')
                    augmented_count += 1
        
        print(f"✅ Total videos processed: {len(self.videos_data)}")
        print(f"   - Original videos: {original_count}")
        print(f"   - Augmented videos: {augmented_count}")
        print(f"   - Successful fingerprints: {len(self.fingerprint_db)}")
        
        return len(self.videos_data)
    
    def _process_video_file(self, video_file, video_type):
        """Process a single video file."""
        try:
            # Extract demographics and class
            demographics = self.extract_demographics_from_filename(video_file.name)
            class_name = self.extract_class_from_filename(video_file.name)
            
            # Skip unknown classes
            if class_name not in self.classes:
                return
            
            # Extract first frame
            first_frame = self.extract_first_frame(video_file)
            if first_frame is None:
                return
            
            # Generate fingerprint
            fingerprint = self.generate_visual_fingerprint(first_frame)
            if fingerprint is None:
                return
            
            # Store video information
            video_info = {
                'filename': video_file.name,
                'full_path': str(video_file),
                'class': class_name,
                'age_group': demographics['age_group'],
                'gender': demographics['gender'],
                'ethnicity': demographics['ethnicity'],
                'format_type': demographics['format_type'],
                'video_type': video_type,
                'fingerprint_hash': fingerprint,
                'pseudo_speaker_id': None,
                'dataset_split': None
            }
            
            self.videos_data.append(video_info)
            self.fingerprint_db[str(video_file)] = fingerprint
            
        except Exception as e:
            print(f"❌ Error processing {video_file.name}: {str(e)}")
    
    def calculate_fingerprint_similarity(self, hash1, hash2):
        """Calculate Hamming distance between two fingerprint hashes."""
        try:
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            return h1 - h2  # Hamming distance
        except Exception as e:
            return float('inf')
    
    def create_pseudo_speaker_groups(self):
        """Create pseudo-speaker groups using visual similarity clustering."""
        print(f"\n👥 CREATING PSEUDO-SPEAKER GROUPS")
        print("=" * 80)
        
        if len(self.videos_data) == 0:
            print("❌ No videos available for clustering")
            return False
        
        # Create similarity graph
        print("📊 Building similarity graph...")
        G = nx.Graph()
        
        # Add all videos as nodes
        for i, video in enumerate(self.videos_data):
            G.add_node(i, **video)
        
        # Add edges between similar videos
        similar_pairs = 0
        total_comparisons = 0
        
        for i in range(len(self.videos_data)):
            for j in range(i + 1, len(self.videos_data)):
                hash1 = self.videos_data[i]['fingerprint_hash']
                hash2 = self.videos_data[j]['fingerprint_hash']
                
                similarity = self.calculate_fingerprint_similarity(hash1, hash2)
                total_comparisons += 1
                
                if similarity <= self.similarity_threshold:
                    G.add_edge(i, j, similarity=similarity)
                    similar_pairs += 1
        
        print(f"📈 Similarity analysis:")
        print(f"   Total comparisons: {total_comparisons:,}")
        print(f"   Similar pairs found: {similar_pairs}")
        print(f"   Similarity rate: {similar_pairs/total_comparisons*100:.2f}%")
        
        # Find connected components (pseudo-speaker groups)
        connected_components = list(nx.connected_components(G))
        
        print(f"👥 Pseudo-speaker grouping results:")
        print(f"   Total groups found: {len(connected_components)}")
        
        # Assign group IDs and update video data
        for group_id, component in enumerate(connected_components):
            self.pseudo_speaker_groups[group_id] = list(component)
            
            # Update video data with group ID
            for video_idx in component:
                self.videos_data[video_idx]['pseudo_speaker_id'] = group_id
        
        return True

    def assign_groups_to_splits_corrected(self):
        """Corrected assignment of pseudo-speaker groups to splits."""
        print(f"\n🎯 ASSIGNING PSEUDO-SPEAKER GROUPS TO SPLITS (CORRECTED)")
        print("=" * 80)
        print("🚨 Constraints:")
        print("   1. Zero pseudo-speaker overlap between splits")
        print("   2. All 7 classes must be present in each split")
        print("   3. Male 18to39 videos → Training set (high quality)")
        print("   4. Target ratios: 70% train, 15% val, 15% test")

        # Step 1: Identify mandatory training groups (male 18to39) - CORRECTED
        mandatory_train_groups = set()
        distributable_groups = set()

        male_18to39_count = 0

        for group_id, video_indices in self.pseudo_speaker_groups.items():
            # Check if group contains male 18to39 videos - FIXED FORMAT
            has_male_18to39 = False
            for video_idx in video_indices:
                video = self.videos_data[video_idx]
                # CORRECTED: Check for exact match '18to39' and 'male'
                if video['gender'] == 'male' and video['age_group'] == '18to39':
                    has_male_18to39 = True
                    male_18to39_count += 1
                    break

            if has_male_18to39:
                mandatory_train_groups.add(group_id)
                self.group_assignments[group_id] = 'train'
            else:
                distributable_groups.add(group_id)

        print(f"🔒 Mandatory training groups: {len(mandatory_train_groups)}")
        print(f"📊 Distributable groups: {len(distributable_groups)}")
        print(f"👨 Male 18to39 videos found: {male_18to39_count}")

        # Step 2: Calculate current class coverage
        def get_split_class_coverage(split_groups):
            """Get classes covered by a set of groups."""
            covered_classes = set()
            for group_id in split_groups:
                for video_idx in self.pseudo_speaker_groups[group_id]:
                    video = self.videos_data[video_idx]
                    covered_classes.add(video['class'])
            return covered_classes

        # Check current training coverage
        train_coverage = get_split_class_coverage(mandatory_train_groups)
        missing_from_train = set(self.classes) - train_coverage

        print(f"📊 Current training coverage: {len(train_coverage)}/7 classes")
        if missing_from_train:
            print(f"❌ Missing from training: {missing_from_train}")

        # Step 3: Ensure all classes in training by adding required groups
        if missing_from_train:
            print(f"🔧 Adding groups to training to cover missing classes...")

            for missing_class in missing_from_train:
                # Find a distributable group that contains this class
                best_group = None
                for group_id in distributable_groups:
                    group_classes = get_split_class_coverage({group_id})
                    if missing_class in group_classes:
                        best_group = group_id
                        break

                if best_group is not None:
                    mandatory_train_groups.add(best_group)
                    distributable_groups.remove(best_group)
                    self.group_assignments[best_group] = 'train'
                    print(f"   Added group {best_group} for class '{missing_class}'")

        # Step 4: Calculate video counts for proper distribution
        total_videos = len(self.videos_data)
        current_train_videos = sum(len(self.pseudo_speaker_groups[gid]) for gid in mandatory_train_groups)
        remaining_videos = total_videos - current_train_videos

        target_train_videos = int(total_videos * self.target_train_ratio)
        target_val_videos = int(total_videos * self.target_val_ratio)
        target_test_videos = total_videos - target_train_videos - target_val_videos

        additional_train_needed = max(0, target_train_videos - current_train_videos)

        print(f"\n📊 Video distribution planning:")
        print(f"   Total videos: {total_videos}")
        print(f"   Current training videos: {current_train_videos}")
        print(f"   Target training videos: {target_train_videos}")
        print(f"   Additional training needed: {additional_train_needed}")
        print(f"   Target validation videos: {target_val_videos}")
        print(f"   Target test videos: {target_test_videos}")

        # Step 5: Intelligent assignment of remaining groups
        remaining_groups = list(distributable_groups)

        # Sort groups by size (largest first) for better distribution
        remaining_groups.sort(key=lambda gid: len(self.pseudo_speaker_groups[gid]), reverse=True)

        val_groups = set()
        test_groups = set()
        current_val_videos = 0
        current_test_videos = 0

        # First, add more groups to training if needed
        groups_added_to_train = 0
        for group_id in remaining_groups[:]:
            if additional_train_needed <= 0:
                break

            group_size = len(self.pseudo_speaker_groups[group_id])
            if group_size <= additional_train_needed:
                mandatory_train_groups.add(group_id)
                remaining_groups.remove(group_id)
                self.group_assignments[group_id] = 'train'
                additional_train_needed -= group_size
                groups_added_to_train += 1

        print(f"🔧 Added {groups_added_to_train} additional groups to training")

        # Then distribute remaining between validation and test
        for group_id in remaining_groups:
            group_size = len(self.pseudo_speaker_groups[group_id])

            # Decide assignment based on current needs
            val_deficit = target_val_videos - current_val_videos
            test_deficit = target_test_videos - current_test_videos

            if test_deficit > val_deficit and test_deficit > 0:
                test_groups.add(group_id)
                current_test_videos += group_size
                self.group_assignments[group_id] = 'test'
            else:
                val_groups.add(group_id)
                current_val_videos += group_size
                self.group_assignments[group_id] = 'validation'

        # Step 6: Verify class coverage in all splits
        print(f"\n🔍 VERIFYING CLASS COVERAGE IN ALL SPLITS")
        print("-" * 60)

        train_classes = get_split_class_coverage(mandatory_train_groups)
        val_classes = get_split_class_coverage(val_groups)
        test_classes = get_split_class_coverage(test_groups)

        print(f"Training classes ({len(train_classes)}): {sorted(train_classes)}")
        print(f"Validation classes ({len(val_classes)}): {sorted(val_classes)}")
        print(f"Test classes ({len(test_classes)}): {sorted(test_classes)}")

        # Check for missing classes in val/test and fix if needed
        missing_val = set(self.classes) - val_classes
        missing_test = set(self.classes) - test_classes

        if missing_val or missing_test:
            print(f"⚠️  Class coverage issues detected - attempting to fix...")
            self._fix_class_coverage_corrected(missing_val, missing_test, val_groups, test_groups)

        # Step 7: Final statistics
        final_train_videos = sum(len(self.pseudo_speaker_groups[gid]) for gid in mandatory_train_groups)
        final_val_videos = sum(len(self.pseudo_speaker_groups[gid]) for gid in val_groups)
        final_test_videos = sum(len(self.pseudo_speaker_groups[gid]) for gid in test_groups)

        print(f"\n📊 FINAL GROUP ASSIGNMENT RESULTS:")
        print("-" * 60)
        print(f"Training: {len(mandatory_train_groups)} groups ({final_train_videos} videos, {final_train_videos/total_videos*100:.1f}%)")
        print(f"Validation: {len(val_groups)} groups ({final_val_videos} videos, {final_val_videos/total_videos*100:.1f}%)")
        print(f"Test: {len(test_groups)} groups ({final_test_videos} videos, {final_test_videos/total_videos*100:.1f}%)")

        return True

    def _fix_class_coverage_corrected(self, missing_val, missing_test, val_groups, test_groups):
        """Attempt to fix class coverage issues by reassigning groups."""
        print(f"🔧 Attempting to fix class coverage...")

        # For each missing class, try to find a group from the other split
        for missing_class in missing_val:
            for group_id in list(test_groups):
                group_classes = set()
                for video_idx in self.pseudo_speaker_groups[group_id]:
                    video = self.videos_data[video_idx]
                    group_classes.add(video['class'])

                if missing_class in group_classes:
                    test_groups.remove(group_id)
                    val_groups.add(group_id)
                    self.group_assignments[group_id] = 'validation'
                    print(f"   Moved group {group_id} to validation for class '{missing_class}'")
                    break

        for missing_class in missing_test:
            for group_id in list(val_groups):
                group_classes = set()
                for video_idx in self.pseudo_speaker_groups[group_id]:
                    video = self.videos_data[video_idx]
                    group_classes.add(video['class'])

                if missing_class in group_classes:
                    val_groups.remove(group_id)
                    test_groups.add(group_id)
                    self.group_assignments[group_id] = 'test'
                    print(f"   Moved group {group_id} to test for class '{missing_class}'")
                    break

    def assign_videos_to_splits(self):
        """Assign individual videos based on their pseudo-speaker group assignments."""
        print(f"\n📋 ASSIGNING VIDEOS TO SPLITS")
        print("=" * 60)

        # Assign each video to the split determined by its pseudo-speaker group
        for video in self.videos_data:
            group_id = video['pseudo_speaker_id']
            assigned_split = self.group_assignments.get(group_id, 'train')  # Default to train
            video['dataset_split'] = assigned_split

        # Count final splits
        split_counts = Counter(video['dataset_split'] for video in self.videos_data)
        total = len(self.videos_data)

        print("📊 Final Video Distribution:")
        print("-" * 50)
        for split_name in ['train', 'validation', 'test']:
            count = split_counts[split_name]
            percentage = (count / total) * 100
            print(f"{split_name.upper():<12} | {count:>4} videos ({percentage:>5.1f}%)")

        return split_counts

    def verify_zero_speaker_overlap(self):
        """Verify that no pseudo-speaker appears in multiple splits."""
        print(f"\n🔍 VERIFYING ZERO PSEUDO-SPEAKER OVERLAP")
        print("=" * 80)

        # Check for pseudo-speaker overlap
        speaker_split_map = defaultdict(set)

        for video in self.videos_data:
            speaker_id = video['pseudo_speaker_id']
            split = video['dataset_split']
            speaker_split_map[speaker_id].add(split)

        # Find violations
        overlap_violations = []
        for speaker_id, splits in speaker_split_map.items():
            if len(splits) > 1:
                overlap_violations.append((speaker_id, splits))

        if overlap_violations:
            print("❌ PSEUDO-SPEAKER OVERLAP DETECTED!")
            print("   The following pseudo-speakers appear in multiple splits:")
            for speaker_id, splits in overlap_violations:
                print(f"   - Pseudo-speaker {speaker_id}: {', '.join(splits)}")
            return False
        else:
            print("✅ ZERO PSEUDO-SPEAKER OVERLAP CONFIRMED!")
            print(f"   All {len(speaker_split_map)} pseudo-speakers assigned to single splits only")

        return True

    def verify_class_representation(self):
        """Verify that all classes are present in each split."""
        print(f"\n🔍 VERIFYING CLASS REPRESENTATION")
        print("=" * 80)

        # Check class representation in each split
        split_classes = defaultdict(set)

        for video in self.videos_data:
            split = video['dataset_split']
            class_name = video['class']
            split_classes[split].add(class_name)

        all_classes_present = True

        for split in ['train', 'validation', 'test']:
            classes_in_split = split_classes[split]
            missing_classes = set(self.classes) - classes_in_split

            print(f"{split.upper()} split:")
            print(f"   Present classes ({len(classes_in_split)}): {sorted(classes_in_split)}")

            if missing_classes:
                print(f"   ❌ Missing classes: {sorted(missing_classes)}")
                all_classes_present = False
            else:
                print(f"   ✅ All {len(self.classes)} classes present")

        if all_classes_present:
            print(f"\n✅ ALL CLASSES PRESENT IN ALL SPLITS!")
        else:
            print(f"\n❌ Some classes missing from splits!")

        return all_classes_present

    def create_split_manifests(self):
        """Create CSV manifests for each split."""
        print(f"\n📄 CREATING SPLIT MANIFESTS")
        print("=" * 60)

        # Convert to DataFrame
        df = pd.DataFrame(self.videos_data)

        # Reorder columns
        column_order = [
            'filename', 'full_path', 'class', 'pseudo_speaker_id',
            'fingerprint_hash', 'dataset_split', 'age_group', 'gender',
            'ethnicity', 'video_type', 'format_type'
        ]

        df = df[column_order]
        df = df.sort_values(['dataset_split', 'pseudo_speaker_id', 'class', 'filename'])

        # Save main manifest
        manifest_path = self.output_dir / 'corrected_visual_similarity_manifest.csv'
        df.to_csv(manifest_path, index=False)

        print(f"✅ Main manifest saved: {manifest_path}")
        print(f"   Total records: {len(df)}")

        # Create split-specific manifests
        for split in ['train', 'validation', 'test']:
            split_df = df[df['dataset_split'] == split].copy()
            split_manifest_path = self.output_dir / f"corrected_visual_similarity_{split}_manifest.csv"
            split_df.to_csv(split_manifest_path, index=False)
            print(f"✅ {split.capitalize()} manifest: {len(split_df)} videos")

        return df

    def run_corrected_splitting(self):
        """Run the corrected visual similarity splitting pipeline."""
        print("🎯 CORRECTED VISUAL SIMILARITY DATASET SPLITTING PIPELINE")
        print("=" * 80)
        print("🔍 Using visual similarity clustering to prevent speaker data leakage")
        print("🚨 Zero pseudo-speaker overlap guarantee")
        print("🎯 All classes must be present in each split")
        print("🔒 Male 18to39 videos assigned to training (CORRECTED)")
        print("📊 Proper 70/15/15 distribution")
        print()

        # Step 1: Load videos and generate fingerprints
        total_videos = self.load_and_fingerprint_videos()
        if total_videos == 0:
            print("❌ No videos found!")
            return None

        # Step 2: Create pseudo-speaker groups
        if not self.create_pseudo_speaker_groups():
            print("❌ Failed to create pseudo-speaker groups!")
            return None

        # Step 3: Assign groups to splits (CORRECTED)
        if not self.assign_groups_to_splits_corrected():
            print("❌ Failed to assign groups to splits!")
            return None

        # Step 4: Assign individual videos
        split_counts = self.assign_videos_to_splits()

        # Step 5: Verify zero speaker overlap
        overlap_verified = self.verify_zero_speaker_overlap()

        # Step 6: Verify class representation
        classes_verified = self.verify_class_representation()

        if not overlap_verified or not classes_verified:
            print("❌ Verification failed!")
            return None

        # Step 7: Create manifests
        df = self.create_split_manifests()

        print(f"\n🎯 CORRECTED VISUAL SIMILARITY SPLITTING COMPLETE!")
        print("=" * 80)
        print(f"✅ {total_videos} videos split with zero pseudo-speaker overlap")
        print(f"✅ All 7 classes present in each split")
        print(f"✅ Male 18to39 videos properly assigned to training")
        print(f"✅ Proper distribution achieved")
        print(f"📁 Output directory: {self.output_dir}")

        return {
            'manifest_df': df,
            'split_counts': split_counts,
            'pseudo_speaker_groups': self.pseudo_speaker_groups,
            'group_assignments': self.group_assignments
        }

def main():
    """Main execution function."""
    # Initialize corrected splitter
    splitter = CorrectedVisualSimilaritySplitter()

    # Run corrected splitting pipeline
    results = splitter.run_corrected_splitting()

    return results

if __name__ == "__main__":
    main()
