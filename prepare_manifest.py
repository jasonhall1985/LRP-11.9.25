#!/usr/bin/env python3
"""
Manifest Preparation for 7-Class Lip Reading Trainer
====================================================

Scans all data sources, builds comprehensive manifest.csv with demographic parsing,
video validation, and V3>V2>original prioritization logic.

Features:
- Handles spaces in folder names
- Prioritizes V3 > V2 > original processed videos
- Demographic parsing from ICU dataset naming convention
- Video integrity validation
- Duplicate detection and handling
- Comprehensive statistics and validation

Author: Production Lip Reading System
Date: 2025-09-15
"""

import os
import re
import cv2
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import json
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ManifestBuilder:
    """
    Comprehensive manifest builder with demographic parsing and video validation.
    """
    
    def __init__(self, processed_dir: str = "fixed_temporal_output/full_processed"):
        self.processed_dir = Path(processed_dir)
        
        # Class definitions
        self.class_names = [
            "help", "doctor", "glasses", "phone", "pillow", 
            "i_need_to_move", "my_mouth_is_dry"
        ]
        
        # Class variations for parsing
        self.class_variations = {
            "help": ["help", "HELP", "Help"],
            "doctor": ["doctor", "DOCTOR", "Doctor"],
            "glasses": ["glasses", "GLASSES", "Glasses"],
            "phone": ["phone", "PHONE", "Phone"],
            "pillow": ["pillow", "PILLOW", "Pillow"],
            "i_need_to_move": [
                "i_need_to_move", "I_NEED_TO_MOVE", 
                "i need to move", "I NEED TO MOVE",
                "ineedtomove", "INEEDTOMOVE"
            ],
            "my_mouth_is_dry": [
                "my_mouth_is_dry", "MY_MOUTH_IS_DRY",
                "my mouth is dry", "MY MOUTH IS DRY",
                "mymouthisdry", "MYMOUTHISDRY"
            ]
        }
        
        # Demographic parsing patterns
        self.gender_pattern = re.compile(r'\b(male|female|man|woman|m_|f_)\b', re.IGNORECASE)
        self.age_pattern = re.compile(r'\b(18[-_]?39|40[-_]?64|65\+|65plus)\b', re.IGNORECASE)
        self.ethnicity_pattern = re.compile(
            r'\b(caucasian|asian|african|hispanic|not_specified)\b', re.IGNORECASE
        )
        
        # Statistics
        self.stats = {
            'total_videos': 0,
            'valid_videos': 0,
            'invalid_videos': 0,
            'processed_videos': {'v3': 0, 'v2': 0, 'original': 0},
            'duplicates_found': 0,
            'classes': defaultdict(int),
            'demographics': defaultdict(int),
            'sources': defaultdict(int)
        }
        
    def scan_sources(self, sources: List[str], verify_videos: bool = True) -> pd.DataFrame:
        """
        Scan all data sources and build comprehensive manifest.
        """
        logger.info(f"Scanning {len(sources)} data sources...")
        
        all_videos = []
        
        for source_idx, source_path in enumerate(sources):
            source_path = Path(source_path)
            logger.info(f"Scanning source {source_idx + 1}/{len(sources)}: {source_path}")
            
            if not source_path.exists():
                logger.warning(f"Source path does not exist: {source_path}")
                continue
                
            # Find all video files
            video_files = self._find_video_files(source_path)
            logger.info(f"Found {len(video_files)} video files in {source_path.name}")
            
            # Process each video
            for video_path in tqdm(video_files, desc=f"Processing {source_path.name}"):
                video_info = self._process_video(video_path, source_path.name, verify_videos)
                if video_info:
                    all_videos.append(video_info)
                    
        # Create DataFrame
        df = pd.DataFrame(all_videos)
        
        if len(df) == 0:
            logger.error("No valid videos found in any source!")
            return df
            
        # Apply prioritization logic
        df = self._apply_prioritization(df)
        
        # Remove duplicates
        df = self._remove_duplicates(df)
        
        # Validate and clean data
        df = self._validate_and_clean(df)

        logger.info(f"Manifest after cleaning: {len(df)} videos")
        return df
        
    def _find_video_files(self, source_path: Path) -> List[Path]:
        """Find all video files in source directory."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(source_path.rglob(f"*{ext}"))
            
        return video_files
        
    def _process_video(self, video_path: Path, source_name: str, verify: bool) -> Optional[Dict[str, Any]]:
        """Process a single video file."""
        try:
            # Check for processed version first (V3 > V2 > original)
            processed_version, actual_path = self._find_processed_version(video_path)
            
            # Extract class from filename/path
            class_name = self._extract_class(actual_path)
            if not class_name:
                logger.debug(f"Could not extract class from: {actual_path}")
                return None
                
            # Parse demographics
            demographics = self._parse_demographics(actual_path)
            
            # Get video properties
            if verify:
                video_props = self._get_video_properties(actual_path)
                if not video_props:
                    logger.debug(f"Could not read video properties: {actual_path}")
                    self.stats['invalid_videos'] += 1
                    return None
            else:
                video_props = {
                    'frames': -1, 'fps': -1, 'width': -1, 
                    'height': -1, 'duration': -1
                }
                
            # Update statistics
            self.stats['total_videos'] += 1
            self.stats['valid_videos'] += 1
            self.stats['classes'][class_name] += 1
            self.stats['sources'][source_name] += 1
            self.stats['processed_videos'][processed_version] += 1
            
            demo_key = f"{demographics['gender']}_{demographics['age_band']}_{demographics['ethnicity']}"
            self.stats['demographics'][demo_key] += 1
            
            return {
                'path': str(actual_path),
                'original_path': str(video_path),
                'class': class_name,
                'gender': demographics['gender'],
                'age_band': demographics['age_band'],
                'ethnicity': demographics['ethnicity'],
                'source': source_name,
                'processed_version': processed_version,
                'frames': video_props['frames'],
                'fps': video_props['fps'],
                'width': video_props['width'],
                'height': video_props['height'],
                'duration': video_props['duration']
            }
            
        except Exception as e:
            logger.debug(f"Error processing {video_path}: {e}")
            self.stats['invalid_videos'] += 1
            return None
            
    def _find_processed_version(self, video_path: Path) -> Tuple[str, Path]:
        """Find the best processed version (V3 > V2 > original)."""
        if not self.processed_dir.exists():
            return "original", video_path
            
        base_name = video_path.stem
        
        # Check for V3 version
        v3_pattern = f"processed_v3_{base_name}.mp4"
        v3_path = self.processed_dir / v3_pattern
        if v3_path.exists():
            return "v3", v3_path
            
        # Check for V2 version
        v2_pattern = f"processed_v2_{base_name}.mp4"
        v2_path = self.processed_dir / v2_pattern
        if v2_path.exists():
            return "v2", v2_path
            
        # Check for generic processed version
        processed_pattern = f"processed_{base_name}.mp4"
        processed_path = self.processed_dir / processed_pattern
        if processed_path.exists():
            return "v2", processed_path
            
        return "original", video_path
        
    def _extract_class(self, video_path: Path) -> Optional[str]:
        """Extract class name from video path."""
        path_str = str(video_path).lower()
        filename = video_path.stem.lower()
        
        # Try ICU dataset naming convention first
        if "__" in filename:
            parts = filename.split("__")
            if len(parts) >= 1:
                potential_class = parts[0].replace("processed_v3_", "").replace("processed_v2_", "").replace("processed_", "")
                for class_name, variations in self.class_variations.items():
                    if any(var.lower() == potential_class for var in variations):
                        return class_name
                        
        # Try folder-based class extraction
        for part in video_path.parts:
            part_lower = part.lower()
            for class_name, variations in self.class_variations.items():
                if any(var.lower() in part_lower for var in variations):
                    return class_name
                    
        # Try filename-based extraction
        for class_name, variations in self.class_variations.items():
            if any(var.lower() in filename for var in variations):
                return class_name
                
        return None
        
    def _parse_demographics(self, video_path: Path) -> Dict[str, str]:
        """Parse demographic information from video path."""
        path_str = str(video_path)
        filename = video_path.stem
        
        # Initialize with defaults
        demographics = {
            'gender': 'unknown',
            'age_band': 'unknown', 
            'ethnicity': 'unknown'
        }
        
        # Try ICU dataset naming convention first
        if "__" in filename:
            parts = filename.split("__")
            if len(parts) >= 5:
                try:
                    # Format: class__user__age__gender__ethnicity__timestamp
                    demographics['age_band'] = self._normalize_age(parts[2])
                    demographics['gender'] = self._normalize_gender(parts[3])
                    demographics['ethnicity'] = self._normalize_ethnicity(parts[4])
                    return demographics
                except:
                    pass
                    
        # Fallback to regex parsing
        gender_match = self.gender_pattern.search(path_str)
        if gender_match:
            demographics['gender'] = self._normalize_gender(gender_match.group(1))
            
        age_match = self.age_pattern.search(path_str)
        if age_match:
            demographics['age_band'] = self._normalize_age(age_match.group(1))
            
        ethnicity_match = self.ethnicity_pattern.search(path_str)
        if ethnicity_match:
            demographics['ethnicity'] = self._normalize_ethnicity(ethnicity_match.group(1))
            
        return demographics
        
    def _normalize_gender(self, gender_str: str) -> str:
        """Normalize gender string."""
        gender_lower = gender_str.lower()
        if any(g in gender_lower for g in ['male', 'man', 'm_']):
            return 'male'
        elif any(g in gender_lower for g in ['female', 'woman', 'f_']):
            return 'female'
        return 'unknown'
        
    def _normalize_age(self, age_str: str) -> str:
        """Normalize age band string."""
        age_lower = age_str.lower().replace('-', '_').replace(' ', '_')
        if '18' in age_lower and '39' in age_lower:
            return '18-39'
        elif '40' in age_lower and '64' in age_lower:
            return '40-64'
        elif '65' in age_lower:
            return '65+'
        return 'unknown'
        
    def _normalize_ethnicity(self, ethnicity_str: str) -> str:
        """Normalize ethnicity string."""
        ethnicity_lower = ethnicity_str.lower()
        if 'caucasian' in ethnicity_lower:
            return 'caucasian'
        elif 'asian' in ethnicity_lower:
            return 'asian'
        elif 'african' in ethnicity_lower:
            return 'african'
        elif 'hispanic' in ethnicity_lower:
            return 'hispanic'
        elif 'not_specified' in ethnicity_lower:
            return 'not_specified'
        return 'unknown'
        
    def _get_video_properties(self, video_path: Path) -> Optional[Dict[str, Any]]:
        """Get video properties using OpenCV."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
                
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frames / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'frames': frames,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration
            }
            
        except Exception as e:
            logger.debug(f"Error reading video properties for {video_path}: {e}")
            return None
            
    def _apply_prioritization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply V3 > V2 > original prioritization logic."""
        logger.info("Applying prioritization logic (V3 > V2 > original)...")
        
        # Group by original path to handle duplicates
        grouped = df.groupby('original_path')
        
        prioritized_videos = []
        for original_path, group in grouped:
            # Sort by priority: v3 > v2 > original
            priority_order = {'v3': 0, 'v2': 1, 'original': 2}
            group_sorted = group.sort_values(
                key=lambda x: x['processed_version'].map(priority_order)
            )
            
            # Take the highest priority version
            best_version = group_sorted.iloc[0]
            prioritized_videos.append(best_version)
            
        return pd.DataFrame(prioritized_videos).reset_index(drop=True)
        
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate videos based on content similarity."""
        logger.info("Removing duplicates...")
        
        initial_count = len(df)
        
        # Remove exact path duplicates
        df = df.drop_duplicates(subset=['path']).reset_index(drop=True)
        
        # Remove duplicates based on class + demographics + similar duration
        df['duration_rounded'] = df['duration'].round(1)
        df = df.drop_duplicates(
            subset=['class', 'gender', 'age_band', 'ethnicity', 'duration_rounded']
        ).reset_index(drop=True)
        
        df = df.drop(columns=['duration_rounded'])
        
        duplicates_removed = initial_count - len(df)
        self.stats['duplicates_found'] = duplicates_removed
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate videos")
            
        return df
        
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the manifest data."""
        logger.info("Validating and cleaning manifest data...")

        initial_count = len(df)

        # Remove videos with unknown class
        df = df[df['class'] != 'unknown'].reset_index(drop=True)

        # Remove videos with invalid dimensions (if verified)
        if 'width' in df.columns:
            df = df[(df['width'] > 0) | (df['width'] == -1)].reset_index(drop=True)
            df = df[(df['height'] > 0) | (df['height'] == -1)].reset_index(drop=True)

        # Remove videos with zero duration (if verified)
        if 'duration' in df.columns:
            df = df[(df['duration'] > 0) | (df['duration'] == -1)].reset_index(drop=True)

        cleaned_count = initial_count - len(df)
        if cleaned_count > 0:
            logger.info(f"Removed {cleaned_count} invalid videos during cleaning")

        return df

    def balance_classes_by_duplication(self, df: pd.DataFrame, balance_source_demo: str = "gender=male,age_band=18-39") -> pd.DataFrame:
        """
        Balance classes by duplicating videos from specified demographic group.

        Args:
            df: Input DataFrame
            balance_source_demo: Demographic criteria for source videos (format: "key=value,key=value")

        Returns:
            Balanced DataFrame with duplicated videos
        """
        logger.info("Balancing classes by duplication...")

        # Parse balance source demographic criteria
        balance_criteria = {}
        for criterion in balance_source_demo.split(','):
            key, value = criterion.strip().split('=')
            balance_criteria[key.strip()] = value.strip()

        logger.info(f"Using balance source demographic: {balance_criteria}")

        # Get class distribution
        class_counts = df['class'].value_counts()
        max_count = class_counts.max()

        logger.info(f"Current class distribution:")
        for class_name in self.class_names:
            count = class_counts.get(class_name, 0)
            logger.info(f"  {class_name}: {count}")
        logger.info(f"Target count per class: {max_count}")

        # Find source videos for duplication (male 18-39)
        source_mask = pd.Series([True] * len(df))
        for key, value in balance_criteria.items():
            if key in df.columns:
                source_mask &= (df[key] == value)

        source_videos = df[source_mask].copy()
        logger.info(f"Found {len(source_videos)} source videos for duplication from {balance_criteria}")

        if len(source_videos) == 0:
            logger.warning("No source videos found for duplication! Skipping class balancing.")
            return df

        # Create duplicates for each class that needs balancing
        duplicated_videos = []
        duplication_stats = {}

        for class_name in self.class_names:
            current_count = class_counts.get(class_name, 0)
            needed_count = max_count - current_count

            if needed_count <= 0:
                duplication_stats[class_name] = 0
                continue

            # Get source videos for this class
            class_source_videos = source_videos[source_videos['class'] == class_name]

            if len(class_source_videos) == 0:
                logger.warning(f"No source videos found for class '{class_name}' in demographic {balance_criteria}")
                duplication_stats[class_name] = 0
                continue

            # Duplicate videos to reach target count
            duplicates_created = 0
            for i in range(needed_count):
                # Cycle through available source videos
                source_idx = i % len(class_source_videos)
                source_video = class_source_videos.iloc[source_idx].copy()

                # Modify path to indicate it's a duplicate
                original_path = Path(source_video['path'])
                duplicate_path = original_path.parent / f"duplicate_{i+1}_{original_path.name}"
                source_video['path'] = str(duplicate_path)
                source_video['original_path'] = source_video['original_path']  # Keep original reference
                source_video['is_duplicate'] = True
                source_video['duplicate_id'] = i + 1
                source_video['duplicate_source'] = str(original_path)

                duplicated_videos.append(source_video)
                duplicates_created += 1

            duplication_stats[class_name] = duplicates_created

        # Add duplicated videos to DataFrame
        if duplicated_videos:
            duplicates_df = pd.DataFrame(duplicated_videos)
            # Add duplicate flags to original DataFrame
            df['is_duplicate'] = False
            df['duplicate_id'] = 0
            df['duplicate_source'] = ''

            # Combine original and duplicated videos
            balanced_df = pd.concat([df, duplicates_df], ignore_index=True)
        else:
            # Add duplicate flags even if no duplicates created
            df['is_duplicate'] = False
            df['duplicate_id'] = 0
            df['duplicate_source'] = ''
            balanced_df = df

        # Update statistics
        total_duplicates = sum(duplication_stats.values())
        self.stats['duplicates_created'] = total_duplicates
        self.stats['duplication_stats'] = duplication_stats

        logger.info(f"Class balancing completed:")
        logger.info(f"  Total duplicates created: {total_duplicates}")
        for class_name, count in duplication_stats.items():
            logger.info(f"  {class_name}: +{count} duplicates")

        # Verify final class distribution
        final_class_counts = balanced_df['class'].value_counts()
        logger.info(f"Final class distribution:")
        for class_name in self.class_names:
            count = final_class_counts.get(class_name, 0)
            logger.info(f"  {class_name}: {count}")

        return balanced_df
        
    def print_statistics(self, df: pd.DataFrame):
        """Print comprehensive statistics."""
        logger.info("\n" + "="*80)
        logger.info("MANIFEST STATISTICS")
        logger.info("="*80)
        
        # Overall statistics
        logger.info(f"\n📊 Overall Statistics:")
        logger.info(f"  Total videos found: {self.stats['total_videos']}")
        logger.info(f"  Valid videos: {self.stats['valid_videos']}")
        logger.info(f"  Invalid videos: {self.stats['invalid_videos']}")
        logger.info(f"  Final manifest size: {len(df)}")
        logger.info(f"  Duplicates removed: {self.stats['duplicates_found']}")

        # Class balancing information
        if 'duplicates_created' in self.stats:
            logger.info(f"  Duplicates created for balancing: {self.stats['duplicates_created']}")
            if 'duplication_stats' in self.stats:
                logger.info(f"  Duplication breakdown:")
                for class_name, count in self.stats['duplication_stats'].items():
                    if count > 0:
                        logger.info(f"    {class_name}: +{count}")

        # Processed versions
        logger.info(f"\n🔄 Processed Versions:")
        for version, count in self.stats['processed_videos'].items():
            logger.info(f"  {version.upper()}: {count}")
            
        # Class distribution
        logger.info(f"\n📋 Class Distribution:")
        class_counts = df['class'].value_counts()
        for class_name in self.class_names:
            count = class_counts.get(class_name, 0)
            logger.info(f"  {class_name}: {count}")
            
        # Demographic distribution
        logger.info(f"\n👥 Demographic Distribution:")
        logger.info(f"  Gender: {dict(df['gender'].value_counts())}")
        logger.info(f"  Age Band: {dict(df['age_band'].value_counts())}")
        logger.info(f"  Ethnicity: {dict(df['ethnicity'].value_counts())}")
        
        # Source distribution
        logger.info(f"\n📁 Source Distribution:")
        for source, count in df['source'].value_counts().items():
            logger.info(f"  {source}: {count}")
            
        logger.info("="*80)


def main():
    """Main function for manifest preparation."""
    parser = argparse.ArgumentParser(description="Prepare manifest for lip reading training")
    
    parser.add_argument(
        '--sources', 
        nargs='+', 
        required=True,
        help='Data source directories'
    )
    parser.add_argument(
        '--processed_dir',
        default='fixed_temporal_output/full_processed',
        help='Directory containing processed videos'
    )
    parser.add_argument(
        '--out',
        default='manifest.csv',
        help='Output manifest file'
    )
    parser.add_argument(
        '--val_holdout',
        default='gender=male,age_band=40-64',
        help='Validation holdout criteria'
    )
    parser.add_argument(
        '--test_holdout', 
        default='gender=female,age_band=18-39',
        help='Test holdout criteria'
    )
    parser.add_argument(
        '--show_stats',
        action='store_true',
        help='Show detailed statistics'
    )
    parser.add_argument(
        '--verify_videos',
        action='store_true',
        help='Verify video integrity (slower)'
    )
    parser.add_argument(
        '--balance_classes',
        action='store_true',
        help='Balance classes by duplicating videos from male 18-39 demographic'
    )
    parser.add_argument(
        '--balance_source_demo',
        default='gender=male,age_band=18-39',
        help='Demographic criteria for source videos for balancing'
    )

    args = parser.parse_args()

    # Build manifest
    builder = ManifestBuilder(args.processed_dir)
    df = builder.scan_sources(args.sources, args.verify_videos)

    if len(df) == 0:
        logger.error("No valid videos found!")
        return

    # Balance classes if requested
    if args.balance_classes:
        df = builder.balance_classes_by_duplication(df, args.balance_source_demo)
        logger.info(f"Final manifest after balancing: {len(df)} videos")
        
    # Save manifest
    df.to_csv(args.out, index=False)
    logger.info(f"Manifest saved to: {args.out}")
    
    # Show statistics
    if args.show_stats:
        builder.print_statistics(df)
        
    # Save statistics
    stats_file = args.out.replace('.csv', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(builder.stats, f, indent=2, default=str)
    logger.info(f"Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
