#!/usr/bin/env python3
"""
Comprehensive manifest creator for proper training data structure.
Creates manifest from multiple dataset folders with different naming conventions.
"""

import os
import pandas as pd
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_class_from_filename(filename):
    """Extract class from various filename formats."""
    filename_lower = filename.lower()

    # Check for exact class matches
    if "doctor" in filename_lower:
        return "doctor"
    elif "help" in filename_lower:
        return "help"
    elif "glasses" in filename_lower:
        return "glasses"
    elif "phone" in filename_lower:
        return "phone"
    elif "pillow" in filename_lower:
        return "pillow"
    elif "my_mouth_is_dry" in filename_lower or "mouth_is_dry" in filename_lower:
        return "my_mouth_is_dry"
    elif "water" in filename_lower:
        return "water"
    elif "i_need_to_move" in filename_lower or "need_to_move" in filename_lower:
        return "i_need_to_move"

    return "unknown"

def extract_demographics_from_filename(filename):
    """Extract demographics from filename with various formats."""
    gender = "unknown"
    age_band = "unknown"
    ethnicity = "unknown"

    # Extract gender
    if "__male__" in filename or "_male_" in filename:
        gender = "male"
    elif "__female__" in filename or "_female_" in filename:
        gender = "female"

    # Extract age band
    if "__18to39__" in filename or "_18to39_" in filename:
        age_band = "18-39"
    elif "__40to64__" in filename or "_40to64_" in filename:
        age_band = "40-64"
    elif "__65plus__" in filename or "_65plus_" in filename:
        age_band = "65+"

    # Extract ethnicity
    if "__caucasian__" in filename or "_caucasian_" in filename:
        ethnicity = "caucasian"
    elif "__asian__" in filename or "_asian_" in filename:
        ethnicity = "asian"
    elif "__aboriginal__" in filename or "_aboriginal_" in filename:
        ethnicity = "aboriginal"
    elif "__african__" in filename or "_african_" in filename:
        ethnicity = "african"
    elif "__not_specified__" in filename or "_not_specified_" in filename:
        ethnicity = "not_specified"

    return gender, age_band, ethnicity

def process_folder(folder_path, source_name):
    """Process a single folder and extract video information."""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        logger.warning(f"Folder not found: {folder_path}")
        return []

    video_files = list(folder_path.glob("*.mp4"))
    logger.info(f"Found {len(video_files)} video files in {source_name}")

    manifest_data = []

    for video_path in video_files:
        filename = video_path.name

        # Extract class
        class_name = extract_class_from_filename(filename)

        # Extract demographics
        gender, age_band, ethnicity = extract_demographics_from_filename(filename)

        manifest_data.append({
            'path': str(video_path),
            'class': class_name,
            'gender': gender,
            'age_band': age_band,
            'ethnicity': ethnicity,
            'source': source_name,
            'processed_version': 'cropped'
        })

    return manifest_data

def create_comprehensive_manifest():
    """Create comprehensive manifest from all dataset folders."""

    # Define dataset folders
    dataset_folders = [
        {
            'path': '/Users/client/Desktop/13.9.25top7dataset_cropped',
            'name': '13.9.25top7dataset_cropped',
            'expected_count': 1502
        },
        {
            'path': '/Users/client/Desktop/TRAINING SET 2.9.25',
            'name': 'TRAINING_SET_2.9.25',
            'expected_count': None
        }
    ]

    all_manifest_data = []

    # Process each folder
    for folder_info in dataset_folders:
        folder_data = process_folder(folder_info['path'], folder_info['name'])
        all_manifest_data.extend(folder_data)

        if folder_info['expected_count']:
            actual_count = len(folder_data)
            if actual_count != folder_info['expected_count']:
                logger.warning(f"Expected {folder_info['expected_count']} videos in {folder_info['name']}, found {actual_count}")
            else:
                logger.info(f"✅ Verified {actual_count} videos in {folder_info['name']}")

    if not all_manifest_data:
        logger.error("No video files found in any dataset folder")
        return

    # Create DataFrame
    df = pd.DataFrame(all_manifest_data)

    # Save manifest
    output_path = "comprehensive_manifest.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Created comprehensive manifest with {len(df)} videos: {output_path}")

    # Show detailed stats
    logger.info("=== DATASET STATISTICS ===")
    logger.info(f"Total videos: {len(df)}")
    logger.info(f"Sources: {df['source'].value_counts().to_dict()}")
    logger.info(f"Classes: {df['class'].value_counts().to_dict()}")
    logger.info(f"Gender: {df['gender'].value_counts().to_dict()}")
    logger.info(f"Age bands: {df['age_band'].value_counts().to_dict()}")
    logger.info(f"Ethnicity: {df['ethnicity'].value_counts().to_dict()}")

    return output_path

if __name__ == "__main__":
    create_comprehensive_manifest()
