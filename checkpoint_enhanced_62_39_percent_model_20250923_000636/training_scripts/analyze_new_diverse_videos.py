#!/usr/bin/env python3
"""
Analyze New Diverse Videos from Multiple Demographic Groups
Comprehensive analysis of data/extra videos 22.9.25/extra videos 11pm directory
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import re

def normalize_class_name(raw_class_name):
    """Normalize class name variations to standard format"""
    if not raw_class_name:
        return None
    
    # Convert to lowercase and replace spaces with underscores
    normalized = raw_class_name.lower().replace(' ', '_')
    
    # Handle specific variations
    class_mappings = {
        'my_mouth_is_dry': 'my_mouth_is_dry',
        'i_need_to_move': 'i_need_to_move',
        'doctor': 'doctor',
        'pillow': 'pillow',
        # Handle variations
        'my mouth is dry': 'my_mouth_is_dry',
        'i need to move': 'i_need_to_move',
    }
    
    return class_mappings.get(normalized, normalized)

def extract_class_from_filename(filename):
    """Extract and normalize class name from video filename"""
    filename = os.path.basename(filename)
    filename_lower = filename.lower()

    # Try various patterns to extract class name
    potential_class = None

    # Pattern 1: Standard underscore format
    if filename.startswith('pillow_'):
        potential_class = 'pillow'
    elif filename.startswith('doctor_'):
        potential_class = 'doctor'
    elif filename.startswith('i_need_to_move_'):
        potential_class = 'i_need_to_move'
    elif filename.startswith('my_mouth_is_dry_'):
        potential_class = 'my_mouth_is_dry'

    # Pattern 2: Capital I variations (I_need_to_move)
    elif filename.startswith('I_need_to_move_') or filename_lower.startswith('i_need_to_move_'):
        potential_class = 'i_need_to_move'

    # Pattern 3: Space-separated format
    elif 'my mouth is dry' in filename_lower:
        potential_class = 'my mouth is dry'
    elif 'i need to move' in filename_lower:
        potential_class = 'i need to move'

    # Pattern 4: Mixed case variations
    elif filename_lower.startswith('my mouth is dry'):
        potential_class = 'my mouth is dry'
    elif filename_lower.startswith('i need to move'):
        potential_class = 'i need to move'

    # Pattern 5: Check if filename contains class keywords anywhere
    if not potential_class:
        if 'pillow' in filename_lower:
            potential_class = 'pillow'
        elif 'doctor' in filename_lower:
            potential_class = 'doctor'
        elif 'i_need_to_move' in filename_lower or 'i need to move' in filename_lower:
            potential_class = 'i_need_to_move'
        elif 'my_mouth_is_dry' in filename_lower or 'my mouth is dry' in filename_lower:
            potential_class = 'my_mouth_is_dry'

    # Normalize the extracted class name
    if potential_class:
        normalized_class = normalize_class_name(potential_class)
        if normalized_class in ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']:
            return normalized_class

    return None

def extract_demographic_info(filename):
    """Extract demographic information from filename - diverse speakers expected"""
    # Default values
    age_group = 'unknown'
    gender = 'unknown'
    ethnicity = 'unknown'
    
    filename_lower = filename.lower()
    
    # Extract gender
    if 'female' in filename_lower:
        gender = 'female'
    elif 'male' in filename_lower:
        gender = 'male'
    
    # Extract age group - expanded patterns for diverse speakers
    if '18-39' in filename_lower or '18-29' in filename_lower or '30-39' in filename_lower:
        age_group = '18to39'
    elif '40-64' in filename_lower or '40-49' in filename_lower or '50-64' in filename_lower:
        age_group = '40to64'
    elif '65plus' in filename_lower or '65+' in filename_lower or 'senior' in filename_lower:
        age_group = '65plus'
    elif 'young' in filename_lower or 'teen' in filename_lower:
        age_group = '18to39'  # Map young/teen to 18to39
    elif 'middle' in filename_lower or 'mid' in filename_lower:
        age_group = '40to64'  # Map middle-aged to 40to64
    elif 'old' in filename_lower or 'elder' in filename_lower:
        age_group = '65plus'  # Map older terms to 65plus
    
    # Extract ethnicity - expanded patterns for diverse speakers
    if 'caucasian' in filename_lower or 'white' in filename_lower or 'european' in filename_lower:
        ethnicity = 'caucasian'
    elif 'asian' in filename_lower or 'chinese' in filename_lower or 'japanese' in filename_lower or 'korean' in filename_lower:
        ethnicity = 'asian'
    elif 'aboriginal' in filename_lower or 'indigenous' in filename_lower or 'native' in filename_lower:
        ethnicity = 'aboriginal'
    elif 'african' in filename_lower or 'black' in filename_lower:
        ethnicity = 'african'
    elif 'hispanic' in filename_lower or 'latino' in filename_lower or 'latina' in filename_lower:
        ethnicity = 'hispanic'
    elif 'indian' in filename_lower or 'south_asian' in filename_lower:
        ethnicity = 'south_asian'
    elif 'middle_eastern' in filename_lower or 'arab' in filename_lower:
        ethnicity = 'middle_eastern'
    
    return age_group, gender, ethnicity

def analyze_new_video_directory():
    """Analyze the new diverse video directory"""
    print("🔍 ANALYZING NEW DIVERSE VIDEOS FROM MULTIPLE DEMOGRAPHIC GROUPS")
    print("=" * 80)
    print("Directory: data/extra videos 22.9.25/extra videos 11pm")
    print("Expected: Diverse speakers across age groups, genders, and ethnicities")
    
    # Check if directory exists
    new_video_dir = "data/extra videos 22.9.25/extra videos 11pm"
    if not os.path.exists(new_video_dir):
        print(f"❌ Directory not found: {new_video_dir}")
        return None
    
    # Find all video files
    video_files = glob.glob(os.path.join(new_video_dir, "*.mp4"))
    print(f"\n📁 Found {len(video_files)} MP4 files")
    
    if len(video_files) == 0:
        print("❌ No MP4 files found in directory")
        return None
    
    # Analyze each video
    video_analysis = []
    class_counts = defaultdict(int)
    demographic_counts = defaultdict(int)
    age_counts = defaultdict(int)
    gender_counts = defaultdict(int)
    ethnicity_counts = defaultdict(int)
    
    print(f"\n📋 ANALYZING INDIVIDUAL VIDEO FILES:")
    for video_path in video_files:
        filename = os.path.basename(video_path)
        
        # Extract class and demographic info
        class_name = extract_class_from_filename(filename)
        age_group, gender, ethnicity = extract_demographic_info(filename)
        
        if class_name:
            class_counts[class_name] += 1
            demographic_group = f"{age_group}_{gender}_{ethnicity}"
            demographic_counts[demographic_group] += 1
            age_counts[age_group] += 1
            gender_counts[gender] += 1
            ethnicity_counts[ethnicity] += 1
            
            video_analysis.append({
                'filename': filename,
                'class': class_name,
                'age_group': age_group,
                'gender': gender,
                'ethnicity': ethnicity,
                'demographic_group': demographic_group
            })
            
            print(f"  ✅ {filename}")
            print(f"      Class: {class_name}, Age: {age_group}, Gender: {gender}, Ethnicity: {ethnicity}")
        else:
            print(f"  ❌ Could not extract class from: {filename}")
    
    # Summary statistics
    print(f"\n📊 NEW DIVERSE VIDEOS BY CLASS:")
    target_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    total_new_videos = 0
    for class_name in sorted(target_classes):
        count = class_counts[class_name]
        total_new_videos += count
        print(f"  {class_name}: {count} videos")
    
    print(f"\n🌍 DEMOGRAPHIC DIVERSITY ANALYSIS:")
    print(f"  Total new videos: {total_new_videos}")
    print(f"  Unique demographic groups: {len(demographic_counts)}")
    
    print(f"\n📊 AGE GROUP DISTRIBUTION:")
    for age, count in sorted(age_counts.items()):
        percentage = (count / total_new_videos) * 100 if total_new_videos > 0 else 0
        print(f"  {age}: {count} videos ({percentage:.1f}%)")
    
    print(f"\n📊 GENDER DISTRIBUTION:")
    for gender, count in sorted(gender_counts.items()):
        percentage = (count / total_new_videos) * 100 if total_new_videos > 0 else 0
        print(f"  {gender}: {count} videos ({percentage:.1f}%)")
    
    print(f"\n📊 ETHNICITY DISTRIBUTION:")
    for ethnicity, count in sorted(ethnicity_counts.items()):
        percentage = (count / total_new_videos) * 100 if total_new_videos > 0 else 0
        print(f"  {ethnicity}: {count} videos ({percentage:.1f}%)")
    
    print(f"\n📊 TOP DEMOGRAPHIC GROUPS:")
    sorted_demographics = sorted(demographic_counts.items(), key=lambda x: x[1], reverse=True)
    for demo, count in sorted_demographics[:10]:
        percentage = (count / total_new_videos) * 100 if total_new_videos > 0 else 0
        print(f"  {demo}: {count} videos ({percentage:.1f}%)")
    
    return {
        'video_analysis': video_analysis,
        'class_counts': dict(class_counts),
        'demographic_counts': dict(demographic_counts),
        'age_counts': dict(age_counts),
        'gender_counts': dict(gender_counts),
        'ethnicity_counts': dict(ethnicity_counts),
        'total_new_videos': total_new_videos,
        'unique_demographics': len(demographic_counts)
    }

def analyze_existing_dataset():
    """Analyze existing processed videos for comparison"""
    print(f"\n🔄 ANALYZING EXISTING PROCESSED DATASET FOR COMPARISON")
    print("=" * 60)
    
    existing_dir = "data/the_best_videos_so_far"
    video_files = glob.glob(os.path.join(existing_dir, "*.mp4"))
    
    print(f"📁 Found {len(video_files)} existing processed videos")
    
    # Count by class (exclude augmented)
    existing_class_counts = defaultdict(int)
    target_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        
        # Skip augmented videos
        if 'augmented' in filename:
            continue
        
        # Extract class from processed filename
        for class_name in target_classes:
            if filename.startswith(class_name + '__') or filename.startswith(class_name + '_'):
                existing_class_counts[class_name] += 1
                break
    
    print(f"\n📊 EXISTING PROCESSED VIDEOS BY CLASS:")
    total_existing = 0
    for class_name in sorted(target_classes):
        count = existing_class_counts[class_name]
        total_existing += count
        print(f"  {class_name}: {count} videos")
    
    print(f"  Total existing: {total_existing} videos")
    
    return dict(existing_class_counts), total_existing

def project_expanded_dataset_size(new_analysis, existing_counts):
    """Project the expanded dataset size after integration"""
    print(f"\n🎯 PROJECTING EXPANDED DATASET SIZE")
    print("=" * 50)
    
    target_classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    projected_counts = {}
    
    print(f"📊 PROJECTED TOTALS AFTER INTEGRATION:")
    for class_name in target_classes:
        existing = existing_counts.get(class_name, 0)
        new = new_analysis['class_counts'].get(class_name, 0)
        total = existing + new
        projected_counts[class_name] = total
        
        improvement = f"(+{new})" if new > 0 else "(+0)"
        print(f"  {class_name}: {total} videos {improvement}")
    
    # Find maximum class count for balancing
    max_count = max(projected_counts.values())
    min_count = min(projected_counts.values())
    
    print(f"\n🎯 BALANCING ANALYSIS:")
    print(f"  Maximum class count: {max_count}")
    print(f"  Minimum class count: {min_count}")
    print(f"  Recommended balance target: {max_count} videos per class")
    print(f"  Total balanced dataset size: {max_count * 4} videos")
    
    # Compare with previous 85-per-class dataset
    previous_size = 85 * 4  # 340 videos
    improvement = (max_count * 4) - previous_size
    improvement_pct = (improvement / previous_size) * 100 if previous_size > 0 else 0
    
    print(f"\n📈 IMPROVEMENT VS. PREVIOUS 85-PER-CLASS DATASET:")
    print(f"  Previous dataset: {previous_size} videos (85 per class)")
    print(f"  Projected dataset: {max_count * 4} videos ({max_count} per class)")
    print(f"  Improvement: +{improvement} videos ({improvement_pct:+.1f}%)")
    
    return projected_counts, max_count

def main():
    """Execute comprehensive analysis of new diverse videos"""
    print("🎯 COMPREHENSIVE ANALYSIS OF NEW DIVERSE VIDEOS")
    print("=" * 80)
    print("Objective: Analyze diverse speakers across multiple demographic groups")
    print("Target: 100+ videos per class for enhanced balanced training")
    
    # Analyze new diverse videos
    new_analysis = analyze_new_video_directory()
    
    if not new_analysis:
        print("❌ Failed to analyze new video directory")
        return False
    
    # Analyze existing dataset
    existing_counts, total_existing = analyze_existing_dataset()
    
    # Project expanded dataset size
    projected_counts, max_balance_target = project_expanded_dataset_size(new_analysis, existing_counts)
    
    # Save analysis results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'timestamp': timestamp,
        'new_video_analysis': new_analysis,
        'existing_counts': existing_counts,
        'projected_counts': projected_counts,
        'max_balance_target': max_balance_target,
        'diversity_metrics': {
            'unique_demographics': new_analysis['unique_demographics'],
            'age_groups': len(new_analysis['age_counts']),
            'genders': len(new_analysis['gender_counts']),
            'ethnicities': len(new_analysis['ethnicity_counts'])
        }
    }
    
    # Save to file
    import json
    results_path = f"new_diverse_videos_analysis_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Analysis results saved: {results_path}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 ANALYSIS COMPLETE - NEW DIVERSE VIDEOS INTEGRATION")
    print(f"🎉 New diverse videos: {new_analysis['total_new_videos']}")
    print(f"🌍 Demographic diversity: {new_analysis['unique_demographics']} unique groups")
    print(f"🎯 Projected balance target: {max_balance_target} videos per class")
    print(f"📈 Dataset expansion: {((max_balance_target * 4) - 340) / 340 * 100:+.1f}% vs. previous 85-per-class")
    
    if max_balance_target >= 100:
        print("✅ Target of 100+ videos per class achievable!")
    else:
        print(f"⚠️  Target of 100+ videos per class not quite reached (max: {max_balance_target})")
    
    print("🚀 Ready for Phase 1b: Video Preprocessing")
    
    return True, results_path

if __name__ == "__main__":
    success, results_path = main()
    exit(0 if success else 1)
