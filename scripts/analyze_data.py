"""
Data Analysis and Validation Script for ICU Lip Reading Dataset.
Analyzes class distribution, validates data integrity, and checks for data leakage.
"""

import os
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import logging
from typing import Dict, List, Tuple, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ICUDataAnalyzer:
    """Comprehensive data analysis for ICU lip reading dataset."""
    
    def __init__(self, 
                 train_dir: str = "/Users/client/Desktop/TRAINING SET 2.9.25",
                 val_dir: str = "/Users/client/Desktop/VAL SET", 
                 test_dir: str = "/Users/client/Desktop/TEST SET"):
        """
        Initialize data analyzer.
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory  
            test_dir: Path to test data directory
        """
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.test_dir = Path(test_dir)
        
        # Target classes for ICU communication
        self.target_classes = {"doctor", "glasses", "phone", "pillow", "help"}
        
        # Data storage
        self.data_info = {
            'train': [],
            'val': [],
            'test': []
        }
        
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def extract_label_from_filename(self, filename: str) -> str:
        """Extract class label from filename."""
        filename_lower = filename.lower()
        for target_class in self.target_classes:
            if filename_lower.startswith(target_class):
                return target_class
        return "unknown"
    
    def get_video_info(self, video_path: Path) -> Dict:
        """Extract video metadata."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            return {
                'frame_count': frame_count,
                'fps': fps,
                'duration': duration,
                'width': width,
                'height': height
            }
        except Exception as e:
            logger.error(f"Error reading video {video_path}: {e}")
            return {
                'frame_count': 0,
                'fps': 0,
                'duration': 0,
                'width': 0,
                'height': 0
            }
    
    def scan_directory(self, directory: Path, split_name: str) -> List[Dict]:
        """Scan directory and collect file information."""
        logger.info(f"Scanning {split_name} directory: {directory}")
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return []
        
        file_info = []
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                # Extract label from filename
                label = self.extract_label_from_filename(file_path.name)
                
                # Skip files not in target classes
                if label not in self.target_classes:
                    logger.warning(f"Skipping file with unknown label: {file_path.name}")
                    continue
                
                # Get video metadata
                video_info = self.get_video_info(file_path)
                
                # Calculate file hash
                file_hash = self.calculate_file_hash(file_path)
                
                # Extract speaker ID from filename (assuming format: "class speaker_id.mp4")
                speaker_id = file_path.stem.split()[-1] if ' ' in file_path.stem else "unknown"
                
                file_info.append({
                    'filepath': str(file_path),
                    'filename': file_path.name,
                    'label': label,
                    'split': split_name,
                    'file_hash': file_hash,
                    'speaker_id': speaker_id,
                    'file_size': file_path.stat().st_size,
                    **video_info
                })
        
        logger.info(f"Found {len(file_info)} valid files in {split_name} split")
        return file_info
    
    def analyze_data(self) -> Dict:
        """Perform comprehensive data analysis."""
        logger.info("Starting comprehensive data analysis...")
        
        # Scan all directories
        self.data_info['train'] = self.scan_directory(self.train_dir, 'train')
        self.data_info['val'] = self.scan_directory(self.val_dir, 'val')
        self.data_info['test'] = self.scan_directory(self.test_dir, 'test')
        
        # Combine all data
        all_data = []
        for split_data in self.data_info.values():
            all_data.extend(split_data)
        
        df = pd.DataFrame(all_data)
        
        if df.empty:
            logger.error("No valid data found!")
            return {}
        
        # Analysis results
        analysis = {
            'total_files': len(df),
            'splits': {},
            'classes': {},
            'speakers': {},
            'video_stats': {},
            'data_leakage': {},
            'quality_issues': []
        }
        
        # Split analysis
        for split in ['train', 'val', 'test']:
            split_df = df[df['split'] == split]
            analysis['splits'][split] = {
                'count': len(split_df),
                'classes': split_df['label'].value_counts().to_dict(),
                'speakers': split_df['speaker_id'].nunique(),
                'total_duration': split_df['duration'].sum(),
                'avg_duration': split_df['duration'].mean()
            }
        
        # Class distribution analysis
        for class_name in self.target_classes:
            class_df = df[df['label'] == class_name]
            analysis['classes'][class_name] = {
                'total_count': len(class_df),
                'train_count': len(class_df[class_df['split'] == 'train']),
                'val_count': len(class_df[class_df['split'] == 'val']),
                'test_count': len(class_df[class_df['split'] == 'test']),
                'avg_duration': class_df['duration'].mean(),
                'speakers': class_df['speaker_id'].nunique()
            }
        
        # Speaker analysis
        speaker_splits = defaultdict(set)
        for _, row in df.iterrows():
            speaker_splits[row['speaker_id']].add(row['split'])
        
        analysis['speakers'] = {
            'total_speakers': len(speaker_splits),
            'cross_split_speakers': sum(1 for splits in speaker_splits.values() if len(splits) > 1),
            'speaker_distribution': dict(Counter(df['speaker_id']))
        }
        
        # Video statistics
        analysis['video_stats'] = {
            'avg_duration': df['duration'].mean(),
            'std_duration': df['duration'].std(),
            'min_duration': df['duration'].min(),
            'max_duration': df['duration'].max(),
            'avg_frame_count': df['frame_count'].mean(),
            'resolution_distribution': df.groupby(['width', 'height']).size().to_dict()
        }
        
        # Data leakage detection
        hash_splits = defaultdict(set)
        for _, row in df.iterrows():
            if row['file_hash']:
                hash_splits[row['file_hash']].add(row['split'])
        
        duplicate_hashes = {h: splits for h, splits in hash_splits.items() if len(splits) > 1}
        analysis['data_leakage'] = {
            'duplicate_files': len(duplicate_hashes),
            'affected_files': sum(len(df[df['file_hash'] == h]) for h in duplicate_hashes),
            'duplicate_details': duplicate_hashes
        }
        
        # Quality issues
        quality_issues = []
        
        # Check for very short videos
        short_videos = df[df['duration'] < 0.5]  # Less than 0.5 seconds
        if len(short_videos) > 0:
            quality_issues.append(f"Found {len(short_videos)} videos shorter than 0.5 seconds")
        
        # Check for very long videos
        long_videos = df[df['duration'] > 10]  # More than 10 seconds
        if len(long_videos) > 0:
            quality_issues.append(f"Found {len(long_videos)} videos longer than 10 seconds")
        
        # Check for missing video info
        invalid_videos = df[(df['frame_count'] == 0) | (df['duration'] == 0)]
        if len(invalid_videos) > 0:
            quality_issues.append(f"Found {len(invalid_videos)} videos with invalid metadata")
        
        analysis['quality_issues'] = quality_issues
        
        return analysis, df
    
    def generate_visualizations(self, df: pd.DataFrame, output_dir: str = "analysis_plots"):
        """Generate visualization plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Class distribution by split
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall class distribution
        class_counts = df['label'].value_counts()
        axes[0, 0].bar(class_counts.index, class_counts.values)
        axes[0, 0].set_title('Overall Class Distribution')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Class distribution by split
        split_class_counts = df.groupby(['split', 'label']).size().unstack(fill_value=0)
        split_class_counts.plot(kind='bar', ax=axes[0, 1], stacked=True)
        axes[0, 1].set_title('Class Distribution by Split')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend(title='Class')
        
        # Duration distribution
        axes[1, 0].hist(df['duration'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Video Duration Distribution')
        axes[1, 0].set_xlabel('Duration (seconds)')
        axes[1, 0].set_ylabel('Count')
        
        # Duration by class
        df.boxplot(column='duration', by='label', ax=axes[1, 1])
        axes[1, 1].set_title('Duration Distribution by Class')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Duration (seconds)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/data_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Speaker analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Speaker distribution
        speaker_counts = df['speaker_id'].value_counts().head(20)  # Top 20 speakers
        axes[0].bar(range(len(speaker_counts)), speaker_counts.values)
        axes[0].set_title('Top 20 Speaker Distribution')
        axes[0].set_xlabel('Speaker Rank')
        axes[0].set_ylabel('Video Count')
        
        # Speakers per class
        speaker_class = df.groupby('label')['speaker_id'].nunique()
        axes[1].bar(speaker_class.index, speaker_class.values)
        axes[1].set_title('Unique Speakers per Class')
        axes[1].set_ylabel('Number of Speakers')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/speaker_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}/")
    
    def save_analysis_report(self, analysis: Dict, df: pd.DataFrame, output_file: str = "data_analysis_report.txt"):
        """Save detailed analysis report."""
        with open(output_file, 'w') as f:
            f.write("ICU LIP READING DATASET ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary
            f.write(f"Total Files: {analysis['total_files']}\n")
            f.write(f"Target Classes: {', '.join(self.target_classes)}\n\n")
            
            # Split information
            f.write("SPLIT DISTRIBUTION:\n")
            f.write("-" * 20 + "\n")
            for split, info in analysis['splits'].items():
                f.write(f"{split.upper()}:\n")
                f.write(f"  Total files: {info['count']}\n")
                f.write(f"  Unique speakers: {info['speakers']}\n")
                f.write(f"  Total duration: {info['total_duration']:.2f} seconds\n")
                f.write(f"  Average duration: {info['avg_duration']:.2f} seconds\n")
                f.write(f"  Class distribution: {info['classes']}\n\n")
            
            # Class analysis
            f.write("CLASS ANALYSIS:\n")
            f.write("-" * 15 + "\n")
            for class_name, info in analysis['classes'].items():
                f.write(f"{class_name.upper()}:\n")
                f.write(f"  Total: {info['total_count']} files\n")
                f.write(f"  Train: {info['train_count']}, Val: {info['val_count']}, Test: {info['test_count']}\n")
                f.write(f"  Average duration: {info['avg_duration']:.2f} seconds\n")
                f.write(f"  Unique speakers: {info['speakers']}\n\n")
            
            # Data quality
            f.write("DATA QUALITY ASSESSMENT:\n")
            f.write("-" * 25 + "\n")
            
            if analysis['data_leakage']['duplicate_files'] > 0:
                f.write(f"⚠️  DATA LEAKAGE DETECTED: {analysis['data_leakage']['duplicate_files']} duplicate files\n")
                f.write(f"   Affected files: {analysis['data_leakage']['affected_files']}\n")
            else:
                f.write("✅ No data leakage detected\n")
            
            if analysis['speakers']['cross_split_speakers'] > 0:
                f.write(f"⚠️  SPEAKER LEAKAGE: {analysis['speakers']['cross_split_speakers']} speakers appear in multiple splits\n")
            else:
                f.write("✅ No speaker leakage detected\n")
            
            if analysis['quality_issues']:
                f.write("\nQuality Issues:\n")
                for issue in analysis['quality_issues']:
                    f.write(f"⚠️  {issue}\n")
            else:
                f.write("✅ No quality issues detected\n")
            
            # Recommendations
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 15 + "\n")
            
            # Check class balance
            class_counts = [analysis['classes'][cls]['total_count'] for cls in self.target_classes]
            min_count, max_count = min(class_counts), max(class_counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 3:
                f.write("⚠️  Significant class imbalance detected. Consider:\n")
                f.write("   - Data augmentation for underrepresented classes\n")
                f.write("   - Weighted loss functions\n")
                f.write("   - Stratified sampling\n")
            
            # Check data size
            if analysis['total_files'] < 100:
                f.write("⚠️  Small dataset size. Consider:\n")
                f.write("   - Data augmentation strategies\n")
                f.write("   - Transfer learning from larger datasets\n")
                f.write("   - Cross-validation for robust evaluation\n")
        
        logger.info(f"Analysis report saved to {output_file}")


def main():
    """Main analysis function."""
    analyzer = ICUDataAnalyzer()
    
    # Perform analysis
    analysis, df = analyzer.analyze_data()
    
    if not analysis:
        logger.error("Analysis failed - no data found")
        return
    
    # Generate visualizations
    analyzer.generate_visualizations(df)
    
    # Save analysis report
    analyzer.save_analysis_report(analysis, df)
    
    # Save data manifest
    df.to_csv("data_manifest.csv", index=False)
    logger.info("Data manifest saved to data_manifest.csv")
    
    # Print summary
    print("\n" + "="*50)
    print("DATA ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total files: {analysis['total_files']}")
    print(f"Train: {analysis['splits']['train']['count']}")
    print(f"Val: {analysis['splits']['val']['count']}")
    print(f"Test: {analysis['splits']['test']['count']}")
    print(f"Classes: {list(analysis['classes'].keys())}")
    
    if analysis['data_leakage']['duplicate_files'] > 0:
        print(f"⚠️  WARNING: {analysis['data_leakage']['duplicate_files']} duplicate files detected!")
    
    if analysis['speakers']['cross_split_speakers'] > 0:
        print(f"⚠️  WARNING: {analysis['speakers']['cross_split_speakers']} speakers in multiple splits!")
    
    print("="*50)


if __name__ == "__main__":
    main()
