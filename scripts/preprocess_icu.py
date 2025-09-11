"""
ICU Data Preprocessing Pipeline
Extracts lip ROIs, standardizes video format, and generates training manifests.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import argparse

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from lip_roi import LipROIExtractor, load_video_frames

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ICUPreprocessor:
    """Preprocessor for ICU lip reading dataset."""
    
    def __init__(self, 
                 output_dir: str = "processed_data",
                 target_size: Tuple[int, int] = (96, 96),
                 target_frames: int = 24,
                 target_classes: List[str] = None):
        """
        Initialize preprocessor.
        
        Args:
            output_dir: Directory to save processed data
            target_size: Target size for lip ROI (width, height)
            target_frames: Target number of frames per video
            target_classes: List of target class names
        """
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.target_frames = target_frames
        self.target_classes = target_classes or ["doctor", "glasses", "phone", "pillow", "help"]
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(exist_ok=True)
            for class_name in self.target_classes:
                (self.output_dir / split / class_name).mkdir(exist_ok=True)
        
        # Initialize lip ROI extractor
        self.lip_extractor = LipROIExtractor(roi_size=target_size)
        
        # Statistics
        self.stats = {
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'class_counts': {cls: 0 for cls in self.target_classes}
        }
    
    def extract_label_from_filename(self, filename: str) -> Optional[str]:
        """Extract class label from filename."""
        filename_lower = filename.lower()
        for target_class in self.target_classes:
            if filename_lower.startswith(target_class):
                return target_class
        return None
    
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
    
    def process_single_video(self, video_path: Path, output_path: Path) -> Dict:
        """
        Process a single video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video
            
        Returns:
            Dictionary with processing results and metadata
        """
        try:
            # Load video frames
            frames = load_video_frames(str(video_path))
            if not frames:
                return {'success': False, 'error': 'Failed to load video frames'}
            
            # Extract lip ROIs
            lip_rois = self.lip_extractor.process_video_frames(frames)
            if lip_rois is None:
                return {'success': False, 'error': 'Failed to extract lip ROIs'}
            
            # Standardize sequence length
            standardized = self.lip_extractor.standardize_sequence_length(
                lip_rois, target_length=self.target_frames
            )
            
            # Preprocess for model (grayscale + normalize)
            processed = self.lip_extractor.preprocess_for_model(
                standardized, grayscale=True, normalize=True
            )
            
            # Save as numpy array
            np.save(output_path, processed)
            
            # Get video metadata
            cap = cv2.VideoCapture(str(video_path))
            original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = original_frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            return {
                'success': True,
                'processed_shape': processed.shape,
                'original_frames': len(frames),
                'extracted_rois': len(lip_rois),
                'final_frames': len(processed),
                'original_frame_count': original_frame_count,
                'fps': fps,
                'duration': duration,
                'original_width': width,
                'original_height': height,
                'processed_size': os.path.getsize(output_path)
            }
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_directory(self, input_dir: Path, split_name: str) -> List[Dict]:
        """
        Process all videos in a directory.
        
        Args:
            input_dir: Input directory path
            split_name: Split name (train/val/test)
            
        Returns:
            List of processing results
        """
        logger.info(f"Processing {split_name} directory: {input_dir}")
        
        if not input_dir.exists():
            logger.error(f"Directory does not exist: {input_dir}")
            return []
        
        results = []
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        # Get all video files
        video_files = [f for f in input_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in video_extensions]
        
        logger.info(f"Found {len(video_files)} video files in {split_name}")
        
        for video_path in tqdm(video_files, desc=f"Processing {split_name}"):
            # Extract label from filename
            label = self.extract_label_from_filename(video_path.name)
            
            if label not in self.target_classes:
                logger.warning(f"Skipping file with invalid label: {video_path.name}")
                self.stats['skipped'] += 1
                continue
            
            # Generate output filename
            output_filename = f"{video_path.stem}.npy"
            output_path = self.output_dir / split_name / label / output_filename
            
            # Skip if already processed
            if output_path.exists():
                logger.info(f"Skipping already processed file: {output_filename}")
                continue
            
            # Process video
            result = self.process_single_video(video_path, output_path)
            
            if result['success']:
                self.stats['processed'] += 1
                self.stats['class_counts'][label] += 1
                
                # Calculate file hashes
                original_hash = self.calculate_file_hash(video_path)
                processed_hash = self.calculate_file_hash(output_path)
                
                # Extract speaker ID from filename
                speaker_id = video_path.stem.split()[-1] if ' ' in video_path.stem else "unknown"
                
                # Store result
                result.update({
                    'original_path': str(video_path),
                    'processed_path': str(output_path),
                    'filename': video_path.name,
                    'processed_filename': output_filename,
                    'label': label,
                    'split': split_name,
                    'speaker_id': speaker_id,
                    'original_hash': original_hash,
                    'processed_hash': processed_hash,
                    'original_size': video_path.stat().st_size
                })
                
                results.append(result)
                
            else:
                self.stats['failed'] += 1
                logger.error(f"Failed to process {video_path.name}: {result.get('error', 'Unknown error')}")
        
        return results
    
    def generate_manifest(self, all_results: List[Dict], output_file: str = "processed_manifest.csv"):
        """Generate CSV manifest for processed data."""
        if not all_results:
            logger.warning("No results to save in manifest")
            return
        
        df = pd.DataFrame(all_results)
        
        # Select relevant columns for manifest
        manifest_columns = [
            'original_path', 'processed_path', 'filename', 'processed_filename',
            'label', 'split', 'speaker_id', 'original_hash', 'processed_hash',
            'processed_shape', 'original_frames', 'final_frames', 'duration',
            'original_size', 'processed_size'
        ]
        
        # Filter columns that exist
        available_columns = [col for col in manifest_columns if col in df.columns]
        manifest_df = df[available_columns]
        
        # Save manifest
        manifest_df.to_csv(output_file, index=False)
        logger.info(f"Manifest saved to {output_file}")
        
        return manifest_df
    
    def validate_processed_data(self, manifest_df: pd.DataFrame) -> Dict:
        """Validate processed data integrity."""
        logger.info("Validating processed data...")
        
        validation_results = {
            'total_files': len(manifest_df),
            'missing_files': 0,
            'shape_inconsistencies': 0,
            'hash_duplicates': 0,
            'speaker_leakage': 0,
            'class_distribution': {},
            'split_distribution': {}
        }
        
        # Check for missing files
        for _, row in manifest_df.iterrows():
            if not Path(row['processed_path']).exists():
                validation_results['missing_files'] += 1
        
        # Check shape consistency
        expected_shape = f"({self.target_frames}, {self.target_size[1]}, {self.target_size[0]}, 1)"
        shape_counts = manifest_df['processed_shape'].value_counts()
        if len(shape_counts) > 1:
            validation_results['shape_inconsistencies'] = len(manifest_df) - shape_counts.iloc[0]
        
        # Check for hash duplicates across splits
        hash_splits = manifest_df.groupby('processed_hash')['split'].apply(set)
        cross_split_hashes = sum(1 for splits in hash_splits if len(splits) > 1)
        validation_results['hash_duplicates'] = cross_split_hashes
        
        # Check for speaker leakage
        speaker_splits = manifest_df.groupby('speaker_id')['split'].apply(set)
        cross_split_speakers = sum(1 for splits in speaker_splits if len(splits) > 1)
        validation_results['speaker_leakage'] = cross_split_speakers
        
        # Class and split distributions
        validation_results['class_distribution'] = manifest_df['label'].value_counts().to_dict()
        validation_results['split_distribution'] = manifest_df['split'].value_counts().to_dict()
        
        return validation_results
    
    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "="*50)
        print("PREPROCESSING STATISTICS")
        print("="*50)
        print(f"Total processed: {self.stats['processed']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Skipped: {self.stats['skipped']}")
        print("\nClass distribution:")
        for class_name, count in self.stats['class_counts'].items():
            print(f"  {class_name}: {count}")
        print("="*50)


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess ICU lip reading dataset")
    parser.add_argument("--train_dir", default="/Users/client/Desktop/TRAINING SET 2.9.25",
                       help="Path to training data directory")
    parser.add_argument("--val_dir", default="/Users/client/Desktop/VAL SET",
                       help="Path to validation data directory")
    parser.add_argument("--test_dir", default="/Users/client/Desktop/TEST SET",
                       help="Path to test data directory")
    parser.add_argument("--output_dir", default="processed_data",
                       help="Output directory for processed data")
    parser.add_argument("--target_size", nargs=2, type=int, default=[96, 96],
                       help="Target size for lip ROI (width height)")
    parser.add_argument("--target_frames", type=int, default=24,
                       help="Target number of frames per video")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = ICUPreprocessor(
        output_dir=args.output_dir,
        target_size=tuple(args.target_size),
        target_frames=args.target_frames
    )
    
    # Process all splits
    all_results = []
    
    # Process training data
    train_results = preprocessor.process_directory(Path(args.train_dir), 'train')
    all_results.extend(train_results)
    
    # Process validation data
    val_results = preprocessor.process_directory(Path(args.val_dir), 'val')
    all_results.extend(val_results)
    
    # Process test data
    test_results = preprocessor.process_directory(Path(args.test_dir), 'test')
    all_results.extend(test_results)
    
    # Generate manifest
    manifest_df = preprocessor.generate_manifest(all_results)
    
    # Validate processed data
    if manifest_df is not None:
        validation_results = preprocessor.validate_processed_data(manifest_df)
        
        print("\n" + "="*50)
        print("VALIDATION RESULTS")
        print("="*50)
        for key, value in validation_results.items():
            print(f"{key}: {value}")
        print("="*50)
    
    # Print statistics
    preprocessor.print_statistics()


if __name__ == "__main__":
    main()
