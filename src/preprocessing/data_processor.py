"""
Data Processing Pipeline

This script processes all training, validation, and test videos to extract lip sequences
and prepare them for machine learning training.
"""

import os
import numpy as np
import pickle
from typing import Dict, List, Tuple
from lip_detector import LipDetector
from sklearn.preprocessing import LabelEncoder
import json


class DataProcessor:
    """
    Handles the complete data processing pipeline for the lipreading dataset.
    """
    
    def __init__(self, data_root: str = "data", target_words: List[str] = None):
        """
        Initialize the DataProcessor.
        
        Args:
            data_root: Root directory containing training/validation/test folders
            target_words: List of target words to recognize
        """
        self.data_root = data_root
        self.target_words = target_words or ["doctor", "glasses", "help", "pillow", "phone"]
        self.lip_detector = LipDetector()
        self.label_encoder = LabelEncoder()
        
        # Fit label encoder with target words
        self.label_encoder.fit(self.target_words)
    
    def process_dataset_split(self, split_name: str, max_frames: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a dataset split (training, validation, or test).
        
        Args:
            split_name: Name of the split ('training_set', 'validation_set', 'test_set')
            max_frames: Maximum number of frames per video sequence
            
        Returns:
            Tuple of (X, y) where X is the feature array and y is the label array
        """
        split_path = os.path.join(self.data_root, split_name)
        
        if not os.path.exists(split_path):
            print(f"Split directory not found: {split_path}")
            return np.array([]), np.array([])
        
        all_sequences = []
        all_labels = []
        
        print(f"\nProcessing {split_name}...")
        print("=" * 50)
        
        for word in self.target_words:
            word_folder = os.path.join(split_path, word)
            
            if os.path.exists(word_folder):
                sequences, labels = self.lip_detector.process_video_folder(
                    word_folder, word, max_frames
                )
                all_sequences.extend(sequences)
                all_labels.extend(labels)
                print(f"Word '{word}': {len(sequences)} videos processed")
            else:
                print(f"Warning: Folder not found for word '{word}' in {split_name}")
        
        if len(all_sequences) == 0:
            print(f"No data found in {split_name}")
            return np.array([]), np.array([])
        
        # Convert to numpy arrays
        X = np.array(all_sequences)
        y = self.label_encoder.transform(all_labels)
        
        print(f"\n{split_name} Summary:")
        print(f"Total sequences: {len(X)}")
        print(f"Sequence shape: {X.shape}")
        print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y
    
    def process_all_data(self, max_frames: int = 30, save_processed: bool = True) -> Dict:
        """
        Process all dataset splits and optionally save the processed data.
        
        Args:
            max_frames: Maximum number of frames per video sequence
            save_processed: Whether to save processed data to disk
            
        Returns:
            Dictionary containing all processed data splits
        """
        data = {}
        
        # Process each split
        splits = ['training_set', 'validation_set', 'test_set']
        
        for split in splits:
            X, y = self.process_dataset_split(split, max_frames)
            if len(X) > 0:
                data[split] = {'X': X, 'y': y}
        
        # Save processed data if requested
        if save_processed and data:
            self.save_processed_data(data)
        
        # Save label encoder
        self.save_label_encoder()
        
        return data
    
    def save_processed_data(self, data: Dict):
        """
        Save processed data to disk.
        
        Args:
            data: Dictionary containing processed data splits
        """
        os.makedirs('processed_data', exist_ok=True)
        
        for split_name, split_data in data.items():
            filename = f'processed_data/{split_name}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"Saved {split_name} to {filename}")
    
    def save_label_encoder(self):
        """
        Save the label encoder for later use in the web app.
        """
        os.makedirs('processed_data', exist_ok=True)
        
        # Save as pickle for Python use
        with open('processed_data/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save as JSON for easy inspection
        label_mapping = {str(i): word for i, word in enumerate(self.label_encoder.classes_)}
        with open('processed_data/label_mapping.json', 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print("Saved label encoder and mapping")
    
    def load_processed_data(self, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load previously processed data from disk.
        
        Args:
            split_name: Name of the split to load
            
        Returns:
            Tuple of (X, y) arrays
        """
        filename = f'processed_data/{split_name}.pkl'
        
        if not os.path.exists(filename):
            print(f"Processed data file not found: {filename}")
            return np.array([]), np.array([])
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        return data['X'], data['y']
    
    def get_data_statistics(self, data: Dict):
        """
        Print comprehensive statistics about the processed data.
        
        Args:
            data: Dictionary containing processed data splits
        """
        print("\n" + "="*60)
        print("DATA STATISTICS")
        print("="*60)
        
        total_sequences = 0
        
        for split_name, split_data in data.items():
            X, y = split_data['X'], split_data['y']
            total_sequences += len(X)
            
            print(f"\n{split_name.upper()}:")
            print(f"  Sequences: {len(X)}")
            print(f"  Shape: {X.shape}")
            print(f"  Memory usage: {X.nbytes / 1024 / 1024:.2f} MB")
            
            # Label distribution
            unique_labels, counts = np.unique(y, return_counts=True)
            print("  Label distribution:")
            for label_idx, count in zip(unique_labels, counts):
                word = self.label_encoder.classes_[label_idx]
                print(f"    {word}: {count} sequences")
        
        print(f"\nTOTAL SEQUENCES: {total_sequences}")
        print(f"TARGET WORDS: {', '.join(self.target_words)}")
        print(f"SEQUENCE LENGTH: {max_frames} frames")
        print(f"FRAME SIZE: 64x64 pixels")


def main():
    """
    Main function to process all data.
    """
    # Initialize processor
    processor = DataProcessor()
    
    # Process all data
    print("Starting data processing pipeline...")
    data = processor.process_all_data(max_frames=30, save_processed=True)
    
    if data:
        # Print statistics
        processor.get_data_statistics(data)
        print("\nData processing completed successfully!")
        print("Processed data saved in 'processed_data/' directory")
    else:
        print("No data was processed. Please check your data directory structure.")
        print("Expected structure:")
        print("data/")
        print("├── training_set/")
        print("│   ├── doctor/")
        print("│   ├── glasses/")
        print("│   ├── help/")
        print("│   ├── pillow/")
        print("│   └── phone/")
        print("├── validation_set/")
        print("│   └── [same structure]")
        print("└── test_set/")
        print("    └── [same structure]")


if __name__ == "__main__":
    main()
