#!/usr/bin/env python3
"""
Simplified Lipreading Training Pipeline
"""

import numpy as np
import os
import json
import pickle

# Set random seed for reproducibility
np.random.seed(42)

print("🎯 Lipreading Model Training Pipeline")
print("=" * 60)

# Configuration
target_words = ["doctor", "glasses", "help", "pillow", "phone"]
samples_per_word = 20
sequence_length = 30
image_size = (64, 64)

print(f"Target words: {target_words}")
print(f"Samples per word: {samples_per_word}")
print(f"Sequence length: {sequence_length} frames")
print(f"Image size: {image_size}")

def generate_synthetic_data():
    """Generate synthetic lip movement data."""
    print("\n📊 Generating synthetic dataset...")
    
    X = []
    y = []
    
    for word_idx, word in enumerate(target_words):
        print(f"Generating data for '{word}'...")
        
        for sample_idx in range(samples_per_word):
            # Generate a synthetic sequence
            sequence = []
            
            for frame_idx in range(sequence_length):
                # Create a simple pattern based on word and frame
                frame = np.zeros(image_size)
                
                # Different patterns for different words
                if word == "doctor":
                    # Opening/closing mouth pattern
                    center_y, center_x = 32, 32
                    t = frame_idx / sequence_length * 2 * np.pi
                    mouth_height = int(8 + 6 * np.sin(t))
                    mouth_width = 12
                    
                    y_coords, x_coords = np.ogrid[:64, :64]
                    mask = ((x_coords - center_x)**2 / mouth_width**2 + 
                           (y_coords - center_y)**2 / mouth_height**2) <= 1
                    frame[mask] = 0.8 + 0.2 * np.sin(t)
                
                elif word == "glasses":
                    # Lateral movement pattern
                    center_y, center_x = 32, 32
                    t = frame_idx / sequence_length * 2 * np.pi
                    offset_x = int(3 * np.sin(t))
                    mouth_width = 10
                    mouth_height = 6
                    
                    y_coords, x_coords = np.ogrid[:64, :64]
                    mask = ((x_coords - (center_x + offset_x))**2 / mouth_width**2 + 
                           (y_coords - center_y)**2 / mouth_height**2) <= 1
                    frame[mask] = 0.7 + 0.3 * np.cos(t)
                
                elif word == "help":
                    # Vertical emphasis pattern
                    center_y, center_x = 32, 32
                    t = frame_idx / sequence_length * 2 * np.pi
                    mouth_width = 8
                    mouth_height = int(6 + 4 * np.abs(np.sin(t)))
                    
                    y_coords, x_coords = np.ogrid[:64, :64]
                    mask = ((x_coords - center_x)**2 / mouth_width**2 + 
                           (y_coords - center_y)**2 / mouth_height**2) <= 1
                    frame[mask] = 0.6 + 0.4 * np.abs(np.sin(t))
                
                elif word == "pillow":
                    # Rounded pattern
                    center_y, center_x = 32, 32
                    t = frame_idx / sequence_length * 2 * np.pi
                    radius = 8 + 3 * np.sin(t)
                    
                    y_coords, x_coords = np.ogrid[:64, :64]
                    mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) <= radius**2
                    frame[mask] = 0.5 + 0.5 * np.sin(t + np.pi/4)
                
                elif word == "phone":
                    # Plosive pattern
                    center_y, center_x = 32, 32
                    if frame_idx < 10:
                        mouth_size = 2
                    elif frame_idx < 20:
                        mouth_size = 12
                    else:
                        mouth_size = 8 - (frame_idx - 20) * 0.3
                    
                    y_coords, x_coords = np.ogrid[:64, :64]
                    mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) <= mouth_size**2
                    frame[mask] = 0.7 + 0.3 * np.sin(frame_idx * 0.5)
                
                # Add some noise for realism
                noise = np.random.normal(0, 0.05, image_size)
                frame = np.clip(frame + noise, 0, 1)
                
                # Add person variation
                person_factor = (sample_idx % 5) * 0.1
                frame = np.clip(frame + person_factor * np.random.normal(0, 0.02, image_size), 0, 1)
                
                sequence.append(frame)
            
            X.append(np.array(sequence))
            y.append(word_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Dataset generated: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    return X, y

def create_splits(X, y):
    """Create train/validation/test splits."""
    print("\n📊 Creating data splits...")
    
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_data_and_labels():
    """Save processed data and label mappings."""
    print("\n💾 Saving data and labels...")
    
    # Create directories
    os.makedirs('processed_data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save label mapping
    label_mapping = {str(i): word for i, word in enumerate(target_words)}
    with open('processed_data/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    # Create a simple label encoder equivalent
    class SimpleLabelEncoder:
        def __init__(self, classes):
            self.classes_ = classes
        
        def transform(self, labels):
            return [self.classes_.index(label) for label in labels]
        
        def inverse_transform(self, indices):
            return [self.classes_[idx] for idx in indices]
    
    label_encoder = SimpleLabelEncoder(target_words)
    
    with open('processed_data/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("✅ Labels saved")

def main():
    """Main execution."""
    print("\n" + "="*60)
    print("EXECUTING TRAINING PIPELINE")
    print("="*60)
    
    # Generate data
    X, y = generate_synthetic_data()
    
    # Create splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_splits(X, y)
    
    # Save data and labels
    save_data_and_labels()
    
    # Display statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"Total samples: {len(X)}")
    print(f"Sequence shape: {X.shape[1:]}")
    print(f"Number of classes: {len(target_words)}")
    
    print(f"\n📊 Class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for word_idx, count in zip(unique, counts):
        print(f"  {target_words[word_idx]}: {count} samples")
    
    print(f"\n✅ Data preparation completed!")
    print(f"Ready for model training...")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    main()
