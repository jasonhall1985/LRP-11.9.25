#!/usr/bin/env python3
"""
Simplified Lipreading Training Pipeline
Creates a trained TensorFlow.js model for the 5-word vocabulary
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import json
from tqdm import tqdm
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

print("🎯 Simplified Lipreading Training Pipeline")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

class LipMovementDataGenerator:
    """
    Generates synthetic lip movement coordinate data for training.
    Simulates MediaPipe-style lip landmark coordinates.
    """
    
    def __init__(self, target_words=None):
        self.target_words = target_words or ["doctor", "glasses", "help", "pillow", "phone"]
        self.sequence_length = 30  # 30 frames
        self.num_landmarks = 24   # 24 lip landmarks
        self.coords_per_landmark = 2  # x, y coordinates
        
    def generate_word_pattern(self, word_idx, person_variation=0.0):
        """Generate lip movement pattern for a specific word."""
        
        # Base patterns for each word (normalized coordinates 0-1)
        patterns = {
            0: self._doctor_movement(),    # doctor - vertical emphasis
            1: self._glasses_movement(),   # glasses - lateral movement  
            2: self._help_movement(),      # help - quick open/close
            3: self._pillow_movement(),    # pillow - rounded shapes
            4: self._phone_movement()      # phone - plosive pattern
        }
        
        base_pattern = patterns[word_idx]
        
        # Add person-specific variations
        person_noise = np.random.normal(0, person_variation, base_pattern.shape)
        
        # Add temporal noise for realism
        temporal_noise = np.random.normal(0, 0.02, base_pattern.shape)
        
        # Combine patterns
        final_pattern = base_pattern + person_noise + temporal_noise
        
        # Ensure coordinates stay in valid range [0, 1]
        final_pattern = np.clip(final_pattern, 0, 1)
        
        return final_pattern
    
    def _doctor_movement(self):
        """Generate movement pattern for 'doctor' - vertical mouth opening."""
        frames = []
        for i in range(self.sequence_length):
            t = i / self.sequence_length * 2 * np.pi
            
            # Create lip landmark coordinates
            coords = []
            for landmark_idx in range(self.num_landmarks):
                # Base position around mouth center
                base_x = 0.5 + 0.1 * np.cos(landmark_idx * 2 * np.pi / self.num_landmarks)
                base_y = 0.5 + 0.05 * np.sin(landmark_idx * 2 * np.pi / self.num_landmarks)
                
                # Add vertical movement for "doctor"
                movement_y = 0.03 * np.sin(t) * (1 if landmark_idx < 12 else -1)
                
                coords.extend([base_x, base_y + movement_y])
            
            frames.append(coords)
        
        return np.array(frames)
    
    def _glasses_movement(self):
        """Generate movement pattern for 'glasses' - lateral emphasis."""
        frames = []
        for i in range(self.sequence_length):
            t = i / self.sequence_length * 2 * np.pi
            
            coords = []
            for landmark_idx in range(self.num_landmarks):
                base_x = 0.5 + 0.1 * np.cos(landmark_idx * 2 * np.pi / self.num_landmarks)
                base_y = 0.5 + 0.05 * np.sin(landmark_idx * 2 * np.pi / self.num_landmarks)
                
                # Add lateral movement for "glasses"
                movement_x = 0.02 * np.sin(t) * np.cos(landmark_idx * 2 * np.pi / self.num_landmarks)
                
                coords.extend([base_x + movement_x, base_y])
            
            frames.append(coords)
        
        return np.array(frames)
    
    def _help_movement(self):
        """Generate movement pattern for 'help' - quick open/close."""
        frames = []
        for i in range(self.sequence_length):
            t = i / self.sequence_length * 4 * np.pi  # Faster movement
            
            coords = []
            for landmark_idx in range(self.num_landmarks):
                base_x = 0.5 + 0.1 * np.cos(landmark_idx * 2 * np.pi / self.num_landmarks)
                base_y = 0.5 + 0.05 * np.sin(landmark_idx * 2 * np.pi / self.num_landmarks)
                
                # Quick vertical movement
                movement_y = 0.04 * np.abs(np.sin(t)) * (1 if landmark_idx < 12 else -1)
                
                coords.extend([base_x, base_y + movement_y])
            
            frames.append(coords)
        
        return np.array(frames)
    
    def _pillow_movement(self):
        """Generate movement pattern for 'pillow' - rounded shapes."""
        frames = []
        for i in range(self.sequence_length):
            t = i / self.sequence_length * 2 * np.pi
            
            coords = []
            for landmark_idx in range(self.num_landmarks):
                base_x = 0.5 + 0.1 * np.cos(landmark_idx * 2 * np.pi / self.num_landmarks)
                base_y = 0.5 + 0.05 * np.sin(landmark_idx * 2 * np.pi / self.num_landmarks)
                
                # Circular movement pattern
                radius_variation = 0.02 * np.sin(t)
                angle_offset = t * 0.5
                
                movement_x = radius_variation * np.cos(landmark_idx * 2 * np.pi / self.num_landmarks + angle_offset)
                movement_y = radius_variation * np.sin(landmark_idx * 2 * np.pi / self.num_landmarks + angle_offset)
                
                coords.extend([base_x + movement_x, base_y + movement_y])
            
            frames.append(coords)
        
        return np.array(frames)
    
    def _phone_movement(self):
        """Generate movement pattern for 'phone' - plosive pattern."""
        frames = []
        for i in range(self.sequence_length):
            # Plosive pattern: closed -> open -> closed
            if i < 8:  # Initial closure
                mouth_opening = 0.01
            elif i < 18:  # Quick opening
                mouth_opening = 0.05 * (i - 8) / 10
            else:  # Gradual closing
                mouth_opening = 0.05 * (self.sequence_length - i) / 12
            
            coords = []
            for landmark_idx in range(self.num_landmarks):
                base_x = 0.5 + 0.1 * np.cos(landmark_idx * 2 * np.pi / self.num_landmarks)
                base_y = 0.5 + 0.05 * np.sin(landmark_idx * 2 * np.pi / self.num_landmarks)
                
                # Plosive movement
                movement_y = mouth_opening * (1 if landmark_idx < 12 else -1)
                
                coords.extend([base_x, base_y + movement_y])
            
            frames.append(coords)
        
        return np.array(frames)
    
    def generate_dataset(self, samples_per_word=50, num_people=10):
        """Generate complete dataset."""
        print(f"\n📊 Generating lip movement dataset...")
        print(f"Words: {self.target_words}")
        print(f"Samples per word: {samples_per_word}")
        print(f"Number of people variations: {num_people}")
        
        X = []
        y = []
        
        total_samples = len(self.target_words) * samples_per_word
        pbar = tqdm(total=total_samples, desc="Generating samples")
        
        for word_idx, word in enumerate(self.target_words):
            for sample_idx in range(samples_per_word):
                # Vary person characteristics
                person_variation = (sample_idx % num_people) * 0.01
                
                # Generate lip movement sequence
                sequence = self.generate_word_pattern(word_idx, person_variation)
                
                X.append(sequence)
                y.append(word_idx)
                pbar.update(1)
        
        pbar.close()
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Dataset generated: {X.shape}")
        print(f"   Sequence length: {X.shape[1]} frames")
        print(f"   Features per frame: {X.shape[2]} coordinates")
        
        return X, y


class LipreadingNeuralNetwork:
    """Neural network for lipreading classification."""
    
    def __init__(self, input_shape, num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build the neural network architecture."""
        print("\n🏗️  Building Neural Network...")
        
        inputs = layers.Input(shape=self.input_shape, name='lip_coordinates')
        
        # LSTM layers for temporal modeling
        x = layers.LSTM(128, return_sequences=True, name='lstm_1')(inputs)
        x = layers.Dropout(0.3, name='dropout_1')(x)
        
        x = layers.LSTM(64, return_sequences=False, name='lstm_2')(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)
        
        # Dense layers for classification
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.3, name='dropout_3')(x)
        
        x = layers.Dense(32, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.2, name='dropout_4')(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='lipreading_model')
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model built and compiled")
        print(f"   Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model."""
        print(f"\n🚀 Starting training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.history
    
    def evaluate(self, X_test, y_test, target_words):
        """Evaluate the model."""
        print(f"\n📊 Evaluating model...")
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=target_words))
        
        return test_loss, test_accuracy, y_pred_classes
    
    def save_model(self, filepath='models/lipreading_model.h5'):
        """Save the trained model."""
        self.model.save(filepath)
        print(f"✅ Model saved to {filepath}")
        
        # Save as TensorFlow.js format
        tfjs_path = 'models/tfjs_model'
        os.makedirs(tfjs_path, exist_ok=True)
        
        # Convert to TensorFlow.js
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(self.model, tfjs_path)
        print(f"✅ TensorFlow.js model saved to {tfjs_path}")


def create_data_splits(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create train/validation/test splits."""
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"\n📊 Data splits:")
    print(f"Training: {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/n_samples*100:.1f}%)")
    print(f"Test: {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_training_results(model, history, test_accuracy, target_words):
    """Save training results and metadata."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('processed_data', exist_ok=True)
    
    # Save label mapping
    label_mapping = {str(i): word for i, word in enumerate(target_words)}
    with open('processed_data/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    # Save training results
    results = {
        "model_type": "LSTM Neural Network",
        "input_shape": list(model.input_shape[1:]),
        "num_classes": len(target_words),
        "target_words": target_words,
        "test_accuracy": float(test_accuracy),
        "total_parameters": int(model.count_params()),
        "training_epochs": len(history.history['loss']),
        "final_train_accuracy": float(history.history['accuracy'][-1]),
        "final_val_accuracy": float(history.history['val_accuracy'][-1]),
        "files_generated": [
            "models/lipreading_model.h5",
            "models/tfjs_model/",
            "processed_data/label_mapping.json"
        ]
    }
    
    with open('models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Training results saved")


def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("🎯 LIPREADING MODEL TRAINING")
    print("="*60)
    
    # Configuration
    target_words = ["doctor", "glasses", "help", "pillow", "phone"]
    samples_per_word = 60
    num_people = 12
    epochs = 40
    batch_size = 32
    
    print(f"\n⚙️  Configuration:")
    print(f"Target words: {target_words}")
    print(f"Samples per word: {samples_per_word}")
    print(f"Total samples: {len(target_words) * samples_per_word}")
    print(f"Training epochs: {epochs}")
    
    # Step 1: Generate dataset
    print("\n" + "="*40)
    print("STEP 1: DATA GENERATION")
    print("="*40)
    
    data_generator = LipMovementDataGenerator(target_words)
    X, y = data_generator.generate_dataset(samples_per_word, num_people)
    
    # Step 2: Create data splits
    print("\n" + "="*40)
    print("STEP 2: DATA SPLITTING")
    print("="*40)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_data_splits(X, y)
    
    # Step 3: Build and train model
    print("\n" + "="*40)
    print("STEP 3: MODEL TRAINING")
    print("="*40)
    
    model = LipreadingNeuralNetwork(
        input_shape=X_train.shape[1:],
        num_classes=len(target_words)
    )
    
    model.build_model()
    model.model.summary()
    
    # Train model
    history = model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
    
    # Step 4: Evaluate model
    print("\n" + "="*40)
    print("STEP 4: EVALUATION")
    print("="*40)
    
    test_loss, test_accuracy, y_pred = model.evaluate(X_test, y_test, target_words)
    
    # Step 5: Save model and results
    print("\n" + "="*40)
    print("STEP 5: SAVING MODEL")
    print("="*40)
    
    model.save_model()
    save_training_results(model.model, history, test_accuracy, target_words)
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED!")
    print("="*60)
    print(f"Final Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    print(f"Model saved and ready for web app integration!")
    
    return model, history, test_accuracy


if __name__ == "__main__":
    model, history, accuracy = main()
