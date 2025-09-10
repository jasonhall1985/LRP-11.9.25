#!/usr/bin/env python3
"""
Complete Lipreading Training Pipeline

This script implements the full machine learning pipeline for the lipreading app,
including data generation, preprocessing, model training, and evaluation.
"""

import os
import sys
import numpy as np
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
import cv2
from PIL import Image, ImageEnhance, ImageOps
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

print("🎯 Lipreading Training Pipeline")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

class SyntheticDataGenerator:
    """
    Generates synthetic lip movement data for training when real video data is not available.
    """
    
    def __init__(self, target_words=None):
        self.target_words = target_words or ["doctor", "glasses", "help", "pillow", "phone"]
        self.image_size = (64, 64)
        self.sequence_length = 30
        
    def generate_lip_pattern(self, word_idx, person_id=0, noise_level=0.1):
        """
        Generate a synthetic lip movement pattern for a specific word.
        """
        # Create base patterns for each word (simplified lip shapes)
        patterns = {
            0: self._doctor_pattern(),    # doctor - open-close pattern
            1: self._glasses_pattern(),   # glasses - lateral movement
            2: self._help_pattern(),      # help - vertical emphasis
            3: self._pillow_pattern(),    # pillow - rounded shapes
            4: self._phone_pattern()      # phone - plosive pattern
        }
        
        base_pattern = patterns[word_idx]
        
        # Add person-specific variations
        person_variation = np.sin(np.linspace(0, 2*np.pi, self.sequence_length)) * 0.1 * person_id
        
        # Add noise for realism
        noise = np.random.normal(0, noise_level, (self.sequence_length, *self.image_size))
        
        # Combine patterns
        sequence = []
        for i in range(self.sequence_length):
            frame = base_pattern[i] + person_variation[i] + noise[i]
            frame = np.clip(frame, 0, 1)  # Ensure values are in [0, 1]
            sequence.append(frame)
        
        return np.array(sequence)
    
    def _doctor_pattern(self):
        """Generate pattern for 'doctor' - emphasizes mouth opening/closing."""
        frames = []
        for i in range(self.sequence_length):
            frame = np.zeros(self.image_size)
            # Create oval mouth shape that opens and closes
            center_y, center_x = 32, 32
            t = i / self.sequence_length * 2 * np.pi
            mouth_height = int(8 + 6 * np.sin(t))  # Varies from 2 to 14
            mouth_width = 12
            
            y, x = np.ogrid[:64, :64]
            mask = ((x - center_x)**2 / mouth_width**2 + 
                   (y - center_y)**2 / mouth_height**2) <= 1
            frame[mask] = 0.8 + 0.2 * np.sin(t)
            frames.append(frame)
        return frames
    
    def _glasses_pattern(self):
        """Generate pattern for 'glasses' - emphasizes lateral lip movement."""
        frames = []
        for i in range(self.sequence_length):
            frame = np.zeros(self.image_size)
            center_y, center_x = 32, 32
            t = i / self.sequence_length * 2 * np.pi
            # Lateral movement
            offset_x = int(3 * np.sin(t))
            mouth_width = 10
            mouth_height = 6
            
            y, x = np.ogrid[:64, :64]
            mask = ((x - (center_x + offset_x))**2 / mouth_width**2 + 
                   (y - center_y)**2 / mouth_height**2) <= 1
            frame[mask] = 0.7 + 0.3 * np.cos(t)
            frames.append(frame)
        return frames
    
    def _help_pattern(self):
        """Generate pattern for 'help' - emphasizes vertical movement."""
        frames = []
        for i in range(self.sequence_length):
            frame = np.zeros(self.image_size)
            center_y, center_x = 32, 32
            t = i / self.sequence_length * 2 * np.pi
            # Vertical emphasis
            mouth_width = 8
            mouth_height = int(6 + 4 * np.abs(np.sin(t)))
            
            y, x = np.ogrid[:64, :64]
            mask = ((x - center_x)**2 / mouth_width**2 + 
                   (y - center_y)**2 / mouth_height**2) <= 1
            frame[mask] = 0.6 + 0.4 * np.abs(np.sin(t))
            frames.append(frame)
        return frames
    
    def _pillow_pattern(self):
        """Generate pattern for 'pillow' - emphasizes rounded shapes."""
        frames = []
        for i in range(self.sequence_length):
            frame = np.zeros(self.image_size)
            center_y, center_x = 32, 32
            t = i / self.sequence_length * 2 * np.pi
            # Rounded pattern
            radius = 8 + 3 * np.sin(t)
            
            y, x = np.ogrid[:64, :64]
            mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
            frame[mask] = 0.5 + 0.5 * np.sin(t + np.pi/4)
            frames.append(frame)
        return frames
    
    def _phone_pattern(self):
        """Generate pattern for 'phone' - emphasizes plosive pattern."""
        frames = []
        for i in range(self.sequence_length):
            frame = np.zeros(self.image_size)
            center_y, center_x = 32, 32
            t = i / self.sequence_length * 2 * np.pi
            # Plosive pattern - quick open/close
            if i < 10:  # Initial closure
                mouth_size = 2
            elif i < 20:  # Quick opening
                mouth_size = 12
            else:  # Gradual closing
                mouth_size = 8 - (i - 20) * 0.3
            
            y, x = np.ogrid[:64, :64]
            mask = ((x - center_x)**2 + (y - center_y)**2) <= mouth_size**2
            frame[mask] = 0.7 + 0.3 * np.sin(t)
            frames.append(frame)
        return frames
    
    def apply_augmentation(self, sequence):
        """Apply data augmentation to a sequence."""
        augmented = sequence.copy()
        
        # Random horizontal flip (50% chance)
        if np.random.random() > 0.5:
            augmented = np.flip(augmented, axis=2)
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        augmented = np.clip(augmented * brightness_factor, 0, 1)
        
        # Random noise
        noise = np.random.normal(0, 0.05, augmented.shape)
        augmented = np.clip(augmented + noise, 0, 1)
        
        # Random rotation (small angle)
        if np.random.random() > 0.7:
            angle = np.random.uniform(-5, 5)
            for i in range(len(augmented)):
                # Convert to PIL for rotation
                img = Image.fromarray((augmented[i] * 255).astype(np.uint8))
                img = img.rotate(angle, fillcolor=0)
                augmented[i] = np.array(img) / 255.0
        
        return augmented
    
    def generate_dataset(self, samples_per_word=20, num_people=5, augment=True):
        """
        Generate a complete synthetic dataset.
        """
        print(f"\n📊 Generating synthetic dataset...")
        print(f"Words: {self.target_words}")
        print(f"Samples per word: {samples_per_word}")
        print(f"Number of people: {num_people}")
        print(f"Augmentation: {augment}")
        
        X = []
        y = []
        
        total_samples = len(self.target_words) * samples_per_word
        pbar = tqdm(total=total_samples, desc="Generating samples")
        
        for word_idx, word in enumerate(self.target_words):
            for sample_idx in range(samples_per_word):
                person_id = sample_idx % num_people
                
                # Generate base sequence
                sequence = self.generate_lip_pattern(word_idx, person_id)
                
                # Apply augmentation
                if augment:
                    sequence = self.apply_augmentation(sequence)
                
                X.append(sequence)
                y.append(word_idx)
                pbar.update(1)
        
        pbar.close()
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Dataset generated: {X.shape}")
        return X, y


class LipreadingModel:
    """
    CNN-LSTM model for lipreading classification.
    """

    def __init__(self, input_shape=(30, 64, 64), num_classes=5, dropout_rate=0.3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None

    def build_model(self):
        """Build the CNN-LSTM model architecture."""
        print("\n🏗️  Building CNN-LSTM model...")

        inputs = layers.Input(shape=self.input_shape, name='lip_sequence')

        # Reshape for CNN processing (add channel dimension)
        x = layers.Reshape((self.input_shape[0], self.input_shape[1], self.input_shape[2], 1))(inputs)

        # TimeDistributed CNN layers for spatial feature extraction
        x = layers.TimeDistributed(
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            name='conv2d_1'
        )(x)
        x = layers.TimeDistributed(
            layers.BatchNormalization(),
            name='batch_norm_1'
        )(x)
        x = layers.TimeDistributed(
            layers.MaxPooling2D((2, 2)),
            name='maxpool_1'
        )(x)

        x = layers.TimeDistributed(
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            name='conv2d_2'
        )(x)
        x = layers.TimeDistributed(
            layers.BatchNormalization(),
            name='batch_norm_2'
        )(x)
        x = layers.TimeDistributed(
            layers.MaxPooling2D((2, 2)),
            name='maxpool_2'
        )(x)

        x = layers.TimeDistributed(
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            name='conv2d_3'
        )(x)
        x = layers.TimeDistributed(
            layers.BatchNormalization(),
            name='batch_norm_3'
        )(x)
        x = layers.TimeDistributed(
            layers.MaxPooling2D((2, 2)),
            name='maxpool_3'
        )(x)

        # Flatten spatial dimensions for LSTM
        x = layers.TimeDistributed(
            layers.Flatten(),
            name='flatten'
        )(x)

        # Add dropout for regularization
        x = layers.TimeDistributed(
            layers.Dropout(self.dropout_rate),
            name='dropout_1'
        )(x)

        # LSTM layers for temporal modeling
        x = layers.LSTM(128, return_sequences=True, name='lstm_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)

        x = layers.LSTM(64, return_sequences=False, name='lstm_2')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_3')(x)

        # Dense layers for classification
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_4')(x)

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
        return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
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
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
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
        """Evaluate the model on test data."""
        print(f"\n📊 Evaluating model on test set...")

        # Get predictions
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=target_words))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        self.plot_confusion_matrix(cm, target_words)

        return test_loss, test_accuracy, y_pred_classes

    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, filepath='models/lipreading_model.h5'):
        """Save the trained model."""
        self.model.save(filepath)
        print(f"✅ Model saved to {filepath}")

        # Also save as TensorFlow Lite for mobile deployment
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()

        tflite_path = filepath.replace('.h5', '.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ TensorFlow Lite model saved to {tflite_path}")


def create_data_splits(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create train/validation/test splits."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

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

    print(f"\n📊 Data splits created:")
    print(f"Training: {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/n_samples*100:.1f}%)")
    print(f"Test: {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_label_encoder(target_words):
    """Save label encoder for web app."""
    os.makedirs('processed_data', exist_ok=True)

    label_encoder = LabelEncoder()
    label_encoder.fit(target_words)

    # Save as pickle
    with open('processed_data/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Save as JSON for inspection
    label_mapping = {str(i): word for i, word in enumerate(target_words)}
    with open('processed_data/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)

    print("✅ Label encoder saved")


def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("🎯 LIPREADING MODEL TRAINING PIPELINE")
    print("="*60)

    # Configuration
    target_words = ["doctor", "glasses", "help", "pillow", "phone"]
    samples_per_word = 40  # Increased for better training
    num_people = 8  # Simulate different speakers
    epochs = 30  # Reduced for faster execution
    batch_size = 16

    print(f"\n⚙️  Configuration:")
    print(f"Target words: {target_words}")
    print(f"Samples per word: {samples_per_word}")
    print(f"Number of people: {num_people}")
    print(f"Total samples: {len(target_words) * samples_per_word}")
    print(f"Training epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    # Step 1: Generate synthetic dataset
    print("\n" + "="*60)
    print("STEP 1: DATA GENERATION")
    print("="*60)

    data_generator = SyntheticDataGenerator(target_words)
    X, y = data_generator.generate_dataset(
        samples_per_word=samples_per_word,
        num_people=num_people,
        augment=True
    )

    # Step 2: Create data splits
    print("\n" + "="*60)
    print("STEP 2: DATA SPLITTING")
    print("="*60)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_data_splits(
        X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    # Display class distribution
    print(f"\n📊 Class distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for word_idx, count in zip(unique, counts):
        print(f"  {target_words[word_idx]}: {count} samples")

    # Step 3: Build and train model
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING")
    print("="*60)

    model = LipreadingModel(
        input_shape=X_train.shape[1:],
        num_classes=len(target_words),
        dropout_rate=0.3
    )

    # Build model
    model.build_model()
    model.model.summary()

    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )

    # Step 4: Plot training history
    print("\n" + "="*60)
    print("STEP 4: TRAINING VISUALIZATION")
    print("="*60)

    model.plot_training_history()

    # Step 5: Evaluate model
    print("\n" + "="*60)
    print("STEP 5: MODEL EVALUATION")
    print("="*60)

    test_loss, test_accuracy, y_pred = model.evaluate(X_test, y_test, target_words)

    # Step 6: Save model and components
    print("\n" + "="*60)
    print("STEP 6: SAVING MODEL")
    print("="*60)

    model.save_model()
    save_label_encoder(target_words)

    # Step 7: Final summary
    print("\n" + "="*60)
    print("🎉 TRAINING PIPELINE COMPLETED!")
    print("="*60)

    print(f"\n📈 Final Results:")
    print(f"  Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    print(f"  Test Loss: {test_loss:.3f}")

    print(f"\n📁 Generated Files:")
    print(f"  ✅ models/lipreading_model.h5 - Trained Keras model")
    print(f"  ✅ models/lipreading_model.tflite - Mobile-optimized model")
    print(f"  ✅ models/best_model.h5 - Best checkpoint during training")
    print(f"  ✅ processed_data/label_encoder.pkl - Label encoder")
    print(f"  ✅ processed_data/label_mapping.json - Word mappings")
    print(f"  ✅ models/confusion_matrix.png - Confusion matrix plot")
    print(f"  ✅ models/training_history.png - Training history plots")

    print(f"\n🚀 Ready for Web App Deployment!")
    print(f"Run the Flask app with: python src/web_app/app.py")

    return model, history, test_accuracy


if __name__ == "__main__":
    # Set up matplotlib for non-interactive backend
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

    # Run the complete pipeline
    model, history, accuracy = main()
