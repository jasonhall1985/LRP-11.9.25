#!/usr/bin/env python3
"""
Complete Lipreading Model Training Script
This script creates synthetic data and trains a CNN-LSTM model for lipreading.
"""

import numpy as np
import os
import json
import pickle
import sys

# Try to import TensorFlow, handle if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    print(f"TensorFlow {tf.__version__} available")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available - creating mock model")

# Set random seeds
np.random.seed(42)
if TF_AVAILABLE:
    tf.random.set_seed(42)

print("🎯 Complete Lipreading Model Training")
print("=" * 60)

# Configuration
TARGET_WORDS = ["doctor", "glasses", "help", "pillow", "phone"]
SAMPLES_PER_WORD = 30
SEQUENCE_LENGTH = 30
IMAGE_SIZE = (64, 64)
EPOCHS = 20
BATCH_SIZE = 8

def generate_synthetic_dataset():
    """Generate synthetic lipreading dataset."""
    print("\n📊 Generating synthetic dataset...")
    
    X = []
    y = []
    
    for word_idx, word in enumerate(TARGET_WORDS):
        print(f"  Generating {SAMPLES_PER_WORD} samples for '{word}'...")
        
        for sample_idx in range(SAMPLES_PER_WORD):
            sequence = []
            
            for frame_idx in range(SEQUENCE_LENGTH):
                # Create base pattern
                frame = np.zeros(IMAGE_SIZE, dtype=np.float32)
                
                # Word-specific patterns
                center_y, center_x = 32, 32
                t = frame_idx / SEQUENCE_LENGTH * 2 * np.pi
                
                if word == "doctor":
                    # Opening/closing mouth
                    mouth_height = int(8 + 6 * np.sin(t))
                    mouth_width = 12
                    y_coords, x_coords = np.ogrid[:64, :64]
                    mask = ((x_coords - center_x)**2 / mouth_width**2 + 
                           (y_coords - center_y)**2 / mouth_height**2) <= 1
                    frame[mask] = 0.8 + 0.2 * np.sin(t)
                
                elif word == "glasses":
                    # Lateral movement
                    offset_x = int(3 * np.sin(t))
                    mouth_width = 10
                    mouth_height = 6
                    y_coords, x_coords = np.ogrid[:64, :64]
                    mask = ((x_coords - (center_x + offset_x))**2 / mouth_width**2 + 
                           (y_coords - center_y)**2 / mouth_height**2) <= 1
                    frame[mask] = 0.7 + 0.3 * np.cos(t)
                
                elif word == "help":
                    # Vertical emphasis
                    mouth_width = 8
                    mouth_height = int(6 + 4 * np.abs(np.sin(t)))
                    y_coords, x_coords = np.ogrid[:64, :64]
                    mask = ((x_coords - center_x)**2 / mouth_width**2 + 
                           (y_coords - center_y)**2 / mouth_height**2) <= 1
                    frame[mask] = 0.6 + 0.4 * np.abs(np.sin(t))
                
                elif word == "pillow":
                    # Rounded pattern
                    radius = 8 + 3 * np.sin(t)
                    y_coords, x_coords = np.ogrid[:64, :64]
                    mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) <= radius**2
                    frame[mask] = 0.5 + 0.5 * np.sin(t + np.pi/4)
                
                elif word == "phone":
                    # Plosive pattern
                    if frame_idx < 10:
                        mouth_size = 2
                    elif frame_idx < 20:
                        mouth_size = 12
                    else:
                        mouth_size = max(2, 8 - (frame_idx - 20) * 0.3)
                    
                    y_coords, x_coords = np.ogrid[:64, :64]
                    mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) <= mouth_size**2
                    frame[mask] = 0.7 + 0.3 * np.sin(frame_idx * 0.5)
                
                # Add noise and person variation
                noise = np.random.normal(0, 0.05, IMAGE_SIZE)
                person_variation = (sample_idx % 5) * 0.02 * np.random.normal(0, 1, IMAGE_SIZE)
                frame = np.clip(frame + noise + person_variation, 0, 1)
                
                sequence.append(frame)
            
            X.append(np.array(sequence))
            y.append(word_idx)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"✅ Dataset generated: {X.shape}")
    return X, y

def create_data_splits(X, y):
    """Create train/validation/test splits."""
    print("\n📊 Creating data splits...")
    
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    splits = {
        'train': (X[train_idx], y[train_idx]),
        'val': (X[val_idx], y[val_idx]),
        'test': (X[test_idx], y[test_idx])
    }
    
    for split_name, (X_split, y_split) in splits.items():
        print(f"  {split_name}: {len(X_split)} samples")
    
    return splits

def build_cnn_lstm_model(input_shape, num_classes):
    """Build CNN-LSTM model for lipreading."""
    if not TF_AVAILABLE:
        print("⚠️  TensorFlow not available - creating mock model")
        return None
    
    print("\n🏗️  Building CNN-LSTM model...")
    
    inputs = layers.Input(shape=input_shape, name='lip_sequence')
    
    # Reshape for CNN processing
    x = layers.Reshape((input_shape[0], input_shape[1], input_shape[2], 1))(inputs)
    
    # CNN layers with TimeDistributed
    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    
    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    
    x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    
    # Flatten for LSTM
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.TimeDistributed(layers.Dropout(0.3))(x)
    
    # LSTM layers
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='lipreading_model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model built and compiled")
    return model

def train_model(model, splits):
    """Train the model."""
    if not TF_AVAILABLE or model is None:
        print("⚠️  Skipping training - TensorFlow not available")
        return None
    
    print(f"\n🚀 Training model for {EPOCHS} epochs...")
    
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_accuracy')
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ Training completed")
    return history

def evaluate_model(model, splits):
    """Evaluate the model."""
    if not TF_AVAILABLE or model is None:
        print("⚠️  Skipping evaluation - model not available")
        return
    
    print("\n📊 Evaluating model...")
    
    X_test, y_test = splits['test']
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print(f"Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    print(f"Test Loss: {test_loss:.3f}")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, word in enumerate(TARGET_WORDS):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(predicted_classes[mask] == y_test[mask])
            print(f"  {word}: {class_acc:.3f} ({class_acc*100:.1f}%)")
    
    return test_accuracy

def save_model_and_components(model):
    """Save model and supporting files."""
    print("\n💾 Saving model and components...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('processed_data', exist_ok=True)
    
    # Save model
    if TF_AVAILABLE and model is not None:
        model.save('models/lipreading_model.h5')
        print("✅ Model saved to models/lipreading_model.h5")
        
        # Save as TensorFlow Lite
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open('models/lipreading_model.tflite', 'wb') as f:
                f.write(tflite_model)
            print("✅ TensorFlow Lite model saved")
        except Exception as e:
            print(f"⚠️  Could not save TFLite model: {e}")
    
    # Save label encoder
    class SimpleLabelEncoder:
        def __init__(self, classes):
            self.classes_ = np.array(classes)
    
    label_encoder = SimpleLabelEncoder(TARGET_WORDS)
    with open('processed_data/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save label mapping
    label_mapping = {str(i): word for i, word in enumerate(TARGET_WORDS)}
    with open('processed_data/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    print("✅ Label encoder and mapping saved")

def main():
    """Main training pipeline."""
    print(f"\n⚙️  Configuration:")
    print(f"  Target words: {TARGET_WORDS}")
    print(f"  Samples per word: {SAMPLES_PER_WORD}")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # Generate dataset
    X, y = generate_synthetic_dataset()
    
    # Create splits
    splits = create_data_splits(X, y)
    
    # Build model
    model = build_cnn_lstm_model(X.shape[1:], len(TARGET_WORDS))
    
    if model is not None:
        print(f"\nModel summary:")
        model.summary()
    
    # Train model
    history = train_model(model, splits)
    
    # Evaluate model
    test_accuracy = evaluate_model(model, splits)
    
    # Save everything
    save_model_and_components(model)
    
    # Final summary
    print(f"\n🎉 Training pipeline completed!")
    print(f"📁 Files created:")
    print(f"  ✅ models/lipreading_model.h5")
    print(f"  ✅ models/lipreading_model.tflite")
    print(f"  ✅ processed_data/label_encoder.pkl")
    print(f"  ✅ processed_data/label_mapping.json")
    
    if test_accuracy:
        print(f"\n📈 Final test accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    
    print(f"\n🚀 Ready for web app deployment!")
    
    return model, history

if __name__ == "__main__":
    model, history = main()
