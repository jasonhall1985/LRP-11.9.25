"""
Model Training Script

This script handles the complete training pipeline for the lipreading model,
including data loading, model training, validation, and evaluation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import pickle
import json

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.lipreading_model import LipreadingModel, create_lightweight_model
from preprocessing.data_processor import DataProcessor


class ModelTrainer:
    """
    Handles the complete model training pipeline.
    """
    
    def __init__(self, target_words: list = None):
        """
        Initialize the ModelTrainer.
        
        Args:
            target_words: List of target words to recognize
        """
        self.target_words = target_words or ["doctor", "glasses", "help", "pillow", "phone"]
        self.model = None
        self.history = None
        self.data_processor = DataProcessor(target_words=self.target_words)
        
    def load_or_process_data(self, force_reprocess: bool = False):
        """
        Load processed data or process raw data if needed.
        
        Args:
            force_reprocess: Whether to force reprocessing of raw data
            
        Returns:
            Dictionary containing train, validation, and test data
        """
        processed_data_dir = 'processed_data'
        
        # Check if processed data exists
        if not force_reprocess and os.path.exists(processed_data_dir):
            try:
                data = {}
                for split in ['training_set', 'validation_set', 'test_set']:
                    X, y = self.data_processor.load_processed_data(split)
                    if len(X) > 0:
                        data[split] = {'X': X, 'y': y}
                
                if data:
                    print("Loaded processed data from disk")
                    return data
            except Exception as e:
                print(f"Error loading processed data: {e}")
                print("Will reprocess raw data...")
        
        # Process raw data
        print("Processing raw data...")
        data = self.data_processor.process_all_data(max_frames=30, save_processed=True)
        return data
    
    def train_model(self, 
                   data: dict,
                   epochs: int = 50,
                   batch_size: int = 16,
                   validation_split: float = 0.2,
                   use_lightweight: bool = False):
        """
        Train the lipreading model.
        
        Args:
            data: Dictionary containing processed data
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of training data to use for validation
            use_lightweight: Whether to use the lightweight model architecture
        """
        if 'training_set' not in data:
            raise ValueError("Training data not found")
        
        X_train, y_train = data['training_set']['X'], data['training_set']['y']
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        
        # Create model
        if use_lightweight:
            self.model = create_lightweight_model(
                input_shape=X_train.shape[1:],
                num_classes=len(self.target_words)
            )
            print("Using lightweight model architecture")
        else:
            model_builder = LipreadingModel(
                input_shape=X_train.shape[1:],
                num_classes=len(self.target_words)
            )
            self.model = model_builder.build_model()
            print("Using full model architecture")
        
        # Print model summary
        self.model.summary()
        
        # Prepare validation data
        if 'validation_set' in data:
            X_val, y_val = data['validation_set']['X'], data['validation_set']['y']
            validation_data = (X_val, y_val)
            print(f"Using separate validation set: {X_val.shape}")
        else:
            validation_data = None
            print(f"Using validation split: {validation_split}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
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
        print("\nStarting training...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split if validation_data is None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
    
    def evaluate_model(self, data: dict):
        """
        Evaluate the trained model on test data.
        
        Args:
            data: Dictionary containing processed data
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        if 'test_set' not in data:
            print("No test set available for evaluation")
            return
        
        X_test, y_test = data['test_set']['X'], data['test_set']['y']
        
        print("\nEvaluating model on test set...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred_classes,
            target_names=self.target_words
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        self.plot_confusion_matrix(cm, self.target_words)
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
        """
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
        """
        Plot training history.
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath: str = 'models/lipreading_model.h5'):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Also save as TensorFlow Lite for mobile deployment
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        
        tflite_path = filepath.replace('.h5', '.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TensorFlow Lite model saved to {tflite_path}")


def main():
    """
    Main training function.
    """
    print("Lipreading Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load or process data
    data = trainer.load_or_process_data(force_reprocess=False)
    
    if not data:
        print("No data available for training. Please check your data directory.")
        return
    
    # Print data statistics
    trainer.data_processor.get_data_statistics(data)
    
    # Train model
    trainer.train_model(
        data=data,
        epochs=50,
        batch_size=16,
        use_lightweight=False  # Set to True for faster training/mobile deployment
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    trainer.evaluate_model(data)
    
    # Save model
    trainer.save_model()
    
    print("\nTraining pipeline completed successfully!")
    print("Model saved in 'models/' directory")
    print("Ready for web app integration!")


if __name__ == "__main__":
    main()
