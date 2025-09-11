"""
Lipreading CNN-LSTM Model

This module defines the neural network architecture for lipreading classification.
The model combines CNN layers for spatial feature extraction with LSTM layers
for temporal sequence modeling.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class LipreadingModel:
    """
    A CNN-LSTM model for lipreading classification.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (30, 64, 64),
                 num_classes: int = 5,
                 dropout_rate: float = 0.3):
        """
        Initialize the LipreadingModel.
        
        Args:
            input_shape: Shape of input sequences (frames, height, width)
            num_classes: Number of word classes to predict
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self) -> keras.Model:
        """
        Build the CNN-LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
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
        model = keras.Model(inputs=inputs, outputs=outputs, name='lipreading_model')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def get_model_summary(self) -> str:
        """
        Get a string representation of the model architecture.
        
        Returns:
            Model summary as string
        """
        if self.model is None:
            self.build_model()
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be built before saving")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """
        Make predictions on lip sequences.
        
        Args:
            sequences: Input sequences of shape (batch_size, frames, height, width)
            
        Returns:
            Prediction probabilities of shape (batch_size, num_classes)
        """
        if self.model is None:
            raise ValueError("Model must be built or loaded before making predictions")
        
        return self.model.predict(sequences)
    
    def predict_single(self, sequence: np.ndarray) -> Tuple[int, float]:
        """
        Make a prediction on a single lip sequence.
        
        Args:
            sequence: Input sequence of shape (frames, height, width)
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if self.model is None:
            raise ValueError("Model must be built or loaded before making predictions")
        
        # Add batch dimension
        sequence_batch = np.expand_dims(sequence, axis=0)
        
        # Get prediction
        predictions = self.model.predict(sequence_batch, verbose=0)
        
        # Get class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return predicted_class, confidence


def create_lightweight_model(input_shape: Tuple[int, int, int] = (30, 64, 64),
                            num_classes: int = 5) -> keras.Model:
    """
    Create a lightweight version of the model for mobile deployment.
    
    Args:
        input_shape: Shape of input sequences (frames, height, width)
        num_classes: Number of word classes to predict
        
    Returns:
        Compiled lightweight Keras model
    """
    inputs = layers.Input(shape=input_shape, name='lip_sequence')
    
    # Reshape for CNN processing
    x = layers.Reshape((input_shape[0], input_shape[1], input_shape[2], 1))(inputs)
    
    # Lighter CNN layers
    x = layers.TimeDistributed(
        layers.Conv2D(16, (3, 3), activation='relu', padding='same')
    )(x)
    x = layers.TimeDistributed(
        layers.MaxPooling2D((2, 2))
    )(x)
    
    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), activation='relu', padding='same')
    )(x)
    x = layers.TimeDistributed(
        layers.MaxPooling2D((2, 2))
    )(x)
    
    # Flatten and reduce dimensions
    x = layers.TimeDistributed(layers.Flatten())(x)
    
    # Simpler LSTM
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='lightweight_lipreading_model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    """
    Example usage of the LipreadingModel class.
    """
    # Create model
    model_builder = LipreadingModel()
    model = model_builder.build_model()
    
    # Print model summary
    print("Model Architecture:")
    print("=" * 50)
    print(model_builder.get_model_summary())
    
    # Create lightweight model
    lightweight_model = create_lightweight_model()
    print("\nLightweight Model Architecture:")
    print("=" * 50)
    lightweight_model.summary()


if __name__ == "__main__":
    main()
