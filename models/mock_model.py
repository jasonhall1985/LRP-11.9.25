"""
Mock Lipreading Model for Demonstration

This module provides a mock implementation of the lipreading model
that can be used for demonstration purposes when TensorFlow is not available.
"""

import numpy as np
import json
import os

class MockLipreadingModel:
    """
    Mock model that simulates lipreading predictions for demonstration.
    """
    
    def __init__(self, target_words=None):
        self.target_words = target_words or ["doctor", "glasses", "help", "pillow", "phone"]
        self.num_classes = len(self.target_words)
        
        # Set seed for consistent demo results
        np.random.seed(42)
        
        # Create word-specific patterns for more realistic predictions
        self.word_patterns = {
            "doctor": [0.7, 0.1, 0.1, 0.05, 0.05],
            "glasses": [0.1, 0.75, 0.05, 0.05, 0.05],
            "help": [0.05, 0.05, 0.8, 0.05, 0.05],
            "pillow": [0.05, 0.05, 0.05, 0.75, 0.1],
            "phone": [0.05, 0.05, 0.05, 0.1, 0.75]
        }
    
    def predict(self, sequences, verbose=0):
        """
        Generate mock predictions for lip sequences.
        
        Args:
            sequences: Input sequences (batch_size, frames, height, width)
            verbose: Verbosity level (ignored)
            
        Returns:
            Prediction probabilities of shape (batch_size, num_classes)
        """
        if hasattr(sequences, 'shape'):
            batch_size = sequences.shape[0]
        else:
            batch_size = len(sequences) if hasattr(sequences, '__len__') else 1
        
        predictions = []
        
        for i in range(batch_size):
            # Simulate analysis of lip movement patterns
            # In a real model, this would analyze the actual sequence
            
            # For demo, create semi-realistic predictions
            if hasattr(sequences, 'shape') and len(sequences.shape) == 4:
                # Analyze some basic properties of the sequence
                sequence = sequences[i]
                
                # Simple heuristic based on sequence properties
                mean_intensity = np.mean(sequence)
                std_intensity = np.std(sequence)
                
                # Map intensity patterns to words (very simplified)
                if mean_intensity > 0.6:
                    base_word = "doctor"  # High intensity -> mouth opening
                elif std_intensity > 0.3:
                    base_word = "glasses"  # High variation -> lateral movement
                elif mean_intensity < 0.3:
                    base_word = "help"  # Low intensity -> closed mouth sounds
                elif np.mean(sequence[:, 20:40, 20:40]) > 0.5:
                    base_word = "pillow"  # Center activity -> rounded sounds
                else:
                    base_word = "phone"  # Default to phone
            else:
                # Random selection for non-array inputs
                base_word = np.random.choice(self.target_words)
            
            # Get base pattern for the selected word
            base_probs = np.array(self.word_patterns[base_word])
            
            # Add some noise for realism
            noise = np.random.normal(0, 0.05, self.num_classes)
            pred = base_probs + noise
            
            # Ensure probabilities are positive and sum to 1
            pred = np.maximum(pred, 0.01)
            pred = pred / np.sum(pred)
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def evaluate(self, X_test, y_test, verbose=0):
        """
        Mock evaluation that returns realistic performance metrics.
        
        Args:
            X_test: Test sequences
            y_test: Test labels
            verbose: Verbosity level (ignored)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        # Simulate realistic performance
        mock_accuracy = 0.82 + np.random.normal(0, 0.02)  # Around 82% accuracy
        mock_loss = 0.45 + np.random.normal(0, 0.05)  # Around 0.45 loss
        
        # Ensure reasonable bounds
        mock_accuracy = np.clip(mock_accuracy, 0.7, 0.95)
        mock_loss = np.clip(mock_loss, 0.2, 0.8)
        
        return mock_loss, mock_accuracy
    
    def save(self, filepath):
        """
        Save mock model metadata.
        
        Args:
            filepath: Path to save the model metadata
        """
        model_data = {
            'model_type': 'mock_lipreading_model',
            'target_words': self.target_words,
            'num_classes': self.num_classes,
            'version': '1.0',
            'description': 'Mock model for demonstration purposes'
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def summary(self):
        """Print model summary."""
        print("Mock Lipreading Model Summary")
        print("=" * 40)
        print(f"Target words: {', '.join(self.target_words)}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Model type: Mock/Demonstration")
        print("Input shape: (batch_size, 30, 64, 64)")
        print("Output shape: (batch_size, 5)")


def load_mock_model(model_path=None):
    """
    Load or create a mock model.
    
    Args:
        model_path: Path to model metadata (optional)
        
    Returns:
        MockLipreadingModel instance
    """
    if model_path and os.path.exists(model_path):
        try:
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            target_words = model_data.get('target_words', ["doctor", "glasses", "help", "pillow", "phone"])
        except:
            target_words = ["doctor", "glasses", "help", "pillow", "phone"]
    else:
        target_words = ["doctor", "glasses", "help", "pillow", "phone"]
    
    return MockLipreadingModel(target_words)


if __name__ == "__main__":
    # Demo usage
    print("🎯 Mock Lipreading Model Demo")
    print("=" * 40)
    
    # Create model
    model = MockLipreadingModel()
    model.summary()
    
    # Test prediction
    print("\n📊 Testing prediction...")
    mock_sequence = np.random.random((1, 30, 64, 64))
    predictions = model.predict(mock_sequence)
    
    print("Predictions:")
    for i, (word, prob) in enumerate(zip(model.target_words, predictions[0])):
        print(f"  {word}: {prob:.3f} ({prob*100:.1f}%)")
    
    predicted_word = model.target_words[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    print(f"\nPredicted word: {predicted_word} (confidence: {confidence:.3f})")
    
    # Save model
    model.save('models/mock_model_metadata.json')
    print("\n✅ Mock model demo completed!")
