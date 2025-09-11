"""
Flask Web Application for Lipreading

This is the main web application that provides a mobile-friendly interface
for real-time lipreading using the trained model.
"""

import os
import sys
import numpy as np
import cv2
import base64
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import json
from io import BytesIO
from PIL import Image

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.lip_detector import LipDetector

app = Flask(__name__)
CORS(app)

# Global variables for model and components
model = None
lip_detector = None
label_encoder = None
target_words = ["doctor", "glasses", "help", "pillow", "phone"]


def load_model_and_components():
    """
    Load the trained model and required components.
    """
    global model, lip_detector, label_encoder

    try:
        # Try to load TensorFlow model first
        model_path = os.path.join('models', 'lipreading_model.h5')
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                print(f"TensorFlow model loaded from {model_path}")
            except Exception as e:
                print(f"Could not load TensorFlow model: {e}")
                model = None
        else:
            print(f"TensorFlow model file not found: {model_path}")
            model = None

        # If TensorFlow model not available, use mock model
        if model is None:
            print("Loading mock model for demonstration...")
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
            try:
                from mock_model import MockLipreadingModel
                model = MockLipreadingModel(target_words)
                print("Mock model loaded successfully")
            except Exception as e:
                print(f"Could not load mock model: {e}")
                return False

        # Initialize lip detector (simplified for demo)
        try:
            lip_detector = LipDetector(target_size=(64, 64))
            print("Lip detector initialized")
        except Exception as e:
            print(f"Could not initialize lip detector: {e}")
            # Create a simple mock lip detector
            class MockLipDetector:
                def detect_lips_in_frame(self, frame):
                    # Return a simple mock lip region
                    return np.random.random((64, 64)).astype(np.float32)

            lip_detector = MockLipDetector()
            print("Mock lip detector initialized")

        # Load label encoder
        label_encoder_path = os.path.join('processed_data', 'label_encoder.pkl')
        if os.path.exists(label_encoder_path):
            try:
                with open(label_encoder_path, 'rb') as f:
                    label_encoder = pickle.load(f)
                print("Label encoder loaded")
            except Exception as e:
                print(f"Could not load label encoder: {e}")
                label_encoder = None
        else:
            print("Label encoder not found, using default word order")
            label_encoder = None

        return True

    except Exception as e:
        print(f"Error loading model components: {e}")
        return False


def decode_base64_image(base64_string):
    """
    Decode base64 image string to numpy array.
    
    Args:
        base64_string: Base64 encoded image
        
    Returns:
        Numpy array representing the image
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_data))
        
        # Convert to numpy array (RGB)
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
        
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


@app.route('/')
def index():
    """
    Main page route.
    """
    return render_template('index.html', target_words=target_words)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from the frontend.
    """
    global model, lip_detector, label_encoder
    
    if model is None or lip_detector is None:
        return jsonify({
            'error': 'Model not loaded',
            'success': False
        })
    
    try:
        data = request.get_json()
        
        if 'frames' not in data:
            return jsonify({
                'error': 'No frames provided',
                'success': False
            })
        
        frames = data['frames']
        
        if len(frames) == 0:
            return jsonify({
                'error': 'Empty frames list',
                'success': False
            })
        
        # Process frames to extract lip regions
        lip_sequences = []
        
        for frame_data in frames:
            # Decode base64 image
            frame = decode_base64_image(frame_data)
            
            if frame is not None:
                # Extract lip region
                lip_region = lip_detector.detect_lips_in_frame(frame)
                
                if lip_region is not None:
                    lip_sequences.append(lip_region)
        
        if len(lip_sequences) == 0:
            return jsonify({
                'error': 'No lip regions detected in frames',
                'success': False
            })
        
        # Pad or truncate to 30 frames
        target_frames = 30
        if len(lip_sequences) < target_frames:
            # Pad with last frame
            last_frame = lip_sequences[-1]
            padding_needed = target_frames - len(lip_sequences)
            for _ in range(padding_needed):
                lip_sequences.append(last_frame)
        elif len(lip_sequences) > target_frames:
            # Truncate to target frames
            lip_sequences = lip_sequences[:target_frames]
        
        # Convert to numpy array and add batch dimension
        sequence = np.array(lip_sequences)
        sequence_batch = np.expand_dims(sequence, axis=0)
        
        # Make prediction
        predictions = model.predict(sequence_batch, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get word label
        if label_encoder is not None:
            predicted_word = label_encoder.classes_[predicted_class_idx]
        else:
            predicted_word = target_words[predicted_class_idx]
        
        # Get all class probabilities
        class_probabilities = {}
        for i, prob in enumerate(predictions[0]):
            if label_encoder is not None:
                word = label_encoder.classes_[i]
            else:
                word = target_words[i]
            class_probabilities[word] = float(prob)
        
        return jsonify({
            'success': True,
            'predicted_word': predicted_word,
            'confidence': confidence,
            'all_probabilities': class_probabilities,
            'frames_processed': len(lip_sequences)
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        })


@app.route('/health')
def health():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'lip_detector_loaded': lip_detector is not None,
        'target_words': target_words
    })


@app.route('/demo')
def demo():
    """
    Demo page with example predictions.
    """
    return render_template('demo.html', target_words=target_words)


if __name__ == '__main__':
    print("Starting Lipreading Web App...")
    print("=" * 50)
    
    # Load model and components
    if load_model_and_components():
        print("All components loaded successfully!")
        print("\nTarget words:", ", ".join(target_words))
        print("\nStarting Flask server...")
        print("Access the app at: http://localhost:5000")
        print("For mobile testing, use your computer's IP address")
        
        # Run the app
        app.run(
            host='0.0.0.0',  # Allow external connections
            port=5000,
            debug=True
        )
    else:
        print("Failed to load model components.")
        print("Please ensure you have:")
        print("1. Trained model in 'models/lipreading_model.h5'")
        print("2. Label encoder in 'processed_data/label_encoder.pkl'")
        print("3. Run the training script first: python src/training/train_model.py")
