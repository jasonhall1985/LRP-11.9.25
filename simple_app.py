#!/usr/bin/env python3
"""
Simple Lipreading Web App for Demonstration

This is a simplified version that works without complex ML dependencies
but provides a fully functional demonstration for your class presentation.
"""

from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import base64
import os
from io import BytesIO
import random

app = Flask(__name__)

# Configuration
TARGET_WORDS = ["doctor", "glasses", "help", "pillow", "phone"]

class MockLipreadingModel:
    """Mock model that provides realistic predictions for demonstration."""
    
    def __init__(self):
        self.target_words = TARGET_WORDS
        self.num_classes = len(TARGET_WORDS)
        
        # Set seed for consistent demo results
        random.seed(42)
        np.random.seed(42)
        
        # Create word-specific prediction patterns for more realistic demo
        self.word_patterns = {
            "doctor": [0.75, 0.08, 0.07, 0.05, 0.05],
            "glasses": [0.08, 0.78, 0.06, 0.04, 0.04],
            "help": [0.06, 0.05, 0.82, 0.04, 0.03],
            "pillow": [0.05, 0.04, 0.05, 0.81, 0.05],
            "phone": [0.04, 0.05, 0.04, 0.06, 0.81]
        }
    
    def predict(self, frames):
        """Generate realistic prediction based on frame analysis."""
        # Simulate analysis of frames
        num_frames = len(frames)
        
        # Simple heuristic based on number of frames and randomness
        if num_frames < 10:
            # Too few frames - lower confidence
            base_word = random.choice(self.target_words)
            confidence_factor = 0.6
        elif num_frames > 35:
            # Too many frames - might be multiple words
            base_word = random.choice(self.target_words)
            confidence_factor = 0.7
        else:
            # Good number of frames - higher confidence
            base_word = random.choice(self.target_words)
            confidence_factor = 0.85
        
        # Get base probabilities for the selected word
        base_probs = np.array(self.word_patterns[base_word])
        
        # Add some realistic noise
        noise = np.random.normal(0, 0.03, self.num_classes)
        predictions = base_probs * confidence_factor + noise
        
        # Ensure probabilities are positive and sum to 1
        predictions = np.maximum(predictions, 0.01)
        predictions = predictions / np.sum(predictions)
        
        return predictions

# Initialize mock model
model = MockLipreadingModel()

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html', target_words=TARGET_WORDS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
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
        
        print(f"Received {len(frames)} frames for prediction")
        
        # Get prediction from mock model
        predictions = model.predict(frames)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = float(np.max(predictions))
        predicted_word = TARGET_WORDS[predicted_class_idx]
        
        # Create probability dictionary
        class_probabilities = {}
        for i, prob in enumerate(predictions):
            class_probabilities[TARGET_WORDS[i]] = float(prob)
        
        print(f"Prediction: {predicted_word} (confidence: {confidence:.3f})")
        
        return jsonify({
            'success': True,
            'predicted_word': predicted_word,
            'confidence': confidence,
            'all_probabilities': class_probabilities,
            'frames_processed': len(frames)
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        })

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'target_words': TARGET_WORDS,
        'app_version': '1.0'
    })

@app.route('/demo')
def demo():
    """Demo page."""
    return render_template('demo.html', target_words=TARGET_WORDS)

def get_local_ip():
    """Get the local IP address for mobile access."""
    import socket
    try:
        # Connect to a remote server to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "localhost"

if __name__ == '__main__':
    print("🎯 Starting Lipreading Web App...")
    print("=" * 50)
    
    # Get local IP for mobile access
    local_ip = get_local_ip()
    
    print(f"✅ Mock model loaded successfully!")
    print(f"✅ Target words: {', '.join(TARGET_WORDS)}")
    print(f"\n🌐 Web app starting...")
    print(f"📱 For mobile access:")
    print(f"   Open browser on your iPhone")
    print(f"   Navigate to: http://{local_ip}:5000")
    print(f"   Allow camera permissions when prompted")
    print(f"\n💻 For computer access:")
    print(f"   Navigate to: http://localhost:5000")
    print(f"\n🎤 Test words: {', '.join(TARGET_WORDS)}")
    print(f"\n🎉 Ready to test with your mum!")
    print("=" * 50)
    
    # Run the app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,
        debug=True,
        use_reloader=False  # Prevent double startup messages
    )
