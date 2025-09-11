#!/usr/bin/env python3
"""
Mobile Backend Server for Lipreading iOS App

This Flask server handles video uploads from the React Native app,
processes them through our CNN-LSTM model, and returns predictions.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import tempfile
import json
import random
import time
from werkzeug.utils import secure_filename
import socket

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native requests

# Configuration
TARGET_WORDS = ["doctor", "glasses", "help", "pillow", "phone"]
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class MockLipreadingModel:
    """Mock model for demonstration - simulates our trained CNN-LSTM model."""
    
    def __init__(self):
        self.target_words = TARGET_WORDS
        random.seed(42)  # For consistent demo results
        
        # Word-specific prediction patterns for realistic demo
        self.word_patterns = {
            "doctor": [0.78, 0.08, 0.06, 0.04, 0.04],
            "glasses": [0.07, 0.81, 0.05, 0.04, 0.03],
            "help": [0.05, 0.04, 0.84, 0.04, 0.03],
            "pillow": [0.04, 0.03, 0.05, 0.83, 0.05],
            "phone": [0.03, 0.04, 0.04, 0.05, 0.84]
        }
    
    def predict_from_video(self, video_path):
        """Process video and return prediction."""
        try:
            # Extract frames from video
            frames = self.extract_frames(video_path)
            
            if len(frames) < 10:
                # Too few frames - lower confidence
                base_word = random.choice(self.target_words)
                confidence_factor = 0.65
            elif len(frames) > 150:
                # Too many frames - might be multiple words
                base_word = random.choice(self.target_words)
                confidence_factor = 0.72
            else:
                # Good number of frames - higher confidence
                base_word = random.choice(self.target_words)
                confidence_factor = 0.85
            
            # Get base probabilities
            base_probs = np.array(self.word_patterns[base_word])
            
            # Add realistic noise
            noise = np.random.normal(0, 0.02, len(self.target_words))
            predictions = base_probs * confidence_factor + noise
            
            # Ensure positive probabilities that sum to 1
            predictions = np.maximum(predictions, 0.01)
            predictions = predictions / np.sum(predictions)
            
            # Get predicted class
            predicted_idx = np.argmax(predictions)
            predicted_word = self.target_words[predicted_idx]
            confidence = float(np.max(predictions))
            
            # Create probability dictionary
            all_probabilities = {}
            for i, prob in enumerate(predictions):
                all_probabilities[self.target_words[i]] = float(prob)
            
            return {
                'predicted_word': predicted_word,
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'frames_processed': len(frames)
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def extract_frames(self, video_path):
        """Extract frames from video file."""
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale and resize (simulating preprocessing)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 64))
                frames.append(resized)
            
            cap.release()
            
        except Exception as e:
            print(f"Frame extraction error: {e}")
        
        return frames

# Initialize model
model = MockLipreadingModel()

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Health check endpoint."""
    return jsonify({
        'status': 'Lipreading Mobile Backend Server Running',
        'model_loaded': True,
        'target_words': TARGET_WORDS,
        'version': '1.0'
    })

@app.route('/health')
def health():
    """Detailed health check."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'target_words': TARGET_WORDS,
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'server_info': 'Ready for iOS app connections'
    })

@app.route('/predict_video', methods=['POST'])
def predict_video():
    """Handle video upload and prediction from React Native app."""
    try:
        print("Received video prediction request")
        
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = str(int(time.time() * 1000))
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            file.save(filepath)
            print(f"Video saved: {filepath}")
            
            # Process video through model
            result = model.predict_from_video(filepath)
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            if result:
                print(f"Prediction: {result['predicted_word']} ({result['confidence']:.3f})")
                
                return jsonify({
                    'success': True,
                    'predicted_word': result['predicted_word'],
                    'confidence': result['confidence'],
                    'all_probabilities': result['all_probabilities'],
                    'frames_processed': result['frames_processed'],
                    'processing_time_ms': 45  # Simulated processing time
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to process video'
                }), 500
        
        else:
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Supported: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
            
    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/test_prediction', methods=['GET'])
def test_prediction():
    """Test endpoint for quick prediction testing."""
    # Generate a test prediction
    word = random.choice(TARGET_WORDS)
    confidence = random.uniform(0.75, 0.95)
    
    all_probs = {}
    for w in TARGET_WORDS:
        if w == word:
            all_probs[w] = confidence
        else:
            all_probs[w] = random.uniform(0.01, 0.1)
    
    # Normalize probabilities
    total = sum(all_probs.values())
    all_probs = {k: v/total for k, v in all_probs.items()}
    
    return jsonify({
        'success': True,
        'predicted_word': word,
        'confidence': confidence,
        'all_probabilities': all_probs,
        'frames_processed': 75,
        'test_mode': True
    })

def get_local_ip():
    """Get the local IP address for mobile connections."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

if __name__ == '__main__':
    import time
    
    local_ip = get_local_ip()
    port = 5000
    
    print("🎯 LIPREADING MOBILE BACKEND SERVER")
    print("=" * 50)
    print(f"✅ Mock CNN-LSTM model loaded")
    print(f"✅ Target words: {', '.join(TARGET_WORDS)}")
    print(f"✅ Video processing ready")
    print(f"\n🌐 Server starting...")
    print(f"📱 FOR REACT NATIVE APP:")
    print(f"   Update SERVER_URL in App.js to:")
    print(f"   http://{local_ip}:{port}")
    print(f"\n💻 FOR TESTING:")
    print(f"   Health check: http://{local_ip}:{port}/health")
    print(f"   Test prediction: http://{local_ip}:{port}/test_prediction")
    print(f"\n🎉 Ready for iOS app connections!")
    print("=" * 50)
    
    # Run server
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=port,
        debug=True,
        use_reloader=False
    )
