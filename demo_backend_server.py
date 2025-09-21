#!/usr/bin/env python3
"""
🎯 DEMO BACKEND SERVER - 75.9% Checkpoint Integration
Flask server that uses the restored 75.9% validation accuracy model
for real-time lip-reading predictions via iOS Expo Go app.
"""

import os
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import logging
from datetime import datetime
from pathlib import Path
import socket

# Import our restored checkpoint loader
from load_75_9_checkpoint import load_checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable CORS for Expo Go requests

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}
MAX_CONTENT_LENGTH = 3 * 1024 * 1024  # 3MB max file size
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.50'))
MODEL_PATH = os.getenv('MODEL_PATH', './checkpoint_75_9_percent.pth')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Global model variables
model = None
class_to_idx = None
idx_to_class = None
checkpoint = None

def get_local_ip():
    """Get the local IP address for network connections."""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the 75.9% checkpoint model."""
    global model, class_to_idx, idx_to_class, checkpoint
    
    try:
        logger.info("Loading 75.9% validation accuracy checkpoint...")
        model, class_to_idx, idx_to_class, checkpoint = load_checkpoint()
        model.eval()
        logger.info(f"✅ Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        logger.info(f"✅ Classes: {list(class_to_idx.keys())}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

def preprocess_video(video_path, target_frames=32):
    """
    Preprocess video to match training format:
    - Extract exactly 32 frames
    - Resize to 64x96
    - Convert to grayscale
    - Normalize to [0,1]
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        # Read all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames found in video")
        
        # Sample exactly target_frames frames
        if len(frames) >= target_frames:
            # Evenly sample frames
            indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            # Pad with last frame if needed
            sampled_frames = frames[:]
            while len(sampled_frames) < target_frames:
                sampled_frames.append(frames[-1])
        
        # Process frames: resize to 64x96, convert to grayscale, normalize
        processed_frames = []
        for frame in sampled_frames:
            # Resize to 64x96 (height x width)
            resized = cv2.resize(frame, (96, 64))  # cv2.resize uses (width, height)
            
            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Normalize to [0,1]
            normalized = gray.astype(np.float32) / 255.0
            
            processed_frames.append(normalized)
        
        # Convert to numpy array and add batch/channel dimensions
        video_tensor = np.array(processed_frames)  # Shape: (32, 64, 96)
        video_tensor = video_tensor[np.newaxis, np.newaxis, ...]  # Shape: (1, 1, 32, 64, 96)
        
        return torch.FloatTensor(video_tensor)
        
    except Exception as e:
        logger.error(f"Error preprocessing video: {e}")
        return None

def predict_video(video_tensor):
    """Make prediction using the loaded model."""
    try:
        with torch.no_grad():
            output = model(video_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            # Get top 2 predictions
            top2_probs, top2_indices = torch.topk(probabilities, 2, dim=1)
            
            results = []
            for i in range(2):
                class_idx = top2_indices[0, i].item()
                confidence = top2_probs[0, i].item()
                class_name = idx_to_class[class_idx]
                results.append({
                    "class": class_name,
                    "confidence": float(confidence)
                })
            
            # Determine if we should abstain (low confidence)
            abstain = results[0]["confidence"] < CONFIDENCE_THRESHOLD
            
            return {
                "top2": results,
                "abstain": abstain
            }
            
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": list(class_to_idx.keys()) if class_to_idx else [],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model_parameters": sum(p.numel() for p in model.parameters()) if model else 0,
        "server_info": "75.9% Checkpoint Demo Server"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint for video files."""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded",
                "success": False
            }), 500
        
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({
                "error": "No video file provided",
                "success": False
            }), 400
        
        file = request.files['video']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                "error": "No file selected",
                "success": False
            }), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({
                "error": f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}",
                "success": False
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing video: {filename}")
        
        # Preprocess video
        video_tensor = preprocess_video(filepath)
        if video_tensor is None:
            return jsonify({
                "error": "Failed to preprocess video",
                "success": False
            }), 400
        
        # Make prediction
        prediction_result = predict_video(video_tensor)
        if prediction_result is None:
            return jsonify({
                "error": "Failed to make prediction",
                "success": False
            }), 500
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        # Log prediction
        logger.info(f"Prediction: {prediction_result['top2'][0]['class']} "
                   f"({prediction_result['top2'][0]['confidence']:.3f})")
        
        # Return results
        return jsonify({
            "success": True,
            "top2": prediction_result["top2"],
            "abstain": prediction_result["abstain"],
            "timestamp": datetime.now().isoformat(),
            "model_info": "75.9% Validation Accuracy Checkpoint"
        })
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "success": False
        }), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint with dummy prediction."""
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "success": False
        }), 500
    
    # Create dummy input for testing
    dummy_input = torch.randn(1, 1, 32, 64, 96)
    
    try:
        prediction_result = predict_video(dummy_input)
        return jsonify({
            "success": True,
            "test_mode": True,
            "top2": prediction_result["top2"],
            "abstain": prediction_result["abstain"],
            "message": "Model is working correctly"
        })
    except Exception as e:
        return jsonify({
            "error": f"Model test failed: {str(e)}",
            "success": False
        }), 500

if __name__ == '__main__':
    print("🎯 DEMO BACKEND SERVER - 75.9% Checkpoint")
    print("=" * 50)
    
    # Load the model
    if not load_model():
        print("❌ Failed to load model. Exiting.")
        exit(1)
    
    # Get network info
    local_ip = get_local_ip()
    port = 5000
    
    print(f"✅ Model loaded: 75.9% validation accuracy checkpoint")
    print(f"✅ Classes: {', '.join(class_to_idx.keys())}")
    print(f"✅ Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"✅ Max file size: {MAX_CONTENT_LENGTH // (1024*1024)}MB")
    
    print(f"\n🌐 Server starting on:")
    print(f"   Local: http://localhost:{port}")
    print(f"   Network: http://{local_ip}:{port}")
    
    print(f"\n📱 FOR EXPO GO APP:")
    print(f"   Set EXPO_PUBLIC_API_URL to: http://{local_ip}:{port}")
    
    print(f"\n🧪 Test endpoints:")
    print(f"   Health: http://{local_ip}:{port}/health")
    print(f"   Test: http://{local_ip}:{port}/test")
    
    print(f"\n🚀 Ready for iOS demo app connections!")
    print("=" * 50)
    
    # Start the server
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=port,
        debug=False,  # Disable debug for demo
        use_reloader=False
    )
