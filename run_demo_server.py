#!/usr/bin/env python3
"""
Simple Demo Server for Lipreading App

This creates a basic HTTP server to demonstrate the lipreading app
without requiring Flask installation.
"""

import http.server
import socketserver
import json
import urllib.parse
import socket
import os
import random
import numpy as np

# Configuration
PORT = 5000
TARGET_WORDS = ["doctor", "glasses", "help", "pillow", "phone"]

class MockLipreadingModel:
    """Mock model for demonstration."""
    
    def __init__(self):
        self.target_words = TARGET_WORDS
        random.seed(42)
        
        # Word-specific patterns for realistic predictions
        self.patterns = {
            "doctor": [0.75, 0.08, 0.07, 0.05, 0.05],
            "glasses": [0.08, 0.78, 0.06, 0.04, 0.04],
            "help": [0.06, 0.05, 0.82, 0.04, 0.03],
            "pillow": [0.05, 0.04, 0.05, 0.81, 0.05],
            "phone": [0.04, 0.05, 0.04, 0.06, 0.81]
        }
    
    def predict(self, frames):
        """Generate realistic prediction."""
        # Select a word based on frame count and randomness
        base_word = random.choice(self.target_words)
        base_probs = self.patterns[base_word]
        
        # Add noise for realism
        noise = [random.gauss(0, 0.03) for _ in range(5)]
        predictions = [max(0.01, p + n) for p, n in zip(base_probs, noise)]
        
        # Normalize
        total = sum(predictions)
        predictions = [p / total for p in predictions]
        
        return predictions

# Initialize model
model = MockLipreadingModel()

class DemoRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for the demo."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/' or self.path == '/index.html':
            self.serve_index()
        elif self.path == '/health':
            self.serve_health()
        elif self.path.startswith('/static/'):
            # Serve static files
            super().do_GET()
        else:
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/predict':
            self.handle_prediction()
        else:
            self.send_error(404)
    
    def serve_index(self):
        """Serve the main page."""
        html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lipreading Demo App</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #4a5568;
            margin-bottom: 10px;
        }
        .target-words {
            text-align: center;
            margin-bottom: 30px;
            font-size: 18px;
            color: #666;
        }
        .demo-section {
            text-align: center;
            margin: 30px 0;
        }
        .demo-btn {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 18px;
            border-radius: 25px;
            cursor: pointer;
            margin: 10px;
            transition: transform 0.3s ease;
        }
        .demo-btn:hover {
            transform: translateY(-2px);
        }
        .results {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            display: none;
        }
        .prediction {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            margin: 10px 0;
        }
        .confidence {
            font-size: 18px;
            color: #2196F3;
            margin: 10px 0;
        }
        .instructions {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
        }
        .instructions h3 {
            color: #1976d2;
            margin-bottom: 15px;
        }
        .instructions ol {
            color: #666;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Lipreading Demo App</h1>
        <div class="target-words">
            <strong>Target Words:</strong> doctor, glasses, help, pillow, phone
        </div>
        
        <div class="demo-section">
            <h2>Demo Mode</h2>
            <p>Click a word below to simulate lipreading prediction:</p>
            <button class="demo-btn" onclick="simulateWord('doctor')">Doctor</button>
            <button class="demo-btn" onclick="simulateWord('glasses')">Glasses</button>
            <button class="demo-btn" onclick="simulateWord('help')">Help</button>
            <button class="demo-btn" onclick="simulateWord('pillow')">Pillow</button>
            <button class="demo-btn" onclick="simulateWord('phone')">Phone</button>
        </div>
        
        <div id="results" class="results">
            <h3>Prediction Results</h3>
            <div class="prediction" id="predicted-word"></div>
            <div class="confidence" id="confidence"></div>
        </div>
        
        <div class="instructions">
            <h3>How It Works</h3>
            <ol>
                <li>The app uses computer vision to detect lip movements</li>
                <li>A CNN-LSTM neural network analyzes the lip sequences</li>
                <li>The model predicts which of the 5 words was spoken</li>
                <li>Results show the predicted word and confidence level</li>
            </ol>
            
            <h3>For Your Class Presentation</h3>
            <p>This demonstrates the core functionality of your lipreading app. In the full version, it would use your phone's camera to capture real lip movements and make live predictions!</p>
        </div>
    </div>
    
    <script>
        function simulateWord(word) {
            // Simulate API call
            const mockData = {
                frames: Array(25).fill('mock_frame_data')
            };
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(mockData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('predicted-word').textContent = 
                        `Predicted: ${data.predicted_word.toUpperCase()}`;
                    document.getElementById('confidence').textContent = 
                        `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
                    document.getElementById('results').style.display = 'block';
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    </script>
</body>
</html>
        '''
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_health(self):
        """Serve health check."""
        response = {
            'status': 'healthy',
            'model_loaded': True,
            'target_words': TARGET_WORDS
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def handle_prediction(self):
        """Handle prediction requests."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            frames = data.get('frames', [])
            predictions = model.predict(frames)
            
            predicted_idx = predictions.index(max(predictions))
            predicted_word = TARGET_WORDS[predicted_idx]
            confidence = max(predictions)
            
            response = {
                'success': True,
                'predicted_word': predicted_word,
                'confidence': confidence,
                'all_probabilities': dict(zip(TARGET_WORDS, predictions)),
                'frames_processed': len(frames)
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            error_response = {
                'success': False,
                'error': str(e)
            }
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())

def get_local_ip():
    """Get local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def main():
    """Start the demo server."""
    print("🎯 Starting Lipreading Demo Server...")
    print("=" * 50)
    
    local_ip = get_local_ip()
    
    print(f"✅ Mock model loaded successfully!")
    print(f"✅ Target words: {', '.join(TARGET_WORDS)}")
    print(f"\n🌐 Demo server starting on port {PORT}...")
    print(f"\n📱 FOR YOUR MUM TO TEST ON IPHONE:")
    print(f"   1. Open Safari browser")
    print(f"   2. Navigate to: http://{local_ip}:{PORT}")
    print(f"   3. Click the word buttons to see predictions")
    print(f"\n💻 FOR COMPUTER TESTING:")
    print(f"   Navigate to: http://localhost:{PORT}")
    print(f"\n🎤 Demo words: {', '.join(TARGET_WORDS)}")
    print(f"\n🎉 Ready for your class presentation!")
    print("=" * 50)
    
    # Start server
    with socketserver.TCPServer(("", PORT), DemoRequestHandler) as httpd:
        print(f"Server running at http://{local_ip}:{PORT}")
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped. Great job on your presentation!")

if __name__ == "__main__":
    main()
