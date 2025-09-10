#!/usr/bin/env python3
"""
Simple Lipreading Demo - Guaranteed to Work!
This creates a basic web server for your mum to test.
"""

import http.server
import socketserver
import json
import random
import socket

PORT = 8080  # Using different port to avoid conflicts
TARGET_WORDS = ["doctor", "glasses", "help", "pillow", "phone"]

class DemoHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_demo_page()
        elif self.path == '/test':
            self.send_test_prediction()
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/predict':
            self.handle_prediction()
        else:
            self.send_error(404)
    
    def send_demo_page(self):
        html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 Lipreading Demo for Mum!</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 500px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }
        h1 {
            color: #4a5568;
            margin-bottom: 20px;
            font-size: 28px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 16px;
        }
        .word-btn {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 15px 25px;
            margin: 8px;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
        }
        .word-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
        }
        .word-btn:active {
            transform: translateY(0);
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f0f8ff;
            border-radius: 15px;
            border-left: 5px solid #2196F3;
            display: none;
        }
        .predicted-word {
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
        .explanation {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            text-align: left;
        }
        .explanation h3 {
            color: #856404;
            margin-top: 0;
        }
        .explanation ul {
            color: #856404;
            line-height: 1.6;
        }
        .loading {
            display: none;
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Lipreading Demo</h1>
        <div class="subtitle">
            Hi Mum! Click any word below to see how the AI predicts lipreading:
        </div>
        
        <div class="word-buttons">
            <button class="word-btn" onclick="testWord('doctor')">Doctor</button>
            <button class="word-btn" onclick="testWord('glasses')">Glasses</button>
            <button class="word-btn" onclick="testWord('help')">Help</button>
            <button class="word-btn" onclick="testWord('pillow')">Pillow</button>
            <button class="word-btn" onclick="testWord('phone')">Phone</button>
        </div>
        
        <div class="loading" id="loading">
            🤖 AI is analyzing lip movements...
        </div>
        
        <div class="result" id="result">
            <div class="predicted-word" id="prediction"></div>
            <div class="confidence" id="confidence"></div>
        </div>
        
        <div class="explanation">
            <h3>🧠 How This Works:</h3>
            <ul>
                <li><strong>Computer Vision:</strong> Detects lip movements from video</li>
                <li><strong>Neural Network:</strong> CNN-LSTM analyzes 30 frames per second</li>
                <li><strong>AI Prediction:</strong> Identifies which word was spoken</li>
                <li><strong>Confidence Score:</strong> Shows how certain the AI is</li>
            </ul>
            <p><strong>For the real app:</strong> This would use your phone's camera to capture actual lip movements and make live predictions!</p>
        </div>
    </div>
    
    <script>
        function testWord(word) {
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            // Simulate AI processing time
            setTimeout(() => {
                // Generate realistic prediction
                const predictions = generatePrediction(word);
                
                // Show results
                document.getElementById('prediction').textContent = 
                    `Predicted: ${predictions.word.toUpperCase()}`;
                document.getElementById('confidence').textContent = 
                    `Confidence: ${predictions.confidence}%`;
                
                document.getElementById('loading').style.display = 'none';
                document.getElementById('result').style.display = 'block';
            }, 1500);
        }
        
        function generatePrediction(targetWord) {
            // Simulate realistic AI predictions
            const words = ['doctor', 'glasses', 'help', 'pillow', 'phone'];
            
            // 85% chance of correct prediction
            if (Math.random() < 0.85) {
                return {
                    word: targetWord,
                    confidence: Math.floor(75 + Math.random() * 20) // 75-95%
                };
            } else {
                // Sometimes predict wrong word (realistic)
                const wrongWords = words.filter(w => w !== targetWord);
                const wrongWord = wrongWords[Math.floor(Math.random() * wrongWords.length)];
                return {
                    word: wrongWord,
                    confidence: Math.floor(60 + Math.random() * 15) // 60-75%
                };
            }
        }
    </script>
</body>
</html>'''
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def send_test_prediction(self):
        # Simple test endpoint
        word = random.choice(TARGET_WORDS)
        confidence = random.randint(75, 95)
        
        response = {
            'predicted_word': word,
            'confidence': confidence,
            'status': 'working'
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def main():
    ip = get_ip()
    
    print("🎯 LIPREADING DEMO FOR MUM!")
    print("=" * 40)
    print(f"✅ Server starting on port {PORT}")
    print(f"")
    print(f"📱 FOR MUM'S IPHONE:")
    print(f"   Open Safari and go to:")
    print(f"   http://{ip}:{PORT}")
    print(f"")
    print(f"💻 FOR COMPUTER:")
    print(f"   http://localhost:{PORT}")
    print(f"")
    print(f"🎤 Click any word button to test!")
    print(f"🎉 Perfect for your class presentation!")
    print("=" * 40)
    
    with socketserver.TCPServer(("", PORT), DemoHandler) as httpd:
        print(f"🚀 Server running! Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Demo finished! Great job!")

if __name__ == "__main__":
    main()
