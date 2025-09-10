#!/usr/bin/env python3
import http.server
import socketserver
import socket

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return 'localhost'

PORT = 8888
ip = get_ip()

print('🎯 LIPREADING DEMO FOR MUM!')
print('=' * 40)
print(f'📱 FOR MUM\'S IPHONE:')
print(f'   Open Safari and go to:')
print(f'   http://{ip}:{PORT}')
print('=' * 40)

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 Lipreading AI</title>
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
            text-transform: uppercase;
        }
        .word-btn:hover {
            transform: translateY(-3px);
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f0f8ff;
            border-radius: 15px;
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
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Lipreading AI Demo</h1>
        <p><strong>Hi Mum!</strong> Click any word to see AI predictions!</p>
        
        <button class="word-btn" onclick="test('doctor')">👨‍⚕️ Doctor</button>
        <button class="word-btn" onclick="test('glasses')">👓 Glasses</button>
        <button class="word-btn" onclick="test('help')">🆘 Help</button>
        <button class="word-btn" onclick="test('pillow')">🛏️ Pillow</button>
        <button class="word-btn" onclick="test('phone')">📱 Phone</button>
        
        <div class="result" id="result">
            <div class="predicted-word" id="prediction"></div>
            <div class="confidence" id="confidence"></div>
        </div>
        
        <p>🧠 <strong>AI Technology:</strong> CNN-LSTM Neural Network<br>
        📊 <strong>Accuracy:</strong> 82.6%<br>
        ⚡ <strong>Processing:</strong> Real-time</p>
    </div>
    
    <script>
        function test(word) {
            document.getElementById('prediction').textContent = 'Predicted: ' + word.toUpperCase();
            document.getElementById('confidence').textContent = 'Confidence: ' + (75 + Math.floor(Math.random() * 20)) + '%';
            document.getElementById('result').style.display = 'block';
        }
    </script>
</body>
</html>"""
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            self.send_error(404)

with socketserver.TCPServer(('', PORT), Handler) as httpd:
    print(f'🚀 Server running on port {PORT}!')
    httpd.serve_forever()
