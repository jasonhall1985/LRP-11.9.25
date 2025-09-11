#!/usr/bin/env python3
"""
Simple HTTP server to serve the trained lipreading app
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to serve files with proper MIME types."""
    
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def guess_type(self, path):
        """Override to handle .js files properly."""
        mimetype, encoding = super().guess_type(path)
        if path.endswith('.js'):
            return 'application/javascript', encoding
        return mimetype, encoding

def main():
    """Start the HTTP server."""
    
    # Configuration
    PORT = 8080
    HOST = 'localhost'
    
    # Check if required files exist
    required_files = [
        'real_ai_lipreading_app.html',
        'models/lipreading_model.js',
        'models/lipreading_model.h5'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease run the training pipeline first:")
        print("   python simplified_training_pipeline.py")
        print("   python convert_model_to_web.py")
        return
    
    print("🚀 Starting Trained AI Lipreading App Server")
    print("=" * 50)
    print(f"📁 Serving from: {os.getcwd()}")
    print(f"🌐 Server: http://{HOST}:{PORT}")
    print(f"📱 App URL: http://{HOST}:{PORT}/real_ai_lipreading_app.html")
    print("=" * 50)
    
    # Change to the current directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create server
    with socketserver.TCPServer((HOST, PORT), CustomHTTPRequestHandler) as httpd:
        print(f"✅ Server started successfully!")
        print(f"📊 Model Info:")
        
        # Try to read model metadata
        try:
            import json
            with open('models/model_metadata.json', 'r') as f:
                metadata = json.load(f)
                print(f"   - Model Type: {metadata.get('model_type', 'Unknown')}")
                print(f"   - Target Words: {', '.join(metadata.get('target_words', []))}")
                print(f"   - Parameters: {metadata.get('total_parameters', 'Unknown'):,}")
                print(f"   - Input Shape: {metadata.get('input_shape', 'Unknown')}")
        except (FileNotFoundError, json.JSONDecodeError):
            print("   - Metadata not available")
        
        print("\n🎯 Instructions:")
        print("1. Open the app URL in your browser")
        print("2. Allow camera access when prompted")
        print("3. Wait for AI models to load")
        print("4. Record yourself mouthing words: doctor, glasses, help, pillow, phone")
        print("5. Watch the TRAINED AI analyze your lip movements!")
        
        print(f"\n🔗 Opening browser automatically...")
        
        # Open browser automatically
        try:
            webbrowser.open(f'http://{HOST}:{PORT}/real_ai_lipreading_app.html')
        except Exception as e:
            print(f"   Could not open browser automatically: {e}")
            print(f"   Please open manually: http://{HOST}:{PORT}/real_ai_lipreading_app.html")
        
        print(f"\n⚡ Server running... Press Ctrl+C to stop")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n🛑 Server stopped by user")
            httpd.shutdown()

if __name__ == "__main__":
    main()
