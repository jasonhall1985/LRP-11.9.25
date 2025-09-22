#!/usr/bin/env python3
"""
Simple web server to display QR code for Expo Go connection
"""

from flask import Flask, render_template_string
import qrcode
import io
import base64

app = Flask(__name__)

# The Expo URL we need to connect to
EXPO_URL = "exp://192.168.1.100:8081"

@app.route('/')
def show_qr():
    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(EXPO_URL)
    qr.make(fit=True)
    
    # Create QR code image
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64 for web display
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lip-Reading Demo - QR Code</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 600px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .qr-code {
                margin: 20px 0;
            }
            .url-box {
                background: #f0f0f0;
                padding: 15px;
                border-radius: 5px;
                font-family: monospace;
                font-size: 16px;
                margin: 20px 0;
                word-break: break-all;
            }
            .status {
                color: #28a745;
                font-weight: bold;
                margin: 10px 0;
            }
            .instructions {
                text-align: left;
                margin: 20px 0;
                padding: 20px;
                background: #e3f2fd;
                border-radius: 5px;
            }
            .instructions ol {
                margin: 10px 0;
            }
            .instructions li {
                margin: 8px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Lip-Reading AI Demo</h1>
            <h2>📱 Connect Your iPhone</h2>
            
            <div class="status">✅ Backend Server Running (75.9% Model Loaded)</div>
            <div class="status">✅ QR Code Ready for Scanning</div>
            
            <div class="qr-code">
                <img src="data:image/png;base64,{{ qr_image }}" alt="QR Code" style="max-width: 400px;">
            </div>
            
            <div class="url-box">
                {{ expo_url }}
            </div>
            
            <div class="instructions">
                <h3>📱 How to Connect:</h3>
                <ol>
                    <li><strong>Install Expo Go</strong> from App Store (if not installed)</li>
                    <li><strong>Scan QR Code:</strong>
                        <ul>
                            <li>Open iPhone Camera app</li>
                            <li>Point at QR code above</li>
                            <li>Tap notification → "Open in Expo Go"</li>
                        </ul>
                    </li>
                    <li><strong>OR Manual Entry:</strong>
                        <ul>
                            <li>Open Expo Go app</li>
                            <li>Tap "Enter URL manually"</li>
                            <li>Copy/paste URL from gray box above</li>
                        </ul>
                    </li>
                    <li><strong>Allow camera permissions</strong> when prompted</li>
                    <li><strong>Start testing!</strong> Record lip movements for:
                        <ul>
                            <li>"my mouth is dry"</li>
                            <li>"I need to move"</li>
                            <li>"doctor"</li>
                            <li>"pillow"</li>
                        </ul>
                    </li>
                </ol>
            </div>
            
            <div class="instructions">
                <h3>🎬 Expected Demo Flow:</h3>
                <ol>
                    <li><strong>Recording:</strong> 3-second countdown → record 1-3 seconds</li>
                    <li><strong>Processing:</strong> Video uploads → AI processes → returns predictions</li>
                    <li><strong>Results:</strong> Top 2 predictions with confidence badges</li>
                    <li><strong>Confidence:</strong> Green (≥75%), Yellow (50-74%), Red (<50%)</li>
                    <li><strong>Speech:</strong> Tap to confirm → text-to-speech output</li>
                    <li><strong>Repeat:</strong> "Try Again" for more tests</li>
                </ol>
            </div>
            
            <div style="margin-top: 30px; padding: 15px; background: #fff3cd; border-radius: 5px;">
                <strong>🔧 Troubleshooting:</strong><br>
                • Ensure iPhone and computer on same WiFi network<br>
                • If connection fails, try restarting Expo Go app<br>
                • Backend server logs will show connection attempts<br>
                • Good lighting helps with lip-reading accuracy
            </div>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html_template, qr_image=img_str, expo_url=EXPO_URL)

if __name__ == '__main__':
    print("🎯 QR CODE DISPLAY SERVER")
    print("=" * 30)
    print("✅ Starting web server for QR code display...")
    print("📱 Open browser to: http://localhost:3000")
    print("🔗 Expo URL: " + EXPO_URL)
    print("=" * 30)
    
    app.run(host='0.0.0.0', port=3000, debug=False)
