#!/usr/bin/env python3
"""
Mobile-friendly HTTPS server for lip-reading demo.
Creates self-signed certificate for mobile camera access.
"""

import http.server
import ssl
import socketserver
import os
from pathlib import Path
import subprocess

def create_self_signed_cert():
    """Create a self-signed certificate for HTTPS."""
    cert_file = "server.crt"
    key_file = "server.key"
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print("✅ SSL certificate already exists")
        return cert_file, key_file
    
    print("🔐 Creating self-signed SSL certificate...")
    
    # Create self-signed certificate
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:4096", "-keyout", key_file,
        "-out", cert_file, "-days", "365", "-nodes", "-subj",
        "/C=US/ST=State/L=City/O=Organization/CN=192.168.1.100"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("✅ SSL certificate created successfully")
        return cert_file, key_file
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create SSL certificate: {e}")
        print("📝 You may need to install OpenSSL")
        return None, None
    except FileNotFoundError:
        print("❌ OpenSSL not found. Please install OpenSSL to create HTTPS server.")
        print("📝 Alternative: Use Chrome with --unsafely-treat-insecure-origin-as-secure flag")
        return None, None

class MobileHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler with mobile-friendly headers."""
    
    def end_headers(self):
        # Add mobile-friendly headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight requests."""
        self.send_response(200)
        self.end_headers()

def start_mobile_server():
    """Start HTTPS server for mobile demo."""
    port = 8443
    
    print("🚀 MOBILE LIP-READING DEMO SERVER")
    print("=" * 50)
    
    # Try to create SSL certificate
    cert_file, key_file = create_self_signed_cert()
    
    if not cert_file or not key_file:
        print("\n⚠️  FALLBACK: Starting HTTP server (may not work on mobile)")
        print("📱 For mobile camera access, you need HTTPS")
        print("🔧 Alternative solutions:")
        print("   1. Use Chrome with --unsafely-treat-insecure-origin-as-secure=http://192.168.1.100:8000")
        print("   2. Install OpenSSL and restart this script")
        print("   3. Use a different device/browser")
        
        # Start HTTP server as fallback
        with socketserver.TCPServer(("", 8000), MobileHTTPRequestHandler) as httpd:
            print(f"\n🌐 HTTP Server running at:")
            print(f"   Local: http://localhost:8000/mobile_demo.html")
            print(f"   Network: http://192.168.1.100:8000/mobile_demo.html")
            print("\n📱 Note: Camera may not work over HTTP on mobile devices")
            print("🔄 Press Ctrl+C to stop")
            httpd.serve_forever()
    else:
        # Start HTTPS server
        with socketserver.TCPServer(("", port), MobileHTTPRequestHandler) as httpd:
            # Create SSL context
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert_file, key_file)
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
            
            print(f"\n🔐 HTTPS Server running at:")
            print(f"   Local: https://localhost:{port}/mobile_demo.html")
            print(f"   Network: https://192.168.1.100:{port}/mobile_demo.html")
            print(f"\n📱 MOBILE ACCESS:")
            print(f"   Scan QR code or visit: https://192.168.1.100:{port}/mobile_demo.html")
            print(f"\n⚠️  You'll see a security warning - click 'Advanced' → 'Proceed'")
            print(f"   (This is normal for self-signed certificates)")
            print(f"\n🔄 Press Ctrl+C to stop")
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n🛑 Server stopped")

if __name__ == "__main__":
    start_mobile_server()
