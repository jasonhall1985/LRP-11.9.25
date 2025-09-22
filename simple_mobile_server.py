#!/usr/bin/env python3
"""
Simple mobile server with camera access workarounds.
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path

class MobileHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    port = 8080
    
    print("📱 MOBILE LIP-READING DEMO")
    print("=" * 40)
    print(f"🌐 Server starting on port {port}")
    
    with socketserver.TCPServer(("", port), MobileHandler) as httpd:
        print(f"\n✅ Server running at:")
        print(f"   📱 Mobile: http://192.168.1.100:{port}/mobile_demo.html")
        print(f"   💻 Desktop: http://localhost:{port}/mobile_demo.html")
        
        print(f"\n📱 MOBILE CAMERA ACCESS SOLUTIONS:")
        print(f"   1. 🔐 Use HTTPS (recommended)")
        print(f"   2. 🌐 Try different browsers (Firefox, Safari)")
        print(f"   3. ⚙️  Enable 'Insecure origins' in Chrome flags")
        print(f"   4. 📍 Use localhost if testing locally")
        
        print(f"\n🎯 CHROME MOBILE FIX:")
        print(f"   1. Open Chrome on mobile")
        print(f"   2. Go to chrome://flags")
        print(f"   3. Search 'insecure origins'")
        print(f"   4. Add: http://192.168.1.100:{port}")
        print(f"   5. Restart Chrome")
        
        print(f"\n🔄 Press Ctrl+C to stop server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")

if __name__ == "__main__":
    main()
