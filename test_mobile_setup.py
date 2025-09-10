#!/usr/bin/env python3
"""
Test script to verify mobile app setup is working correctly.
"""

import requests
import json
import socket
import os
import subprocess
import sys

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

def check_node_installation():
    """Check if Node.js is installed."""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Node.js not found")
            return False
    except FileNotFoundError:
        print("❌ Node.js not installed")
        return False

def check_expo_installation():
    """Check if Expo CLI is installed."""
    try:
        result = subprocess.run(['expo', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Expo CLI installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Expo CLI not found")
            return False
    except FileNotFoundError:
        print("❌ Expo CLI not installed")
        return False

def check_app_structure():
    """Check if app files exist."""
    required_files = [
        'LipreadingApp/package.json',
        'LipreadingApp/App.js',
        'LipreadingApp/app.json',
        'mobile_backend_server.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_backend_server():
    """Test if backend server is accessible."""
    local_ip = get_local_ip()
    server_url = f"http://{local_ip}:5000"
    
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend server accessible at {server_url}")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Target words: {data.get('target_words', [])}")
            return True, server_url
        else:
            print(f"❌ Backend server returned status {response.status_code}")
            return False, server_url
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to backend server at {server_url}")
        print(f"   Error: {e}")
        return False, server_url

def check_app_config(server_url):
    """Check if App.js has correct server URL."""
    app_js_path = 'LipreadingApp/App.js'
    
    if not os.path.exists(app_js_path):
        print("❌ App.js not found")
        return False
    
    with open(app_js_path, 'r') as f:
        content = f.read()
    
    if server_url in content:
        print(f"✅ App.js configured with correct server URL: {server_url}")
        return True
    else:
        print(f"⚠️  App.js may need server URL update")
        print(f"   Update line ~29 to: const SERVER_URL = '{server_url}';")
        return False

def main():
    """Run all tests."""
    print("🎯 LIPREADING iOS APP SETUP TEST")
    print("=" * 50)
    
    # Test 1: Node.js installation
    print("\n1. Checking Node.js installation...")
    node_ok = check_node_installation()
    
    # Test 2: Expo CLI installation
    print("\n2. Checking Expo CLI installation...")
    expo_ok = check_expo_installation()
    
    # Test 3: App file structure
    print("\n3. Checking app file structure...")
    files_ok = check_app_structure()
    
    # Test 4: Backend server
    print("\n4. Testing backend server...")
    server_ok, server_url = test_backend_server()
    
    # Test 5: App configuration
    print("\n5. Checking app configuration...")
    config_ok = check_app_config(server_url)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SETUP TEST SUMMARY")
    print("=" * 50)
    
    tests = [
        ("Node.js Installation", node_ok),
        ("Expo CLI Installation", expo_ok),
        ("App File Structure", files_ok),
        ("Backend Server", server_ok),
        ("App Configuration", config_ok)
    ]
    
    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)
    
    for test_name, ok in tests:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your iOS app setup is ready!")
        print("\n📱 Next steps:")
        print("1. Start backend server: python3 mobile_backend_server.py")
        print("2. Start Expo app: cd LipreadingApp && expo start")
        print("3. Scan QR code with Expo Go on iPhone")
        print("4. Test with your mum for cross-person validation!")
    else:
        print(f"\n⚠️  {total - passed} issues need to be resolved")
        print("Please fix the failed tests before proceeding.")
        
        if not node_ok:
            print("\n🔧 To install Node.js:")
            print("   Visit: https://nodejs.org/")
        
        if not expo_ok:
            print("\n🔧 To install Expo CLI:")
            print("   npm install -g @expo/cli")
        
        if not server_ok:
            print("\n🔧 To start backend server:")
            print("   python3 mobile_backend_server.py")

if __name__ == "__main__":
    main()
