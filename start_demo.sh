#!/bin/bash

# 🎯 LIP-READING DEMO STARTUP SCRIPT
# One-tap script to start backend → verify health → start Expo → scan QR

set -e  # Exit on any error

echo "🎯 LIP-READING DEMO STARTUP"
echo "=========================="

# Get LAN IP
LAN_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
echo "🌐 LAN IP: $LAN_IP"

# Update API URLs in frontend files
echo "🔧 Updating API URLs..."

# Update web demo
sed -i '' "s|const API_URL = 'http://.*:5000'|const API_URL = 'http://$LAN_IP:5000'|g" web_demo.html
echo "✅ Updated web_demo.html API_URL to http://$LAN_IP:5000"

# Update Expo app
sed -i '' "s|const API_URL = 'http://.*:5000'|const API_URL = 'http://$LAN_IP:5000'|g" DemoLipreadingApp/App.js
echo "✅ Updated DemoLipreadingApp/App.js API_URL to http://$LAN_IP:5000"

# Kill any existing servers
echo "🧹 Cleaning up existing servers..."
pkill -f "demo_backend_server.py" 2>/dev/null || true
pkill -f "expo start" 2>/dev/null || true
sleep 2

# Start backend server
echo "🚀 Starting backend server..."
python demo_backend_server.py &
BACKEND_PID=$!
echo "📊 Backend PID: $BACKEND_PID"

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Test health endpoint
echo "🏥 Testing health endpoint..."
if curl -s "http://$LAN_IP:5000/health" | grep -q "healthy"; then
    echo "✅ Backend health check passed"
else
    echo "❌ Backend health check failed"
    exit 1
fi

# Test from localhost
echo "🏠 Testing localhost health..."
if curl -s "http://localhost:5000/health" | grep -q "healthy"; then
    echo "✅ Localhost health check passed"
else
    echo "❌ Localhost health check failed"
fi

echo ""
echo "🎯 BACKEND READY!"
echo "=================="
echo "📱 iPhone: Test http://$LAN_IP:5000/health in Safari"
echo "💻 Laptop: curl http://$LAN_IP:5000/health"
echo "🌐 Web Demo: file://$(pwd)/web_demo.html"
echo ""

# Check if Expo CLI is available
if command -v npx expo &> /dev/null; then
    echo "🎬 Starting Expo in LAN mode..."
    cd DemoLipreadingApp
    
    # Try LAN mode first
    echo "📡 Attempting LAN mode..."
    timeout 30 npx expo start --lan --port 8081 &
    EXPO_PID=$!
    
    sleep 10
    
    # Check if Expo is running
    if ps -p $EXPO_PID > /dev/null; then
        echo "✅ Expo running in LAN mode"
        echo "📱 Scan QR code with Expo Go app"
        echo "🔗 Manual URL: exp://$LAN_IP:8081"
    else
        echo "⚠️ LAN mode failed, trying tunnel mode..."
        npx expo start --tunnel --port 8081 &
        EXPO_PID=$!
        echo "🌐 Expo running in tunnel mode"
        echo "📱 Scan QR code with Expo Go app"
    fi
    
    cd ..
else
    echo "⚠️ Expo CLI not found. Install with: npm install -g @expo/cli"
    echo "📱 Use web demo instead: file://$(pwd)/web_demo.html"
fi

echo ""
echo "🎯 DEMO READY!"
echo "=============="
echo "Backend: http://$LAN_IP:5000"
echo "Web Demo: file://$(pwd)/web_demo.html"
echo "Debug uploads: $(pwd)/debug_uploads/"
echo ""
echo "📋 TEST CHECKLIST:"
echo "1. Record 'doctor' - should get high confidence"
echo "2. Record 'pillow' - should get medium confidence"  
echo "3. Record unclear mouth - should get low confidence"
echo ""
echo "Press Ctrl+C to stop all servers"

# Keep script running
wait
