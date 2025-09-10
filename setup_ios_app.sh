#!/bin/bash

echo "🎯 Setting up Lipreading iOS App"
echo "================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first:"
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

# Check if Expo CLI is installed
if ! command -v expo &> /dev/null; then
    echo "📱 Installing Expo CLI..."
    npm install -g @expo/cli
fi

# Navigate to app directory
cd LipreadingApp

echo "📦 Installing dependencies..."
npm install

echo "✅ Setup complete!"
echo ""
echo "🚀 To start the app:"
echo "   1. Start the backend server:"
echo "      python3 ../mobile_backend_server.py"
echo ""
echo "   2. In a new terminal, start the Expo app:"
echo "      cd LipreadingApp"
echo "      expo start"
echo ""
echo "   3. On your iPhone:"
echo "      - Install 'Expo Go' from App Store"
echo "      - Scan the QR code that appears"
echo "      - Allow camera permissions"
echo ""
echo "📱 The app will work on both your iPhone and your mum's iPhone!"
echo "🎓 Perfect for your Year 10 Computer Science presentation!"
