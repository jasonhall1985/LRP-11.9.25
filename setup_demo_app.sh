#!/bin/bash

echo "🎯 SETTING UP LIP-READING AI DEMO APP"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed.${NC}"
    echo "Please install Node.js from: https://nodejs.org/"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm is not installed.${NC}"
    echo "Please install npm first."
    exit 1
fi

echo -e "${GREEN}✅ Node.js version: $(node --version)${NC}"
echo -e "${GREEN}✅ npm version: $(npm --version)${NC}"

# Install Expo CLI globally if not already installed
if ! command -v expo &> /dev/null; then
    echo -e "${YELLOW}📦 Installing Expo CLI globally...${NC}"
    npm install -g @expo/cli
else
    echo -e "${GREEN}✅ Expo CLI already installed: $(expo --version)${NC}"
fi

# Navigate to demo app directory
cd DemoLipreadingApp

# Install dependencies
echo -e "${YELLOW}📦 Installing project dependencies...${NC}"
npm install

# Create placeholder assets if they don't exist
echo -e "${YELLOW}🎨 Creating placeholder assets...${NC}"
mkdir -p assets

# Create simple placeholder files (these will work for development)
if [ ! -f "assets/icon.png" ]; then
    echo "Creating placeholder icon.png..."
    # Create a simple 1024x1024 placeholder (this is just a text file for demo)
    echo "# Placeholder icon - replace with actual 1024x1024 PNG" > assets/icon.png
fi

if [ ! -f "assets/splash.png" ]; then
    echo "Creating placeholder splash.png..."
    echo "# Placeholder splash - replace with actual splash screen" > assets/splash.png
fi

if [ ! -f "assets/adaptive-icon.png" ]; then
    echo "Creating placeholder adaptive-icon.png..."
    echo "# Placeholder adaptive icon - replace with actual adaptive icon" > assets/adaptive-icon.png
fi

if [ ! -f "assets/favicon.png" ]; then
    echo "Creating placeholder favicon.png..."
    echo "# Placeholder favicon - replace with actual favicon" > assets/favicon.png
fi

echo ""
echo -e "${GREEN}✅ Demo app setup complete!${NC}"
echo ""
echo -e "${BLUE}🚀 NEXT STEPS:${NC}"
echo ""
echo -e "${YELLOW}1. Start the backend server:${NC}"
echo "   cd .."
echo "   python demo_backend_server.py"
echo ""
echo -e "${YELLOW}2. Note the IP address shown (e.g., 192.168.1.100)${NC}"
echo ""
echo -e "${YELLOW}3. Update the API URL in App.js:${NC}"
echo "   Edit line 18 in App.js:"
echo "   const API_URL = 'http://YOUR_IP_ADDRESS:5000';"
echo ""
echo -e "${YELLOW}4. Start the Expo development server:${NC}"
echo "   expo start"
echo ""
echo -e "${YELLOW}5. Connect your iPhone:${NC}"
echo "   - Install 'Expo Go' from App Store"
echo "   - Scan the QR code that appears"
echo "   - Allow camera permissions when prompted"
echo ""
echo -e "${GREEN}📱 The app will demonstrate the 75.9% accuracy model!${NC}"
echo -e "${GREEN}🎯 Perfect for live demonstrations!${NC}"
echo ""
echo -e "${BLUE}📋 TESTING CHECKLIST:${NC}"
echo "✅ Backend health check: curl http://YOUR_IP:5000/health"
echo "✅ Model test: curl http://YOUR_IP:5000/test"
echo "✅ Record each phrase: 'my mouth is dry', 'I need to move', 'doctor', 'pillow'"
echo "✅ Test confidence levels and abstain logic"
echo "✅ Verify text-to-speech works"
echo ""
echo -e "${GREEN}🎉 Ready for your demo presentation!${NC}"
