#!/bin/bash

echo "🚀 Setting up AI Lipreading Expo App"
echo "===================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first:"
    echo "   https://nodejs.org/"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ Node.js version: $(node --version)"
echo "✅ npm version: $(npm --version)"

# Install Expo CLI globally if not already installed
if ! command -v expo &> /dev/null; then
    echo "📦 Installing Expo CLI globally..."
    npm install -g @expo/cli
else
    echo "✅ Expo CLI already installed: $(expo --version)"
fi

# Install dependencies
echo "📦 Installing project dependencies..."
npm install

# Create assets directory and placeholder files
echo "🎨 Creating assets directory..."
mkdir -p assets

# Create placeholder icon (you can replace this with a real icon)
echo "Creating placeholder assets..."
cat > assets/icon.png << 'EOF'
# This is a placeholder. Replace with actual 1024x1024 PNG icon
EOF

cat > assets/splash.png << 'EOF'
# This is a placeholder. Replace with actual splash screen image
EOF

cat > assets/adaptive-icon.png << 'EOF'
# This is a placeholder. Replace with actual adaptive icon
EOF

cat > assets/favicon.png << 'EOF'
# This is a placeholder. Replace with actual favicon
EOF

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📱 To run on your mum's iPhone:"
echo "1. Install 'Expo Go' app from App Store on her iPhone"
echo "2. Run: npm start"
echo "3. Scan the QR code with iPhone camera or Expo Go app"
echo "4. The app will load with trained AI lipreading!"
echo ""
echo "🧠 Features:"
echo "✅ Real camera access"
echo "✅ Trained neural network (146K parameters)"
echo "✅ Live lip movement analysis"
echo "✅ 5 word recognition: doctor, glasses, help, pillow, phone"
echo "✅ Professional UI for class presentation"
echo ""
echo "🚀 Ready to start: npm start"
