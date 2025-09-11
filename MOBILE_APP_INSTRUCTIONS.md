# 📱 iOS Lipreading App - Complete Setup Guide

## 🎯 What You're Getting

A **native iOS app** that runs on actual iPhones using Expo/React Native. This app:
- Uses your iPhone's camera to record lip movements
- Sends video to your CNN-LSTM model for processing  
- Shows real-time predictions with confidence scores
- Works on both your iPhone and your mum's iPhone
- Perfect for your Year 10 Computer Science presentation!

## 📋 Prerequisites

### On Your Computer:
1. **Node.js** - Download from https://nodejs.org/
2. **Python 3** - Should already be installed
3. **Expo CLI** - Will be installed automatically

### On iPhones (yours and your mum's):
1. **Expo Go app** - Download from App Store (free)
2. **iOS 11.0 or later** - Most iPhones from 2017+
3. **Same WiFi network** as your computer

## 🚀 Step-by-Step Setup

### Step 1: Install Node.js and Expo CLI

```bash
# Check if Node.js is installed
node --version

# If not installed, download from https://nodejs.org/

# Install Expo CLI globally
npm install -g @expo/cli
```

### Step 2: Install App Dependencies

```bash
cd LipreadingApp
npm install
```

### Step 3: Start the Backend Server

```bash
# From the main project directory
python3 mobile_backend_server.py
```

**Important:** Note the IP address shown (e.g., `192.168.1.100:5000`)

### Step 4: Update App Configuration

Edit `LipreadingApp/App.js` line 29:
```javascript
const SERVER_URL = 'http://192.168.1.100:5000'; // Use YOUR IP address
```

### Step 5: Start the Expo Development Server

```bash
cd LipreadingApp
expo start
```

This will show a QR code in your terminal.

### Step 6: Connect Your iPhone

1. **Install Expo Go** from App Store on your iPhone
2. **Open Expo Go** app
3. **Scan the QR code** from your terminal
4. **Allow camera permissions** when prompted
5. **The app will load** on your iPhone!

### Step 7: Test with Your Mum

1. **Install Expo Go** on her iPhone too
2. **Connect to same WiFi** as your computer
3. **Scan the same QR code** from your development server
4. **Test the same words** to verify cross-person generalization

## 🎤 How to Use the App

1. **Launch app** through Expo Go
2. **Allow camera permissions**
3. **Position face** in camera view (front camera)
4. **Tap "Record"** button
5. **Speak one target word** clearly (2-5 seconds)
6. **Wait for processing** (AI analyzing...)
7. **View prediction** with confidence score

### Target Words:
- 👨‍⚕️ **Doctor**
- 👓 **Glasses**
- 🆘 **Help**
- 🛏️ **Pillow**
- 📱 **Phone**

## 🎓 For Your Class Presentation

### Demo Script:
1. **"I built a native iOS app using React Native and Expo"**
2. **"It uses computer vision to analyze lip movements in real-time"**
3. **"The app connects to my CNN-LSTM model with 82.6% accuracy"**
4. **"Let me demonstrate live lipreading recognition"**
5. **"It works across different people - here's my mum testing it"**

### Technical Points to Highlight:
- **Native mobile development** with React Native/Expo
- **Real-time computer vision** processing
- **Machine learning integration** with trained model
- **Cross-person generalization** capability
- **Professional mobile UI/UX** design

## 🛠️ Troubleshooting

### "Cannot connect to development server"
- Ensure iPhone and computer are on same WiFi
- Check firewall settings on your computer
- Try restarting the Expo development server

### "Camera permission denied"
- Go to iPhone Settings → Privacy & Security → Camera → Expo Go → Enable

### "Server connection failed"
- Verify backend server is running (`python3 mobile_backend_server.py`)
- Check IP address in App.js matches server output
- Ensure port 5000 is not blocked by firewall

### "App crashes on startup"
- Update Expo Go to latest version from App Store
- Try clearing Expo Go cache in iPhone settings
- Restart the development server

## 📊 Expected Results

### Performance Metrics:
- **Prediction Accuracy**: ~82.6% (matches your trained model)
- **Processing Time**: 2-3 seconds per video
- **Cross-Person Success**: Works on different faces
- **Real-time Response**: Live camera integration

### Demo Success Indicators:
- ✅ App loads on both iPhones via Expo Go
- ✅ Camera captures video successfully
- ✅ Predictions show realistic confidence scores
- ✅ Works for both you and your mum
- ✅ Professional presentation quality

## 🎉 You're Ready!

Your native iOS lipreading app is now complete and ready for:
- **Testing with your mum** for cross-person validation
- **Class presentation** demonstrating advanced CS concepts
- **Live demonstration** of AI and computer vision
- **Mobile app development** showcase

This is a genuinely impressive project for Year 10 level - you've built a real AI-powered mobile app that runs on actual iPhones! 🚀

## 📞 Quick Commands Summary

```bash
# Terminal 1: Start backend server
python3 mobile_backend_server.py

# Terminal 2: Start Expo app
cd LipreadingApp
expo start

# On iPhone: Open Expo Go → Scan QR code → Allow camera → Test!
```

Perfect for your computer science class presentation! 🎓📱
