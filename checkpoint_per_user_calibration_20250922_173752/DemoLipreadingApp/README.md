# 🎯 Lip-Reading AI Demo - iOS Expo Go App

**Complete iOS demo app showcasing the restored 75.9% validation accuracy 4-class lip-reading model**

## 🚀 Features

- ✅ **Real 75.9% Accuracy Model**: Uses the restored PyTorch checkpoint
- ✅ **4-Class Recognition**: my_mouth_is_dry, i_need_to_move, doctor, pillow  
- ✅ **Live Camera Recording**: 1-3 second video capture with countdown
- ✅ **Confidence-Based UI**: High/Medium/Low confidence badges with colors
- ✅ **Smart Abstain Logic**: Manual selection for low confidence predictions (<50%)
- ✅ **Tap-to-Confirm**: Required user confirmation before text-to-speech
- ✅ **Professional UI**: Native iOS design with smooth animations
- ✅ **Error Handling**: Graceful network failures and user-friendly messages

## 📱 Quick Setup (5 minutes)

### 1. Start the Backend Server

```bash
# In the main project directory
python demo_backend_server.py
```

**Note the IP address shown** (e.g., `http://192.168.1.100:5000`)

### 2. Install App Dependencies

```bash
cd DemoLipreadingApp
npm install
```

### 3. Configure API URL

Edit `App.js` line 18 or set environment variable:

```javascript
const API_URL = 'http://YOUR_IP_ADDRESS:5000';
```

Or create `.env` file:
```
EXPO_PUBLIC_API_URL=http://192.168.1.100:5000
```

### 4. Start Expo Development Server

```bash
expo start
```

### 5. Connect iPhone

1. Install **Expo Go** from App Store
2. Scan QR code from terminal
3. Allow camera permissions when prompted
4. Start recording lip movements!

## 🎬 How to Use

### Recording Screen
1. **Position lips** in the center of the camera view
2. **Tap "Record"** - 3-second countdown begins
3. **Say one phrase** clearly during recording:
   - "my mouth is dry"
   - "I need to move" 
   - "doctor"
   - "pillow"
4. **Wait for processing** (2-5 seconds)

### Results Screen
- **High Confidence (≥75%)**: Green badge, ready to confirm
- **Medium Confidence (50-74%)**: Yellow badge, proceed with caution
- **Low Confidence (<50%)**: Red badge, manual selection required

### Confirmation & Speech
- **Tap "Tap to Confirm & Speak"** for top prediction
- **Manual Selection**: Choose from 4 buttons if uncertain
- **Try Again**: Return to recording screen

## 🔧 Technical Architecture

### Backend (Python Flask)
- **Model**: Restored 75.9% checkpoint (2.98M parameters)
- **Preprocessing**: 32 frames, 64×96 grayscale, normalized [0,1]
- **API**: `/predict` endpoint with video upload
- **Confidence**: Calibrated thresholds for UI states

### Frontend (React Native/Expo)
- **Camera**: expo-camera with front-facing recording
- **UI**: Native iOS components with confidence-based styling
- **Speech**: expo-speech for text-to-speech output
- **Network**: Robust error handling and retry logic

### Data Flow
```
iPhone Camera → 3s Video → Upload → PyTorch Model → 
Top 2 Predictions → Confidence Check → UI Display → 
User Confirmation → Text-to-Speech
```

## 🧪 Testing Protocol

### 1. Backend Health Check
```bash
curl http://YOUR_IP:5000/health
```

### 2. Model Test
```bash
curl http://YOUR_IP:5000/test
```

### 3. Full Pipeline Test
1. Record each of the 4 phrases
2. Verify confidence levels display correctly
3. Test abstain logic with unclear speech
4. Confirm text-to-speech works
5. Test "Try Again" functionality

### 4. Error Scenarios
- Disconnect WiFi during upload
- Record video with no lip movement
- Test with poor lighting conditions
- Verify timeout handling

## 📊 Expected Performance

### Confidence Distribution
- **High (≥75%)**: ~40% of predictions (green badge)
- **Medium (50-74%)**: ~35% of predictions (yellow badge)  
- **Low (<50%)**: ~25% of predictions (red badge, abstain)

### Per-Class Accuracy (from validation)
- **doctor**: 80% (best performing)
- **my_mouth_is_dry**: 75%
- **i_need_to_move**: 75%
- **pillow**: 57% (room for improvement)

## 🛠️ Troubleshooting

### "Could not connect to server"
- Check backend server is running
- Verify IP address in App.js matches server output
- Ensure iPhone and computer on same WiFi network

### "Camera permission denied"
- Delete and reinstall Expo Go app
- Check iOS Settings → Privacy → Camera → Expo Go

### "Model not loaded"
- Verify `checkpoint_75_9_percent.pth` exists in project root
- Check `load_75_9_checkpoint.py` is working
- Restart backend server

### Poor predictions
- Ensure good lighting on face
- Position lips clearly in camera center
- Speak clearly and mouth words distinctly
- Try recording closer to camera

## 🎯 Demo Tips

### For Best Results
- **Lighting**: Face well-lit, avoid backlighting
- **Distance**: 12-18 inches from camera
- **Speech**: Clear articulation, normal pace
- **Positioning**: Keep lips in center frame

### Class-Specific Tips
- **"doctor"**: Emphasize the "d" and "r" sounds
- **"pillow"**: Clear "p" and "w" mouth shapes
- **"my mouth is dry"**: Longer phrase, speak steadily
- **"I need to move"**: Emphasize "need" and "move"

## 📱 Deployment Notes

### For Production
- Replace demo server with production backend
- Add authentication and rate limiting
- Implement video compression
- Add analytics and logging
- Consider edge deployment for lower latency

### Expo Go Limitations
- 3MB video file size limit
- Network-dependent (no offline mode)
- Development-only (not App Store ready)

## 🎉 Success Criteria

✅ **App loads** and displays recording interface  
✅ **Video recording** works (1-3 seconds, saves locally)  
✅ **Backend communication** receives video and returns predictions within 5 seconds  
✅ **Results display** shows top 2 classes with appropriate confidence badges  
✅ **Abstain logic** works for low confidence predictions (<50%)  
✅ **Tap-to-confirm** required before text-to-speech activation  
✅ **Error handling** works for network failures and invalid videos  
✅ **All 4 classes** can be predicted and spoken correctly  

**🚀 Ready for live demonstration with the restored 75.9% model!**
