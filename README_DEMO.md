# 🎯 Lip-Reading AI Demo - 75.9% Accuracy Model

Real-time lip-reading demonstration using the restored 75.9% validation accuracy checkpoint with 2.98M parameters.

## 🚀 Quick Start

### One-Tap Demo Launch
```bash
./start_demo.sh
```

This script will:
1. ✅ Get your LAN IP address
2. ✅ Update all frontend API URLs automatically  
3. ✅ Start backend server with 75.9% model
4. ✅ Verify health endpoints (laptop + iPhone)
5. ✅ Launch Expo in LAN mode (or tunnel fallback)
6. ✅ Display QR code for iPhone connection

### Manual Steps
```bash
# 1. Start backend
python demo_backend_server.py

# 2. Test health
curl http://YOUR_LAN_IP:5000/health

# 3. Open web demo
open web_demo.html

# 4. Start Expo (optional)
cd DemoLipreadingApp && npx expo start --lan
```

## 📱 How to Frame Your Mouth (1-Minute Guide)

### 🎯 Perfect Lip Placement

The model expects lips in a **96×64 landscape rectangle**. Follow this guide:

#### ✅ Correct Positioning:
- **Lips centered** in the dashed rectangle overlay
- **Landscape orientation** (wider than tall)
- **Mouth fills 60-80%** of the guide box
- **Chin relatively still** during recording
- **Minimal head turning** - face the camera directly

#### ❌ Common Mistakes:
- Lips too high/low in frame
- Head tilted or turned sideways  
- Mouth too small in the guide box
- Moving head instead of just lips
- Poor lighting on mouth area

#### 💡 Pro Tips:
- **"doctor"** typically gets highest accuracy (~80%)
- **Good lighting** on your face improves results
- **Clear mouth movements** with distinct lip shapes
- **3-second recordings** are optimal length
- **Keep chin still** - only move lips and jaw

### 🎬 Recording Flow:
1. **Position face** so lips align with dashed rectangle
2. **Wait for 3-second countdown**
3. **Say phrase clearly** with exaggerated lip movements
4. **Keep head still** throughout recording
5. **Wait for AI processing** and confidence results

## 🎯 Test Phrases & Expected Results

| Phrase | Expected Confidence | Notes |
|--------|-------------------|-------|
| **"doctor"** | High (75-85%) | Best accuracy, clear lip movements |
| **"pillow"** | Medium (50-70%) | Good accuracy, distinctive 'p' sound |
| **"my mouth is dry"** | Medium (45-65%) | Longer phrase, moderate accuracy |
| **"I need to move"** | Medium (40-60%) | Complex phrase, variable results |
| **Unclear/mumbled** | Low (<40%) | Should trigger low confidence |

## 🔧 System Architecture

### Backend Server (`demo_backend_server.py`)
- **Model**: 75.9% validation accuracy (2.98M parameters)
- **Input**: 32 frames, 64×96 grayscale, normalized [0,1]
- **Output**: Top-2 predictions with confidence scores
- **Formats**: MP4, MOV, AVI, WebM
- **Logging**: Detailed processing info + debug uploads

### Web Demo (`web_demo.html`)
- **Camera**: WebRTC MediaRecorder API
- **Guide**: 96×64 lip placement overlay
- **Processing**: Real-time upload + inference
- **Results**: Confidence badges + debug info

### Expo App (`DemoLipreadingApp/`)
- **Platform**: React Native + Expo Go
- **Camera**: expo-camera with video recording
- **UI**: Professional iOS-style interface
- **Features**: Text-to-speech, confidence display

## 🐛 Troubleshooting

### Connection Issues
- ✅ **Same WiFi**: Ensure iPhone and laptop on same network
- ✅ **Firewall**: Allow Python through macOS firewall
- ✅ **IP Address**: Verify LAN IP with `ifconfig`
- ✅ **Health Check**: Test `http://YOUR_IP:5000/health` in Safari

### Low Accuracy
- ✅ **Lip Placement**: Use the dashed rectangle guide
- ✅ **Lighting**: Ensure good lighting on face
- ✅ **Clarity**: Speak clearly with distinct mouth movements
- ✅ **Stillness**: Keep head still, only move lips

### No Frames Processed
- ✅ **Backend Running**: Check terminal for server logs
- ✅ **Correct URL**: Verify API_URL matches your LAN IP
- ✅ **File Upload**: Check debug_uploads/ folder for saved clips
- ✅ **CORS**: Backend should show CORS enabled

## 📊 Debug Information

### Backend Logs Show:
```
🎯 Received prediction request
📁 Received file: recording.webm
📊 File size: 245.3 KB (251,187 bytes)
🔍 Debug copy saved: debug_uploads/latest_recording.webm
🎬 Processing video: recording.webm
📊 Video info: 90 frames, 30.00 FPS, 3.00s duration
⏱️ Total processing time: 1.23s
🎯 Top predictions: doctor (0.847), pillow (0.098)
```

### Frontend Shows:
- **Backend Status**: Connected ✅
- **Frames Processed**: Running count
- **Latency**: Upload + inference time
- **Debug Info**: Endpoint hit, timing details

## 📁 File Structure

```
├── demo_backend_server.py     # Flask server with 75.9% model
├── web_demo.html             # Web-based demo with lip guide
├── start_demo.sh             # One-tap startup script
├── debug_uploads/            # Saved clips for inspection
├── DemoLipreadingApp/        # Expo React Native app
│   └── App.js               # Main app with camera + UI
└── README_DEMO.md           # This guide
```

## 🎯 Success Metrics

- **High Confidence**: ≥75% (Green badge)
- **Medium Confidence**: 50-74% (Yellow badge)  
- **Low Confidence**: <50% (Red badge)
- **Processing Time**: ~1-2 seconds per clip
- **Frame Count**: Should show ~32 frames processed

---

**🚀 Ready to test your lip-reading AI! Run `./start_demo.sh` and start recording.**
