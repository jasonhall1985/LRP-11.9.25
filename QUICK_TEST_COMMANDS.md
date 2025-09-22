# 🎯 Quick Test Commands - One-Tap Demo

## ✅ SYSTEM STATUS (All Running!)

**Backend Server**: ✅ Running at `http://192.168.1.100:5000`
- Model: 75.9% validation accuracy (2.98M parameters)
- Classes: my_mouth_is_dry, i_need_to_move, doctor, pillow
- Debug folder: `debug_uploads/` (saves all clips)

**Web Demo**: ✅ Ready at browser window
- Lip placement guide: 96×64 dashed rectangle
- Real-time frame counting and latency display
- Enhanced error handling and troubleshooting

## 📱 iPhone Testing

### 1. Test Backend Connection
Open Safari on iPhone → Navigate to:
```
http://192.168.1.100:5000/health
```
Should show JSON with "status": "healthy"

### 2. Test Web Demo (if iPhone has camera)
Open Safari on iPhone → Navigate to:
```
file:///Users/client/Desktop/LRP classifier 11.9.25/web_demo.html
```
(Or use the browser window already open on laptop)

## 💻 Laptop Testing

### 1. Web Demo (Primary Method)
- Browser window already open with web demo
- Click "Start Camera" → Allow permissions
- Position lips in dashed rectangle guide
- Click "Record (3s)" and say test phrases

### 2. Backend Monitoring
Watch terminal output for detailed logs:
```bash
# Backend logs show:
🎯 Received prediction request
📁 Received file: recording.webm
📊 File size: 245.3 KB (251,187 bytes)
🔍 Debug copy saved: debug_uploads/latest_recording.webm
🎬 Processing video: recording.webm
📊 Video info: 90 frames, 30.00 FPS, 3.00s duration
⏱️ Total processing time: 1.23s
🎯 Top predictions: doctor (0.847), pillow (0.098)
```

## 🎬 Test Sequence

### Test 1: "doctor" (High Confidence Expected)
1. Position lips in dashed rectangle
2. Click "Record (3s)"
3. Wait for countdown
4. Say "doctor" clearly with exaggerated lip movements
5. Expected: Green badge, 75-85% confidence

### Test 2: "pillow" (Medium Confidence Expected)  
1. Same positioning
2. Say "pillow" with clear 'p' sound
3. Expected: Yellow badge, 50-70% confidence

### Test 3: Unclear mouth (Low Confidence Expected)
1. Same positioning  
2. Mumble or speak unclearly
3. Expected: Red badge, <40% confidence

## 🔍 Debug Verification

### Check Processing:
- **Frames Status**: Should show increasing count (1, 2, 3...)
- **Backend Logs**: Should show file size, frame count, processing time
- **Debug Files**: Check `debug_uploads/` folder for saved clips
- **Latency**: Should show ~1000-2000ms processing time

### Troubleshooting Checklist:
- [ ] Backend server running (terminal shows requests)
- [ ] Same WiFi network (iPhone can reach health endpoint)
- [ ] Camera permissions granted
- [ ] Lips positioned in dashed rectangle
- [ ] Non-zero file size in backend logs
- [ ] ~32 frames processed per clip
- [ ] Debug files saved to debug_uploads/

## 🚀 Success Indicators

**✅ Working Correctly:**
- Backend logs show incoming requests
- File sizes > 0 bytes
- Frame counts ~90 frames (3 seconds × 30 FPS)
- Processing time 1-3 seconds
- Confidence scores vary by phrase quality
- Debug files saved with timestamps

**❌ Issues to Check:**
- No backend logs = connection problem
- Zero file size = upload problem  
- No frames processed = video processing problem
- Same confidence every time = model problem

---

**🎯 The system is ready! Test the web demo in your browser and watch the terminal for detailed processing logs.**
