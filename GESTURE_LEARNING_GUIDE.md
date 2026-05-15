# IHearYou Gesture Learning System - Quick Start Guide

## ✅ System Status: FULLY OPERATIONAL

Backend server running on **http://localhost:8000**

All gesture learning endpoints tested and working:
- ✓ `/gesture/list` - View recorded gestures
- ✓ `/gesture/record` - Store new gesture (via WebSocket)
- ✓ `/gesture/recognize` - Match current pose against stored gestures
- ✓ WebSocket `/ws` - Real-time landmark streaming + mode switching

---

## How to Use Gesture Learning

### Setup
1. **Backend must be running:**
   ```bash
   cd d:\Programming\ihear-you-app\backend
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

2. **Frontend browser:**
   Open: `http://localhost:8000`

### Recording a New Gesture

1. **Change Mode** → Select "Gesture Learning" from dropdown
2. **Gesture Controls** panel appears below video
3. **Label Input** → Type gesture name (e.g., "halo", "terima_kasih", "tolong")
4. **Position Camera** → Show your gesture to camera (5+ frames captured automatically)
5. **Click "Record Gesture"** → Button sends signal to backend
6. **Status** → Shows "✓ Recorded: [label] (N samples)"

### Recognizing Gestures

1. **Ensure Gesture Learning mode is active**
2. **Click "Recognize Mode"** → Button turns orange/active
3. **Perform a gesture** → System buffers frames in real-time
4. **5+ frames captured** → Automatic recognition triggered
5. **Result** → Displays "🎯 Terdeteksi: [label] (XX.X%)" with confidence

### Storage

- Gestures saved to: `d:\Programming\ihear-you-app\backend\data\gesture_db.json`
- Persists across server restarts
- View via `/gesture/list` endpoint

---

## Technical Details

### Gesture Matching Algorithm
- **Method**: Cosine similarity of mean frame embeddings
- **Threshold**: 0.65 (tunable in websockets.py line ~175)
- **Frame requirement**: Minimum 5 frames per gesture
- **Real-time processing**: ~30fps from MediaPipe Holistic

### Landmark Normalization
- 1530-dim vectors (pose 33 + face 468 + hands 42 points × 3 coords)
- Center on wrist reference point (translation invariance)
- Scale by max norm (scale invariance)

### WebSocket Message Protocol

**Record Gesture:**
```json
{"type": "gesture_record", "label": "halo"}
```

**Recognize Mode:**
```json
{"type": "gesture_recognize", "data": {"pose": [...], "face": [...], ...}}
```

**Response (recognized):**
```json
{"type": "gesture_recognized", "label": "halo", "confidence": 0.92}
```

---

## Alternative: Model Inference

To enable trained model inference instead of gesture learning:

1. **Obtain trained checkpoint**: `bisindo_translator.pth` (valid PyTorch file)
2. **Replace placeholder**: 
   ```
   Current: d:\Programming\ihear-you-app\backend\models\bisindo_translator.pth (117 bytes, placeholder)
   Replace with: Valid STGT model checkpoint
   ```
3. **Restart server** → `model_loaded` becomes `true`
4. **Switch to "Translasi Model"** mode → Uses real predictions instead of dummy

---

## Troubleshooting

### "Label diperlukan, minimal 5 frame"
- Wait longer before clicking Record (need at least 5 frames)
- Ensure camera is capturing body motion

### Low recognition confidence
- Record more samples of the gesture (more diversity)
- Ensure consistent lighting and camera angle
- Lower threshold if needed (editing threshold value in websockets.py)

### Gestures not persisting
- Check: `d:\Programming\ihear-you-app\backend\data/` directory exists
- Verify file write permissions
- Check browser console for errors

---

## Development Notes

**Files involved:**
- Backend: `app/websockets.py`, `app/core/gesture_learning.py`
- Frontend: `frontend/js/app.js`, `frontend/index.html`, `frontend/css/style.css`
- ML: `app/ml/feature_extractor.py` (landmark normalization)

**Test script:**
Run: `python d:\Programming\ihear-you-app\backend\test_gesture_learning.py`

**Next enhancements:**
- Gesture persistence UI (load/delete operations)
- Confidence threshold tuning slider
- Gesture taxonomy (categories)
- Export/import gesture sets
