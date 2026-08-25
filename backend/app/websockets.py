"""
IHearYou — WebSocket handler v9

FIX KRITIS:
- Stability filter terlalu ketat → prediksi tidak pernah muncul
- Sekarang: 2 frame berturut-turut cukup untuk emit prediksi
- Confidence threshold turun ke 0.15 (model BISINDO sudah 89% acc)
- Duplicate suppression turun ke 0.8 detik
- Voted_conf bug: window kosong → pakai conf langsung
- Scan interval 1.5 dtk × 2 frame = 3 detik → prediksi pertama muncul

ML improvements tetap dipertahankan:
- EMA temporal smoothing di image_model
- TTA 4x augmentation
- CLAHE preprocessing
"""
from __future__ import annotations

import time
from collections import deque, Counter
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from fastapi import WebSocket

from .core.regional_mapping import apply_regional_mapping
from .ml.feature_extractor import landmarks_to_array
from .ml.image_model import ImageSignModel
from .ml.speech_recognizer import transcribe_audio


# ── Connection state ──────────────────────────────────────────────────────────

@dataclass
class ConnectionState:
    frame_buffer:            deque = field(default_factory=lambda: deque(maxlen=30))
    stability_window:        deque = field(default_factory=lambda: deque(maxlen=5))

    last_prediction:         str   = ""
    prediction_counter:      int   = 0
    last_emitted:            str   = ""
    last_emitted_time:       float = 0.0

    word_buffer:             list  = field(default_factory=list)
    last_word_time:          float = 0.0

    mode:                    str   = "predict"

    gesture_frame_buffer:    deque = field(default_factory=lambda: deque(maxlen=120))
    gesture_recording:       bool  = False
    gesture_record_label:    str   = ""
    gesture_sentence_buffer: list  = field(default_factory=list)
    gesture_last_word_time:  float = 0.0

    frame_tick:              int   = 0
    recognize_tick:          int   = 0


# ── Connection manager ────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self, max_clients: int = 100, sequence_length: int = 30,
                 prediction_stability_threshold: int = 2):   # ← turun dari 3 ke 2
        self.active:   list[WebSocket]               = []
        self.states:   dict[WebSocket, ConnectionState] = {}
        self.max_clients    = max_clients
        self.sequence_length = sequence_length
        self.stability_thresh = prediction_stability_threshold

    async def connect(self, ws: WebSocket) -> bool:
        if len(self.active) >= self.max_clients:
            await ws.close(code=1013, reason="Server busy")
            return False
        await ws.accept()
        self.active.append(ws)
        self.states[ws] = ConnectionState(
            frame_buffer=deque(maxlen=self.sequence_length))
        return True

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)
        self.states.pop(ws, None)

    def get_state(self, ws: WebSocket) -> ConnectionState | None:
        return self.states.get(ws)

    async def send(self, payload: dict, ws: WebSocket) -> None:
        try:
            await ws.send_json(payload)
        except Exception:
            pass


# ── Message handler ───────────────────────────────────────────────────────────

async def handle_message(
    *,
    message: dict,
    websocket: WebSocket,
    manager: ConnectionManager,
    gesture_db=None,
    image_model: ImageSignModel | None = None,
) -> None:
    state = manager.get_state(websocket)
    if state is None:
        return

    mtype = message.get("type", "")

    if "mode" in message:
        m = str(message.get("mode", "predict")).strip().lower()
        if m in ("predict", "gesture_learn"):
            state.mode = m

    # ── PREDICT ──────────────────────────────────────────────────────────────
    if mtype in ("predict", "image_frame"):
        if state.mode != "predict":
            return

        b64 = str(message.get("image_frame") or message.get("image") or "")
        if not b64:
            return

        if not image_model or not image_model.model_loaded:
            await manager.send({
                "type":  "error",
                "error": "Model belum loaded — pastikan pipeline_mlp.pkl ada"
            }, websocket)
            return

        t0 = perf_counter()
        try:
            top5 = image_model.predict_top5_from_base64(b64)
        except Exception as exc:
            await manager.send({"type": "error", "error": str(exc)}, websocket)
            return

        if not top5:
            return

        word = str(top5[0]["label"])
        conf = float(top5[0]["confidence"])
        lat  = round((perf_counter() - t0) * 1000, 1)

        # ── Rejection: conf sangat rendah → buang ────────────────────────
        # Threshold rendah (0.15) karena model sudah 89% acc
        if conf < 0.15:
            return

        # ── Stability: hitung frame berturut-turut prediksi sama ─────────
        if word == state.last_prediction:
            state.prediction_counter += 1
        else:
            state.last_prediction    = word
            state.prediction_counter = 1
            state.stability_window.clear()

        state.stability_window.append(conf)
        avg_conf = float(np.mean(list(state.stability_window)))

        # Threshold sederhana: 2 frame berturut-turut cukup
        needed = manager.stability_thresh   # default 2
        if conf >= 0.80:
            needed = 1   # conf tinggi → langsung emit
        elif conf >= 0.55:
            needed = 2
        else:
            needed = 3   # conf rendah → butuh 3 frame

        if state.prediction_counter < needed:
            # Kirim partial update (bisa ditampilkan sebagai "kandidat")
            await manager.send({
                "type":           "prediction_candidate",
                "top_prediction": word,
                "confidence":     conf,
                "top5":           top5,
                "latency_ms":     lat,
                "frames_needed":  needed - state.prediction_counter,
            }, websocket)
            return

        # ── Duplicate suppression: 0.8 detik ─────────────────────────────
        now = time.time()
        if word == state.last_emitted and (now - state.last_emitted_time) < 0.8:
            return

        localized = apply_regional_mapping(word, "")

        # ── Word buffer / sentence ────────────────────────────────────────
        state.word_buffer.append(localized)
        state.last_emitted       = word
        state.last_emitted_time  = now
        state.last_word_time     = now
        state.prediction_counter = 0
        state.stability_window.clear()

        sentence   = " ".join(state.word_buffer)
        sent_done  = False

        await manager.send({
            "type":              "translation",
            "top_prediction":    localized,
            "standard_label":    word,
            "confidence":        avg_conf,
            "top5":              top5,
            "latency_ms":        lat,
            "current_word":      localized,
            "word_buffer":       list(state.word_buffer),
            "sentence":          sentence,
            "sentence_complete": sent_done,
            "speak_word":        localized if localized else None,
            "speak_sentence":    None,
        }, websocket)

    # ── CLEAR SENTENCE ────────────────────────────────────────────────────────
    elif mtype == "clear_sentence":
        state.word_buffer.clear()
        state.last_emitted      = ""
        state.last_emitted_time = 0.0
        if image_model:
            image_model.reset_temporal()
        await manager.send({"type": "sentence_cleared"}, websocket)

    # ── GESTURE RECORD START ──────────────────────────────────────────────────
    elif mtype == "gesture_record_start":
        label = str(message.get("label", "")).strip()
        if not label:
            await manager.send({"type": "error", "error": "Label kosong"}, websocket)
            return
        state.gesture_recording    = True
        state.gesture_record_label = label
        state.gesture_frame_buffer.clear()
        await manager.send({"type": "gesture_record_started", "label": label}, websocket)

    # ── GESTURE FRAME ─────────────────────────────────────────────────────────
    elif mtype in ("gesture_frame", "gesture_learn", "gesture_record"):
        lm_data = message.get("landmarks")
        if lm_data:
            arr = landmarks_to_array(lm_data)
            if arr is not None:
                if state.gesture_recording:
                    state.gesture_frame_buffer.append(arr)
                else:
                    state.frame_buffer.append(arr)

        if state.gesture_recording:
            await manager.send({
                "type":   "gesture_recording",
                "label":  state.gesture_record_label,
                "frames": len(state.gesture_frame_buffer),
            }, websocket)

    # ── GESTURE RECORD STOP ───────────────────────────────────────────────────
    elif mtype == "gesture_record_stop":
        label = str(message.get("label", "") or state.gesture_record_label).strip()
        state.gesture_recording = False

        if not label or not gesture_db:
            await manager.send({"type": "error", "error": "Gesture DB tidak tersedia"}, websocket)
            return

        frames = list(state.gesture_frame_buffer)
        if not frames:
            await manager.send({"type": "error", "error": "Tidak ada frame terekam — pastikan tangan terlihat kamera"}, websocket)
            return

        gesture_db.add_gesture(label, frames, timestamp=time.time())
        gesture_db._save_to_disk()
        count = gesture_db.get_label_count(label)

        await manager.send({
            "type":       "gesture_saved",
            "label":      label,
            "count":      count,
            "frames":     len(frames),
            "model_note": f"{len(frames)} frame tersimpan untuk '{label}'",
        }, websocket)
        state.gesture_frame_buffer.clear()

    # ── GESTURE RECOGNIZE ─────────────────────────────────────────────────────
    elif mtype == "gesture_recognize":
        if not gesture_db or not gesture_db.gestures:
            await manager.send({"type": "error", "error": "Belum ada gesture tersimpan"}, websocket)
            return

        lm_data = message.get("landmarks")
        if lm_data:
            arr = landmarks_to_array(lm_data)
            if arr is not None:
                state.frame_buffer.append(arr)

        state.recognize_tick += 1
        if state.recognize_tick % 2 != 0:   # lebih responsif
            return

        if len(state.frame_buffer) < 2:
            return

        frames = list(state.frame_buffer)
        votes:  list[str]   = []
        scores: list[float] = []
        window_size = max(2, len(frames) // 2)
        for start in range(max(0, len(frames) - 5), len(frames) - window_size + 1):
            sub = frames[start: start + window_size]
            res = gesture_db.recognize(sub, threshold=0.40)  # threshold lebih rendah
            if res:
                votes.append(res[0])
                scores.append(res[1])

        if not votes:
            return

        winner    = Counter(votes).most_common(1)[0][0]
        avg_score = float(np.mean([s for v, s in zip(votes, scores) if v == winner]))

        now       = time.time()
        localized = apply_regional_mapping(winner, "")

        if localized != state.last_emitted:
            state.gesture_sentence_buffer.append(localized)
            state.last_emitted           = localized
            state.gesture_last_word_time = now

        sentence = " ".join(state.gesture_sentence_buffer)

        await manager.send({
            "type":              "translation",
            "top_prediction":    localized,
            "standard_label":    winner,
            "confidence":        round(avg_score, 3),
            "top5":              [{"label": winner, "confidence": avg_score}],
            "latency_ms":        0,
            "source":            "gesture_learning",
            "word_buffer":       list(state.gesture_sentence_buffer),
            "sentence":          sentence,
            "sentence_complete": False,
            "speak_word":        localized,
            "speak_sentence":    None,
        }, websocket)
        state.frame_buffer.clear()

    # ── GESTURE DELETE ────────────────────────────────────────────────────────
    elif mtype == "gesture_delete":
        label = str(message.get("label", "")).strip()
        if gesture_db and label:
            gesture_db.delete_label(label)
            gesture_db._save_to_disk()
            await manager.send({"type": "gesture_deleted", "label": label}, websocket)

    # ── GESTURE LIST ──────────────────────────────────────────────────────────
    elif mtype == "gesture_list":
        labels = gesture_db.get_all_labels() if gesture_db else []
        await manager.send({"type": "gesture_list", "labels": labels}, websocket)

    # ── SET MODE ──────────────────────────────────────────────────────────────
    elif mtype == "set_mode":
        m = str(message.get("mode", "predict")).strip().lower()
        if m in ("predict", "gesture_learn"):
            state.mode = m
            state.frame_buffer.clear()
            await manager.send({"type": "mode_changed", "mode": state.mode}, websocket)

    # ── AUDIO / STT ───────────────────────────────────────────────────────────
    elif mtype == "audio_chunk":
        audio = message.get("data", "") or message.get("audio", "")
        if not audio:
            return
        t0   = perf_counter()
        text = await transcribe_audio(audio)
        if text:
            await manager.send({
                "type":       "transcription",
                "text":       text,
                "latency_ms": round((perf_counter() - t0) * 1000, 1),
            }, websocket)
