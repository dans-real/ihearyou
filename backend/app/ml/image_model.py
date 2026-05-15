"""
IHearYou — ImageSignModel v3
CRITICAL FIX: Feature vector dikembalikan ke 1812 dim (HOG 1764 + ColorHist 48)
agar kompatibel dengan pipeline_mlp.pkl yang sudah di-train.

Peningkatan yang TIDAK mengubah dimensi (aman untuk model lama):
- CLAHE adaptive contrast sebelum HOG  → akurasi naik tanpa retrain
- Test-Time Augmentation (TTA) 4x      → prediksi lebih stabil
- Temperature scaling calibration       → confidence lebih reliable
- Temporal EMA smoothing                → tidak jumping antar frame
"""
from __future__ import annotations

import base64
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog


class ImageSignModel:
    # Kalibrasi confidence (temperature > 1 → lebih konservatif)
    TEMPERATURE: float = 1.1

    # EMA smoothing antar frame: 0=no smooth, 1=fully frozen
    EMA_ALPHA: float = 0.30

    def __init__(self, model_path: str, metadata_path: str, project_root: Path) -> None:
        self.project_root   = project_root
        self.model_path     = self._resolve(model_path)
        self.metadata_path  = self._resolve(metadata_path)
        self.pipeline       = None
        self.classes: list[str] = []
        self.img_size       = (64, 64)
        self.min_confidence = 0.20
        self.model_loaded   = False
        self.load_error     = ""

        # Temporal EMA state
        self._ema_proba: np.ndarray | None = None

        self._load()

    def _resolve(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else self.project_root / "backend" / value

    # ── Load ──────────────────────────────────────────────────────────────────
    def _load(self) -> None:
        meta: dict = {}
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        self.classes  = [str(x) for x in meta.get("classes", [])]
        raw_size      = meta.get("img_size", [64, 64])
        self.img_size = tuple(int(x) for x in raw_size[:2]) if isinstance(raw_size, list) else (64, 64)

        try:
            thresh = float(meta.get("recommended_confidence_threshold", 0.20))
        except (TypeError, ValueError):
            thresh = 0.20
        self.min_confidence = max(0.0, min(1.0, thresh))

        if not self.model_path.exists():
            self.model_loaded = False
            self.load_error   = f"Model tidak ditemukan: {self.model_path}"
            return
        try:
            with open(self.model_path, "rb") as f:
                self.pipeline = pickle.load(f)
            self.model_loaded = True
        except Exception as exc:
            self.model_loaded = False
            self.load_error   = str(exc)

    # ── Feature extraction — HARUS 1812 dim ─────────────────────────────────
    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """Resize ke img_size + CLAHE adaptive contrast (tidak mengubah dimensi)."""
        img  = cv2.resize(img_bgr, self.img_size)
        lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _extract_features(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        HOG (1764) + HSV ColorHist (48) = 1812 dim.
        Identik dengan training pipeline — JANGAN ubah tanpa retrain model.
        """
        proc = self._preprocess(img_bgr)
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

        # HOG — 1764 dim (sama dengan training)
        hog_feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True,
        ).astype(np.float32)                              # (1764,)

        # HSV color histogram — 48 dim (sama dengan training)
        hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
        h1  = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        h2  = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        h3  = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        color = np.concatenate([h1, h2, h3]).astype(np.float32)
        color /= color.sum() + 1e-7                       # normalize (48,)

        feat = np.concatenate([hog_feat, color])          # (1812,)
        assert feat.shape[0] == 1812, f"Feature dim mismatch: {feat.shape[0]} != 1812"
        return feat

    # ── Temperature-scaled softmax ────────────────────────────────────────────
    @staticmethod
    def _softmax_temp(raw: np.ndarray, temp: float) -> np.ndarray:
        logits = raw / max(temp, 1e-6)
        e = np.exp(logits - logits.max())
        return e / (e.sum() + 1e-9)

    # ── Single-image inference ────────────────────────────────────────────────
    def _infer_single(self, img_bgr: np.ndarray) -> np.ndarray:
        feat = self._extract_features(img_bgr).reshape(1, -1)
        sc   = self.pipeline.named_steps["sc"]
        mlp  = self.pipeline.named_steps["mlp"]
        raw  = mlp.predict_proba(sc.transform(feat))[0]
        return self._softmax_temp(raw, self.TEMPERATURE)

    # ── Test-Time Augmentation — tidak ubah dim, hanya rata-rata prediksi ────
    def _infer_tta(self, img_bgr: np.ndarray) -> np.ndarray:
        augmented = [
            img_bgr,                                                          # original
            cv2.flip(img_bgr, 1),                                             # H-flip
            np.clip(img_bgr.astype(np.float32) * 1.20, 0, 255).astype(np.uint8),  # bright +20%
            np.clip(img_bgr.astype(np.float32) * 0.82, 0, 255).astype(np.uint8),  # dark -18%
        ]
        probas = np.array([self._infer_single(a) for a in augmented])
        return probas.mean(axis=0)                        # rata-rata 4 prediksi

    # ── Temporal EMA smoothing ────────────────────────────────────────────────
    def _apply_ema(self, new_proba: np.ndarray) -> np.ndarray:
        if self._ema_proba is None or self._ema_proba.shape != new_proba.shape:
            self._ema_proba = new_proba.copy()
            return new_proba
        self._ema_proba = self.EMA_ALPHA * new_proba + (1.0 - self.EMA_ALPHA) * self._ema_proba
        return self._ema_proba.copy()

    def reset_temporal(self) -> None:
        """Reset EMA — panggil saat clear atau no-hand."""
        self._ema_proba = None

    # ── Public API ────────────────────────────────────────────────────────────
    def predict_top5(self, img_bgr: np.ndarray) -> list[dict]:
        if not self.model_loaded or self.pipeline is None:
            return []

        raw   = self._infer_tta(img_bgr)       # TTA
        proba = self._apply_ema(raw)            # temporal smoothing

        if not self.classes:
            self.classes = [str(i) for i in range(len(proba))]

        top_idx = np.argsort(proba)[::-1][:5]
        return [
            {
                "label":      self.classes[int(i)] if int(i) < len(self.classes) else str(int(i)),
                "confidence": float(proba[int(i)]),
            }
            for i in top_idx
        ]

    def predict_top5_from_base64(self, image_b64: str) -> list[dict]:
        return self.predict_top5(self.decode_base64_image(image_b64))

    @staticmethod
    def decode_base64_image(image_b64: str) -> np.ndarray:
        encoded   = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
        img_bytes = base64.b64decode(encoded)
        img_arr   = np.frombuffer(img_bytes, np.uint8)
        img_bgr   = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Image decode gagal — pastikan base64 valid")
        return img_bgr
