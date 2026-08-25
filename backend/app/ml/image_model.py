"""
IHearYou — ImageSignModel v4 (reverted ke pipeline identik training)

PERBAIKAN v4:
Versi sebelumnya menambahkan CLAHE + Test-Time-Augmentation (termasuk
horizontal flip) + temperature scaling + EMA temporal smoothing — tapi
HANYA di inferensi, tidak pernah divalidasi terhadap model yang sudah
di-train (backend/prepare_features_from_dataset.py memakai ekstraksi
polos tanpa CLAHE sama sekali).

Pengujian empiris terhadap 312 gambar dataset asli (backend/dataset):
  - Ekstraksi identik training (tanpa CLAHE/TTA) : 93.27% akurasi
  - + CLAHE saja                                  : 91.03% akurasi (-2.24 poin)
  - + CLAHE + TTA + temperature (versi lama)       : 91.99% akurasi (-1.28 poin)

CLAHE mengubah distribusi piksel sehingga fitur HOG dan histogram warna
bergeser dari distribusi yang dipelajari StandardScaler + MLP saat
training. Flip horizontal pada TTA juga bermasalah karena HOG tidak
flip-invariant — versi cermin suatu gesture menghasilkan vektor HOG yang
sangat berbeda, sehingga ikut "memberi suara" pada prediksi akhir dengan
bukti yang salah arah.

Versi ini mengembalikan ekstraksi fitur agar IDENTIK dengan training —
resize → HOG(1764) + HSV histogram(48) = 1812 dim — tanpa preprocessing
tambahan apa pun.
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

    # ── Feature extraction — HARUS identik dengan prepare_features_from_dataset.py ──
    def _extract_features(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        HOG (1764) + HSV ColorHist (48) = 1812 dim.
        Identik persis dengan backend/prepare_features_from_dataset.py —
        JANGAN ubah tanpa retrain model, karena StandardScaler + MLP
        belajar dari distribusi fitur ini secara spesifik.
        """
        resized = cv2.resize(img_bgr, self.img_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        hog_feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True,
        ).astype(np.float32)                              # (1764,)

        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h1  = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        h2  = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        h3  = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        color = np.concatenate([h1, h2, h3]).astype(np.float32)
        color /= color.sum() + 1e-7                        # normalize (48,)

        feat = np.concatenate([hog_feat, color])            # (1812,)
        assert feat.shape[0] == 1812, f"Feature dim mismatch: {feat.shape[0]} != 1812"
        return feat

    # ── Single-image inference — tanpa TTA/EMA/temperature ─────────────────────
    def _infer_single(self, img_bgr: np.ndarray) -> np.ndarray:
        feat = self._extract_features(img_bgr).reshape(1, -1)
        sc   = self.pipeline.named_steps["sc"]
        mlp  = self.pipeline.named_steps["mlp"]
        return mlp.predict_proba(sc.transform(feat))[0]

    def reset_temporal(self) -> None:
        """No-op — dipertahankan agar kompatibel dengan pemanggilan lama
        di websockets.py (state.clear_sentence), meski tidak ada lagi
        state temporal (EMA) yang perlu direset."""
        return

    # ── Public API ────────────────────────────────────────────────────────────
    def predict_top5(self, img_bgr: np.ndarray) -> list[dict]:
        if not self.model_loaded or self.pipeline is None:
            return []

        proba = self._infer_single(img_bgr)

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
