"""
IHearYou — GestureDatabase v2

v2 improvements:
- Recognize: DTW-lite (per-frame nearest-neighbor) menggantikan pure cosine mean
- Multi-record averaging: ambil top-k similar records, rata-rata skor
- Threshold adaptif berdasarkan jumlah rekaman per label
- _save/_load robust terhadap format lama
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np


class GestureRecord(NamedTuple):
    label:       str
    embedding:   list[float]    # mean frame embedding
    frames:      list[list[float]] | None  # per-frame embeddings (v2)
    timestamp:   float
    frame_count: int


class GestureDatabase:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path  = Path(db_path) if db_path else None
        self.gestures: dict[str, list[GestureRecord]] = {}
        if self.db_path and self.db_path.exists():
            self._load_from_disk()

    # ── Add ────────────────────────────────────────────────────────────────
    def add_gesture(self, label: str, landmarks_sequence: list[np.ndarray],
                    timestamp: float = 0.0) -> None:
        if not label or not landmarks_sequence:
            return
        flat_frames = [lm.flatten().tolist() for lm in landmarks_sequence]
        arr         = np.array([lm.flatten() for lm in landmarks_sequence], dtype=np.float32)
        mean_emb    = arr.mean(axis=0).tolist()

        record = GestureRecord(
            label=label, embedding=mean_emb, frames=flat_frames,
            timestamp=timestamp or time.time(), frame_count=len(landmarks_sequence)
        )
        self.gestures.setdefault(label, []).append(record)

    # ── Query helpers ──────────────────────────────────────────────────────
    def get_label_count(self, label: str) -> int:
        return len(self.gestures.get(label, []))

    def get_all_labels(self) -> list[str]:
        return list(self.gestures.keys())

    def delete_label(self, label: str) -> bool:
        if label in self.gestures:
            del self.gestures[label]
            return True
        return False

    # ── Recognize ─────────────────────────────────────────────────────────
    def recognize(self, landmarks_sequence: list[np.ndarray],
                  threshold: float = 0.55) -> tuple[str, float] | None:
        if not landmarks_sequence or not self.gestures:
            return None

        query_frames = np.array([lm.flatten() for lm in landmarks_sequence], dtype=np.float32)
        query_mean   = query_frames.mean(axis=0)

        best_label: str | None = None
        best_score: float      = 0.0

        for label, records in self.gestures.items():
            label_scores: list[float] = []

            for record in records:
                # 1. Mean-embedding cosine (fast)
                stored = np.array(record.embedding, dtype=np.float32)
                cos    = self._cosine(query_mean, stored)

                # 2. Per-frame DTW-lite (only if record has frame data)
                if record.frames and len(record.frames) >= 2:
                    ref_frames = np.array(record.frames, dtype=np.float32)
                    dtw_score  = self._dtw_lite(query_frames, ref_frames)
                    # Combine: 60% cosine, 40% DTW
                    combined = 0.60 * cos + 0.40 * dtw_score
                else:
                    combined = cos

                label_scores.append(combined)

            # Take top-k scores (ignore outlier recordings)
            k          = max(1, len(label_scores) - 1) if len(label_scores) > 2 else len(label_scores)
            top_k      = sorted(label_scores, reverse=True)[:k]
            label_avg  = float(np.mean(top_k))

            # Adaptive threshold: more recordings → slightly stricter
            n_rec       = len(records)
            adj_thresh  = threshold + min(0.05 * (n_rec // 5), 0.10)

            if label_avg > best_score and label_avg >= adj_thresh:
                best_score = label_avg
                best_label = label

        if best_label is not None:
            return (best_label, round(best_score, 3))
        return None

    # ── Similarity helpers ─────────────────────────────────────────────────
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _dtw_lite(query: np.ndarray, ref: np.ndarray) -> float:
        """
        Lightweight DTW: map each query frame to its nearest ref frame,
        average cosine similarity → [0, 1]
        """
        scores: list[float] = []
        ref_norms = np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8
        ref_n     = ref / ref_norms

        for q_frame in query:
            qn = np.linalg.norm(q_frame)
            if qn < 1e-8:
                continue
            q_n   = q_frame / qn
            sims  = ref_n @ q_n                  # shape (n_ref,)
            scores.append(float(sims.max()))

        return float(np.mean(scores)) if scores else 0.0

    # ── Persistence ────────────────────────────────────────────────────────
    def _save_to_disk(self) -> None:
        if not self.db_path:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            label: [
                {
                    "embedding":   r.embedding,
                    "frames":      r.frames or [],
                    "timestamp":   r.timestamp,
                    "frame_count": r.frame_count,
                }
                for r in records
            ]
            for label, records in self.gestures.items()
        }
        self.db_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_from_disk(self) -> None:
        try:
            raw = json.loads(self.db_path.read_text(encoding="utf-8"))
        except Exception:
            return

        for label, records in raw.items():
            self.gestures[label] = [
                GestureRecord(
                    label=label,
                    embedding=r["embedding"],
                    frames=r.get("frames"),        # None for old-format records
                    timestamp=r.get("timestamp", 0.0),
                    frame_count=r.get("frame_count", len(r["embedding"])),
                )
                for r in records
            ]
