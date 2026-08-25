"""
IHearYou — Feature Extractor v3

v3 fixes:
- Validasi: jika semua landmark nol (tangan tidak terdeteksi), return None
- Log warning saat landmark kosong
- Z default ke 0.0 jika tidak ada (MediaPipe kadang tidak kirim z)
"""
from __future__ import annotations
import numpy as np


def _extract_xyz(landmark_list: list[dict] | None, n: int) -> np.ndarray:
    if not landmark_list:
        return np.zeros((n, 3), dtype=np.float32)
    coords = np.array(
        [[lm.get("x", 0.0), lm.get("y", 0.0), lm.get("z", 0.0)]
         for lm in landmark_list],
        dtype=np.float32,
    )
    if coords.shape[0] < n:
        pad = np.zeros((n - coords.shape[0], 3), dtype=np.float32)
        return np.concatenate([coords, pad], axis=0)
    return coords[:n]


def _has_valid_landmarks(lm: np.ndarray) -> bool:
    """Cek apakah landmark tidak semua nol."""
    return bool(np.any(lm != 0))


def landmarks_to_array(
    landmarks_data: dict | None,
    target_dim: int = (33 + 21 + 21) * 3,
) -> np.ndarray | None:
    """
    Konversi landmark dict → flat float32 array.
    Return None jika tidak ada tangan terdeteksi (semua nol).
    """
    if not landmarks_data:
        return None

    pose = _extract_xyz(landmarks_data.get("pose"), 33)
    lh   = _extract_xyz(landmarks_data.get("left_hand"), 21)
    rh   = _extract_xyz(landmarks_data.get("right_hand"), 21)

    # Validasi: minimal satu tangan harus terdeteksi
    if not _has_valid_landmarks(lh) and not _has_valid_landmarks(rh):
        return None   # sinyal ke caller: tidak ada tangan

    # Wrist-relative normalization
    if _has_valid_landmarks(lh):
        ref = lh[0]
    elif _has_valid_landmarks(rh):
        ref = rh[0]
    else:
        lw, rw = pose[15], pose[16]
        ref = (lw + rw) / 2.0 if np.any(lw) and np.any(rw) else np.zeros(3, dtype=np.float32)

    pose -= ref
    lh   -= ref
    rh   -= ref

    merged = np.concatenate([pose, lh, rh], axis=0)   # (75, 3)
    scale  = np.max(np.linalg.norm(merged, axis=1))
    if scale > 1e-6:
        merged /= scale

    flat = merged.reshape(-1).astype(np.float32)       # 225

    out = np.zeros(target_dim, dtype=np.float32)
    n   = min(len(flat), target_dim)
    out[:n] = flat[:n]
    return out
