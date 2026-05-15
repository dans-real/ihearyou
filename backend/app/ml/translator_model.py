from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


class SpatialTemporalGraphTransformer(nn.Module if nn is not None else object):
    def __init__(
        self, input_dim: int, num_classes: int = 6, d_model: int = 128
    ) -> None:
        if nn is None:
            return
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if nn is None:
            raise RuntimeError("Torch is not available")
        x = self.input_proj(x)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)


class SignTranslator:
    def __init__(
        self,
        model_path: str,
        sequence_length: int = 30,
        min_frames_for_inference: int = 15,
        input_dim: int = 1530,  # Disesuaikan dengan output baru dari landmarks_to_array
    ) -> None:
        self.backend_root = Path(__file__).resolve().parents[2]
        path = Path(model_path)
        self.model_path = path if path.is_absolute() else self.backend_root / path

        self.sequence_length = sequence_length
        self.min_frames_for_inference = min_frames_for_inference
        self.input_dim = input_dim
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch is not None
            else "cpu"
        )
        self.id_to_text = {
            0: "Halo",
            1: "Apa kabar",
            2: "Terima kasih",
            3: "Tolong",
            4: "Ya",
            5: "Tidak",
        }

        self.model = None
        self.model_loaded = False
        if torch is not None and nn is not None:
            self.model = SpatialTemporalGraphTransformer(input_dim=self.input_dim).to(
                self.device
            )
        self._load_model_if_available()

    def _load_model_if_available(self) -> None:
        if not self.model_path.exists() or self.model_path.stat().st_size == 0:
            return

        if torch is None or self.model is None:
            return

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
        except Exception:
            try:
                # PyTorch 2.6 changes torch.load default to weights_only=True.
                checkpoint = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except Exception:
                self.model_loaded = False
                return

        try:
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                self.model_loaded = True
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint, strict=False)
                self.model_loaded = True
        except Exception:
            self.model_loaded = False

    def _stack_sequence(
        self, sequence_of_landmarks: Sequence[np.ndarray]
    ) -> np.ndarray:
        seq: list[np.ndarray] = []
        for frame in sequence_of_landmarks:
            frame_arr = np.asarray(frame, dtype=np.float32).reshape(-1)
            if frame_arr.shape[0] < self.input_dim:
                pad = np.zeros((self.input_dim - frame_arr.shape[0],), dtype=np.float32)
                frame_arr = np.concatenate([frame_arr, pad], axis=0)
            elif frame_arr.shape[0] > self.input_dim:
                frame_arr = frame_arr[: self.input_dim]
            seq.append(frame_arr)

        return np.stack(seq, axis=0).astype(np.float32)

    def _prepare_tensor(
        self, sequence_of_landmarks: Sequence[np.ndarray] | deque[np.ndarray]
    ) -> torch.Tensor:
        stacked = self._stack_sequence(sequence_of_landmarks)
        arr = stacked[None, :, :]
        if torch is None:
            return arr
        return torch.from_numpy(arr).to(self.device)

    def predict(
        self, sequence_of_landmarks: Sequence[np.ndarray] | deque[np.ndarray]
    ) -> str:
        if len(sequence_of_landmarks) < self.min_frames_for_inference:
            return ""

        if torch is None or self.model is None or not self.model_loaded:
            return "Halo (dummy)"

        self.model.eval()
        with torch.no_grad():
            tensor = self._prepare_tensor(sequence_of_landmarks)
            logits = self.model(tensor)
            pred = int(torch.argmax(logits, dim=-1).item())

        return self.id_to_text.get(pred, "Terjemahan tidak dikenali")
