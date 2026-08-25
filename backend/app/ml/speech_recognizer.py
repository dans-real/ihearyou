from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
except ImportError:  # pragma: no cover
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None
    pipeline = None

_ASR_PIPELINE = None
_MODEL_ID = "openai/whisper-base"
_ASR_DISABLED = False
_ASR_DISABLE_REASON = ""


def _get_asr_pipeline():
    global _ASR_PIPELINE, _ASR_DISABLED, _ASR_DISABLE_REASON
    if _ASR_PIPELINE is not None:
        return _ASR_PIPELINE

    if _ASR_DISABLED:
        return None

    if pipeline is None or AutoProcessor is None or AutoModelForSpeechSeq2Seq is None:
        _ASR_DISABLED = True
        _ASR_DISABLE_REASON = "Dependency transformers/torch belum siap"
        return None

    has_cuda = torch.cuda.is_available()
    torch_dtype = torch.float16 if has_cuda else torch.float32

    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to("cuda" if has_cuda else "cpu")

        processor = AutoProcessor.from_pretrained(_MODEL_ID)

        _ASR_PIPELINE = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=0 if has_cuda else -1,
        )
    except Exception as exc:
        _ASR_DISABLED = True
        _ASR_DISABLE_REASON = f"Whisper belum tersedia offline: {exc}"
        return None

    return _ASR_PIPELINE


def get_stt_runtime_status() -> dict[str, str]:
    if _ASR_PIPELINE is not None:
        return {
            "status": "ready",
            "note": f"Whisper aktif ({_MODEL_ID})",
        }

    if _ASR_DISABLED:
        return {
            "status": "unavailable",
            "note": _ASR_DISABLE_REASON or "Whisper tidak tersedia di environment ini",
        }

    return {
        "status": "lazy",
        "note": "Whisper akan diinisialisasi saat audio pertama diterima",
    }


def _transcribe_audio_sync(audio_data: str) -> str:
    if not audio_data:
        return ""

    asr = _get_asr_pipeline()
    if asr is None:
        return ""

    temp_path: Path | None = None
    try:
        encoded = audio_data.split(",", 1)[1] if "," in audio_data else audio_data
        audio_bytes = base64.b64decode(encoded)

        with NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = Path(temp_file.name)

        result = asr(str(temp_path), generate_kwargs={"task": "transcribe"})
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(result).strip()
    except Exception:
        return ""
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


async def transcribe_audio(audio_data: str) -> str:
    return await asyncio.to_thread(_transcribe_audio_sync, audio_data)
