from pathlib import Path
from time import perf_counter
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .core.config import settings
from .core.gesture_learning import GestureDatabase
from .core.monitoring import InferenceMonitor
from .core.regional_mapping import (
    apply_regional_mapping,
    get_region_mapping,
    list_regions,
)
from .ml.feature_extractor import landmarks_to_array
from .ml.image_model import ImageSignModel
from .ml.speech_recognizer import get_stt_runtime_status, transcribe_audio
from .ml.translator_model import SignTranslator
from .schemas import (
    ClassesResponse,
    GestureListResponse,
    ImagePredictRequest,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    RecognizeGestureRequest,
    RecognizeGestureResponse,
    RecordGestureRequest,
    RecordGestureResponse,
    RegionResponse,
    STTStatusResponse,
    STTRequest,
    STTResponse,
)
from .websockets import ConnectionManager, handle_message

app = FastAPI(title=settings.app_name, version=settings.app_version)

project_root = Path(__file__).resolve().parents[2]
frontend_root = project_root / "frontend"
metadata_path = project_root / "backend" / "data" / "model_metadata.json"


def _load_model_metadata() -> dict:
    if metadata_path.exists():
        try:
            with open(metadata_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "classes": ["Halo", "Apa kabar", "Terima kasih", "Tolong", "Ya", "Tidak"],
        "num_classes": 6,
        "test_accuracy": 0.0,
        "test_macro_f1": 0.0,
        "top5_accuracy": 0.0,
        "train_samples": 0,
        "architecture": "SpatialTemporalGraphTransformer",
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ws_manager = ConnectionManager(
    max_clients=settings.ws_max_clients,
    sequence_length=settings.sequence_length,
    prediction_stability_threshold=settings.prediction_stability_threshold,
)
monitor = InferenceMonitor()
gesture_db = GestureDatabase(db_path="data/gesture_db.json")
translator = SignTranslator(
    model_path=settings.model_path,
    sequence_length=settings.sequence_length,
    min_frames_for_inference=settings.min_frames_for_inference,
)
image_model = ImageSignModel(
    model_path=settings.image_model_path,
    metadata_path=settings.image_metadata_path,
    project_root=project_root,
)

app.mount("/css", StaticFiles(directory=str(frontend_root / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(frontend_root / "js")), name="js")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(frontend_root / "index.html")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": translator.model_loaded,
        "image_model_loaded": image_model.model_loaded,
        "image_model_error": image_model.load_error,
        "active_ws_clients": len(ws_manager.active),
    }


@app.get("/metrics")
async def metrics() -> dict:
    return monitor.snapshot()


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    connected = await ws_manager.connect(websocket)
    if not connected:
        return

    try:
        while True:
            message = await websocket.receive_json()
            await handle_message(
                message=message,
                websocket=websocket,
                manager=ws_manager,
                gesture_db=gesture_db,
                image_model=image_model,
            )
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as exc:
        monitor.record_error()
        await ws_manager.send(
            {
                "type": "error",
                "text": f"Translation failed: {exc}",
            },
            websocket,
        )
        ws_manager.disconnect(websocket)


@app.websocket("/ws/predict")
async def ws_predict_endpoint(websocket: WebSocket) -> None:
    """Alias /ws/predict — digunakan frontend app.js."""
    await ws_endpoint(websocket)


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    if not payload.sequence:
        raise HTTPException(status_code=400, detail="sequence must not be empty")

    start = perf_counter()
    sequence = [landmarks_to_array(frame) for frame in payload.sequence]
    base_text = translator.predict(sequence)
    top5: list[dict[str, float | str]] = []

    # Fallback yang lebih berguna saat model belum siap.
    if base_text == "Halo (dummy)":
        top5 = gesture_db.top_matches(sequence, top_k=5, min_confidence=0.2)
        if top5:
            base_text = str(top5[0]["label"])
        else:
            base_text = "Model belum tersedia. Gunakan Gesture Learning."

    regional_text = apply_regional_mapping(base_text, payload.region)
    latency_ms = round((perf_counter() - start) * 1000, 2)

    monitor.record_sign(latency_ms)
    return PredictResponse(
        text=base_text,
        regional_text=regional_text,
        region=payload.region,
        latency_ms=latency_ms,
        top5=top5,
    )


@app.post("/predict-image", response_model=PredictResponse)
async def predict_image(payload: ImagePredictRequest) -> PredictResponse:
    if not payload.image:
        raise HTTPException(status_code=400, detail="image must not be empty")

    if not image_model.model_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model image belum tersedia. Jalankan training untuk menghasilkan "
                "backend/models/pipeline_mlp.pkl"
            ),
        )

    start = perf_counter()
    try:
        top5 = image_model.predict_top5_from_base64(payload.image)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"image decode/inference failed: {exc}"
        )

    if not top5:
        raise HTTPException(
            status_code=500, detail="Inference gagal menghasilkan prediksi"
        )

    top_confidence = float(top5[0].get("confidence", 0.0))
    if top_confidence < image_model.min_confidence:
        base_text = "Tidak yakin"
    else:
        base_text = str(top5[0]["label"])
    regional_text = apply_regional_mapping(base_text, payload.region)
    latency_ms = round((perf_counter() - start) * 1000, 2)

    monitor.record_sign(latency_ms)
    return PredictResponse(
        text=base_text,
        regional_text=regional_text,
        region=payload.region,
        latency_ms=latency_ms,
        top5=top5,
    )


@app.post("/stt", response_model=STTResponse)
async def stt(payload: STTRequest) -> STTResponse:
    start = perf_counter()
    text = await transcribe_audio(payload.audio_data)
    latency_ms = round((perf_counter() - start) * 1000, 2)

    monitor.record_stt(latency_ms)
    return STTResponse(text=text, latency_ms=latency_ms)


@app.post("/api/stt", response_model=STTResponse)
async def api_stt(payload: STTRequest) -> STTResponse:
    """Frontend-friendly STT endpoint (alias /stt)."""
    start = perf_counter()
    audio = payload.audio_data or payload.audio
    text = await transcribe_audio(audio)
    latency_ms = round((perf_counter() - start) * 1000, 2)
    monitor.record_stt(latency_ms)
    return STTResponse(text=text, latency_ms=latency_ms)


@app.get("/regional", response_model=RegionResponse)
async def regional(region: str | None = None) -> RegionResponse:
    return RegionResponse(
        regions=list_regions(),
        mapping=get_region_mapping(region),
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    metadata = _load_model_metadata()
    return ModelInfoResponse(
        num_classes=int(metadata.get("num_classes", 6)),
        test_accuracy=float(metadata.get("test_accuracy", 0.0)),
        test_macro_f1=float(metadata.get("test_macro_f1", 0.0)),
        top5_accuracy=float(metadata.get("top5_accuracy", 0.0)),
        train_samples=int(metadata.get("train_samples", 0)),
        val_samples=int(metadata.get("val_samples", 0)),
        test_samples=int(metadata.get("test_samples", 0)),
        architecture=str(metadata.get("architecture", "HOG+ColorHist+MLP")),
        classes_sample=list(metadata.get("classes", []))[:10],
        regions=list_regions(),
    )


@app.get("/api/classes", response_model=ClassesResponse)
async def classes() -> ClassesResponse:
    metadata = _load_model_metadata()
    classes_payload = [str(item) for item in metadata.get("classes", [])]
    return ClassesResponse(classes=classes_payload, total=len(classes_payload))


@app.get("/api/stt-status", response_model=STTStatusResponse)
async def stt_status() -> STTStatusResponse:
    stt_runtime = get_stt_runtime_status()
    return STTStatusResponse(
        engine="Whisper",
        language="id",
        status=stt_runtime["status"],
        note=stt_runtime["note"],
    )


@app.post("/gesture/record", response_model=RecordGestureResponse)
async def record_gesture(payload: RecordGestureRequest) -> RecordGestureResponse:
    """Simpan gesture baru dengan label user-defined."""
    if not payload.label or not payload.sequence:
        raise HTTPException(status_code=400, detail="label dan sequence diperlukan")

    label = payload.label.strip().lower()
    sequence = [landmarks_to_array(frame) for frame in payload.sequence]
    gesture_db.add_gesture(label, sequence)
    gesture_db._save_to_disk()

    return RecordGestureResponse(
        label=label,
        count=gesture_db.get_label_count(label),
        status="recorded",
    )


@app.post("/gesture/recognize", response_model=RecognizeGestureResponse)
async def recognize_gesture(
    payload: RecognizeGestureRequest,
) -> RecognizeGestureResponse:
    """Recognize gesture dari stored database dengan similarity matching."""
    if not payload.sequence:
        raise HTTPException(status_code=400, detail="sequence diperlukan")

    sequence = [landmarks_to_array(frame) for frame in payload.sequence]
    result = gesture_db.recognize(sequence, threshold=payload.threshold)

    if result:
        label, confidence = result
        return RecognizeGestureResponse(
            recognized=True,
            label=label,
            confidence=confidence,
            message=f"Terdeteksi: {label} ({confidence * 100:.1f}%)",
        )

    return RecognizeGestureResponse(
        recognized=False,
        message="Gesture tidak dikenali. Silakan record gesture terlebih dahulu.",
    )


@app.delete("/gesture/{label}")
async def delete_gesture_endpoint(label: str) -> dict:
    """Hapus gesture by label."""
    label = label.strip().lower()
    if gesture_db.delete_gesture(label):
        return {"status": "deleted", "label": label}
    raise HTTPException(status_code=404, detail=f"Gesture '{label}' tidak ditemukan")


@app.get("/gesture/list", response_model=GestureListResponse)
async def list_gestures() -> GestureListResponse:
    """Daftar semua gesture yang sudah direcord."""
    return GestureListResponse(
        labels=gesture_db.list_labels(),
        database=gesture_db.export(),
    )
