from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionCandidate(BaseModel):
    label: str
    confidence: float


class PredictRequest(BaseModel):
    sequence: list[dict] = Field(default_factory=list)
    region: str = ""


class ImagePredictRequest(BaseModel):
    image: str
    region: str = ""


class PredictResponse(BaseModel):
    text: str
    regional_text: str
    region: str
    latency_ms: float
    top5: list[PredictionCandidate] = Field(default_factory=list)


class STTRequest(BaseModel):
    audio_data: str = ""
    audio: str = ""  # alias dari frontend


class STTResponse(BaseModel):
    text: str
    latency_ms: float


class RegionResponse(BaseModel):
    regions: list[str]
    mapping: dict


class RecordGestureRequest(BaseModel):
    label: str
    sequence: list[dict] = []


class RecordGestureResponse(BaseModel):
    label: str
    count: int
    status: str


class RecognizeGestureRequest(BaseModel):
    sequence: list[dict] = []
    threshold: float = 0.7


class RecognizeGestureResponse(BaseModel):
    recognized: bool
    label: str | None = None
    confidence: float | None = None
    message: str


class GestureListResponse(BaseModel):
    labels: list[str]
    database: dict


class ModelInfoResponse(BaseModel):
    num_classes: int
    test_accuracy: float
    test_macro_f1: float
    top5_accuracy: float
    train_samples: int
    val_samples: int = 0
    test_samples: int = 0
    architecture: str
    classes_sample: list[str]
    regions: list[str]


class ClassesResponse(BaseModel):
    classes: list[str]
    total: int


class STTStatusResponse(BaseModel):
    engine: str
    language: str
    status: str
    note: str
