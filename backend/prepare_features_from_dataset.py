from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalize_label(label: str) -> str:
    cleaned = " ".join(label.strip().split())
    cleaned = re.sub(r"\s*([-+])\s*", r" \1", cleaned)
    if len(cleaned) == 1:
        return cleaned.upper()
    return cleaned


def _extract_features_from_bgr(
    img_bgr: np.ndarray, img_size: tuple[int, int]
) -> np.ndarray:
    resized = cv2.resize(img_bgr, img_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h1 = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    h2 = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    h3 = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    color = np.concatenate([h1, h2, h3]).astype(np.float32)
    color /= color.sum() + 1e-7

    return np.concatenate([hog_feat.astype(np.float32), color]).astype(np.float32)


def _build_image_features(
    image_root: Path, img_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    label_to_features: dict[str, list[np.ndarray]] = defaultdict(list)

    if image_root.exists():
        for class_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
            label = _normalize_label(class_dir.name)
            for file_path in sorted(class_dir.iterdir()):
                if (
                    not file_path.is_file()
                    or file_path.suffix.lower() not in IMAGE_EXTS
                ):
                    continue
                img = cv2.imread(str(file_path))
                if img is None:
                    continue
                feat = _extract_features_from_bgr(img, img_size)
                label_to_features[label].append(feat)

    classes = sorted(label_to_features.keys())
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for label in classes:
        idx = class_to_idx[label]
        for feat in label_to_features[label]:
            x_rows.append(feat)
            y_rows.append(idx)

    x = (
        np.array(x_rows, dtype=np.float32)
        if x_rows
        else np.zeros((0, 1812), dtype=np.float32)
    )
    y = np.array(y_rows, dtype=np.int32) if y_rows else np.zeros((0,), dtype=np.int32)
    return x, y, classes


def _sample_video_frames(video_path: Path, max_frames: int = 6) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    candidate_factor = 3
    candidate_count = max(max_frames * candidate_factor, max_frames)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        step = max(total // candidate_count, 1)
        target_indices = list(range(0, total, step))[:candidate_count]
    else:
        target_indices = list(range(candidate_count))

    indexed_frames: list[tuple[int, np.ndarray]] = []
    index_set = set(target_indices)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in index_set:
            indexed_frames.append((i, frame))
        i += 1

    cap.release()

    if not indexed_frames:
        return []

    ranked: list[tuple[float, int, np.ndarray]] = []
    for idx, frame in indexed_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        ranked.append((sharpness, idx, frame))

    top = sorted(ranked, key=lambda x: x[0], reverse=True)[:max_frames]
    top_sorted_by_time = sorted(top, key=lambda x: x[1])
    return [item[2] for item in top_sorted_by_time]


def _build_video_features(
    video_root: Path, img_size: tuple[int, int], frames_per_video: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    label_to_features: dict[str, list[np.ndarray]] = defaultdict(list)

    if video_root.exists():
        for video_path in sorted(video_root.rglob("*.mp4")):
            label = _normalize_label(video_path.stem)
            frames = _sample_video_frames(video_path, max_frames=frames_per_video)
            for frame in frames:
                feat = _extract_features_from_bgr(frame, img_size)
                label_to_features[label].append(feat)

    classes = sorted(label_to_features.keys())
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for label in classes:
        idx = class_to_idx[label]
        for feat in label_to_features[label]:
            x_rows.append(feat)
            y_rows.append(idx)

    x = (
        np.array(x_rows, dtype=np.float32)
        if x_rows
        else np.zeros((0, 1812), dtype=np.float32)
    )
    y = np.array(y_rows, dtype=np.int32) if y_rows else np.zeros((0,), dtype=np.int32)
    return x, y, classes


def main() -> None:
    backend_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=backend_root.parent / "dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=backend_root / "data",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=6,
    )
    args = parser.parse_args()

    img_size = (64, 64)
    image_root = args.dataset_root / "Citra BISINDO"
    video_root = (
        args.dataset_root
        / "Data base BOSINDO -20260423T105629Z-3-001"
        / "Data base BOSINDO"
    )

    x_img, y_img, img_classes = _build_image_features(image_root, img_size)
    x_vid, y_vid, vid_classes = _build_video_features(
        video_root,
        img_size,
        frames_per_video=args.frames_per_video,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_dir / "img_features.pkl", "wb") as f:
        pickle.dump({"X": x_img, "y": y_img}, f)
    with open(args.output_dir / "vid_features.pkl", "wb") as f:
        pickle.dump({"X": x_vid, "y": y_vid}, f)
    with open(args.output_dir / "img_classes.json", "w", encoding="utf-8") as f:
        json.dump(img_classes, f, ensure_ascii=False, indent=2)
    with open(args.output_dir / "vid_classes.json", "w", encoding="utf-8") as f:
        json.dump(vid_classes, f, ensure_ascii=False, indent=2)

    print("FEATURE PREP COMPLETE")
    print(f"img_samples: {len(x_img)} | img_classes: {len(img_classes)}")
    print(f"vid_samples: {len(x_vid)} | vid_classes: {len(vid_classes)}")
    print(f"output_dir : {args.output_dir}")


if __name__ == "__main__":
    main()
