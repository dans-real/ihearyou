"""
IHearYou - Metadata Trainer (legacy HOG/MLP dataset compatible)

Script ini melatih MLP dari feature yang sudah diekstrak lalu menyimpan metadata
ke backend/data/model_metadata.json agar langsung dipakai endpoint /api/model-info
dan /api/classes pada FastAPI app.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    img_features = data_dir / "img_features.pkl"
    vid_features = data_dir / "vid_features.pkl"
    img_classes_path = data_dir / "img_classes.json"
    vid_classes_path = data_dir / "vid_classes.json"

    required = [img_features, vid_features, img_classes_path, vid_classes_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        missing_str = "\n- ".join([""] + missing)
        raise FileNotFoundError(
            "Dataset pre-extracted belum lengkap. File yang belum ada:" + missing_str
        )

    with open(img_features, "rb") as f:
        img_data = pickle.load(f)
    with open(vid_features, "rb") as f:
        vid_data = pickle.load(f)
    with open(img_classes_path, encoding="utf-8") as f:
        img_classes = json.load(f)
    with open(vid_classes_path, encoding="utf-8") as f:
        vid_classes = json.load(f)

    all_classes = list(img_classes)
    for cls in vid_classes:
        if cls not in all_classes:
            all_classes.append(cls)

    x_all: list[np.ndarray] = []
    y_all: list[int] = []

    img_remap = {i: all_classes.index(img_classes[i]) for i in range(len(img_classes))}
    for feat, idx in zip(img_data["X"], img_data["y"]):
        x_all.append(feat)
        y_all.append(img_remap[idx])

    vid_remap = {i: all_classes.index(vid_classes[i]) for i in range(len(vid_classes))}
    for feat, idx in zip(vid_data["X"], vid_data["y"]):
        x_all.append(feat)
        y_all.append(vid_remap[idx])

    return (
        np.array(x_all, dtype=np.float32),
        np.array(y_all, dtype=np.int32),
        all_classes,
    )


def _augment_min_samples(
    x_all: np.ndarray, y_all: np.ndarray, target_per_class: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    counts = Counter(y_all.tolist())
    x_aug, y_aug = list(x_all), list(y_all)

    for cls_idx, count in counts.items():
        if count >= target_per_class:
            continue
        existing = x_all[y_all == cls_idx]
        need = target_per_class - count
        for _ in range(need):
            base = existing[rng.randint(0, len(existing))]
            noise = rng.normal(0, 0.015, base.shape).astype(np.float32)
            scaled = base * rng.uniform(0.98, 1.02)
            x_aug.append(scaled + noise)
            y_aug.append(cls_idx)

    return np.array(x_aug, dtype=np.float32), np.array(y_aug, dtype=np.int32)


def _top_k_accuracy(pipe: Pipeline, x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    x_scaled = pipe.named_steps["sc"].transform(x)
    proba = pipe.named_steps["mlp"].predict_proba(x_scaled)
    topk = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([y[i] in topk[i] for i in range(len(y))]))


def _build_pipeline(
    *,
    hidden_layers: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    random_state: int,
) -> Pipeline:
    return Pipeline(
        [
            ("sc", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size=32,
                    learning_rate="adaptive",
                    learning_rate_init=learning_rate_init,
                    max_iter=400,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=25,
                    random_state=random_state,
                    verbose=False,
                ),
            ),
        ]
    )


def _calibrate_confidence_threshold(
    pipe: Pipeline,
    x_val: np.ndarray,
    y_val: np.ndarray,
    min_precision: float = 0.85,
) -> tuple[float, float, float]:
    x_scaled = pipe.named_steps["sc"].transform(x_val)
    proba = pipe.named_steps["mlp"].predict_proba(x_scaled)
    conf = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)

    best_threshold = 0.0
    best_precision = float(np.mean(pred == y_val))
    best_coverage = 1.0
    best_score = -1.0
    has_precision_constrained = False

    for threshold in np.arange(0.20, 0.96, 0.05):
        mask = conf >= threshold
        covered = int(mask.sum())
        if covered == 0:
            continue

        precision = float(np.mean(pred[mask] == y_val[mask]))
        coverage = float(covered / len(y_val))

        if precision >= min_precision:
            score = coverage
            if (not has_precision_constrained) or score > best_score:
                has_precision_constrained = True
                best_score = score
                best_threshold = float(threshold)
                best_precision = precision
                best_coverage = coverage
        elif not has_precision_constrained:
            score = precision * coverage
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
                best_precision = precision
                best_coverage = coverage

    return best_threshold, best_precision, best_coverage


def train(
    data_dir: Path,
    output_model: Path,
    output_metadata: Path,
    output_report: Path,
    target_per_class: int,
    min_calibration_precision: float,
) -> None:
    x_all, y_all, all_classes = _load_dataset(data_dir)
    x_aug, y_aug = _augment_min_samples(
        x_all, y_all, target_per_class=max(2, target_per_class)
    )

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x_aug, y_aug, test_size=0.30, random_state=42, stratify=y_aug
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    candidates = [
        {
            "name": "MLP(512-256-128)",
            "hidden_layers": (512, 256, 128),
            "alpha": 1e-4,
            "learning_rate_init": 1e-3,
        },
        {
            "name": "MLP(512-256)",
            "hidden_layers": (512, 256),
            "alpha": 2e-4,
            "learning_rate_init": 8e-4,
        },
        {
            "name": "MLP(384-192-96)",
            "hidden_layers": (384, 192, 96),
            "alpha": 3e-4,
            "learning_rate_init": 8e-4,
        },
    ]

    best_pipe = None
    best_name = ""
    best_val_acc = -1.0
    best_val_f1 = -1.0

    for i, cfg in enumerate(candidates):
        pipe = _build_pipeline(
            hidden_layers=cfg["hidden_layers"],
            alpha=float(cfg["alpha"]),
            learning_rate_init=float(cfg["learning_rate_init"]),
            random_state=42 + i,
        )
        pipe.fit(x_train, y_train)

        y_pred_val_candidate = pipe.predict(x_val)
        val_acc_candidate = accuracy_score(y_val, y_pred_val_candidate)
        val_f1_candidate = f1_score(
            y_val, y_pred_val_candidate, average="macro", zero_division=0
        )
        print(
            f"Candidate {cfg['name']}: "
            f"val_acc={val_acc_candidate:.4f}, val_f1={val_f1_candidate:.4f}"
        )

        if val_f1_candidate > best_val_f1 or (
            abs(val_f1_candidate - best_val_f1) < 1e-8
            and val_acc_candidate > best_val_acc
        ):
            best_pipe = pipe
            best_name = str(cfg["name"])
            best_val_acc = float(val_acc_candidate)
            best_val_f1 = float(val_f1_candidate)

    if best_pipe is None:
        raise RuntimeError("Training gagal: tidak ada kandidat model yang berhasil")

    y_pred_val = best_pipe.predict(x_val)
    y_pred_test = best_pipe.predict(x_test)
    val_acc = accuracy_score(y_val, y_pred_val)
    test_acc = accuracy_score(y_test, y_pred_test)
    val_f1 = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    top5 = _top_k_accuracy(best_pipe, x_test, y_test, k=5)
    calibrated_threshold, calibrated_precision, calibrated_coverage = (
        _calibrate_confidence_threshold(
            best_pipe,
            x_val,
            y_val,
            min_precision=min_calibration_precision,
        )
    )

    output_model.parent.mkdir(parents=True, exist_ok=True)
    with open(output_model, "wb") as f:
        pickle.dump(best_pipe, f)

    test_classes_present = sorted(set(y_test.tolist()))
    report = classification_report(
        y_test,
        y_pred_test,
        labels=test_classes_present,
        target_names=[all_classes[i] for i in test_classes_present],
        zero_division=0,
    )

    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(f"Best Model: {best_name}\n")
        f.write(f"Val  Acc: {val_acc:.4f} | F1: {val_f1:.4f}\n")
        f.write(f"Test Acc: {test_acc:.4f} | F1: {test_f1:.4f}\n")
        f.write(f"Top-5   : {top5:.4f}\n\n")
        f.write(
            "Calibrated confidence threshold: "
            f"{calibrated_threshold:.2f} "
            f"(val precision={calibrated_precision:.4f}, "
            f"coverage={calibrated_coverage:.4f})\n\n"
        )
        f.write(report)

    metadata = {
        "classes": all_classes,
        "num_classes": len(all_classes),
        "feature_dim": int(x_all.shape[1]),
        "val_accuracy": round(float(val_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_macro_f1": round(float(test_f1), 4),
        "top5_accuracy": round(float(top5), 4),
        "recommended_confidence_threshold": round(float(calibrated_threshold), 2),
        "val_precision_at_threshold": round(float(calibrated_precision), 4),
        "val_coverage_at_threshold": round(float(calibrated_coverage), 4),
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "img_size": [64, 64],
        "architecture": f"HOG(9,8,2)+ColorHist(HSV16) -> {best_name}",
        "augmentation_target_per_class": int(max(2, target_per_class)),
    }

    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("TRAINING COMPLETE")
    print(f"Model saved     : {output_model}")
    print(f"Metadata saved  : {output_metadata}")
    print(f"Report saved    : {output_report}")


def main() -> None:
    backend_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=backend_root / "data",
        help="Folder berisi img_features.pkl, vid_features.pkl, img_classes.json, vid_classes.json",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=backend_root / "models" / "pipeline_mlp.pkl",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=backend_root / "data" / "model_metadata.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=backend_root / "models" / "classification_report.txt",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=10,
        help="Jumlah minimum sampel per kelas setelah augmentasi noise.",
    )
    parser.add_argument(
        "--min-calibration-precision",
        type=float,
        default=0.85,
        help="Target precision minimum saat kalibrasi confidence threshold.",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_model=args.output_model,
        output_metadata=args.output_metadata,
        output_report=args.output_report,
        target_per_class=args.target_per_class,
        min_calibration_precision=args.min_calibration_precision,
    )


if __name__ == "__main__":
    main()
