#!/usr/bin/env python3
"""Train arousal/valence classifiers on EmotionalCanines (Hugging Face)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_extraction import FEATURE_NAMES, extract_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "emotion-classifier.json"
DATASET_ID = "ArlingtonCL2/BarkopediaDogEmotionClassification_Data"


def serialize_pipeline(name: str, pipeline: Pipeline, classes: list[str]) -> dict:
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    model: LogisticRegression = pipeline.named_steps["model"]
    return {
        "name": name,
        "classes": classes,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }


def main() -> None:
    print(f"Loading {DATASET_ID} ...")
    dataset = load_dataset(DATASET_ID, split="train")

    features = []
    arousal_labels = []
    valence_labels = []

    for index, row in enumerate(dataset):
        audio = row["audio"]
        path = audio["path"] if isinstance(audio, dict) else audio
        try:
            features.append(extract_features(path))
            arousal_labels.append(row["arousal"])
            valence_labels.append(row["valence"])
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping clip {index}: {exc}")

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(dataset)} clips ...")

    if len(features) < 50:
        raise RuntimeError(f"Only {len(features)} usable clips found.")

    x = np.vstack(features)

    arousal_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000)),
        ]
    )
    valence_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000)),
        ]
    )

    arousal_pipe.fit(x, arousal_labels)
    valence_pipe.fit(x, valence_labels)

    arousal_cv = cross_val_score(arousal_pipe, x, arousal_labels, cv=5).mean()
    valence_cv = cross_val_score(valence_pipe, x, valence_labels, cv=5).mean()

    export = {
        "version": 1,
        "dataset": DATASET_ID,
        "citation": "Dang et al. 2025, EmotionalCanines (ACM MM '25)",
        "featureNames": FEATURE_NAMES,
        "samplesTrained": len(features),
        "crossValidation": {
            "arousalAccuracy": round(float(arousal_cv), 4),
            "valenceAccuracy": round(float(valence_cv), 4),
        },
        "arousal": serialize_pipeline(
            "arousal",
            arousal_pipe,
            list(arousal_pipe.named_steps["model"].classes_),
        ),
        "valence": serialize_pipeline(
            "valence",
            valence_pipe,
            list(valence_pipe.named_steps["model"].classes_),
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(export, indent=2), encoding="utf-8")
    print(f"Saved {MODEL_PATH}")
    print(
        "CV accuracy — arousal: "
        f"{export['crossValidation']['arousalAccuracy']}, "
        f"valence: {export['crossValidation']['valenceAccuracy']}"
    )


if __name__ == "__main__":
    main()
