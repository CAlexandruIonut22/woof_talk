#!/usr/bin/env python3
"""Build a bootstrap emotion model from EmotionalCanines label schema + literature feature ranges."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_extraction import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "emotion-classifier.json"

# Feature ranges informed by EmotionalCanines paper + Yin/Molnár acoustic trends.
PROTOTYPES = {
    "arousal": {
        "Low": [0.8, 0.18, 900, 1600, 0.04, 0.28, 0.42, 0.3, 220, 25, 0.58, 1, 900, 1.1],
        "Medium": [0.55, 0.32, 1200, 2200, 0.06, 0.3, 0.4, 0.3, 280, 55, 0.52, 3, 550, 1.8],
        "High": [0.35, 0.52, 1450, 2600, 0.08, 0.34, 0.36, 0.3, 320, 85, 0.42, 5, 320, 3.1],
    },
    "valence": {
        "Negative": [0.65, 0.48, 850, 1700, 0.085, 0.46, 0.34, 0.2, 170, 20, 0.32, 4, 300, 3.3],
        "Neutral": [0.5, 0.34, 1150, 2100, 0.065, 0.31, 0.4, 0.29, 260, 45, 0.5, 3, 600, 1.7],
        "Positive": [0.42, 0.36, 1400, 2500, 0.055, 0.24, 0.43, 0.33, 330, 80, 0.66, 2, 650, 1.5],
    },
}


def synthesize(prototypes: dict[str, list[float]], samples_per_class: int = 120) -> tuple[np.ndarray, list[str]]:
    rows = []
    labels = []
    rng = np.random.default_rng(42)
    for label, center in prototypes.items():
        center_arr = np.array(center, dtype=np.float64)
        for _ in range(samples_per_class):
            noise = rng.normal(0, 0.08, size=center_arr.shape)
            sample = np.clip(center_arr * (1 + noise), 0, None)
            rows.append(sample)
            labels.append(label)
    return np.vstack(rows), labels


def serialize(pipeline: Pipeline, classes: list[str]) -> dict:
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    model: LogisticRegression = pipeline.named_steps["model"]
    return {
        "classes": classes,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }


def main() -> None:
    x_arousal, y_arousal = synthesize(PROTOTYPES["arousal"])
    x_valence, y_valence = synthesize(PROTOTYPES["valence"])
    x = np.vstack([x_arousal, x_valence[: len(x_arousal)]])

    arousal_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    valence_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )

    arousal_pipe.fit(x_arousal, y_arousal)
    valence_pipe.fit(x_valence, y_valence)

    export = {
        "version": 1,
        "dataset": "literature-prototype-bootstrap",
        "citation": "Bootstrap model from EmotionalCanines label schema + Yin/Molnár/Pongrácz acoustic trends",
        "featureNames": FEATURE_NAMES,
        "samplesTrained": len(x_arousal) + len(x_valence),
        "crossValidation": {
            "arousalAccuracy": 0.0,
            "valenceAccuracy": 0.0,
            "note": "Replace by running train_emotion_classifier.py on EmotionalCanines when Hugging Face access is available.",
        },
        "arousal": serialize(arousal_pipe, list(arousal_pipe.named_steps["model"].classes_)),
        "valence": serialize(valence_pipe, list(valence_pipe.named_steps["model"].classes_)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(export, indent=2), encoding="utf-8")
    print(f"Saved bootstrap model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
