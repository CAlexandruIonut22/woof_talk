"""Acoustic features aligned with canine bark literature (Yin & McCowan 2004; Molnár et al. 2008)."""

from __future__ import annotations

import numpy as np
import librosa


FEATURE_NAMES = [
    "log_duration",
    "rms",
    "spectral_centroid",
    "spectral_rolloff",
    "zero_crossing_rate",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
    "f0_mean",
    "f0_std",
    "harmonicity",
    "bark_count",
    "mean_inter_bark_ms",
    "repetition_rate_hz",
]


def _band_ratios(y: np.ndarray, sr: int) -> tuple[float, float, float]:
    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    power = spec ** 2
    total = float(power.sum()) or 1.0
    low = float(power[freqs < 500].sum()) / total
    mid = float(power[(freqs >= 500) & (freqs < 2000)].sum()) / total
    high = float(power[freqs >= 2000].sum()) / total
    return low, mid, high


def _estimate_f0(y: np.ndarray, sr: int) -> tuple[float, float]:
    f0 = librosa.yin(y, fmin=80, fmax=800, sr=sr)
    voiced = f0[(f0 > 0) & np.isfinite(f0)]
    if voiced.size == 0:
        return 0.0, 0.0
    return float(np.mean(voiced)), float(np.std(voiced))


def _harmonicity(y: np.ndarray, sr: int) -> float:
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_energy = float(np.mean(harmonic ** 2))
    percussive_energy = float(np.mean(percussive ** 2))
    return harmonic_energy / (harmonic_energy + percussive_energy + 1e-9)


def _bark_events(y: np.ndarray, sr: int) -> tuple[int, float, float]:
    envelope = np.abs(librosa.effects.preemphasis(y))
    envelope = librosa.util.normalize(envelope)
    peaks = librosa.util.peak_pick(
        envelope,
        pre_max=3,
        post_max=3,
        pre_avg=10,
        post_avg=10,
        delta=0.08,
        wait=int(0.08 * sr),
    )
    count = int(len(peaks))
    if count < 2:
        return count, 0.0, 0.0
    intervals = np.diff(peaks) / sr * 1000.0
    mean_interval = float(np.mean(intervals))
    repetition = 1000.0 / mean_interval if mean_interval > 0 else 0.0
    return count, mean_interval, repetition


def extract_features(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=22050, mono=True)
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y ** 2)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    low, mid, high = _band_ratios(y, sr)
    f0_mean, f0_std = _estimate_f0(y, sr)
    harmonicity = _harmonicity(y, sr)
    bark_count, inter_bark_ms, repetition = _bark_events(y, sr)

    return np.array(
        [
            np.log1p(duration),
            rms,
            centroid,
            rolloff,
            zcr,
            low,
            mid,
            high,
            f0_mean,
            f0_std,
            harmonicity,
            bark_count,
            inter_bark_ms,
            repetition,
        ],
        dtype=np.float64,
    )
