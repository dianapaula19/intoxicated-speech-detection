"""MFCC feature extraction."""

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16_000
    n_mfcc: int = 13
    win_ms: float = 25.0
    hop_ms: float = 10.0
    n_frames: int = 300  # 3 s at a 10 ms hop
    deltas: bool = True

    @property
    def n_fft(self):
        return int(round(self.sample_rate * self.win_ms / 1000))

    @property
    def hop_length(self):
        return int(round(self.sample_rate * self.hop_ms / 1000))

    @property
    def n_features(self):
        return self.n_mfcc * (3 if self.deltas else 1)


def cmvn(features, eps=1e-8):
    """Per-coefficient mean and variance normalisation over time.

    ``features`` has shape (n_coefficients, n_frames). Normalising each row on
    its own removes channel effects without letting the large c0 (energy)
    coefficient dominate the others, which a single global mean/std would.
    """
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)
    return (features - mean) / (std + eps)


def pad_or_trim(features, n_frames):
    """Zero-pad or cut a (n_coefficients, n_frames) matrix to ``n_frames``."""
    current = features.shape[1]
    if current >= n_frames:
        return features[:, :n_frames]
    return np.pad(features, ((0, 0), (0, n_frames - current)), mode="constant")


def mfcc_from_signal(signal, config=FeatureConfig()):
    """MFCCs (+ deltas) for a mono signal already at ``config.sample_rate``."""
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    if config.deltas:
        width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 else mfcc.shape[1] - 1)
        if width >= 3:
            d1 = librosa.feature.delta(mfcc, width=width, order=1)
            d2 = librosa.feature.delta(mfcc, width=width, order=2)
        else:  # too short for deltas
            d1 = d2 = np.zeros_like(mfcc)
        mfcc = np.vstack([mfcc, d1, d2])
    # Normalise before padding so the zero padding means "average frame".
    return pad_or_trim(cmvn(mfcc), config.n_frames).astype(np.float32)


def mfcc_from_file(path, config=FeatureConfig()):
    signal, _ = librosa.load(path, sr=config.sample_rate, mono=True)
    return mfcc_from_signal(signal, config)
