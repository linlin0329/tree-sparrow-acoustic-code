"""Approved manuscript drawing operations with explicit local dependencies."""

from __future__ import annotations
import numpy as np
from scipy.signal.windows import hann

WIN_LENGTH, HOP_LENGTH, N_FFT = (512, 96, 1024)
FREQUENCY_MAX_HZ, DB_MIN = (16000, -70.0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def calculate_spectrogram(samples: np.ndarray, sample_rate: int) -> dict:
    """不中心填充、不末端补齐；FFT 零插值至1024不是音频端点padding。"""
    y = np.asarray(samples)
    require(
        y.ndim == 1 and len(y) >= WIN_LENGTH and np.isfinite(y).all(),
        "音频必须为有限值单声道且不少于512帧",
    )
    require(sample_rate >= 2 * FREQUENCY_MAX_HZ, "采样率不足以显示0–16kHz")
    frames = np.lib.stride_tricks.sliding_window_view(y, WIN_LENGTH)[::HOP_LENGTH]
    spectrum = np.fft.rfft(frames * hann(WIN_LENGTH, sym=False), n=N_FFT, axis=1)
    frequencies = np.fft.rfftfreq(N_FFT, d=1 / sample_rate)
    mask = frequencies <= FREQUENCY_MAX_HZ
    power = np.abs(spectrum[:, mask]).T ** 2
    peak = float(power.max())
    db = np.full(power.shape, DB_MIN, dtype=np.float64)
    if peak > 0:
        db = np.clip(
            10 * np.log10(np.maximum(power / peak, np.finfo(float).tiny)), DB_MIN, 0
        )
    centers = (np.arange(len(frames)) * HOP_LENGTH + WIN_LENGTH / 2) / sample_rate
    df = sample_rate / N_FFT
    extent = [
        float(centers[0] - HOP_LENGTH / sample_rate / 2),
        float(centers[-1] + HOP_LENGTH / sample_rate / 2),
        float(-df / 2 / 1000),
        float((frequencies[mask][-1] + df / 2) / 1000),
    ]
    support_stop = (len(frames) - 1) * HOP_LENGTH + WIN_LENGTH
    return {
        "db": db,
        "extent": extent,
        "frequencies_hz": frequencies[mask],
        "frame_centers_s": centers,
        "stft_frame_count": len(frames),
        "stft_last_window_stop_frame": support_stop,
        "stft_uncovered_tail_frames": len(y) - support_stop,
        "stft_uncovered_tail_s": (len(y) - support_stop) / sample_rate,
        "stft_peak_display_power": peak,
    }
