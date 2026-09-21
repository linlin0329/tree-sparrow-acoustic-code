"""Acoustic measurement functions; exposure grouping is outside this package."""

from __future__ import annotations
import warnings
import numpy as np

BAND_LOW = 1500.0


BAND_HIGH = 15000.0


F0_FMIN = BAND_LOW


F0_FMAX = 12000.0


PARAM_COLUMNS = [
    "duration_ms",
    "bandwidth_box_hz",
    "low_freq_hz",
    "high_freq_hz",
    "peak_freq_hz",
    "spec_centroid_hz",
    "spec_bandwidth_hz",
    "spec_rolloff_hz",
    "spec_flatness",
    "spec_entropy",
    "freq_q25_hz",
    "freq_q50_hz",
    "freq_q75_hz",
    "freq_iqr_hz",
    "f0_mean_hz",
    "f0_std_hz",
    "f0_min_hz",
    "f0_max_hz",
    "f0_range_hz",
    "fm_std_hz",
    "am_cv",
    "rms",  # 距离混淆, 仅记录
]


DISTANCE_CONFOUNDED = {"rms"}


def _safe(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def extract_params(
    y: np.ndarray, sr: int, low_freq=None, high_freq=None, with_f0: bool = True
) -> dict:
    """对单个音节波形 y 计算可解释声学参数。返回参数 dict。"""
    import librosa

    out = {k: np.nan for k in PARAM_COLUMNS}
    out["duration_ms"] = 1000.0 * len(y) / sr
    out["low_freq_hz"] = _safe(low_freq)
    out["high_freq_hz"] = _safe(high_freq)
    if np.isfinite(out["low_freq_hz"]) and np.isfinite(out["high_freq_hz"]):
        out["bandwidth_box_hz"] = out["high_freq_hz"] - out["low_freq_hz"]

    if len(y) < 64:
        return out

    n_fft = 1024 if len(y) >= 1024 else 512
    hop = max(64, n_fft // 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2  # 功率谱 (F,T)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # 限制到分析频带
        band = (freqs >= BAND_LOW) & (freqs <= BAND_HIGH)
        Sb = S[band, :]
        fb = freqs[band]
        psd = Sb.mean(axis=1)  # 平均功率谱
        psum = psd.sum()

        if psum > 0 and len(fb) > 1:
            p = psd / psum
            out["peak_freq_hz"] = float(fb[np.argmax(psd)])
            # 谱熵 (Shannon, 归一化到 0-1)
            ent = -np.sum(p[p > 0] * np.log(p[p > 0]))
            out["spec_entropy"] = float(ent / np.log(len(fb)))
            # 累积功率分位频率
            cdf = np.cumsum(p)
            out["freq_q25_hz"] = float(fb[np.searchsorted(cdf, 0.25)])
            out["freq_q50_hz"] = float(fb[np.searchsorted(cdf, 0.50)])
            out["freq_q75_hz"] = float(fb[min(np.searchsorted(cdf, 0.75), len(fb) - 1)])
            out["freq_iqr_hz"] = out["freq_q75_hz"] - out["freq_q25_hz"]

        cent = librosa.feature.spectral_centroid(S=Sb, sr=sr, freq=fb)[0]
        out["spec_centroid_hz"] = float(np.nanmean(cent))
        out["fm_std_hz"] = float(np.nanstd(cent))  # 调频: 质心轮廓波动
        out["spec_bandwidth_hz"] = float(
            np.nanmean(librosa.feature.spectral_bandwidth(S=Sb, sr=sr, freq=fb)[0])
        )
        out["spec_rolloff_hz"] = float(
            np.nanmean(
                librosa.feature.spectral_rolloff(
                    S=Sb,
                    sr=sr,
                    freq=fb,
                    roll_percent=0.85,
                )[0]
            )
        )
        out["spec_flatness"] = float(
            np.nanmean(librosa.feature.spectral_flatness(S=np.sqrt(Sb))[0])
        )

        # 幅度包络 (RMS): 绝对值受距离混淆(仅记录), CV 为录音内相对量
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
        out["rms"] = float(np.nanmean(rms))
        m = np.nanmean(rms)
        out["am_cv"] = float(np.nanstd(rms) / m) if m > 0 else np.nan

        if with_f0:
            try:
                f0, vflag, _ = librosa.pyin(
                    y, fmin=F0_FMIN, fmax=F0_FMAX, sr=sr, frame_length=n_fft
                )
                f0v = f0[np.isfinite(f0)]
                if f0v.size:
                    out["f0_mean_hz"] = float(np.mean(f0v))
                    out["f0_std_hz"] = float(np.std(f0v))
                    out["f0_min_hz"] = float(np.min(f0v))
                    out["f0_max_hz"] = float(np.max(f0v))
                    out["f0_range_hz"] = out["f0_max_hz"] - out["f0_min_hz"]
            except Exception:
                pass
    return out
