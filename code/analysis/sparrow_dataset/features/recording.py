#!/usr/bin/env python3
"""歌级/bout 级声学特征提取（不依赖音节边界）。"""

import argparse
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

BAND_LOW, BAND_HIGH = 1500.0, 15000.0
SR = 48000
N_FFT = 1024
HOP = 256
ENERGY_GATE = 0.15


def _band_idx(sr, n_fft):
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (frequencies >= BAND_LOW) & (frequencies <= BAND_HIGH)
    return band, frequencies[band]


def extract(y, sr):
    out = {}
    out["dur_total_s"] = len(y) / sr
    if len(y) < N_FFT * 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spectrum = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    band_mask, band_frequencies = _band_idx(sr, N_FFT)
    band_spectrum = spectrum[band_mask, :]
    n_frames = band_spectrum.shape[1]
    energy = band_spectrum.sum(axis=0)
    if n_frames < 4 or energy.max() <= 0:
        return None
    threshold = ENERGY_GATE * np.percentile(energy, 95)
    active = energy > threshold
    if active.sum() < 3:
        active = energy > (0.1 * energy.max())
    if active.sum() < 3:
        return None
    active_spectrum = band_spectrum[:, active]
    active_energy = energy[active]
    n_active = active_spectrum.shape[1]
    out["dur_active_s"] = n_active * HOP / sr
    out["duty_cycle"] = active.mean()

    normalized = active_spectrum / (active_spectrum.sum(axis=0, keepdims=True) + 1e-12)
    centroid = (band_frequencies[:, None] * normalized).sum(axis=0)
    out["spec_centroid_mean"] = float(np.mean(centroid))
    out["spec_centroid_std"] = float(np.std(centroid))
    bandwidth = np.sqrt(
        (((band_frequencies[:, None] - centroid[None, :]) ** 2) * normalized).sum(
            axis=0
        )
    )
    out["spec_bandwidth_mean"] = float(np.mean(bandwidth))
    cdf = np.cumsum(normalized, axis=0)
    rolloff = band_frequencies[(cdf >= 0.85).argmax(axis=0)]
    out["spec_rolloff_mean"] = float(np.mean(rolloff))
    geometric_mean = np.exp(np.mean(np.log(active_spectrum + 1e-12), axis=0))
    arithmetic_mean = np.mean(active_spectrum, axis=0) + 1e-12
    out["spec_flatness_mean"] = float(np.mean(geometric_mean / arithmetic_mean))
    spectral_entropy = -np.sum(
        normalized * np.log(normalized + 1e-12), axis=0
    ) / np.log(len(band_frequencies))
    out["spec_entropy_mean"] = float(np.mean(spectral_entropy))
    psd = active_spectrum.mean(axis=1)
    probability = psd / psd.sum()
    cumulative = np.cumsum(probability)
    out["freq_q25_hz"] = float(band_frequencies[np.searchsorted(cumulative, 0.25)])
    out["freq_q50_hz"] = float(band_frequencies[np.searchsorted(cumulative, 0.50)])
    out["freq_q75_hz"] = float(
        band_frequencies[
            min(np.searchsorted(cumulative, 0.75), len(band_frequencies) - 1)
        ]
    )
    out["freq_iqr_hz"] = out["freq_q75_hz"] - out["freq_q25_hz"]

    flux = np.abs(np.diff(normalized, axis=1)).sum(axis=0)
    out["spec_flux_mean"] = float(np.mean(flux)) if flux.size else np.nan
    out["spec_flux_std"] = float(np.std(flux)) if flux.size else np.nan
    mean_frame = normalized.mean(axis=1, keepdims=True)
    numerator = (normalized * mean_frame).sum(axis=0)
    denominator = (
        np.linalg.norm(normalized, axis=0) * np.linalg.norm(mean_frame) + 1e-12
    )
    out["spec_frame_dispersion"] = float(np.mean(1.0 - numerator / denominator))
    energy_probability = active_energy / active_energy.sum()
    out["temporal_entropy"] = float(
        -np.sum(energy_probability * np.log(energy_probability + 1e-12))
        / np.log(len(energy_probability))
    )
    intensity_change = np.abs(np.diff(active_spectrum, axis=1)).sum(axis=1)
    intensity_sum = active_spectrum.sum(axis=1) + 1e-12
    out["aci"] = float(np.sum(intensity_change / intensity_sum))

    first = np.argmax(active)
    last = len(active) - np.argmax(active[::-1])
    envelope = energy[first:last].astype(float)
    if len(envelope) >= 8:
        envelope -= envelope.mean()
        modulation_power = (
            np.abs(np.fft.rfft(envelope * np.hanning(len(envelope)))) ** 2
        )
        modulation_frequency = np.fft.rfftfreq(len(envelope), d=HOP / sr)
        modulation_band = (modulation_frequency >= 1.0) & (modulation_frequency <= 25.0)
        if modulation_band.sum() >= 2 and modulation_power[modulation_band].sum() > 0:
            power = modulation_power[modulation_band]
            frequency = modulation_frequency[modulation_band]
            probability = power / power.sum()
            out["mod_peak_hz"] = float(frequency[np.argmax(power)])
            out["mod_centroid_hz"] = float((frequency * probability).sum())
            out["mod_entropy"] = float(
                -np.sum(probability * np.log(probability + 1e-12))
                / np.log(len(probability))
            )
    out.setdefault("mod_peak_hz", np.nan)
    out.setdefault("mod_centroid_hz", np.nan)
    out.setdefault("mod_entropy", np.nan)

    try:
        onset_envelope = librosa.onset.onset_strength(
            S=librosa.amplitude_to_db(np.sqrt(band_spectrum), ref=np.max),
            sr=sr,
            hop_length=HOP,
        )
        peaks = librosa.util.peak_pick(
            onset_envelope,
            pre_max=3,
            post_max=3,
            pre_avg=5,
            post_avg=5,
            delta=0.5,
            wait=int(0.04 * sr / HOP),
        )
        out["n_env_peaks"] = int(len(peaks))
        out["peak_rate"] = (
            float(len(peaks) / out["dur_active_s"])
            if out["dur_active_s"] > 0
            else np.nan
        )
    except Exception:
        out["n_env_peaks"] = np.nan
        out["peak_rate"] = np.nan
    return out


FEATURES = [
    "spec_centroid_mean",
    "spec_centroid_std",
    "spec_bandwidth_mean",
    "spec_rolloff_mean",
    "spec_flatness_mean",
    "spec_entropy_mean",
    "freq_q25_hz",
    "freq_q50_hz",
    "freq_q75_hz",
    "freq_iqr_hz",
    "spec_flux_mean",
    "spec_flux_std",
    "spec_frame_dispersion",
    "temporal_entropy",
    "aci",
    "mod_peak_hz",
    "mod_centroid_hz",
    "mod_entropy",
    "n_env_peaks",
    "peak_rate",
    "dur_total_s",
    "dur_active_s",
    "duty_cycle",
]
CUT_SENSITIVE = {
    "dur_total_s",
    "duty_cycle",
    "dur_active_s",
    "n_env_peaks",
}


def aggregate_recording(group):
    weights = group.dur_active_s.clip(lower=1e-6)
    data = {}
    for feature in FEATURES:
        if feature == "n_env_peaks":
            data[feature] = group[feature].sum()
        elif feature in ("dur_total_s", "dur_active_s"):
            data[feature] = group[feature].sum()
        else:
            values = group[feature]
            finite = np.isfinite(values)
            data[feature] = (
                float(np.average(values[finite], weights=weights[finite]))
                if finite.any()
                else np.nan
            )
    data["n_clip"] = len(group)
    data["site"] = group.site.iloc[0]
    data["year"] = group.year.iloc[0]
    data["kind"] = (
        "manual"
        if (group.kind == "manual").all()
        else "pipeline" if (group.kind == "pipeline").all() else "mixed"
    )
    return pd.Series(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--song-dir", type=Path, required=True)
    ap.add_argument(
        "--song-list",
        type=Path,
        default=None,
        help="默认使用 SONG_DIR/song_final_list.csv",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    song_list = args.song_list or args.song_dir / "song_final_list.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    listing = pd.read_csv(song_list)

    rows = []
    for _, row in listing.iterrows():
        audio_path = args.song_dir / "audio" / row.file
        try:
            y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
        except Exception as exc:
            print("load fail", row.file, exc)
            continue
        features = extract(y, SR)
        if features is None:
            print("skip (too short/quiet):", row.file)
            continue
        features.update(
            file=row.file,
            kind=row.kind,
            base_recording=row.base_recording,
            site=row.site,
            year=row.year,
        )
        rows.append(features)
    clips = pd.DataFrame(rows)
    clips.to_csv(args.output_dir / "song_features_clip.csv", index=False)
    print(
        f"clip 级特征: {len(clips)} -> " f"{args.output_dir / 'song_features_clip.csv'}"
    )

    recordings = (
        clips.groupby("base_recording")
        .apply(aggregate_recording, include_groups=False)
        .reset_index()
    )
    site_map = {
        "liujiaxia": "none",
        "yongxing": "none",
        "liangzhuang": "low",
        "shuanghe": "high",
        "minqin": "high",
        "guanyinya": "high",
    }
    recordings["level"] = recordings.site.map(site_map)
    recordings["rank"] = recordings.level.map({"none": 0, "low": 1, "high": 2})
    recordings.to_csv(args.output_dir / "song_features_recording.csv", index=False)
    print(
        f"录音级特征: {len(recordings)} -> "
        f"{args.output_dir / 'song_features_recording.csv'}"
    )
    print("按 level:", recordings.level.value_counts().to_dict())


if __name__ == "__main__":
    main()
