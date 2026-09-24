"""Measure spectral differences in the fixed WAV/MP3 fixture pairs without encoding new MP3s."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy import signal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def decode(p):
    r = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(p),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(r.stdout, dtype="<f4").astype(float)


def run(fixtures: Path, out: Path):
    manifest = json.loads((fixtures / "manifest.json").read_text(encoding="utf-8"))
    bands = [
        (1000, 5000),
        (5000, 10000),
        (10000, 15000),
        (1000, 15000),
        (15000, 20000),
        (20000, 24000),
    ]
    rows = []
    saved = {}
    for m in manifest:
        name = m["id"]
        x = decode(fixtures / (name + ".wav"))
        y = decode(fixtures / (name + ".mp3"))
        z = signal.correlate(y[:144000], x[:144000], method="fft")
        lags = signal.correlation_lags(min(len(y), 144000), min(len(x), 144000))
        mask = np.abs(lags) < 2400
        lag = int(lags[mask][np.argmax(z[mask])])
        if lag > 0:
            y = y[lag:]
        elif lag < 0:
            x = x[-lag:]
        n = min(len(x), len(y))
        x = x[2400 : n - 2400]
        y = y[2400 : n - 2400]
        f, px = signal.welch(x, 48000, nperseg=4096)
        _, py = signal.welch(y, 48000, nperseg=4096)
        _, pe = signal.welch(y - x, 48000, nperseg=4096)
        result = {
            "id": name,
            "lag_samples": lag,
            "samples_compared": len(x),
            "original_samples": len(decode(fixtures / (name + ".wav"))),
            "decoded_samples": len(decode(fixtures / (name + ".mp3"))),
            "waveform_correlation": float(np.corrcoef(x, y)[0, 1]),
            "bands": {},
        }
        for lo, hi in bands:
            b = (f >= lo) & (f < hi)
            a = px[b].sum()
            d = py[b].sum()
            e = pe[b].sum()
            result["bands"][f"{lo}-{hi}"] = {
                "power_change_db": float(10 * np.log10(d / a)),
                "signal_to_error_db": float(10 * np.log10(a / e)),
                "relative_error_rms_pct": float(100 * np.sqrt(e / a)),
            }
        b = (f >= 1000) & (f <= 15000)
        fb = f[b]

        def measures(p):
            q = p[b]
            cdf = np.cumsum(q) / q.sum()
            return {
                "peak_hz": float(fb[q.argmax()]),
                "centroid_hz": float((q * fb).sum() / q.sum()),
                "bandwidth90_hz": float(
                    np.interp(0.95, cdf, fb) - np.interp(0.05, cdf, fb)
                ),
            }

        result["whole_clip_spectrum_wav"] = measures(px)
        result["whole_clip_spectrum_mp3"] = measures(py)
        fs, ts, sx = signal.spectrogram(
            x, 48000, nperseg=1024, noverlap=768, mode="psd"
        )
        _, _, sy = signal.spectrogram(y, 48000, nperseg=1024, noverlap=768, mode="psd")
        bf = (fs >= 1000) & (fs <= 15000)
        en = sx[bf].sum(axis=0)
        active = en >= np.max(en) * 0.1
        rx = fs[bf][sx[bf].argmax(axis=0)]
        ry = fs[bf][sy[bf].argmax(axis=0)]
        rd = np.abs(ry - rx)[active]
        result["technical_frame_peak_difference_hz"] = {
            "frames": int(active.sum()),
            "median": float(np.median(rd)),
            "p95": float(np.percentile(rd, 95)),
            "max": float(rd.max()),
            "fraction_over_100hz": float(np.mean(rd > 100)),
        }
        rows.append(result)
        saved[name] = (x, y, f, px, py, pe)
    (out / "measurements.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    real = [r for r in rows if r["id"] != "noise"]
    summary = {}
    for band in ["1000-5000", "5000-10000", "10000-15000", "1000-15000"]:
        summary[band] = {}
        for key in ["power_change_db", "signal_to_error_db", "relative_error_rms_pct"]:
            v = [r["bands"][band][key] for r in real]
            summary[band][key] = {
                "min": min(v),
                "median": float(np.median(v)),
                "max": max(v),
            }
    summary["lag_samples"] = [r["lag_samples"] for r in real]
    summary["whole_clip_spectral_differences"] = {
        k: [
            r["whole_clip_spectrum_mp3"][k] - r["whole_clip_spectrum_wav"][k]
            for r in real
        ]
        for k in ["peak_hz", "centroid_hz", "bandwidth90_hz"]
    }
    summary["frame_peaks"] = [r["technical_frame_peak_difference_hz"] for r in real]
    summary["noise_bands"] = next(r["bands"] for r in rows if r["id"] == "noise")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    x, y, f, px, py, pe = saved["noise"]
    axs[0].plot(
        f / 1000,
        10 * np.log10(np.maximum(py, 1e-30) / np.maximum(px, 1e-30)),
        color="#167f82",
        lw=1,
    )
    axs[0].axvspan(
        1, 15, color="#dfeedd", alpha=0.7, label="User target band: 1-15 kHz"
    )
    axs[0].set(
        xlim=(0, 24),
        ylim=(-65, 3),
        xlabel="Frequency (kHz)",
        ylabel="MP3 / WAV power (dB)",
        title="Same Pi encoder, 320 kbit/s mono: white-noise diagnostic",
    )
    axs[0].legend()
    axs[0].grid(alpha=0.2)
    for b, c in zip(
        ["1000-5000", "5000-10000", "10000-15000"], ["#167f82", "#dd9639", "#ad557d"]
    ):
        axs[1].plot(
            range(1, len(real) + 1),
            [r["bands"][b]["signal_to_error_db"] for r in real],
            "o-",
            label=b + " Hz",
            color=c,
        )
    axs[1].set(
        xlabel="Reference WAV sample ID",
        ylabel="Signal-to-compression-error ratio (dB)",
        title=f"{len(real)} fixture clips: higher values mean smaller waveform error",
    )
    axs[1].legend(ncol=3)
    axs[1].grid(alpha=0.2)
    fig.savefig(out / "compression_diagnostic.png", dpi=160)
    plt.close(fig)
    worst = max(
        real,
        key=lambda r: r["technical_frame_peak_difference_hz"]["fraction_over_100hz"],
    )["id"]
    x, y, *_ = saved[worst]
    fig, axs = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True, sharey=True, layout="constrained"
    )
    f, t, z = signal.spectrogram(x, 48000, nperseg=1024, noverlap=768)
    vmax = 10 * np.log10(z.max())
    vmin = vmax - 65
    for ax, a, title in zip(
        axs,
        [x, y, y - x],
        [
            "Original WAV",
            "Decoded 320 kbit/s MP3",
            "Difference (MP3 - WAV), same colour scale",
        ],
    ):
        f, t, z = signal.spectrogram(a, 48000, nperseg=1024, noverlap=768)
        im = ax.pcolormesh(
            t,
            f / 1000,
            10 * np.log10(z + 1e-30),
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            shading="auto",
            rasterized=True,
        )
        ax.set(ylim=(1, 15), ylabel="kHz", title=title)
    axs[-1].set_xlabel("Time (s), with 50 ms edges removed")
    fig.suptitle(f"{worst}: technical example; call boundaries have not been annotated")
    fig.colorbar(im, ax=axs, label="PSD (dB re full scale squared / Hz)")
    fig.savefig(out / "spectrogram_comparison.png", dpi=160)
    plt.close(fig)


def fixture_contract() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "assets/mp3_assessment/fixture_contract.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def validate_fixtures(fixtures: Path) -> dict:
    """Check immutable fixture/reference bytes before any spectral computation."""
    import soundfile as sf
    from ..io import digest, require, safe_path

    contract = fixture_contract()
    for record in contract["files"]:
        path = safe_path(fixtures, record["name"])
        require(path.is_file(), f"Missing MP3 fixture/reference: {record['name']}")
        require(
            path.stat().st_size == record["size_bytes"]
            and digest(path) == record["sha256"],
            f"MP3 fixture/reference identity changed: {record['name']}",
        )
    manifest = json.loads((fixtures / "manifest.json").read_text(encoding="utf-8"))
    require(
        [item["id"] for item in manifest] == contract["fixture_ids"],
        "MP3 fixture order/IDs differ",
    )
    for item in manifest:
        for extension in ("wav", "mp3"):
            path = safe_path(fixtures, item["id"] + "." + extension)
            require(
                path.stat().st_size == item[extension + "_bytes"]
                and digest(path) == item[extension + "_sha256"],
                "Original encoder manifest identity mismatch",
            )
        info = sf.info(safe_path(fixtures, item["id"] + ".wav"))
        require(
            info.samplerate == contract["sample_rate_hz"]
            and info.channels == contract["channels"],
            "MP3 assessment requires native 48 kHz mono inputs",
        )
        require(
            info.frames > 4800,
            "Fixture is too short for the 2400-frame edge removal",
        )
    return {
        "status": "passed",
        "fixture_pairs": len(manifest),
        "files_checked": len(contract["files"]),
        "original_encoder_manifest_hashes_matched": 2 * len(manifest),
        "reencoded": False,
    }


def compare_reference(actual, expected, *, rtol=1e-10, atol=1e-8) -> dict:
    """Compare every field; tolerate floating-point roundoff only."""
    import math
    from ..io import require

    report = {
        "numeric_values": 0,
        "max_absolute_difference": 0.0,
        "relative_tolerance": rtol,
        "absolute_tolerance": atol,
    }

    def compare(left, right, path):
        if isinstance(right, dict):
            require(
                isinstance(left, dict) and left.keys() == right.keys(),
                f"Reference fields differ at {path}",
            )
            for key in right:
                compare(left[key], right[key], f"{path}.{key}")
        elif isinstance(right, list):
            require(
                isinstance(left, list) and len(left) == len(right),
                f"Reference array differs at {path}",
            )
            for index, (value, reference) in enumerate(zip(left, right)):
                compare(value, reference, f"{path}[{index}]")
        elif isinstance(right, float):
            require(
                isinstance(left, (int, float)) and not isinstance(left, bool),
                f"Reference numeric type differs at {path}",
            )
            require(
                (math.isnan(left) and math.isnan(right))
                or math.isclose(left, right, rel_tol=rtol, abs_tol=atol),
                f"Matched numeric reference differs at {path}: {left!r} vs {right!r}",
            )
            report["numeric_values"] += 1
            if math.isfinite(left) and math.isfinite(right):
                report["max_absolute_difference"] = max(
                    report["max_absolute_difference"], abs(left - right)
                )
        else:
            require(
                left == right,
                f"Matched reference differs at {path}: {left!r} vs {right!r}",
            )
            if isinstance(right, int) and not isinstance(right, bool):
                report["numeric_values"] += 1

    compare(actual, expected, "root")
    return dict(status="passed", **report)


def replay(fixtures: Path, output: Path, *, dry_run: bool = False) -> dict:
    """Read matched external fixtures; write all results to a new directory."""
    if output.exists():
        raise FileExistsError(output)
    identity = validate_fixtures(fixtures)
    if dry_run:
        return {"status": "planned", "identity": identity, "output_created": False}
    output.mkdir(parents=True)
    run(fixtures, output)
    contract = fixture_contract()
    comparisons = {}
    for name in ("measurements.json", "summary.json"):
        comparisons[name] = compare_reference(
            json.loads((output / name).read_text()),
            json.loads((fixtures / name).read_text()),
            rtol=contract["reference_relative_tolerance"],
            atol=contract["reference_absolute_tolerance"],
        )
    # Recheck source identities after reading so this is also an immutability receipt.
    identity_after = validate_fixtures(fixtures)
    report = {
        "status": "passed",
        "identity_before": identity,
        "identity_after": identity_after,
        "reference_comparison": comparisons,
        "source_inputs_unchanged": True,
        "reencoded": False,
        "output_directory_is_new": True,
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify fixed fixture/reference hashes without computing spectra",
    )
    args = parser.parse_args()
    report = replay(
        args.fixtures.resolve(), args.output_dir.resolve(), dry_run=args.dry_run
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
