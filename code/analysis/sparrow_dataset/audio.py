"""Sample-exact reconstruction from published integer frame coordinates."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .io import indexed, read_rows, require, safe_path


def syllable_samples(root: Path, syllable_id: str | None = None):
    import numpy as np
    import soundfile as sf

    rows = read_rows(root / "metadata/syllables.csv")
    indexed(rows, "syllable_id")
    selected = [
        row for row in rows if syllable_id is None or row["syllable_id"] == syllable_id
    ]
    require(bool(selected), "No matching syllable ID")
    previous, samples, rate = None, None, None
    for row in selected:
        path = safe_path(root, row["analysis_clip_path"])
        if path != previous:
            samples, rate = sf.read(path, dtype="float32", always_2d=True)
            previous = path
        start, stop = int(row["source_frame_start"]), int(row["source_frame_stop"])
        require(0 <= start < stop <= len(samples), "Frames outside parent")
        cut = samples[start:stop]
        require(
            rate == int(row["sample_rate_hz"])
            and cut.shape == (int(row["decoded_frames"]), int(row["channels"])),
            "PCM shape/rate mismatch",
        )
        require(np.isfinite(cut).all(), "Nonfinite PCM samples")
        actual = hashlib.sha256(
            np.asarray(cut, dtype="<f4").tobytes(order="C")
        ).hexdigest()
        require(
            actual == row["decoded_pcm_f32le_sha256"],
            f"PCM hash mismatch: {row['syllable_id']}",
        )
        yield row, cut, rate


def reconstruct(
    root: Path, syllable_id: str | None = None, output: Path | None = None
) -> dict:
    import soundfile as sf

    if output is not None:
        require(not output.exists(), f"Output must be new: {output}")
        output.mkdir(parents=True)
    checked = 0
    for row, cut, rate in syllable_samples(root, syllable_id):
        if output is not None:
            target = safe_path(output, row["syllable_id"] + ".wav")
            with target.open("xb") as stream:
                sf.write(stream, cut, rate, subtype="FLOAT", format="WAV")
        checked += 1
    return {
        "status": "passed",
        "checked_syllables": checked,
        "all_decoded_samples_exact": True,
        "exported": output is not None,
    }


def extract_features(
    root: Path, output: Path, kind: str, syllable_id: str | None = None
) -> dict:
    import json
    import numpy as np
    from .io import write_rows

    require(not output.exists(), f"Output must be new: {output}")
    output.mkdir(parents=True)
    identities, values = [], []
    if kind == "105d":
        from .features.audio_processor import AudioProcessor
        from .features.config import Config

        config = Config()
        config.feature.feature_types = [
            "mfcc",
            "mfcc_delta",
            "spectral",
            "temporal",
            "amplitude",
        ]
        processor = AudioProcessor(config)
        names = processor.get_feature_names()
        require(len(names) == 105, "Historical feature-name contract changed")
    elif kind == "logmel":
        from .features.logmel import log_mel_patch

        names = [f"mel_{mel}_time_{time}" for mel in range(64) for time in range(32)]
    else:
        from .features.acoustic import extract_params, PARAM_COLUMNS

        names = PARAM_COLUMNS
    for row, cut, rate in syllable_samples(root, syllable_id):
        require(
            cut.shape[1] == 1, "Feature extraction requires the released mono parents"
        )
        waveform = cut[:, 0]
        if kind == "105d":
            values.append(processor.extract_features(waveform, rate))
        elif kind == "logmel":
            values.append(log_mel_patch(waveform, rate).ravel())
        else:
            result = extract_params(
                waveform, rate, row["low_freq_hz"], row["high_freq_hz"]
            )
            values.append([result[name] for name in names])
        identities.append(
            {
                key: row[key]
                for key in (
                    "syllable_id",
                    "clip_id",
                    "data_version",
                    "source_frame_start",
                    "source_frame_stop",
                )
            }
        )
    np.save(output / "features.npy", np.asarray(values))
    write_rows(output / "feature_rows.csv", identities)
    (output / "feature_names.json").write_text(json.dumps(names, indent=2) + "\n")
    return {
        "status": "complete",
        "syllables": len(identities),
        "features": len(names),
        "boundary_basis": "Released integer frames; assessment input bounds are linked separately in data/validation/label_assessment/assessment_input_bounds.csv",
    }
