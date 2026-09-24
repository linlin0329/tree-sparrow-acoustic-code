"""Validate released table relationships and exact example signals before plotting."""

from __future__ import annotations
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from .resources import ASSETS, COMPONENT


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_join(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Unsafe relative path",
    )
    candidate = (root / relative).resolve()
    require(candidate.is_relative_to(root), f"Path escapes data root: {relative}")
    return candidate


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check_assets():
    manifest = json.loads((COMPONENT / "assets_manifest.json").read_text())
    expected = set()
    for item in manifest["files"]:
        path = safe_join(ASSETS, item["path"])
        require(item["path"] not in expected, "Duplicate resource path")
        expected.add(item["path"])
        require(
            path.is_file()
            and path.stat().st_size == item["bytes"]
            and (sha256(path) == item["sha256"]),
            f"Changed or missing resource: {path}",
        )
    actual = {
        p.relative_to(ASSETS).as_posix() for p in ASSETS.rglob("*") if p.is_file()
    }
    require(actual == expected, "Unlisted or missing figure resources")
    return len(expected)


def check_tables(package):
    """Require the published 64,811/581/138/2,556 data contract and Raven joins."""
    names = [
        "raw_recordings",
        "song_recordings",
        "high_quality_clips",
        "syllables",
        "label_families",
        "label_leaves",
        "known_clock_errors",
    ]
    data = {name: rows(package / "metadata" / (name + ".csv")) for name in names}
    raw_ids = {r["raw_recording_id"] for r in data["raw_recordings"]}
    song_ids = {r["raw_recording_id"] for r in data["song_recordings"]}
    clips = {r["clip_id"]: r for r in data["high_quality_clips"]}
    core_ids = {r["raw_recording_id"] for r in clips.values()}
    syllables = {r["syllable_id"]: r for r in data["syllables"]}
    require(len(raw_ids) == len(data["raw_recordings"]) == 64811, "Raw ID count")
    require(len(song_ids) == len(data["song_recordings"]) == 581, "Song ID count")
    require(
        len(core_ids) == 116 and len(clips) == len(data["high_quality_clips"]) == 138,
        "Core count",
    )
    require(core_ids < song_ids < raw_ids, "Source/song/raw inclusion")
    require(len(syllables) == len(data["syllables"]) == 2556, "Syllable count")
    require(
        all(
            (
                Decimal(r["target_max_confidence"]) >= Decimal("0.7")
                for r in data["raw_recordings"]
            )
        ),
        "Raw release threshold",
    )
    flags = {r["raw_recording_id"] for r in data["known_clock_errors"]}
    require(
        len(flags) == len(data["known_clock_errors"]) == 2063
        and flags
        == {
            r["raw_recording_id"]
            for r in data["raw_recordings"]
            if r["time_status"] == "known_clock_error"
        },
        "Clock flags",
    )
    tally = Counter((r["clip_id"] for r in syllables.values()))
    require(set(tally) == set(clips), "Clip association")
    all_raven_ids = set()
    for clip_id, clip in clips.items():
        require(tally[clip_id] == int(clip["canonical_syllable_count"]), "Clip count")
        with safe_join(package, clip["raven_table_path"]).open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            table = list(csv.DictReader(stream, delimiter="\t"))
        ids = {r["syllable_id"] for r in table}
        require(len(table) == len(ids) == tally[clip_id], "Raven row count/uniqueness")
        require(
            ids == {sid for sid, row in syllables.items() if row["clip_id"] == clip_id},
            "Raven membership",
        )
        require(not ids & all_raven_ids, "Syllable repeated across Raven tables")
        all_raven_ids |= ids
        for row in table:
            require(
                all(
                    (
                        row[k] == syllables[row["syllable_id"]][k]
                        for k in ["family_id", "leaf_id", "source_raven_row"]
                    )
                ),
                "Raven label/source mismatch",
            )
    require(all_raven_ids == set(syllables), "Raven coverage")
    counts = {r["metric"]: int(r["count"]) for r in rows(ASSETS / "flow_counts.csv")}
    observed = dict(
        raw_admitted=len(raw_ids),
        clock_raw=len(flags),
        songs=len(song_ids),
        clips=len(clips),
        core_raw=len(core_ids),
        core_syllables=len(syllables),
        families=len(data["label_families"]),
        leaves=len(data["label_leaves"]),
        raven_tables=len(clips),
    )
    require(all((counts[k] == v for k, v in observed.items())), "Flow source counts")
    require(
        counts["logical_archive"] == 337010
        and counts["raw_candidates"] == 64844
        and (counts["raw_held"] == 33)
        and (counts["raw_candidates"] - counts["raw_held"] == len(raw_ids)),
        "Recording screening totals",
    )
    return (
        counts,
        {
            "raven_tables": len(clips),
            "raven_rows": len(all_raven_ids),
            "counts": counts,
        },
    )


def check_selected_examples(selected):
    """Check the ordered 27-family plate and the unchanged Figure 6 pair."""
    require(
        len(selected) == len({r["stable_id"] for r in selected}) == 29,
        "29 distinct examples",
    )
    expected = [
        (structure, f"{structure}_group_{number:02d}", f"{prefix}{number:02d}")
        for structure, prefix, count in [
            ("single", "S", 14), ("double", "D", 6), ("triple", "T", 7)
        ]
        for number in range(1, count + 1)
    ]
    plate = selected[:27]
    require(
        all(row["figure"] == "figure5" for row in plate)
        and [(row["structure"], row["family"], row["panel"]) for row in plate]
        == expected,
        "Figure 5 must contain all 27 families in S01-S14/D01-D06/T01-T07 order",
    )
    require(
        all(row["role"] == "main" and row["leaf"] == row["family"] for row in plate),
        "Figure 5 requires one main-form syllable per family",
    )
    pair = selected[27:]
    require(
        [(row["figure"], row["panel"], row["stable_id"], row["role"], row["leaf"])
         for row in pair]
        == [
            ("figure6", "a", "596382a001f28d14599a", "main", "single_group_08"),
            ("figure6", "b", "bfd4b15418a5db6c425a", "variant", "single_group_08_v1"),
        ]
        and all(row["family"] == "single_group_08" and row["structure"] == "single"
                for row in pair),
        "Figure 6 must retain its original main/variant pair",
    )


def check_spectra(package):
    """Reconstruct 29 examples and compare to supplied STFT arrays, rtol=0."""
    import numpy as np
    import soundfile as sf
    from .stft import calculate_spectrogram

    selected = rows(ASSETS / "selected_syllables.csv")
    metadata = {r["syllable_id"]: r for r in rows(package / "metadata/syllables.csv")}
    clips = {r["clip_id"]: r for r in rows(package / "metadata/high_quality_clips.csv")}
    spectra, parents, checks = ({}, {}, [])
    check_selected_examples(selected)
    require(
        {p.stem for p in (ASSETS / "spectra").glob("*.npz")}
        == {r["stable_id"] for r in selected},
        "Spectrogram assets must match the 29 selected examples exactly",
    )
    for row in selected:
        sid = row["stable_id"]
        meta = metadata[sid]
        clip = clips[meta["clip_id"]]
        require(
            all(
                (
                    row[a] == meta[b]
                    for a, b in [
                        ("family", "family_id"),
                        ("leaf", "leaf_id"),
                        ("structure", "structure"),
                        ("clip_id", "clip_id"),
                        ("source_frame_start", "source_frame_start"),
                        ("source_frame_stop", "source_frame_stop"),
                        ("sample_rate_hz", "sample_rate_hz"),
                        ("reference_wav_sha256", "reference_syllable_wav_sha256"),
                    ]
                )
            ),
            "Example identity/frame association",
        )
        require(
            (row["role"] == "main") == (meta["variant_code"] == ""),
            "Example main/variant association",
        )
        if meta["clip_id"] not in parents:
            path = safe_join(package, meta["analysis_clip_path"])
            require(sha256(path) == clip["sha256"], "Prepared WAV bytes changed")
            parents[meta["clip_id"]] = sf.read(path, dtype="float32", always_2d=True)
        samples, rate = parents[meta["clip_id"]]
        start, stop = (int(meta["source_frame_start"]), int(meta["source_frame_stop"]))
        require(
            0 <= start < stop <= len(samples)
            and rate == int(row["sample_rate_hz"])
            and (samples.shape[1] == 1),
            "Prepared WAV frame bounds/rate/channels",
        )
        sample = samples[start:stop]
        require(
            abs(float(row["duration_s"]) - len(sample) / rate) < 1e-12,
            "Example duration/frame association",
        )
        pcm = hashlib.sha256(
            np.ascontiguousarray(sample, dtype="<f4").tobytes()
        ).hexdigest()
        require(
            pcm == row["pcm_sha256"] == meta["decoded_pcm_f32le_sha256"],
            "Example decoded PCM",
        )
        calculated = calculate_spectrogram(sample[:, 0], rate)
        with np.load(
            safe_join(ASSETS / "spectra", sid + ".npz"), allow_pickle=False
        ) as data:
            reference = {key: data[key].copy() for key in data.files}
        matrix_hash = hashlib.sha256(
            np.ascontiguousarray(reference["db"]).tobytes()
        ).hexdigest()
        require(matrix_hash == row["stft_db_matrix_sha256"], "Original STFT hash")
        require(calculated["db"].shape == reference["db"].shape, "STFT dimensions")
        difference = float(np.max(np.abs(calculated["db"] - reference["db"])))
        require(
            np.allclose(calculated["db"], reference["db"], rtol=0, atol=1e-10),
            f"STFT differs by more than 1e-10 dB: {difference}",
        )
        require(
            np.array_equal(calculated["extent"], reference["extent"]),
            "STFT coordinate extent",
        )
        limit = 0.7 if row["figure"] == "figure5" else 0.4
        require(
            len(sample) / rate <= limit and calculated["extent"][1] <= limit,
            "Example cropped by display axis",
        )
        spectra[sid] = reference
        checks.append(
            {
                "syllable_id": sid,
                "figure": row["figure"],
                "panel": row["panel"],
                "family": row["family"],
                "clip_id": meta["clip_id"],
                "source_frame_start": start,
                "source_frame_stop": stop,
                "duration_s": len(sample) / rate,
                "display_limit_s": limit,
                "pcm_sha256": pcm,
                "stft_sha256": matrix_hash,
                "recomputed_max_absolute_difference_db": difference,
            }
        )
    return (selected, spectra, checks)
