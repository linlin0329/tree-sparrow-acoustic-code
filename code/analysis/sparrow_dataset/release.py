"""Validate the released seven-table schema and regenerate labeled Raven tables."""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from .io import digest, indexed, read_rows, require, safe_path, write_rows

TABLE_COUNTS = {
    "raw_recordings": 64811,
    "song_recordings": 581,
    "high_quality_clips": 138,
    "syllables": 2556,
    "label_families": 27,
    "label_leaves": 41,
    "known_clock_errors": 2063,
}
RAVEN_MAPPING = {
    "Selection": "raven_selection",
    "Channel": "annotation_channel",
    "Begin Time (s)": "begin_s",
    "End Time (s)": "end_s",
    "Low Freq (Hz)": "low_freq_hz",
    "High Freq (Hz)": "high_freq_hz",
    "syllable_id": "syllable_id",
    "family_id": "family_id",
    "leaf_id": "leaf_id",
    "structure": "structure",
    "original_selection_id": "original_selection_id",
    "source_raven_row": "source_raven_row",
    "original_view": "original_view",
}
NUMERIC_RAVEN_FIELDS = {
    "Selection",
    "Channel",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "original_selection_id",
    "source_raven_row",
}


def tables(root: Path) -> dict[str, list[dict[str, str]]]:
    return {name: read_rows(root / "metadata" / f"{name}.csv") for name in TABLE_COUNTS}


def inspect_package(root: Path) -> dict:
    data = tables(root)
    for name, expected in TABLE_COUNTS.items():
        require(len(data[name]) == expected, f"Unexpected {name} row count")
    raw = indexed(data["raw_recordings"], "raw_recording_id")
    songs = indexed(data["song_recordings"], "raw_recording_id")
    indexed(data["song_recordings"], "song_recording_id")
    clips = indexed(data["high_quality_clips"], "clip_id")
    syllables = indexed(data["syllables"], "syllable_id")
    families = indexed(data["label_families"], "family_id")
    leaves = indexed(data["label_leaves"], "leaf_id")
    clocks = indexed(data["known_clock_errors"], "raw_recording_id")
    require(set(songs) <= set(raw), "Song/raw inclusion link is broken")
    source_ids = {row["raw_recording_id"] for row in clips.values()}
    require(
        len(source_ids) == 116 and source_ids <= set(songs),
        "Core/song source links changed",
    )
    require(
        set(clocks)
        == {
            key for key, row in raw.items() if row["time_status"] == "known_clock_error"
        },
        "Clock index differs from raw flags",
    )
    for row in raw.values():
        require(
            row["score_comparison"] == "ge"
            and Decimal(row["score_threshold"]) == Decimal("0.7")
            and Decimal(row["target_max_confidence"]) >= Decimal("0.7"),
            "Raw score gate changed",
        )
        require(
            row["strict_decode_pass"] == "true" and row["corrected_recorded_at"] == "",
            "Raw QC/time contract changed",
        )
        for field in ("audio_path", "detection_csv_path"):
            safe_path(root, row[field])
    for row in songs.values():
        parent = raw[row["raw_recording_id"]]
        for field in ("source_recording_key", "site_id", "audio_path", "time_status"):
            require(row[field] == parent[field], f"Song/raw {field} mismatch")
    for key, row in clocks.items():
        require(
            all(value == raw[key][field] for field, value in row.items()),
            "Clock row/raw mismatch",
        )
    for row in leaves.values():
        require(row["family_id"] in families, "Unknown leaf family")
    counts = Counter()
    for clip_id, clip in clips.items():
        require(
            safe_path(root, clip["analysis_wav_path"]).is_file(), "Missing prepared WAV"
        )
        require(
            clip["raw_start_s"] == clip["raw_end_s"] == clip["release_offset_s"] == "",
            "Unverified raw offsets have been populated",
        )
        annotations = read_rows(safe_path(root, clip["raven_table_path"]), "\t")
        require(
            len(annotations) == int(clip["canonical_syllable_count"]),
            "Clip syllable count differs",
        )
        for annotation in annotations:
            sid = annotation["syllable_id"]
            require(sid in syllables, "Unknown Raven syllable ID")
            row = syllables[sid]
            counts[sid] += 1
            require(
                row["clip_id"] == clip_id
                and row["raw_recording_id"] == clip["raw_recording_id"],
                "Raven source links differ",
            )
            require(
                row["analysis_clip_path"] == clip["analysis_wav_path"]
                and row["raven_table_path"] == clip["raven_table_path"],
                "Syllable paths differ",
            )
            require(
                row["leaf_id"] in leaves
                and leaves[row["leaf_id"]]["family_id"] == row["family_id"],
                "Syllable label hierarchy differs",
            )
            require(
                annotation["View"] == "Spectrogram 1", "Unexpected Raven output view"
            )
            for field, column in RAVEN_MAPPING.items():
                left, right = annotation[field], row[column]
                require(
                    (
                        Decimal(left) == Decimal(right)
                        if field in NUMERIC_RAVEN_FIELDS
                        else left == right
                    ),
                    f"Raven {field} differs: {sid}",
                )
            start, stop = int(row["source_frame_start"]), int(row["source_frame_stop"])
            require(
                0 <= start < stop <= int(clip["local_stop_frame"])
                and stop - start == int(row["decoded_frames"]),
                "Invalid integer frame bounds",
            )
            require(
                Decimal(0)
                <= Decimal(row["begin_s"])
                < Decimal(row["end_s"])
                <= Decimal(clip["local_end_s"]),
                "Invalid time bounds",
            )
            require(
                Decimal(0)
                <= Decimal(row["low_freq_hz"])
                < Decimal(row["high_freq_hz"])
                <= Decimal(clip["sample_rate_hz"]) / 2,
                "Invalid frequency bounds",
            )
    require(
        set(counts) == set(syllables) and all(count == 1 for count in counts.values()),
        "Raven coverage is not one-to-one",
    )
    for field, dictionary in (("family_id", families), ("leaf_id", leaves)):
        actual = Counter(row[field] for row in syllables.values())
        require(
            actual
            == {key: int(row["syllable_count"]) for key, row in dictionary.items()},
            f"{field} counts differ",
        )
    return {
        "status": "passed",
        "counts": {key: len(value) for key, value in data.items()},
        "clip_source_recordings": len(source_ids),
        "raven_rows": sum(counts.values()),
        "coverage": "Seven tables, IDs, score policy, clock flags, source links and all Raven fields; audio hashes and PCM require separate commands",
    }


def check_hashes(root: Path) -> dict:
    checked = set()
    for line in (root / "manifest.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        require(relative not in checked, "Duplicate manifest entry")
        require(
            digest(safe_path(root, relative)) == expected, f"Hash mismatch: {relative}"
        )
        checked.add(relative)
    return {"status": "passed", "files_checked": len(checked)}


def export_raven(root: Path, output: Path) -> dict:
    """Regenerate public TSVs, retaining original row identity and display precision."""
    require(not output.exists(), f"Output must be new: {output}")
    inspect_package(root)
    grouped = defaultdict(list)
    for row in read_rows(root / "metadata/syllables.csv"):
        grouped[row["raven_table_path"]].append(row)
    # Existing public TSVs provide a byte-level regression target, not the data source.
    output.mkdir(parents=True)
    for relative, rows in grouped.items():
        rendered = []
        for row in sorted(rows, key=lambda item: int(item["source_raven_row"])):
            annotation = {"Selection": row["raven_selection"], "View": "Spectrogram 1"}
            annotation.update(
                {
                    key: row[value]
                    for key, value in RAVEN_MAPPING.items()
                    if key != "Selection"
                }
            )
            rendered.append(annotation)
        target = safe_path(output, relative)
        write_rows(target, rendered, delimiter="\t")
        require(
            target.read_bytes() == safe_path(root, relative).read_bytes(),
            f"Regenerated Raven bytes differ: {relative}",
        )
    return {
        "status": "passed",
        "tables_exported": len(grouped),
        "byte_identical_to_release": True,
    }
