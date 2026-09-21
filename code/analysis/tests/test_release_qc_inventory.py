"""发布质控准备器：阈值/未知、坏表、逻辑键和执行框的回归测试。"""

import argparse
import csv
import gzip
import importlib.util
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


from sparrow_dataset.processing import prepare_release_qc as qc

HEADER = "Start (s);End (s);Scientific name;Common name;Confidence\n"


class DetectionTests(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "d.wav.csv"
            path.write_bytes(content.encode("utf-8-sig"))
            return qc.parse_detection_csv(path, "d.wav.csv", Decimal("0.5"))

    def test_exact_threshold_and_decimal_not_float(self):
        result = self.parse(
            HEADER + "0;3;Passer montanus;麻雀;0.50000000000000000000001\n"
        )
        self.assertEqual(
            qc.scope_for(
                result["csv_status"], result["target_max_confidence"], Decimal("0.5")
            )[0],
            qc.SCORE_SELECTED,
        )
        self.assertEqual(
            qc.scope_for("ok", "0.5", Decimal("0.5"))[0], qc.SCORE_EXCLUDED
        )
        self.assertEqual(
            qc.scope_for("ok", "0.5000", Decimal("0.5"))[0], qc.SCORE_EXCLUDED
        )

    def test_maximum_exact_taxon_and_window_diagnostics(self):
        result = self.parse(
            HEADER + "0;3;Passer domesticus;家麻雀;0.99\n"
            "0;3; Passer montanus ;麻雀;0.7\n"
            "28.5;31.5;Passer montanus;麻雀;0.8\n"
            "28.5;31.5;Passer montanus;麻雀;0.6\n"
        )
        self.assertEqual(result["target_max_confidence"], "0.8")
        self.assertEqual(result["target_window_count"], 2)
        self.assertEqual(result["target_duplicate_window_rows"], 1)
        self.assertEqual(result["nominal_end_after_30_rows"], 2)
        self.assertEqual(result["target_max_end_s"], "31.5")
        self.assertEqual(result["csv_status"], "ok")
        self.assertEqual(result["scientific_name_trimmed_rows"], 1)
        self.assertIn("Passer domesticus", result["other_scientific_names"])

    def test_partial_bad_table_is_unknown_despite_earlier_high_score(self):
        for bad in ("nan", "inf", "1.1", "bad", ""):
            result = self.parse(
                HEADER + "0;3;Passer montanus;麻雀;0.9\n"
                f"3;6;Passer montanus;麻雀;{bad}\n"
            )
            self.assertEqual(result["csv_status"], "parse_error")
            self.assertEqual(result["target_max_confidence"], "0.9")
            self.assertEqual(
                qc.scope_for(
                    result["csv_status"],
                    result["target_max_confidence"],
                    Decimal("0.5"),
                )[0],
                qc.SCORE_UNKNOWN,
            )

    def test_missing_empty_no_target_and_bad_column_count(self):
        missing = qc.parse_detection_csv(
            Path("/not/existing/dp11.csv"), "missing", Decimal("0.5")
        )
        self.assertEqual(missing["csv_status"], "missing")
        self.assertEqual(
            qc.scope_for("missing", "", Decimal("0.5"))[0], qc.SCORE_UNKNOWN
        )
        empty = self.parse(HEADER)
        self.assertEqual(empty["csv_status"], "empty")
        no_target = self.parse(HEADER + "0;3;Passer domesticus;家麻雀;0.9\n")
        self.assertEqual(no_target["target_max_confidence"], "")
        self.assertIn("not_verified_absence", qc.scope_for("ok", "", Decimal("0.5"))[1])
        bad = self.parse(HEADER + "0;3;Passer montanus;麻雀;0.7;extra\n")
        self.assertEqual(bad["csv_status"], "parse_error")
        self.assertEqual(bad["malformed_rows"], 1)

    def test_window_error_kept_separate_from_finite_score(self):
        result = self.parse(
            HEADER + "nan;3;Passer montanus;麻雀;0.8\n"
            "5;2;Passer montanus;麻雀;0.6\n"
            "-1;2;Passer montanus;麻雀;0.55\n"
        )
        self.assertEqual(result["csv_status"], "ok")
        self.assertEqual(result["window_parse_error_rows"], 1)
        self.assertEqual(result["nonpositive_window_rows"], 1)
        self.assertEqual(result["negative_window_rows"], 1)

    def test_identity_site_year_and_invalid_name_do_not_collapse(self):
        name = "2023-05-01-birdnet-12_00_00.mp3"
        a = qc.identity(2023, "liujiaxia", "liujiaxia/" + name)[0]
        duplicate = qc.identity(2023, "liujiaxia", "liujiaxia/day/" + name)[0]
        self.assertEqual(a, duplicate)
        self.assertNotEqual(a, qc.identity(2023, "yongxing", "yongxing/" + name)[0])
        self.assertNotEqual(a, qc.identity(2024, "liujiaxia", "liujiaxia/" + name)[0])
        self.assertNotEqual(
            qc.identity(2023, "a", "a/foo.mp3")[0],
            qc.identity(2023, "a", "a/bar.mp3")[0],
        )


class PreparationTests(unittest.TestCase):
    def test_synthetic_end_to_end_keeps_aliases_diagnostics_and_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            inventory = root / "inventory"
            (inventory / "soundscape_inventory").mkdir(parents=True)
            for year in (2023, 2024):
                (data / f"birdsongs_{year}/data/wav/raw").mkdir(parents=True)
            metadata, physical, high = [], [], []

            def add(
                year,
                site,
                time,
                score,
                *,
                anomaly=False,
                duplicate=False,
                missing=False,
            ):
                stem = f"{year}-05-01-birdnet-12_00_{time}"
                relative = f"{site}/{'day/' if duplicate else ''}{stem}.mp3"
                path = data / f"birdsongs_{year}/data/wav/raw/{relative}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fake mp3 bytes")
                if not missing:
                    path.with_name(stem + ".wav.csv").write_text(
                        HEADER + f"0;3;Passer montanus;麻雀;{score}\n", encoding="utf-8"
                    )
                physical.append(
                    {
                        "year": year,
                        "site": site,
                        "timestamp": f"{year}-05-01T12:00:{time}",
                        "relative_path": relative,
                        "file_size_bytes": path.stat().st_size,
                        "is_canonical": not duplicate,
                    }
                )
                if duplicate:
                    return
                metadata.append(
                    {
                        "year": year,
                        "site": site,
                        "relative_path": relative,
                        "timestamp": f"{year}-05-01T12:00:{time}",
                        "bytes": path.stat().st_size,
                        "header_ok": not anomaly,
                        "sample_rate": 48000,
                        "duration_s": 30,
                        "detection_csv_status": "missing" if missing else "ok",
                        "target_max_confidence": "" if missing else score,
                    }
                )
                if score == "0.8":
                    high.append(
                        {
                            "year": year,
                            "site": site,
                            "relative_path": relative,
                            "target_max_confidence": score,
                            "in_existing_high_list": False,
                        }
                    )

            add(2023, "liujiaxia", "00", "0.8")
            add(2023, "liujiaxia", "00", "0.8", duplicate=True)
            add(2023, "liujiaxia", "01", "0.4", anomaly=True)
            add(2024, "yongxing", "00", "0.5")
            add(2024, "yongxing", "01", "", missing=True)
            orphan = data / "birdsongs_2024/data/wav/raw/yongxing/orphan.wav.csv"
            orphan.write_text(
                HEADER + "0;3;Passer montanus;麻雀;0.9\n", encoding="utf-8"
            )
            qc.write_rows(
                inventory / "recording_metadata.csv.gz", metadata, list(metadata[0])
            )
            qc.write_rows(
                inventory / "soundscape_inventory/recording_manifest_physical.csv",
                physical,
                list(physical[0]),
            )
            qc.write_rows(
                inventory / "high_confidence_reconciliation.csv", high, list(high[0])
            )
            qc.write_rows(
                inventory / "duplicate_content_check.csv",
                [],
                ["relative_path", "sha256"],
            )
            output = root / "output"
            args = argparse.Namespace(
                output_dir=output,
                inventory_dir=inventory,
                data_root=data,
                threshold="0.5",
                pilot_per_stratum=1,
                workers=2,
            )
            before = {p: qc.sha256(p) for p in data.rglob("*") if p.is_file()}
            qc.run(args)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["physical_mp3"], 5)
            self.assertEqual(summary["logical_mp3"], 4)
            self.assertEqual(
                summary["current_scope_counts"],
                {qc.SCORE_SELECTED: 1, qc.SCORE_EXCLUDED: 2, qc.SCORE_UNKNOWN: 1},
            )
            self.assertEqual(summary["orphan_detection_csv"], 1)
            self.assertEqual(summary["scope_difference_count"], 0)
            self.assertEqual(
                summary["jobs_full_roles"],
                {
                    "selected_representative": 1,
                    "selected_duplicate": 1,
                    "diagnostic_anomaly": 1,
                },
            )
            self.assertEqual(summary["jobs_pilot"], 3)
            self.assertEqual(summary["old_anomaly_logical"], 1)
            self.assertEqual(summary["exact_threshold_logical_maxima"], 1)
            jobs = list(qc.rows(output / "audio_jobs_full.csv.gz"))
            self.assertEqual(len({job["physical_id"] for job in jobs}), len(jobs))
            self.assertEqual({p: qc.sha256(p) for p in before}, before)
            with self.assertRaises(FileExistsError):
                qc.run(args)


if __name__ == "__main__":
    unittest.main()
