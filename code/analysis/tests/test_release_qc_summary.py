"""DP-11 汇总：完整性门、准入条件、重复与时间联接的合成回归。"""

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


from sparrow_dataset.processing import summarize_release_qc as qc

prep = qc.prep


def dump(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.preparation = self.root / "preparation"
        self.audio = self.root / "audio"
        self.clock = self.root / "clock"
        self.clock.mkdir()
        self.audio.mkdir()
        (self.audio / "batches").mkdir()
        data = self.root / "data"
        inventory = self.root / "inventory"
        (inventory / "soundscape_inventory").mkdir(parents=True)
        for year in (2023, 2024):
            (data / f"birdsongs_{year}/data/wav/raw").mkdir(parents=True)
        metadata, physical, historical = [], [], []
        self.keys = {}

        def add(label, time, score, *, alias=False, anomaly=False, missing=False):
            stem = f"2023-05-01-birdnet-12_00_{time}"
            relative = f"liujiaxia/{'day/' if alias else ''}{stem}.mp3"
            path = data / f"birdsongs_2023/data/wav/raw/{relative}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic fixture")
            if not missing:
                path.with_name(stem + ".wav.csv").write_text(
                    "Start (s);End (s);Scientific name;Common name;Confidence\n"
                    f"28.5;31.5;Passer montanus;麻雀;{score}\n",
                    encoding="utf-8",
                )
            physical.append(
                {
                    "year": 2023,
                    "site": "liujiaxia",
                    "relative_path": relative,
                    "timestamp": f"2023-05-01T12:00:{time}",
                    "file_size_bytes": path.stat().st_size,
                    "is_canonical": not alias,
                }
            )
            if alias:
                return
            self.keys[label] = prep.identity(2023, "liujiaxia", relative)[0]
            metadata.append(
                {
                    "year": 2023,
                    "site": "liujiaxia",
                    "relative_path": relative,
                    "timestamp": f"2023-05-01T12:00:{time}",
                    "bytes": path.stat().st_size,
                    "header_ok": not anomaly,
                    "sample_rate": 48000,
                    "duration_s": 30,
                    "detection_csv_status": "missing" if missing else "ok",
                    "target_max_confidence": "" if missing else score,
                }
            )
            if label in {"A", "B"}:
                historical.append(
                    {
                        "year": 2023,
                        "site": "liujiaxia",
                        "relative_path": relative,
                        "target_max_confidence": score,
                        "in_existing_high_list": False,
                    }
                )

        add("A", "00", "0.6")
        add("A", "00", "0.6", alias=True)
        add("B", "01", "0.7")
        add("C", "02", "0.8", anomaly=True)
        add("D", "03", "0.4", anomaly=True)
        add("E", "04", "", missing=True)
        add("F", "05", "0.9")
        add("F", "05", "0.9", alias=True)
        add("G", "06", "0.9")
        prep.write_rows(
            inventory / "recording_metadata.csv.gz", metadata, list(metadata[0])
        )
        prep.write_rows(
            inventory / "soundscape_inventory/recording_manifest_physical.csv",
            physical,
            list(physical[0]),
        )
        prep.write_rows(
            inventory / "high_confidence_reconciliation.csv",
            historical,
            list(historical[0]),
        )
        prep.write_rows(
            inventory / "duplicate_content_check.csv", [], ["relative_path", "sha256"]
        )
        prep.run(
            argparse.Namespace(
                data_root=data,
                inventory_dir=inventory,
                output_dir=self.preparation,
                threshold="0.5",
                pilot_per_stratum=1,
                workers=1,
            )
        )
        prep.write_rows(
            self.clock / "known_clock_errors.csv.gz",
            [
                {
                    "year": 2023,
                    "site": "liujiaxia",
                    "filename_timestamp": "2023-05-01T12:00:00",
                    "error_evidence": "synthetic_fixture_confirmed_error",
                    "corrected_recorded_at": "",
                }
            ],
            [
                "year",
                "site",
                "filename_timestamp",
                "error_evidence",
                "corrected_recorded_at",
            ],
        )
        self.jobs = list(prep.rows(self.preparation / "audio_jobs_full.csv.gz"))
        self.results = []
        for job in self.jobs:
            row = {
                **job,
                "qc_status": "decoded",
                "strict_decode_pass": True,
                "decode_complete": True,
                "has_decoder_error": False,
                "decoded_frames": 1440000,
                "nonfinite_samples": 0,
                "source_stat_matches": True,
                "expected_size_matches": True,
                "sha256": "a" * 64,
                "actual_size_bytes": job["expected_size_bytes"],
                "duration_seconds": "30.0",
                "native_sample_rate": 48000,
                "native_channels": 1,
                "all_zero": False,
                "samples_abs_ge_1": 0,
                "samples_abs_gt_1": 0,
                "samples_abs_eq_1": 0,
            }
            if job["logical_id"] == self.keys["B"]:
                row.update(
                    duration_seconds="30.0",
                    decoded_frames=1440000,
                    all_zero=False,
                    samples_abs_ge_1=5,
                    samples_abs_gt_1=3,
                    samples_abs_eq_1=2,
                )
            if job["logical_id"] == self.keys["C"]:
                row.update(
                    qc_status="decoder_error_log",
                    strict_decode_pass=False,
                    has_decoder_error=True,
                    decoded_frames=1152,
                    duration_seconds="0.024",
                    sha256="c" * 64,
                )
            if job["logical_id"] == self.keys["F"]:
                row["sha256"] = (
                    "f" if job["role"] == "selected_representative" else "b"
                ) * 64
            if job["logical_id"] == self.keys["G"]:
                # exit-0/strict=True 的错误日志也必须挡住，不能用退出状态冒充成功。
                row.update(has_decoder_error=True, sha256="e" * 64)
            self.results.append(row)
        self.freeze_audio()
        self.args = argparse.Namespace(
            preparation_dir=self.preparation,
            audio_dir=self.audio,
            timestamp_audit_dir=self.clock,
            output_dir=self.root / "final",
        )

    def tearDown(self):
        self.temp.cleanup()

    def freeze_audio(self):
        manifest = self.preparation / "audio_jobs_full.csv.gz"
        fields = list(self.results[0])
        prep.write_rows(self.audio / "results.csv.gz", self.results, fields)
        prep.write_rows(
            self.audio / "source_snapshot.csv.gz",
            [{"physical_id": row["physical_id"]} for row in self.jobs],
            ["physical_id"],
        )
        self.contract = {
            "mode": "full",
            "limit": None,
            "row_count": len(self.jobs),
            "batch_size": 3,
            "manifest_sha256": prep.sha256(manifest),
        }
        self.contract_sha = qc.canonical_hash(self.contract)
        dump(
            self.audio / "run_config.json",
            {
                "contract": self.contract,
                "contract_sha256": self.contract_sha,
                "source_snapshot_sha256": prep.sha256(
                    self.audio / "source_snapshot.csv.gz"
                ),
            },
        )
        batch_count = (len(self.jobs) + 2) // 3
        for number in range(batch_count):
            start, end = number * 3, min((number + 1) * 3, len(self.jobs))
            stem = f"batch_{number:05d}"
            path = self.audio / "batches" / f"{stem}.csv.gz"
            prep.write_rows(path, self.results[start:end], fields)
            dump(
                self.audio / "batches" / f"{stem}.json",
                {
                    "start": start,
                    "end": end,
                    "contract_sha256": self.contract_sha,
                    "results_sha256": prep.sha256(path),
                    "summary": qc.aggregate(self.results[start:end]),
                },
            )
        self.summary = {
            "state": "complete",
            "mode": "full",
            "contract_sha256": self.contract_sha,
            "results_sha256": prep.sha256(self.audio / "results.csv.gz"),
            "batch_count": batch_count,
            **qc.aggregate(self.results),
        }
        dump(self.audio / "summary.json", self.summary)

    def test_end_to_end_preserves_scope_ids_diagnostics_and_time_limits(self):
        summary = qc.run(self.args)
        self.assertEqual(summary["logical_archive"], 7)
        self.assertEqual(summary["selected_candidates"], 5)
        self.assertEqual(summary["technical_eligible_logical_files"], 2)
        self.assertEqual(summary["technical_eligible_distinct_payloads"], 1)
        self.assertEqual(summary["held_candidates"], 3)
        self.assertEqual(summary["eligible_known_clock_errors"], 1)
        self.assertEqual(summary["eligible_other_time_unverified"], 1)
        self.assertEqual(summary["eligible_all_zero_files"], 0)
        self.assertEqual(summary["eligible_duration_not_exactly_30s"], 0)
        self.assertEqual(summary["eligible_finite_amplitude_ge_1_files"], 1)
        self.assertEqual(summary["eligible_finite_amplitude_gt_1_files"], 1)
        self.assertEqual(summary["eligible_finite_amplitude_eq_1_files"], 1)
        self.assertEqual(
            summary["eligible_audio_format_counts"],
            [{"sample_rate_hz": 48000, "channels": 1, "logical_files": 2}],
        )
        self.assertEqual(summary["eligible_duration_seconds_min"], 30.0)
        self.assertEqual(summary["eligible_duration_seconds_max"], 30.0)
        self.assertEqual(summary["eligible_duration_exactly_30s"], 2)
        self.assertEqual(
            summary[
                "all_examined_representatives_cross_logical_nonempty_payload_groups"
            ],
            1,
        )
        self.assertEqual(
            summary["all_examined_representatives_cross_logical_empty_payload_groups"],
            0,
        )
        self.assertEqual(
            summary["known_anomaly_qc_status_counts"],
            {"decoder_error_log": 1, "decoded": 1},
        )
        self.assertEqual(summary["candidate_list_qc_status_counts"], {"decoded": 2})
        self.assertEqual(summary["eligible_target_window_extends_decoded_audio"], 2)
        records = {
            row["logical_id"]: row
            for row in prep.rows(
                self.args.output_dir / "final_logical_inventory.csv.gz"
            )
        }
        self.assertEqual(
            records[self.keys["E"]]["qc_status"], "not_decoded_outside_scope"
        )
        self.assertEqual(records[self.keys["E"]]["technical_status"], "unknown_score")
        self.assertEqual(
            records[self.keys["D"]]["technical_status"], "outside_score_scope"
        )
        self.assertEqual(records[self.keys["D"]]["audio_examined"], "True")
        self.assertIn(
            "duplicate_conflicting_payload", records[self.keys["F"]]["hold_reasons"]
        )
        self.assertIn("decoder_error", records[self.keys["G"]]["hold_reasons"])
        self.assertEqual(
            len(list(prep.rows(self.args.output_dir / "sampling_frame.csv"))), 24
        )
        self.assertEqual(
            len(
                list(prep.rows(self.args.output_dir / "known_anomaly_disposition.csv"))
            ),
            2,
        )
        self.assertEqual(
            len(
                list(prep.rows(self.args.output_dir / "candidate_list_disposition.csv"))
            ),
            2,
        )
        with self.assertRaises(FileExistsError):
            qc.run(self.args)

    def test_versioned_clock_decision_only_changes_clock_membership(self):
        self.args.exclude_known_clock_errors = True
        old = qc.run(self.args)
        source_digest = prep.sha256(self.audio / "results.csv.gz")
        decision_path = self.root / "policy.json"
        dump(
            decision_path,
            dict(
                decision_date="2026-09-18",
                decision_source="synthetic explicit answer",
                user_decision="是，只取消时钟排除",
                policy_id="retain-clock-test-v3",
                change_scope="known_clock_error_only",
                exclude_known_clock_errors=False,
            ),
        )
        self.args.exclude_known_clock_errors = False
        self.args.publication_policy = decision_path
        self.args.output_dir = self.root / "new"
        new = qc.run(self.args)
        self.assertEqual(
            old["technical_eligible_logical_files"] + 1,
            new["technical_eligible_logical_files"],
        )
        self.assertEqual(new["eligible_known_clock_errors"], 1)
        self.assertEqual(
            old["decode_integrity_held_candidates"],
            new["decode_integrity_held_candidates"],
        )
        self.assertEqual(new["publication_policy"]["decision_date"], "2026-09-18")
        self.assertEqual(
            new["publication_policy"]["user_decision"], "是，只取消时钟排除"
        )
        self.assertFalse(new["publication_policy"]["exclude_known_clock_errors"])
        self.assertEqual(prep.sha256(self.audio / "results.csv.gz"), source_digest)
        records = {
            r["logical_id"]: r
            for r in prep.rows(self.args.output_dir / "final_logical_inventory.csv.gz")
        }
        clock_row = records[self.keys["A"]]
        self.assertEqual(clock_row["time_status"], "known_clock_error")
        self.assertEqual(clock_row["corrected_recorded_at"], "")
        self.assertEqual(clock_row["eligibility_scope"], "retain-clock-test-v3")
        for key in ("C", "F", "G"):
            self.assertEqual(records[self.keys[key]]["technical_eligible"], "False")
        self.args.output_dir = self.root / "contradictory"
        self.args.exclude_known_clock_errors = True
        with self.assertRaisesRegex(ValueError, "矛盾"):
            qc.run(self.args)
        self.assertFalse(self.args.output_dir.exists())

    def test_ge07_current_scope_separate_from_existing_qc_coverage(self):
        decision = self.root / "ge07.json"
        dump(
            decision,
            dict(
                decision_date="2026-09-18",
                decision_source="synthetic explicit answer",
                user_decision="raw >=0.7",
                policy_id="ge07-test",
                change_scope="raw_score_threshold_only",
                exclude_known_clock_errors=False,
                score_selection={"comparison": "ge", "threshold": "0.7"},
            ),
        )
        self.args.publication_policy = decision
        summary = qc.run(self.args)
        self.assertEqual(summary["selected_candidates"], 4)
        self.assertEqual(summary["technical_eligible_logical_files"], 1)
        self.assertEqual(summary["held_candidates"], 3)
        self.assertEqual(summary["source_qc_candidate_logical_files"], 5)
        self.assertEqual(summary["exact_threshold_logical_maxima"], 1)
        self.assertEqual(summary["sampling_frame_strata"], 12)
        self.assertEqual(summary["qc_coverage"]["current_candidates_checked"], 4)
        self.assertEqual(
            summary["qc_coverage"]["outside_current_candidates_checked"], 2
        )
        records = {
            r["logical_id"]: r
            for r in prep.rows(self.args.output_dir / "final_logical_inventory.csv.gz")
        }
        self.assertEqual(
            records[self.keys["B"]]["technical_eligible"], "True"
        )  # 恰好0.7纳入。
        lower = records[self.keys["A"]]
        self.assertEqual(lower["scope_status"], "excluded_below_threshold")
        self.assertEqual(lower["scope_reason"], "target_max_below_threshold")
        self.assertEqual(lower["audio_examined"], "True")
        self.assertEqual(lower["strict_decode_pass"], "True")
        self.assertEqual(lower["time_status"], "known_clock_error")
        self.assertEqual(lower["technical_status"], "outside_score_scope")
        self.assertEqual(lower["hold_reasons"], "")

    def test_rejects_incomplete_or_pilot_summary_without_creating_final(self):
        self.summary["state"] = "running"
        dump(self.audio / "summary.json", self.summary)
        with self.assertRaisesRegex(ValueError, "已完成"):
            qc.run(self.args)
        self.assertFalse(self.args.output_dir.exists())

    def test_rejects_tampered_summary_counter(self):
        self.summary["strict_decode_pass_count"] += 1
        dump(self.audio / "summary.json", self.summary)
        with self.assertRaisesRegex(ValueError, "strict_decode_pass_count"):
            qc.run(self.args)

    def test_rejects_tampered_batch_hash(self):
        checkpoint = self.audio / "batches/batch_00000.json"
        content = qc.load_json(checkpoint)
        content["results_sha256"] = "0" * 64
        dump(checkpoint, content)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            qc.run(self.args)

    def test_missing_error_check_and_nonfinite_not_eligible(self):
        record = {"duplicate_scope_conflict": False}
        valid = self.results[0].copy()
        valid.pop("has_decoder_error")
        self.assertIn(
            "decoder_error_or_error_check_unverified",
            qc.technical_reasons(record, valid, "no_checked_alias"),
        )
        valid["has_decoder_error"] = False
        valid["nonfinite_samples"] = 1
        self.assertIn(
            "nonfinite_or_unverified_samples",
            qc.technical_reasons(record, valid, "no_checked_alias"),
        )
        valid["source_stat_matches"] = False
        self.assertIn(
            "source_stat_not_stable",
            qc.technical_reasons(record, valid, "no_checked_alias"),
        )

    def test_empty_payload_group_is_not_reported_as_nonempty_audio_duplicate(self):
        for row in self.results:
            if row["logical_id"] in {self.keys["C"], self.keys["G"]}:
                row.update(
                    qc_status="zero_byte",
                    strict_decode_pass=False,
                    decode_complete=False,
                    actual_size_bytes=0,
                    expected_size_matches=False,
                    sha256="0" * 64,
                    decoded_frames=0,
                    duration_seconds="",
                    has_decoder_error="",
                )
        self.freeze_audio()
        summary = qc.run(self.args)
        self.assertEqual(
            summary["all_examined_representatives_cross_logical_empty_payload_groups"],
            1,
        )
        self.assertEqual(
            summary[
                "all_examined_representatives_in_cross_logical_empty_payload_groups"
            ],
            2,
        )
        self.assertEqual(
            summary[
                "all_examined_representatives_cross_logical_nonempty_payload_groups"
            ],
            1,
        )
        empty_rows = [
            row
            for row in prep.rows(self.args.output_dir / "duplicate_mapping.csv.gz")
            if row["payload_kind"] == "empty_payload"
        ]
        self.assertEqual(len(empty_rows), 2)
        self.assertTrue(
            all(
                row["duplicate_content_status"]
                == "same_empty_payload_not_audio_duplicate"
                for row in empty_rows
            )
        )

    def test_publication_policy_excludes_successful_zero_short_and_warning_files(self):
        for row in self.results:
            if row["logical_id"] == self.keys["A"]:
                row["all_zero"] = True
            if row["logical_id"] == self.keys["B"]:
                row.update(
                    qc_status="decoded_with_warnings",
                    duration_seconds="29.98",
                    decoded_frames=1439040,
                )
        self.freeze_audio()
        summary = qc.run(self.args)
        self.assertEqual(summary["decode_integrity_eligible_logical_files"], 2)
        self.assertEqual(summary["policy_excluded_after_decode_integrity"], 2)
        self.assertEqual(summary["publication_eligible_logical_files"], 0)
        self.assertEqual(
            summary["release_policy_reason_counts_after_decode_integrity"],
            {"all_zero_signal": 1, "non_nominal_duration": 1, "decoder_warning": 1},
        )
        final = {
            row["logical_id"]: row
            for row in prep.rows(
                self.args.output_dir / "final_logical_inventory.csv.gz"
            )
        }
        self.assertEqual(final[self.keys["A"]]["strict_decode_pass"], "True")
        self.assertEqual(final[self.keys["A"]]["decode_integrity_eligible"], "True")
        self.assertEqual(final[self.keys["A"]]["technical_eligible"], "False")
        self.assertEqual(final[self.keys["C"]]["release_policy_exclusion_reasons"], "")
        self.assertEqual(summary["eligible_duration_seconds_min"], None)
        self.assertFalse(
            summary["publication_policy"]["amplitude_abs_gt_or_eq_1_exclusion"]
        )

    def test_known_clock_error_option_preserves_other_unverified_timestamps(self):
        self.args.exclude_known_clock_errors = True
        summary = qc.run(self.args)
        self.assertEqual(summary["decode_integrity_eligible_logical_files"], 2)
        self.assertEqual(summary["publication_eligible_logical_files"], 1)
        self.assertEqual(summary["policy_excluded_after_decode_integrity"], 1)
        self.assertEqual(
            summary["release_policy_reason_counts_after_decode_integrity"],
            {"known_clock_error": 1},
        )
        self.assertEqual(summary["eligible_known_clock_errors"], 0)
        self.assertEqual(summary["eligible_other_time_unverified"], 1)
        self.assertTrue(summary["publication_policy"]["exclude_known_clock_errors"])
        final = {
            row["logical_id"]: row
            for row in prep.rows(
                self.args.output_dir / "final_logical_inventory.csv.gz"
            )
        }
        self.assertIn("known_clock_error", final[self.keys["A"]]["hold_reasons"])
        self.assertEqual(final[self.keys["A"]]["strict_decode_pass"], "True")
        self.assertEqual(final[self.keys["B"]]["technical_eligible"], "True")
        self.assertEqual(
            final[self.keys["B"]]["time_status"], "not_independently_verified"
        )
        self.assertEqual(final[self.keys["B"]]["corrected_recorded_at"], "")
        # 浮点幅值>1与恰等于1不触发损坏/发布排除。
        self.assertEqual(summary["eligible_finite_amplitude_gt_1_files"], 1)

    def test_duration_tolerance_is_one_native_frame_not_nominal_filename(self):
        row = {
            "native_sample_rate": 1000,
            "duration_seconds": "30.001",
            "qc_status": "decoded",
            "all_zero": False,
        }
        self.assertEqual(qc.publication_policy_reasons(row, True, False, False), [])
        row["duration_seconds"] = "29.999"
        self.assertEqual(qc.publication_policy_reasons(row, True, False, False), [])
        row["duration_seconds"] = "29.9989"
        self.assertEqual(
            qc.publication_policy_reasons(row, True, False, False),
            ["non_nominal_duration"],
        )
        # 解码失败时.partial PCM长度不能充当完整文件时长另作科学判断。
        self.assertEqual(qc.publication_policy_reasons(row, False, False, False), [])

    def test_stratum_boundary_at_07_is_in_upper_group(self):
        self.assertEqual(qc.score_stratum("0.5"), "outside_selected_score_range")
        self.assertEqual(qc.score_stratum("0.50000000000000000000001"), "gt_0p5_lt_0p7")
        self.assertEqual(qc.score_stratum("0.69999999999999999999999"), "gt_0p5_lt_0p7")
        self.assertEqual(qc.score_stratum("0.7"), "ge_0p7_le_1")


if __name__ == "__main__":
    unittest.main()
