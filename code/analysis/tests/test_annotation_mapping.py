"""Small synthetic regression coverage for actual final manual-map semantics."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from sparrow_dataset.annotation import refinement as m
from sparrow_dataset.annotation.archive import materialize
from sparrow_dataset.io import digest


class AnnotationMappingTests(unittest.TestCase):
    def fixture(self):
        sources = [
            "single_group_01",
            "single_group_02",
            "single_group_03",
            "single_group_03_v1",
            "double_group_01",
            "single_group_05_v1",
        ]
        destinations = [
            "single_group_01",
            "single_group_01",
            "single_group_04",
            "single_group_04_v1",
            "double_group_01",
            None,
        ]
        actions = [
            "retain_code",
            "merge_into_main_bucket",
            "family_renumber",
            "family_renumber",
            "retain_code",
            "delete_whole_bucket",
        ]
        manifest_rows, mapping_rows = [], []
        for number, (source, target, action) in enumerate(
            zip(sources, destinations, actions)
        ):
            structure, group_id, family, variant = m.parse_target_bucket(source)
            stable_id = f"syllable_{number}"
            manifest_rows.append(
                dict(
                    stable_id=stable_id,
                    micro_type="same_microtype",
                    source_stem="recording_A",
                    final_group_id=group_id,
                    final_group_code=family,
                    final_bucket_code=source,
                    final_variant_code=variant,
                    storage_layout="flat_sdt_prefix_top_level",
                    storage_bucket_code=source,
                    output_relative_path=f"{source}/{stable_id}.wav",
                    output_audio_sha256="a" * 64,
                    syllable_structure=structure,
                    source_raven_row=number + 10,
                    selection_id=number + 100,
                    source_frame_start=number * 48,
                    source_frame_stop=(number + 1) * 48,
                    extra_source_metadata=f"retain-{number}",
                )
            )
            mapping_rows.append(
                dict(
                    source_bucket_code=source,
                    action=action,
                    target_family_code=(
                        m.parse_target_bucket(target)[2] if target else None
                    ),
                    target_bucket_code=target,
                    n_syllables=1,
                )
            )
        manifest = pd.DataFrame(manifest_rows)
        prior = manifest.copy()
        prior["complete_output_status"] = "retained"
        historical = dict(
            manifest_rows[0],
            stable_id="deleted_before",
            source_raven_row=99,
            complete_output_status="deleted_earlier",
            final_bucket_code=None,
        )
        prior = pd.concat([prior, pd.DataFrame([historical])], ignore_index=True)
        return manifest, prior, pd.DataFrame(mapping_rows)

    def run_mapping(self, manifest, mapping):
        return m.build_decision_audit(
            manifest, mapping, design_name="synthetic-confirmed", canonical=False
        )

    def test_explicit_structure_correction_updates_structure_and_family_together(self):
        manifest, prior, mapping = self.fixture()
        chosen = mapping.index[mapping["source_bucket_code"].eq("single_group_02")][0]
        mapping.loc[chosen, ["action", "target_family_code", "target_bucket_code"]] = ["structure_reclassification", "double_group_06", "double_group_06"]
        decisions = m.build_decision_audit(manifest, mapping, design_name="taxonomy-test", canonical=False)
        changed = decisions.loc[decisions["pre_sdt_refined_bucket_code"].eq("single_group_02")].iloc[0]
        self.assertEqual(changed["syllable_structure"], "double")
        self.assertEqual(changed["final_group_code"], "double_group_06")
        self.assertEqual(changed["sdt_refined_decision"], "author_corrected_structure")
        mapping.loc[chosen, "action"] = "family_renumber"
        with self.assertRaisesRegex(ValueError, "explicit structure_reclassification"):
            m.build_decision_audit(manifest, mapping, design_name="wrong-action", canonical=False)

    def test_delete_merge_renumber_and_variant_keep_identity_without_deduplication(
        self,
    ):
        manifest, prior, mapping = self.fixture()
        before = manifest.copy(deep=True)
        decisions = self.run_mapping(manifest, mapping)
        complete = m.update_complete_audit(prior, decisions, canonical=False)
        self.assertEqual(decisions.loc[1, "final_bucket_code"], "single_group_01")
        self.assertEqual(
            decisions.loc[1, "sdt_refined_decision"], "merged_into_main_bucket"
        )
        self.assertEqual(decisions.loc[3, "final_group_code"], "single_group_04")
        self.assertEqual(decisions.loc[3, "final_variant_code"], "single_group_04_v1")
        self.assertTrue(pd.isna(decisions.loc[5, "final_bucket_code"]))
        self.assertTrue(pd.isna(decisions.loc[5, "output_relative_path"]))
        self.assertEqual(
            complete.set_index("stable_id").loc["syllable_5", "complete_output_status"],
            "deleted_sdt_refined",
        )
        old_deleted = complete.set_index("stable_id").loc["deleted_before"]
        self.assertEqual(old_deleted["complete_output_status"], "deleted_earlier")
        self.assertEqual(
            old_deleted["sdt_refined_decision"], "prior_deleted_not_in_output"
        )
        retained = decisions.loc[decisions["final_bucket_code"].notna()]
        self.assertEqual(
            set(retained["stable_id"]), {f"syllable_{i}" for i in range(5)}
        )
        self.assertEqual(retained["output_audio_sha256"].nunique(), 1)
        for column in (
            "source_raven_row",
            "selection_id",
            "source_frame_start",
            "source_frame_stop",
            "extra_source_metadata",
            "output_audio_sha256",
        ):
            self.assertEqual(decisions[column].tolist(), manifest[column].tolist())
        self.assertNotEqual(
            decisions["selection_id"].tolist(), decisions["source_raven_row"].tolist()
        )
        self.assertEqual(int(m.build_crosswalk(decisions)["n_syllables"].sum()), 6)
        pd.testing.assert_frame_equal(manifest, before)

    def test_default_profile_does_not_call_a_small_cohort_canonical2556(self):
        manifest, _, mapping = self.fixture()
        with self.assertRaisesRegex(ValueError, "requires 2560"):
            m.build_decision_audit(manifest, mapping, design_name="small")

    def test_rejects_duplicate_ids_and_duplicate_source_bucket_mapping(self):
        manifest, _, mapping = self.fixture()
        with self.assertRaisesRegex(ValueError, "duplicate stable_id"):
            self.run_mapping(pd.concat([manifest, manifest.iloc[[0]]]), mapping)
        with self.assertRaisesRegex(ValueError, "duplicate source_bucket_code"):
            self.run_mapping(manifest, pd.concat([mapping, mapping.iloc[[0]]]))

    def test_rejects_missing_extra_or_unknown_mapping(self):
        manifest, _, mapping = self.fixture()
        with self.assertRaisesRegex(ValueError, "exactly all source buckets"):
            self.run_mapping(manifest, mapping.iloc[:-1])
        extra = mapping.iloc[[0]].copy()
        extra["source_bucket_code"] = "single_group_31"
        with self.assertRaisesRegex(ValueError, "exactly all source buckets"):
            self.run_mapping(manifest, pd.concat([mapping, extra]))
        mapping.loc[0, "action"] = "invented_action"
        with self.assertRaisesRegex(ValueError, "unknown action"):
            self.run_mapping(manifest, mapping)

    def test_rejects_inconsistent_family_or_changed_declared_bucket_counts(self):
        manifest, _, mapping = self.fixture()
        mapping.loc[0, "target_family_code"] = "single_group_02"
        with self.assertRaisesRegex(ValueError, "family and bucket disagree"):
            self.run_mapping(manifest, mapping)
        _, _, mapping = self.fixture()
        mapping.loc[0, "n_syllables"] = 2
        with self.assertRaisesRegex(ValueError, "source bucket counts disagree"):
            self.run_mapping(manifest, mapping)

    def test_retention_and_main_merge_action_meanings_are_enforced(self):
        manifest, _, mapping = self.fixture()
        mapping.loc[0, ["target_family_code", "target_bucket_code"]] = "single_group_02"
        with self.assertRaisesRegex(ValueError, "retain_code cannot change"):
            self.run_mapping(manifest, mapping)
        _, _, mapping = self.fixture()
        mapping.loc[1, "target_bucket_code"] = "single_group_01_v1"
        with self.assertRaisesRegex(ValueError, "cannot target a variant"):
            self.run_mapping(manifest, mapping)

    def test_prior_audit_cannot_silently_lose_live_ids_or_change_raven_identity(self):
        manifest, prior, mapping = self.fixture()
        decisions = self.run_mapping(manifest, mapping)
        with self.assertRaisesRegex(ValueError, "live IDs disagree"):
            m.update_complete_audit(prior.iloc[1:], decisions, canonical=False)
        with self.assertRaisesRegex(ValueError, "duplicate stable_id"):
            m.update_complete_audit(
                pd.concat([prior, prior.iloc[[0]]]), decisions, canonical=False
            )
        prior.loc[0, "source_raven_row"] = 55
        with self.assertRaisesRegex(ValueError, "identity mismatch: source_raven_row"):
            m.update_complete_audit(prior, decisions, canonical=False)

    def test_missing_source_row_is_not_filled_from_selection(self):
        manifest, prior, mapping = self.fixture()
        manifest = manifest.drop(columns="source_raven_row")
        prior = prior.drop(columns="source_raven_row")
        manifest["source_row"] = [None, 2, None, None, None, None]
        decisions = self.run_mapping(manifest, mapping)
        complete = m.update_complete_audit(prior, decisions, canonical=False)
        self.assertNotIn("source_raven_row", decisions)
        self.assertNotIn("source_raven_row", complete)
        self.assertEqual(decisions["source_row"].isna().sum(), 5)
        self.assertEqual(complete["source_row"].isna().sum(), 6)

    def test_rejects_duplicate_raven_row_per_recording_and_unsafe_id(self):
        manifest, _, mapping = self.fixture()
        manifest.loc[1, "source_raven_row"] = manifest.loc[0, "source_raven_row"]
        with self.assertRaisesRegex(
            ValueError, "duplicate source recording / Raven row"
        ):
            self.run_mapping(manifest, mapping)
        manifest.loc[0, "stable_id"] = "../escape"
        with self.assertRaisesRegex(ValueError, "safe filename"):
            self.run_mapping(manifest, mapping)

    def write_inputs(self, root, audio=False):
        manifest, prior, mapping = self.fixture()
        if audio:
            import numpy as np
            import soundfile as sf

            for index, row in manifest.iterrows():
                path = root / "audio" / row["output_relative_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, np.zeros(48, dtype="float32"), 48000, subtype="FLOAT")
                manifest.loc[index, "output_audio_sha256"] = digest(path)
                prior.loc[index, "output_audio_sha256"] = digest(path)
        paths = [root / name for name in ("input.csv", "prior.csv", "mapping.csv")]
        for path, frame in zip(paths, (manifest, prior, mapping)):
            frame.to_csv(path, index=False)
        return paths

    def test_dry_run_writes_nothing_and_csv_archive_preserves_audit_and_input_hashes(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.write_inputs(root)
            hashes = [digest(path) for path in inputs]
            output = root / "not_created" / "archive"
            plan = materialize(*inputs, output, canonical=False, dry_run=True)
            self.assertFalse(output.parent.exists())
            self.assertFalse(plan["audio_copied"])
            result = materialize(*inputs, output, canonical=False)
            self.assertEqual(result["profile"], "custom_not_canonical2556")
            self.assertEqual(result["retained_rows"], 5)
            self.assertEqual(result["cumulative_deleted_rows"], 2)
            self.assertEqual(result["cumulative_audit_rows"], 7)
            self.assertEqual(len(list(output.glob("*.csv"))), 6)
            self.assertEqual(len(list(output.rglob("*.wav"))), 0)
            self.assertEqual([digest(path) for path in inputs], hashes)
            exported = pd.read_csv(output / "manifest.csv").set_index("stable_id")
            self.assertEqual(exported.loc["syllable_1", "source_raven_row"], 11)
            self.assertEqual(exported.loc["syllable_1", "selection_id"], 101)
            with self.assertRaisesRegex(ValueError, "new directory"):
                materialize(*inputs, output, canonical=False)

    def test_audio_copy_keeps_bytes_and_omits_new_and_historical_deletions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.write_inputs(root, audio=True)
            output = root / "archive"
            result = materialize(
                *inputs, output, audio_root=root / "audio", canonical=False
            )
            self.assertTrue(result["audio_copied"])
            self.assertEqual(len(list(output.rglob("*.wav"))), 5)
            manifest = pd.read_csv(output / "manifest.csv")
            for row in manifest.itertuples(index=False):
                self.assertEqual(
                    digest(output / row.output_relative_path), row.output_audio_sha256
                )
            self.assertFalse(any("syllable_5" in p.name for p in output.rglob("*.wav")))

    def test_audio_hash_failure_leaves_no_output_or_staging_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.write_inputs(root, audio=True)
            next((root / "audio").rglob("*.wav")).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "source audio hash mismatch"):
                materialize(
                    *inputs,
                    root / "archive",
                    audio_root=root / "audio",
                    canonical=False,
                )
            self.assertFalse((root / "archive").exists())
            self.assertEqual(list(root.glob(".archive.*")), [])


if __name__ == "__main__":
    unittest.main()
