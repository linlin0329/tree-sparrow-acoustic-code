"""可移植交付需真实复制、独立字节核验、安全路径及完整清单。"""

import importlib.util
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from sparrow_dataset.processing import build_dataset_release_bundle as m


class PortableBundleTests(unittest.TestCase):
    def test_real_copy_and_resume_require_matching_source_and_destination(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source"
            source.write_bytes(b"original bytes")
            target = root / "package/audio/file.mp3"
            sha = m.digest(source)
            result = m.copy_verified(source, target, sha, source.stat().st_size)
            self.assertFalse(result["reused"])
            self.assertNotEqual(source.stat().st_ino, target.stat().st_ino)
            self.assertEqual(target.stat().st_nlink, 1)
            self.assertTrue(
                m.copy_verified(
                    source, target, sha, source.stat().st_size, resume=True
                )["reused"]
            )
            target.write_bytes(b"corrupt output")
            with self.assertRaisesRegex(ValueError, "断点目标哈希"):
                m.copy_verified(source, target, sha, source.stat().st_size, resume=True)
            self.assertEqual(source.read_bytes(), b"original bytes")

    def test_wrong_source_hash_never_publishes_target(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "source"
            source.write_bytes(b"actual")
            target = Path(d) / "target"
            with self.assertRaisesRegex(ValueError, "源载荷"):
                m.copy_verified(source, target, "0" * 64, 6)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name("target.copy-partial").exists())

    def test_symbolic_or_hardlinked_resume_target_is_not_a_portable_copy(self):
        import os

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source"
            source.write_bytes(b"x")
            target = root / "target"
            target.symlink_to(source)
            with self.assertRaises(ValueError):
                m.copy_verified(source, target, m.digest(source), 1, resume=True)
            target.unlink()
            os.link(source, target)
            with self.assertRaisesRegex(ValueError, "独立字节副本"):
                m.copy_verified(source, target, m.digest(source), 1, resume=True)

    def test_manifest_detects_unlisted_and_changed_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "audio").mkdir()
            (root / "audio/a.wav").write_bytes(b"wave")
            (root / "README.md").write_text("test")
            m.create_manifest(root)
            self.assertEqual(m.verify_package(root)["files_checked"], 2)
            (root / "extra").write_text("extra")
            with self.assertRaisesRegex(ValueError, "未列出"):
                m.verify_package(root)
            (root / "extra").unlink()
            (root / "audio/a.wav").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "校验失败"):
                m.verify_package(root)

    def test_public_paths_cannot_escape_and_metadata_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for relative in ("../source", "/tmp/source", "x\\y"):
                with self.assertRaises(ValueError):
                    m.safe_target(root, relative)
            m.write_new(root / "metadata/test.csv", b"a\n1\n")
            m.write_new(root / "metadata/test.csv", b"a\n1\n")
            with self.assertRaisesRegex(ValueError, "不同内容"):
                m.write_new(root / "metadata/test.csv", b"a\n2\n")

    def test_manifest_checks_retained_copy_against_expected_source_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.mp3").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "源证据"):
                m.create_manifest(root, {"a.mp3": "0" * 64})

    def test_latest_fragment_review_respects_identity_and_other_positive_fragment(self):
        from sparrow_dataset.processing import song_release_index as song

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for path in (
                root / "review_audio/seed/seed.wav",
                root / "review_audio/manual/seed_fragment.wav",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"same reviewed WAV")
            sha = m.digest(root / "review_audio/seed/seed.wav")
            manual = [
                dict(source_stem=stem, raw_recording_id=key, sha256=sha, duration_s="5")
                for stem, key in [
                    ("seed_fragment", "seed"),
                    ("later_fragment", "later"),
                    ("uncertain_other_fragment", "core"),
                ]
            ]
            reviews = [
                dict(
                    source_stem=r["source_stem"],
                    review_group=(
                        "call"
                        if r["raw_recording_id"] != "core"
                        else "uncertain_or_not_stated"
                    ),
                    reason_verbatim="fixture fragment opinion",
                    current_evidence_reference="review.csv",
                    current_evidence_sha256="x",
                    received_on="date",
                    content_revision_received_on="",
                )
                for r in manual
            ]
            sources = dict(
                manual=manual,
                reviews=reviews,
                seed={"seed": [dict(file="seed.wav")]},
                later={"later": {}},
                data_root=root,
            )
            archive = {k: dict(raw_recording_id=k) for k in ("seed", "later", "core")}
            counts = Counter(core=1)
            syllables = Counter(core=26)
            held, audit = song.reconcile_reviews(
                sources,
                set(archive),
                archive,
                archive,
                counts,
                syllables,
                "hold_conflicting_recordings",
            )
            self.assertEqual(held, {"core"})
            self.assertTrue(all(not r["whole_raw_verified_negative"] for r in audit))
            kept, _ = song.reconcile_reviews(
                sources,
                set(archive),
                archive,
                archive,
                counts,
                syllables,
                "retain_historical_raw_positive_with_fragment_note",
            )
            self.assertEqual(kept, {"later", "core"})
            archive["later"]["sha256"] = "b" * 64
            sources["author_rereview"] = {
                key: dict(
                    raw_recording_id=key,
                    source_recording_key=key,
                    affected_manual_source_stem=stem,
                    previous_fragment_review_group="call",
                    reviewed_unit=unit,
                    reviewed_sha256=sha if key == "seed" else "b" * 64,
                    current_reviewed_unit_status=(
                        "low_quality_song" if key == "seed" else "contains_song"
                    ),
                    manual_fragment_judgement_changed=(
                        "true" if key == "seed" else "false"
                    ),
                    correction_id=key,
                    received_on="2026-09-18",
                    evidence_type="author_reported_repeat_listening",
                )
                for key, stem, unit in [
                    ("seed", "seed_fragment", "same_seed_manual_wav"),
                    ("later", "later_fragment", "original_30s_raw_mp3"),
                ]
            }
            restored, audit = song.reconcile_reviews(
                sources,
                set(archive),
                archive,
                archive,
                counts,
                syllables,
                "author_rereview_confirmed_all_11",
            )
            self.assertEqual(restored, {"seed", "later", "core"})
            by_key = {r["source_recording_key"]: r for r in audit}
            self.assertEqual(
                by_key["seed"]["latest_fragment_review_group"], "low_quality_song"
            )
            self.assertEqual(by_key["later"]["latest_fragment_review_group"], "call")
            self.assertEqual(
                by_key["later"]["author_reviewed_unit_status"], "contains_song"
            )
            (root / "review_audio/seed/seed.wav").write_bytes(
                b"a different historical interval"
            )
            with self.assertRaisesRegex(ValueError, "同一字节"):
                song.reconcile_reviews(
                    sources,
                    set(archive),
                    archive,
                    archive,
                    counts,
                    syllables,
                    "hold_conflicting_recordings",
                )


if __name__ == "__main__":
    unittest.main()
