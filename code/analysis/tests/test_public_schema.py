"""Public metadata removes only presentation/history fields, preserving science."""

import unittest
from sparrow_dataset.schema import public_rows


class PublicSchemaTests(unittest.TestCase):
    def test_syllable_projection_preserves_identity_frames_and_hashes(self):
        source = {
            "syllable_id": "abc",
            "source_frame_start": "101",
            "source_frame_stop": "205",
            "family_id": "single_group_01",
            "leaf_id": "single_group_01_v1",
            "begin_s": "0.0021",
            "decoded_pcm_f32le_sha256": "a" * 64,
            "data_version": "input-coordinate-set",
            "label_version": "input-label-set",
            "boundary_status": "reviewed",
            "supersedes_syllable_id": "prior",
        }
        result = public_rows("syllables.csv", [source])[0]
        self.assertEqual(result["data_version"], "1.0.0")
        self.assertEqual(result["label_version"], "sdt-taxonomy-r2-2026-09-24")
        self.assertNotIn("boundary_status", result)
        self.assertNotIn("supersedes_syllable_id", result)
        for key in (
            "syllable_id",
            "source_frame_start",
            "source_frame_stop",
            "family_id",
            "leaf_id",
            "begin_s",
            "decoded_pcm_f32le_sha256",
        ):
            self.assertEqual(result[key], source[key])
        self.assertEqual(source["data_version"], "input-coordinate-set")

    def test_song_projection_preserves_current_confirmation(self):
        source = {
            "raw_recording_id": "abc",
            "confirmation_evidence": "researcher_confirmed",
            "provenance_route": "seed",
            "historical_confirmation_evidence": "earlier",
            "latest_fragment_review_group": "reviewed",
            "latest_review_disposition": "accepted",
            "author_correction_id": "x",
            "author_reviewed_unit": "raw",
        }
        result = public_rows("song_recordings.csv", [source])[0]
        self.assertEqual(
            result,
            {
                "raw_recording_id": "abc",
                "confirmation_evidence": "researcher_confirmed",
                "provenance_route": "seed",
            },
        )


if __name__ == "__main__":
    unittest.main()
