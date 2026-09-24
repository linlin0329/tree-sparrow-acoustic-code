"""Check that the public entry points use the bundled final taxonomy inputs."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sparrow_dataset import assessment, cli
from sparrow_dataset.assessments import taxonomy, taxonomy_inputs


class AssessmentEntrypointTests(unittest.TestCase):
    def test_direct_defaults_match_wrapped_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "assessment"
            plan = assessment.replay("taxonomy", assessment.DEFAULT_ASSETS, output)
            self.assertEqual(plan["status"], "planned")
            self.assertFalse(output.exists())
            command = plan["command"]
            with patch("sys.argv", [command[2], *command[3:]]):
                wrapped = taxonomy.parse_args()
            required = []
            for option in ("--archive-dir", "--run-dir", "--classifier-dir", "--pair-evidence-dir", "--output-dir"):
                index = command.index(option)
                required.extend(command[index:index + 2])
            with patch("sys.argv", [command[2], *required]):
                direct = taxonomy.parse_args()
            self.assertEqual(vars(direct), vars(wrapped))

    def test_bundled_final_inputs_load_with_defaults(self):
        root = assessment.DEFAULT_ASSETS / "label_assessment"
        frame, features, patches, names, indices = taxonomy_inputs.load_analysis_inputs(
            root / "archive", root / "run", root / "classifier"
        )
        frame = taxonomy.add_taxonomy_columns(frame)
        self.assertEqual(frame.groupby("structure").size().to_dict(),
                         {"single": 1153, "double": 768, "triple": 635})
        self.assertEqual((len(frame), frame.family.nunique(), frame.leaf.nunique()),
                         (2556, 27, 41))
        self.assertEqual(features.shape, (2556, 105))
        self.assertEqual(patches.shape[0], 2556)
        self.assertEqual(len(names), 105)
        self.assertEqual(len(indices), 2556)
        summary, counts = taxonomy.reused_pair_summary(root / "pair_evidence/pairwise_similarity_evidence.csv")
        self.assertEqual(int(summary.n_pairs.sum()), 127)
        self.assertEqual(sum(counts.values()), 127)

    def test_previous_taxonomy_profile_is_not_a_public_choice(self):
        with patch("sys.argv", ["taxonomy", "--taxonomy-profile", "2556"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                taxonomy.parse_args()

    def test_removed_mapping_stage_is_not_a_public_command(self):
        self.assertNotIn("reviewed-labels", cli.STAGES)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parser().parse_args(["stage-help", "reviewed-labels"])


if __name__ == "__main__":
    unittest.main()
