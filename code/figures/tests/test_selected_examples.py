"""Regression guards for the released family coverage and preserved example pair."""
import copy
import unittest

from sparrow_figures.resources import ASSETS
from sparrow_figures.validation import check_selected_examples, rows


class SelectedExampleTests(unittest.TestCase):
    def setUp(self):
        self.selected = rows(ASSETS / "selected_syllables.csv")

    def test_current_selection_has_complete_ordered_family_coverage(self):
        check_selected_examples(self.selected)

    def test_missing_family_duplicate_example_and_wrong_order_are_rejected(self):
        missing = self.selected[:-1]
        duplicate = copy.deepcopy(self.selected)
        duplicate[1]["stable_id"] = duplicate[0]["stable_id"]
        unordered = self.selected.copy()
        unordered[0], unordered[1] = unordered[1], unordered[0]
        for changed in [missing, duplicate, unordered]:
            with self.subTest(length=len(changed)), self.assertRaises(ValueError):
                check_selected_examples(changed)

    def test_wrong_structure_panel_or_non_main_form_is_rejected(self):
        for field, value in [
            ("structure", "double"), ("panel", "S02"),
            ("family", "single_group_02"), ("role", "variant"),
            ("leaf", "single_group_01_v1"),
        ]:
            changed = copy.deepcopy(self.selected)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_selected_examples(changed)

    def test_figure6_pair_must_remain_unchanged(self):
        for field, value in [
            ("stable_id", "unapproved-example"), ("role", "main"),
            ("family", "single_group_09"), ("panel", "a"),
        ]:
            changed = copy.deepcopy(self.selected)
            changed[-1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_selected_examples(changed)


if __name__ == "__main__":
    unittest.main()
