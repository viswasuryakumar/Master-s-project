"""Unit checks for benchmark.py answer parsing and confusion-matrix scores."""
import unittest

import numpy as np

from benchmark import Tally, answers_to_flags, first_label_word


class FirstLabelWordTests(unittest.TestCase):
    def test_case_insensitive(self):
        self.assertEqual(first_label_word("The sequence is Anomalous."), "anomalous")

    def test_earliest_keyword_wins(self):
        self.assertEqual(first_label_word("normal, not anomalous"), "normal")
        self.assertEqual(first_label_word("anomalous rather than normal"), "anomalous")

    def test_missing_keyword(self):
        self.assertIsNone(first_label_word("I am not sure"))


class AnswersToFlagsTests(unittest.TestCase):
    def test_unreadable_answers_default_to_normal_and_are_recorded(self):
        unreadable = []
        flags = answers_to_flags(["anomalous", "normal", "???"], unreadable)
        self.assertEqual(flags, [1, 0, 0])
        self.assertEqual(unreadable, ["???"])


class TallyTests(unittest.TestCase):
    def test_counts_and_scores(self):
        tally = Tally.build(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
        self.assertEqual(tally, Tally(tp=1, fp=0, tn=2, fn=1))
        self.assertAlmostEqual(tally.precision, 1.0)
        self.assertAlmostEqual(tally.recall, 0.5)
        self.assertAlmostEqual(tally.f1, 2 / 3)
        self.assertAlmostEqual(tally.accuracy, 0.75)

    def test_no_positive_predictions_gives_zero_precision(self):
        tally = Tally.build(np.array([1, 0]), np.array([0, 0]))
        self.assertEqual(tally.precision, 0.0)
        self.assertEqual(tally.f1, 0.0)

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            Tally.build(np.array([1, 0, 1]), np.array([1, 0]))

    def test_as_dict_keys(self):
        keys = set(Tally(1, 1, 1, 1).as_dict())
        self.assertEqual(keys, {"tp", "fp", "tn", "fn", "precision", "recall", "f1", "accuracy"})


if __name__ == "__main__":
    unittest.main()
