"""Edge cases for windowing: empty input, oversized windows, single rows."""
import unittest

import pandas as pd

from prepareData.helper import fixedSize_window


def logs(n):
    return pd.DataFrame({'Content': [str(i) for i in range(n)], 'Label': [0] * n})


class EdgeCaseTests(unittest.TestCase):
    def test_window_larger_than_data_gives_one_window(self):
        out = fixedSize_window(logs(3), window_size=100, step_size=100)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out['Content'][0]), 3)

    def test_single_row(self):
        out = fixedSize_window(logs(1), window_size=5, step_size=5)
        self.assertEqual(len(out), 1)

    def test_step_larger_than_window_skips_rows(self):
        out = fixedSize_window(logs(10), window_size=2, step_size=5)
        self.assertEqual([list(c) for c in out['Content']], [['0', '1'], ['5', '6']])

    def test_empty_input_gives_no_windows(self):
        out = fixedSize_window(logs(0), window_size=5, step_size=5)
        self.assertEqual(len(out), 0)


if __name__ == '__main__':
    unittest.main()
