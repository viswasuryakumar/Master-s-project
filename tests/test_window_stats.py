"""Unit tests for prepareData.window_stats."""
import unittest

import pandas as pd

from prepareData.helper import fixedSize_window
from prepareData.window_stats import compare_settings, summarize_windows


class WindowStatsTests(unittest.TestCase):
    def test_empty_dataframe(self):
        out = summarize_windows(pd.DataFrame({'Content': [], 'Label': []}))
        self.assertEqual(out['num_windows'], 0)

    def test_summary_values(self):
        df = pd.DataFrame({'Content': [['a', 'b'], ['c', 'd', 'e', 'f']], 'Label': [0, 1]})
        out = summarize_windows(df)
        self.assertEqual(out, {'num_windows': 2, 'mean_length': 3.0,
                               'max_length': 4, 'anomalous_ratio': 0.5})

    def test_compare_settings_one_row_per_setting(self):
        raw = pd.DataFrame({'Content': [str(i) for i in range(20)], 'Label': [0] * 19 + [1]})
        out = compare_settings(raw, [(5, 5), (10, 5)], fixedSize_window)
        self.assertEqual(len(out), 2)
        self.assertEqual(out['num_windows'].tolist(), [4, 4])
        self.assertGreater(out['mean_length'][1], out['mean_length'][0])


if __name__ == '__main__':
    unittest.main()
