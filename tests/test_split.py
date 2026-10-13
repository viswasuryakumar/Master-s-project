import unittest

import pandas as pd

from prepareData.helper import fixedSize_window
from prepareData.split import assert_no_overlap, chronological_split


class SplitTests(unittest.TestCase):
    def logs(self, n=10):
        return pd.DataFrame({'Content': [str(i) for i in range(n)], 'Label': [0] * n})

    def test_split_preserves_order_and_sizes(self):
        train, test = chronological_split(self.logs(10), 0.8)
        self.assertEqual(len(train), 8)
        self.assertEqual(list(test['Content']), ['8', '9'])
        self.assertEqual(list(test.index), [0, 1])

    def test_invalid_ratio(self):
        for r in (0, 1, 1.5):
            with self.assertRaises(ValueError):
                chronological_split(self.logs(), r)

    def test_chronological_split_has_no_window_overlap(self):
        train, test = chronological_split(self.logs(20), 0.8)
        tr = fixedSize_window(train, 4, 2)
        te = fixedSize_window(test, 4, 2)
        assert_no_overlap(tr, te)

    def test_overlap_detected(self):
        a = pd.DataFrame({'Content': [['x', 'y']]})
        with self.assertRaisesRegex(ValueError, 'both'):
            assert_no_overlap(a, a.copy())


if __name__ == '__main__':
    unittest.main()
