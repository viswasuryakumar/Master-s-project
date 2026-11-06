"""Unit tests for prepareData.helper (windowing and log structuring)."""
import os
import tempfile
import unittest

import pandas as pd

from prepareData.helper import (fixedSize_window, generate_logformat_regex,
                                log_to_dataframe, sliding_window, structure_log)


def make_logs(n=10, anomalous=()):
    return pd.DataFrame({
        'Content': ['msg%d' % i for i in range(n)],
        'Label': [1 if i in anomalous else 0 for i in range(n)],
    })


class FixedSizeWindowTests(unittest.TestCase):
    def test_non_overlapping_windows(self):
        out = fixedSize_window(make_logs(10), window_size=5, step_size=5)
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out['Content'][0]), ['msg0', 'msg1', 'msg2', 'msg3', 'msg4'])

    def test_overlapping_windows_and_short_tail(self):
        out = fixedSize_window(make_logs(10), window_size=4, step_size=3)
        self.assertEqual(len(out), 4)  # starts at 0, 3, 6, 9
        self.assertEqual(len(out['Content'][3]), 1)  # last window is truncated

    def test_window_label_is_max_of_item_labels(self):
        out = fixedSize_window(make_logs(10, anomalous={7}), window_size=5, step_size=5)
        self.assertEqual(list(out['Label']), [0, 1])
        self.assertEqual(out['item_Label'][1], [0, 0, 1, 0, 0])


class SlidingWindowTests(unittest.TestCase):
    @staticmethod
    def timed(times, labels):
        return pd.DataFrame({
            'timestamp': times, 'Label': labels, 'deltaT': [1] * len(times),
            'Content': ['m%d' % i for i in range(len(times))],
        })

    def test_time_windows_group_by_timestamp(self):
        df = self.timed([0, 1, 2, 10, 11, 12], [0] * 6)
        out = sliding_window(df, {'window_size': 5, 'step_size': 5})
        sizes = sorted(len(c) for c in out['Content'])
        self.assertEqual(sizes, [3, 3])

    def test_first_delta_in_each_window_is_zero(self):
        df = self.timed([0, 1, 2, 10, 11, 12], [0] * 6)
        out = sliding_window(df, {'window_size': 5, 'step_size': 5})
        for dt in out['deltaT']:
            self.assertEqual(dt[0], 0)

    def test_anomalous_label_propagates(self):
        df = self.timed([0, 1, 2, 10, 11, 12], [0, 0, 0, 0, 1, 0])
        out = sliding_window(df, {'window_size': 5, 'step_size': 5})
        self.assertEqual(sorted(out['Label']), [0, 1])


class LogFormatTests(unittest.TestCase):
    FORMAT = '<Label> <Time> <Content>'

    def test_generate_regex_headers(self):
        headers, regex = generate_logformat_regex(self.FORMAT)
        self.assertEqual(headers, ['Label', 'Time', 'Content'])
        m = regex.search('- 12:00 hello world')
        self.assertEqual(m.group('Label'), '-')
        self.assertEqual(m.group('Content'), 'hello world')

    def test_log_to_dataframe_skips_unparseable_lines(self):
        headers, regex = generate_logformat_regex('<A> <B>')
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'x.log')
            with open(path, 'w') as f:
                f.write('a b\n\nsingle\nc d\n')
            df = log_to_dataframe(path, regex, headers, 0, None)
        self.assertEqual(df['A'].tolist(), ['a', 'c'])

    def test_log_to_dataframe_line_range(self):
        headers, regex = generate_logformat_regex('<A> <B>')
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'x.log')
            with open(path, 'w') as f:
                f.write('a 1\nb 2\nc 3\nd 4\n')
            df = log_to_dataframe(path, regex, headers, 1, 3)
        self.assertEqual(df['A'].tolist(), ['b', 'c'])

    def test_structure_log_writes_csv(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'x.log'), 'w') as f:
                f.write('- hello\n')
            out_dir = os.path.join(d, 'out')
            structure_log(d, out_dir, 'x.log', '<Label> <Content>')
            df = pd.read_csv(os.path.join(out_dir, 'x.log_structured.csv'))
        self.assertEqual(df['Content'].tolist(), ['hello'])


if __name__ == '__main__':
    unittest.main()
