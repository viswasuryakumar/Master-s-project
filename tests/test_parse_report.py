import json
import os
import tempfile
import unittest

from prepareData.parse_report import guess_log_format, parse_quality_report, write_report

FORMAT = '<Label> <Content>'


class ParseReportTests(unittest.TestCase):
    def report_for(self, text, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'x.log')
            with open(path, 'w') as fout:
                fout.write(text)
            return parse_quality_report(path, FORMAT, **kwargs)

    def test_counts_blank_parsed_and_duplicates(self):
        report = self.report_for('- hello\n\n- hello\nALERT boom\n')
        self.assertEqual(report['lines_scanned'], 4)
        self.assertEqual(report['blank_lines'], 1)
        self.assertEqual(report['parsed'], 3)
        self.assertEqual(report['duplicate_content'], 1)
        self.assertEqual(report['unparseable'], 0)
        self.assertEqual(report['parse_rate'], 1.0)

    def test_unparseable_line_is_counted_and_sampled(self):
        report = self.report_for('- ok line\nnospaceline\n')
        self.assertEqual(report['unparseable'], 1)
        self.assertEqual(report['unparseable_samples'], [{'line': 1, 'text': 'nospaceline'}])
        self.assertEqual(report['parse_rate'], 0.5)

    def test_line_range_is_respected(self):
        report = self.report_for('- a\n- b\n- c\n- d\n', start_line=1, end_line=3)
        self.assertEqual(report['lines_scanned'], 2)
        self.assertEqual(report['parsed'], 2)

    def test_empty_file_has_zero_rate(self):
        self.assertEqual(self.report_for('')['parse_rate'], 0.0)

    def test_write_report_round_trips(self):
        report = self.report_for('- a\n')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'r.json')
            write_report(report, path)
            with open(path) as fin:
                self.assertEqual(json.load(fin)['parsed'], 1)

    def test_guess_log_format(self):
        self.assertIn('<Level>', guess_log_format('BGL.log'))
        self.assertIn('<AdminAddr>', guess_log_format('Thunderbird.log'))
        with self.assertRaises(ValueError):
            guess_log_format('HDFS.log')


if __name__ == '__main__':
    unittest.main()
