import json
import os
import tempfile
import unittest

from prepareData.masking import (DEFAULT_SOURCE, SEQUENCE_SEPARATOR, VARIANTS, Baseline, CachedMessages, CachedRows,
                                 Normalizer, Precompiled, audit_masking, benchmark, classify_residual,
                                 load_patterns, main, profile_patterns, read_rows)

SAMPLES = [
    'Connection from 10.251.43.21:50010 refused ... retry in 5 seconds',
    'ciod: Error reading message prefix after LOGIN_MESSAGE on CioStream [2] 0x1a2b3c4d',
    'machine check interrupt (bit=0x1d): L2 dcache unit data parity error',
    'instruction cache parity error corrected',
    'Received block blk_-12345 of size 67108864 from /10.250.19.102 on Monday',
    'user@host-1.example.com logged in at Nov 10 11:22:33 via /usr/lib/foo.so',
    'mac 00:1A:2B:3C:4D:5E and aa:bb:cc:dd:ee:ff:00:11:22:33:44:55 seen',
    '',
    'plain words only here',
    'a....b true False seventeen',
]


class PatternLoadingTests(unittest.TestCase):
    def test_patterns_come_from_customDataset_source(self):
        patterns = load_patterns()
        self.assertGreater(len(patterns), 10)
        self.assertTrue(all(isinstance(pattern, str) for pattern in patterns))

    def test_missing_patterns_list_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'x.py')
            with open(path, 'w') as stream:
                stream.write('other = []\n')
            with self.assertRaises(ValueError):
                load_patterns(path)

    def test_normalizer_source_still_matches_baseline(self):
        with open(DEFAULT_SOURCE) as stream:
            text = stream.read()
        for fragment in ("re.sub(r'[.]{3,}', '.. ', text)", "re.sub(combined_pattern, '<*>', compact_text)",
                         "'|'.join(patterns)"):
            self.assertIn(fragment, text)


class VariantTests(unittest.TestCase):
    def test_baseline_masks_numbers_addresses_and_hex(self):
        masked = Baseline(load_patterns())(SAMPLES[0])
        self.assertNotIn('10.251', masked)
        self.assertIn('<*>', masked)

    def test_every_variant_matches_baseline_on_rows(self):
        patterns = load_patterns()
        rows = [SEQUENCE_SEPARATOR.join(SAMPLES[:5]), SEQUENCE_SEPARATOR.join(SAMPLES[5:]), SAMPLES[0]]
        baseline = Baseline(patterns)
        for variant in VARIANTS:
            normalize = variant(patterns)
            self.assertEqual([normalize(row) for row in rows], [baseline(row) for row in rows], variant.name)

    def test_caches_report_hits_and_reset(self):
        patterns = load_patterns()
        rows = [SEQUENCE_SEPARATOR.join(['same message 1', 'same message 1'])] * 2
        for variant, hits, misses in ((CachedRows, 1, 1), (CachedMessages, 3, 1)):
            normalize = variant(patterns)
            for row in rows:
                normalize(row)
            self.assertEqual(normalize.cache_info(), {'hits': hits, 'misses': misses}, variant.name)
            normalize.reset()
            self.assertEqual(normalize.cache_info(), {'hits': 0, 'misses': 0})


class Unchanged(Normalizer):
    """A deliberately wrong variant: returns its input without masking anything."""
    name = 'unchanged'

    def __init__(self, patterns):
        pass

    def __call__(self, text):
        return text


class BenchmarkTests(unittest.TestCase):
    ROWS = ['value 12345 here', 'no digits']

    def test_one_result_per_variant_and_all_identical(self):
        # Repeat the rows so each run takes measurable time and the speedup is defined.
        results = benchmark(self.ROWS * 500, load_patterns(), repeat=1)
        self.assertEqual([entry['name'] for entry in results], [variant.name for variant in VARIANTS])
        self.assertTrue(all(entry['identical'] for entry in results))
        self.assertTrue(all(entry['seconds'] >= 0 for entry in results))
        by_name = {entry['name']: entry for entry in results}
        self.assertAlmostEqual(by_name['baseline']['speedup_vs_baseline'], 1.0)
        self.assertIsNone(by_name['baseline']['cache'])
        self.assertIsNotNone(by_name['cached_messages']['cache'])

    def test_changed_output_is_reported_as_a_mismatch(self):
        results = benchmark(self.ROWS, load_patterns(), repeat=1, variants=(Baseline, Unchanged))
        wrong = results[1]
        self.assertFalse(wrong['identical'])
        self.assertEqual(wrong['mismatches'], 1)
        self.assertEqual(wrong['first_mismatch']['row'], 0)
        self.assertEqual(wrong['first_mismatch']['got'], 'value 12345 here')

    def test_speedup_is_none_without_a_baseline(self):
        results = benchmark(self.ROWS, load_patterns(), repeat=1, variants=(Precompiled,))
        self.assertIsNone(results[0]['speedup_vs_baseline'])

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            benchmark([], load_patterns())
        with self.assertRaises(ValueError):
            benchmark(self.ROWS, load_patterns(), repeat=0)


class ProfileTests(unittest.TestCase):
    def test_counts_matches_per_pattern_and_sorts_by_time(self):
        entries = profile_patterns(['a1 b22', 'c333'], [r'\d+', r'zzz'], repeat=1)
        by_pattern = {entry['pattern']: entry for entry in entries}
        self.assertEqual(by_pattern[r'\d+']['matches'], 3)
        self.assertEqual(by_pattern['zzz']['matches'], 0)
        self.assertEqual(by_pattern[r'\d+']['index'], 0)
        seconds = [entry['seconds'] for entry in entries]
        self.assertEqual(seconds, sorted(seconds, reverse=True))
        self.assertTrue(all(0.0 <= entry['share'] <= 1.0 for entry in entries))

    def test_invalid_repeat_is_rejected(self):
        with self.assertRaises(ValueError):
            profile_patterns(['a'], ['a'], repeat=0)


def identity(text):
    return text


class AuditTests(unittest.TestCase):
    def test_classify_residual(self):
        self.assertEqual(classify_residual('abc1'), ['has_digit'])
        self.assertEqual(classify_residual('deadbeef'), ['hex_like'])
        self.assertEqual(classify_residual('decade,'), ['hex_like'])
        self.assertEqual(classify_residual('q' * 20), ['long'])
        self.assertEqual(classify_residual('<*>'), [])
        self.assertEqual(classify_residual('operation=<*>'), [])
        self.assertEqual(classify_residual('service'), [])

    def test_audit_counts_residual_and_singleton_tokens(self):
        long_token = 'q' * 20
        messages = ['service ok', 'service ok', 'cache decade failed', 'a very long identifier ' + long_token]
        report = audit_masking(messages, identity)
        self.assertEqual(report['messages'], 4)
        self.assertEqual(report['distinct_raw_messages'], 3)
        self.assertEqual(report['distinct_masked_messages'], 3)
        self.assertEqual(report['masks_per_message'], 0.0)
        self.assertEqual(report['messages_with_unmasked_tokens'], 2)
        self.assertEqual(report['fraction_with_unmasked_tokens'], 0.5)
        self.assertEqual(report['unmasked_occurrences_by_reason'], {'hex_like': 1, 'long': 1})
        self.assertEqual(report['distinct_unmasked_tokens'], 2)
        self.assertEqual([entry['token'] for entry in report['top_unmasked_tokens']], ['decade', long_token])
        self.assertEqual(report['distinct_tokens'], 10)
        self.assertEqual(report['singleton_tokens'], 8)
        self.assertAlmostEqual(report['singleton_fraction'], 0.8)
        self.assertIn('decade', report['singleton_samples'])

    def test_audit_counts_masks_per_message(self):
        report = audit_masking(['a7b7', 'c'], lambda text: text.replace('7', '<*>'))
        self.assertEqual(report['masks_per_message'], 1.0)
        self.assertEqual(report['unmasked_occurrences_by_reason'], {})

    def test_audit_flags_a_surviving_digit(self):
        report = audit_masking(['abc1 ok'], identity)
        self.assertEqual(report['unmasked_occurrences_by_reason'], {'has_digit': 1})

    def test_audit_rejects_no_messages(self):
        with self.assertRaises(ValueError):
            audit_masking([], identity)


class CliTests(unittest.TestCase):
    def write_csv(self, tmp):
        path = os.path.join(tmp, 'train.csv')
        with open(path, 'w') as stream:
            stream.write('Content,Label\n')
            stream.write('a 1 ;-; b 22,0\n')
            stream.write('c ;-; d,1\n')
        return path

    def test_read_rows_returns_one_string_per_window_and_honors_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_csv(tmp)
            self.assertEqual(read_rows(path), ['a 1 ;-; b 22', 'c ;-; d'])
            self.assertEqual(read_rows(path, limit=1), ['a 1 ;-; b 22'])

    def test_main_writes_all_three_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_csv(tmp)
            report = main(['--csv', path, '--repeat', '1'])
            with open(os.path.join(tmp, 'train_masking_report.json')) as stream:
                saved = json.load(stream)
            self.assertEqual(saved, report)
            self.assertEqual(saved['source'], {'csv': path, 'rows': 2, 'messages': 4})
            self.assertEqual([entry['name'] for entry in saved['benchmark']], [variant.name for variant in VARIANTS])
            self.assertTrue(all(entry['identical'] for entry in saved['benchmark']))
            self.assertEqual(len(saved['pattern_profile']), len(load_patterns()))
            self.assertEqual(saved['audit']['messages'], 4)

    def test_invalid_repeat_is_rejected(self):
        with self.assertRaises(SystemExit):
            main(['--csv', 'x.csv', '--repeat', '0'])


if __name__ == '__main__':
    unittest.main()
