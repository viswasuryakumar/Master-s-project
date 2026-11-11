import os
import tempfile
import unittest

from prepareData.masking import (DEFAULT_SOURCE, SEQUENCE_SEPARATOR, VARIANTS, Baseline, CachedMessages, CachedRows,
                                 load_patterns)

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


if __name__ == '__main__':
    unittest.main()
