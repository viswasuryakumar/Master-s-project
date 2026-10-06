"""Unit tests for customDataset (pattern masking, dataset, sampler, collator)."""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

try:
    import torch
    from customDataset import (BalancedSampler, CustomCollator, CustomDataset,
                               merge_data, replace_patterns)
except ImportError:  # torch is optional for the rest of the suite
    torch = None


@unittest.skipIf(torch is None, 'torch not installed')
class ReplacePatternsTests(unittest.TestCase):
    def test_masks_ip_addresses(self):
        self.assertEqual(replace_patterns('conn from 10.0.0.1:8080 failed'), 'conn from <*> failed')

    def test_masks_file_paths(self):
        self.assertIn('<*>', replace_patterns('open /var/log/syslog'))
        self.assertNotIn('/var', replace_patterns('open /var/log/syslog'))

    def test_masks_words_containing_digits(self):
        self.assertEqual(replace_patterns('node R02-M1 up'), 'node <*> up')

    def test_collapses_repeated_dots(self):
        self.assertEqual(replace_patterns('loading.....'), 'loading.. ')

    def test_plain_text_unchanged(self):
        self.assertEqual(replace_patterns('kernel panic'), 'kernel panic')


@unittest.skipIf(torch is None, 'torch not installed')
class MergeDataTests(unittest.TestCase):
    def test_merge_and_start_positions(self):
        merged, starts = merge_data([['a', 'b'], ['c'], ['d', 'e', 'f']])
        self.assertEqual(merged, list('abcdef'))
        self.assertEqual(starts, [0, 2, 3])


@unittest.skipIf(torch is None, 'torch not installed')
class CustomDatasetTests(unittest.TestCase):
    def write_csv(self, rows):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        path = os.path.join(d.name, 'train.csv')
        pd.DataFrame(rows, columns=['Content', 'Label']).to_csv(path, index=False)
        return path

    def test_splits_sequences_on_separator(self):
        ds = CustomDataset(self.write_csv([['a ;-; b ;-; c', 0], ['d', 1]]))
        self.assertEqual(len(ds), 2)
        seq, label = ds[0]
        self.assertEqual(list(seq), ['a', 'b', 'c'])
        self.assertEqual(label, 0)

    def test_drop_duplicates_after_masking(self):
        # 'x 1' and 'x 2' both become 'x <*>' after masking, so only one survives
        ds = CustomDataset(self.write_csv([['x 1', 0], ['x 2', 0], ['y', 1]]), drop_duplicates=True)
        self.assertEqual(len(ds), 2)

    def test_max_samples_limits_rows(self):
        ds = CustomDataset(self.write_csv([['a', 0], ['b', 0], ['c', 1]]), max_samples=2)
        self.assertEqual(len(ds), 2)


class FakeDataset:
    def __init__(self, labels):
        self.labels = np.array(labels)

    def get_label(self):
        return self.labels


@unittest.skipIf(torch is None, 'torch not installed')
class BalancedSamplerTests(unittest.TestCase):
    def test_minority_is_oversampled_towards_target_ratio(self):
        labels = [0] * 90 + [1] * 10
        sampler = BalancedSampler(FakeDataset(labels), target_ratio=0.3, min_samples=0)
        picked = np.array(list(sampler))
        self.assertEqual(len(picked), len(sampler))
        ratio = (np.array(labels)[picked] == 1).mean()
        self.assertAlmostEqual(ratio, 0.3, delta=0.05)

    def test_max_samples_larger_than_dataset_raises(self):
        with self.assertRaises(ValueError):
            BalancedSampler(FakeDataset([0] * 9 + [1]), target_ratio=0.3, max_samples=10 ** 6)


@unittest.skipIf(torch is None, 'torch not installed')
class CustomCollatorTests(unittest.TestCase):
    def test_collates_sequences_and_labels(self):
        calls = {}

        def fake_tokenizer(data, **kwargs):
            calls['data'] = data
            return {'input_ids': torch.zeros(len(data), 2, dtype=torch.long)}

        batch = [(np.array(['a', 'b', 'c']), 0), (np.array(['d']), 1)]
        out = CustomCollator(fake_tokenizer, max_seq_len=2)(batch)
        self.assertEqual(calls['data'], ['a', 'b', 'd'])  # first sequence truncated to 2
        self.assertEqual(out['seq_positions'].tolist(), [2])
        self.assertEqual(list(out['labels']), ['normal', 'anomalous'])


if __name__ == '__main__':
    unittest.main()
