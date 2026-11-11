"""Regression checks for ordered windows and evaluation batch boundaries."""
import unittest

import pandas as pd

from customDataset import CustomCollator
from eval import build_eval_dataloader
from prepareData.helper import fixedSize_window


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((list(messages), kwargs))
        return {'messages': list(messages)}


class SequenceExecutionTests(unittest.TestCase):
    def test_window_columns_do_not_depend_on_input_column_order(self):
        raw = pd.DataFrame({'Label': [0, 1, 0], 'Content': ['a', 'b', 'c'],
                            'timestamp': [10, 20, 30]})
        windows = fixedSize_window(raw, 2, 2)
        self.assertEqual(windows.columns.tolist(), ['Content', 'Label', 'item_Label'])
        self.assertEqual([list(x) for x in windows.Content], [['a', 'b'], ['c']])
        self.assertEqual(windows.Label.tolist(), [1, 0])
        self.assertEqual(windows.item_Label.tolist(), [[0, 1], [0]])

    def test_invalid_window_sizes_fail_clearly(self):
        raw = pd.DataFrame({'Content': ['a'], 'Label': [0]})
        for value in (0, -1, 1.5, True):
            for window, step in ((value, 1), (1, value)):
                with self.subTest(window=window, step=step):
                    with self.assertRaisesRegex(ValueError, 'positive integer'):
                        fixedSize_window(raw, window, step)

    def test_empty_window_input_and_missing_columns(self):
        self.assertTrue(fixedSize_window(pd.DataFrame(columns=['Content', 'Label']), 2, 2).empty)
        with self.assertRaisesRegex(ValueError, 'Missing window input columns'):
            fixedSize_window(pd.DataFrame({'Content': ['a']}), 2, 2)

    def test_collator_rejects_empty_inputs_and_invalid_limits(self):
        tokenizer = RecordingTokenizer()
        for kwargs in ({'max_seq_len': 0}, {'max_content_len': -1}, {'max_seq_len': 1.5}):
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                CustomCollator(tokenizer, **kwargs)
        collator = CustomCollator(tokenizer)
        for batch in ([], [([], 0)]):
            with self.assertRaisesRegex(ValueError, 'empty'):
                collator(batch)
        self.assertEqual(tokenizer.calls, [])

    def test_evaluation_preserves_order_boundaries_and_partial_batch(self):
        tokenizer = RecordingTokenizer()
        dataset = [(['a', 'b', 'discard'], 0), (['c'], 1), (['d', 'e'], 0)]
        batches = list(build_eval_dataloader(dataset, tokenizer, batch_size=2,
                                            max_seq_len=2, max_content_len=8))
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0]['inputs']['messages'], ['a', 'b', 'c'])
        self.assertEqual(batches[0]['seq_positions'].tolist(), [2])
        self.assertEqual(batches[0]['labels'].tolist(), ['normal', 'anomalous'])
        self.assertEqual(batches[1]['inputs']['messages'], ['d', 'e'])
        self.assertEqual(batches[1]['seq_positions'].tolist(), [])
        self.assertEqual(batches[1]['labels'].tolist(), ['normal'])
        self.assertEqual(tokenizer.calls[0][1], dict(return_tensors='pt', max_length=8,
                                                   padding=True, truncation=True))


if __name__ == '__main__':
    unittest.main()
