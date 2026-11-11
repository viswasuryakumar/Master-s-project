import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import customDataset as dataset_api
from prepareData import helper
from prepareData.session_window import collect_block_sessions, partition_block_sessions
from prepareData.sliding_window import prepare_window_splits

ROOT = Path(__file__).resolve().parents[1]


class PipelineRefactorTests(unittest.TestCase):
    def test_legacy_imports_resolve_to_refactored_implementations(self):
        for old, new in (
            ('replace_patterns', 'normalize_log_message'), ('merge_data', 'flatten_log_sequences'),
            ('CustomDataset', 'LogSequenceDataset'), ('CustomCollator', 'LogBatchCollator'),
            ('BalancedSampler', 'ClassBalancedSampler'),
        ):
            self.assertIs(getattr(dataset_api, old), getattr(dataset_api, new))
        for old, new in (
            ('fixedSize_window', 'build_fixed_windows'), ('sliding_window', 'build_time_windows'),
            ('log_to_dataframe', 'read_log_records'), ('structure_log', 'write_structured_log'),
            ('generate_logformat_regex', 'compile_log_format'),
        ):
            self.assertIs(getattr(helper, old), getattr(helper, new))

    def test_window_split_does_not_cross_boundary_or_mutate_input(self):
        records = pd.DataFrame({'Content': list('abcdefghij'), 'Label': ['-'] * 9 + ['ERROR']})
        original = records.copy(deep=True)
        training, testing = prepare_window_splits(records, window_size=3, step_size=3)
        self.assertEqual([list(x) for x in training.Content], [list('abc'), list('def'), list('gh')])
        self.assertEqual([list(x) for x in testing.Content], [list('ij')])
        self.assertEqual(testing.Label.tolist(), [1])
        pd.testing.assert_frame_equal(records, original)

    def test_sessions_deduplicate_block_references_and_keep_message_order(self):
        records = pd.DataFrame({'Content': ['open blk_1 blk_1', 'write blk_1 blk_-2', 'no block']})
        labels = pd.DataFrame({'BlockId': ['blk_1', 'blk_-2'], 'Label': ['Normal', 'Anomaly']})
        sessions = collect_block_sessions(records, labels).set_index('BlockId')
        self.assertEqual(sessions.loc['blk_1', 'Content'], records.Content[:2].tolist())
        self.assertEqual(sessions.loc['blk_-2', 'Content'], [records.Content[1]])
        self.assertEqual(sessions.loc['blk_-2', 'Label'], 1)

    def test_session_partition_matches_existing_seeded_shuffle(self):
        sessions = pd.DataFrame({'BlockId': list('abcde'), 'Content': [['x']] * 5, 'Label': [0]*5})
        np.random.seed(17)
        expected = sessions.sample(frac=1).reset_index(drop=True)
        np.random.seed(17)
        train, test = partition_block_sessions(sessions)
        pd.testing.assert_frame_equal(train, expected.iloc[:4])
        pd.testing.assert_frame_equal(test, expected.iloc[4:].reset_index(drop=True))

    def test_time_windows_leave_source_deltas_unchanged(self):
        records = pd.DataFrame({'timestamp': [0, 1, 8], 'Label': [0, 1, 0],
                                'deltaT': [9, 1, 7], 'Content': ['a', 'b', 'c']})
        before = records.copy(deep=True)
        helper.build_time_windows(records, {'window_size': 3, 'step_size': 3})
        pd.testing.assert_frame_equal(records, before)

    def test_export_preserves_schema_delimiter_and_summary(self):
        records = pd.DataFrame({'Content': [['a', 'b'], ['c']], 'Label': [1, 0],
                                'item_Label': [[0, 1], [0]]})
        with tempfile.TemporaryDirectory() as directory:
            helper.export_sequence_split(records, directory, 'train')
            saved = pd.read_csv(Path(directory) / 'train.csv')
            self.assertEqual(saved.columns.tolist(), ['Content', 'Label', 'item_Label', 'session_length'])
            self.assertEqual(saved.Content.tolist(), ['a ;-; b', 'c'])
            self.assertEqual(saved.session_length.tolist(), [2, 1])
            self.assertEqual((Path(directory) / 'train_info.txt').read_text(),
                'max session length: 2; mean session length: 1.5\n'
                'number of anomalous sessions: 1; number of normal sessions: 1; number of total sessions: 2\n')
        self.assertEqual(records.Content[0], ['a', 'b'])

    def test_preparation_commands_run_as_module_and_script(self):
        for module, name in (('sliding_window', 'BGL.log'), ('session_window', 'HDFS.log')):
            for mode in ('module', 'script'):
                with self.subTest(module=module, mode=mode), tempfile.TemporaryDirectory() as directory:
                    folder = Path(directory)
                    if module == 'sliding_window':
                        lines = [f'- {i} date code time code comp comp INFO event-{i}\n' for i in range(10)]
                    else:
                        lines = [f'081109 203518 143 INFO dfs.DataNode: open blk_{i}\n' for i in range(10)]
                        pd.DataFrame({'BlockId': [f'blk_{i}' for i in range(10)],
                                      'Label': ['Normal']*10}).to_csv(folder/'anomaly_label.csv', index=False)
                    (folder/name).write_text(''.join(lines))
                    entry = ['-m', f'prepareData.{module}'] if mode == 'module' else [f'prepareData/{module}.py']
                    command = [sys.executable, *entry, '--data-dir', directory, '--log-name', name]
                    for cache_args in ([], ['--use-cache']):
                        result = subprocess.run(command + cache_args, cwd=ROOT, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                    train, test = (pd.read_csv(folder/f'{split}.csv') for split in ('train', 'test'))
                    if module == 'sliding_window':
                        self.assertEqual(train.session_length.tolist(), [8])
                        self.assertEqual(test.session_length.tolist(), [2])
                    else:
                        self.assertEqual((len(train), len(test)), (8, 2))
                        self.assertFalse(set(train.BlockId) & set(test.BlockId))


if __name__ == '__main__':
    unittest.main()
