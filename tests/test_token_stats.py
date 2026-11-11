import json
import os
import tempfile
import unittest

from prepareData.token_stats import (ANSWER_PREFIX_TEXT, PREFIX_TEXT, QUESTION_TEXT, describe,
                                     llama_prompt_overhead, main, message_token_lengths, read_sequences,
                                     token_length_report)

BOS = 1
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeBert:
    """One token per word plus [CLS] and [SEP]."""

    def __call__(self, texts, add_special_tokens=True, truncation=False):
        extra = 2 if add_special_tokens else 0
        return {'input_ids': [[0] * (len(text.split()) + extra) for text in texts]}


class FakeLlama:
    """One token per word, plus a BOS token when special tokens are added."""
    bos_token_id = BOS

    def __call__(self, text, add_special_tokens=True):
        ids = [7] * len(text.split())
        return {'input_ids': ([BOS] + ids) if add_special_tokens else ids}


class DescribeTests(unittest.TestCase):
    def test_percentiles_and_limit(self):
        summary = describe(list(range(1, 101)), limit=90)
        self.assertEqual((summary['count'], summary['min'], summary['max']), (100, 1, 100))
        self.assertAlmostEqual(summary['mean'], 50.5)
        self.assertAlmostEqual(summary['p50'], 50.5)
        self.assertEqual(summary['over_limit'], 10)
        self.assertAlmostEqual(summary['over_limit_fraction'], 0.1)

    def test_no_limit_means_no_over_limit_keys(self):
        self.assertNotIn('over_limit', describe([1, 2, 3]))

    def test_empty_is_rejected(self):
        with self.assertRaises(ValueError):
            describe([])


class LengthTests(unittest.TestCase):
    def test_message_token_lengths_include_special_tokens(self):
        self.assertEqual(message_token_lengths(['a b c', 'x'], FakeBert()), [5, 3])

    def test_llama_overhead_drops_the_question_bos_only(self):
        overhead = llama_prompt_overhead(FakeLlama())
        self.assertEqual(overhead['prefix'], 1 + len(PREFIX_TEXT.split()))
        self.assertEqual(overhead['question'], len(QUESTION_TEXT.split()))
        self.assertEqual(overhead['answer_prefix'], len(ANSWER_PREFIX_TEXT.split()))
        self.assertEqual(overhead['total'], overhead['prefix'] + overhead['question'] + overhead['answer_prefix'])

    def test_prompt_text_matches_model_py(self):
        with open(os.path.join(REPO_ROOT, 'model.py')) as source:
            text = source.read()
        for piece in (PREFIX_TEXT, QUESTION_TEXT, ANSWER_PREFIX_TEXT):
            self.assertIn(piece, text.replace('\\\\n', '\\n'))


class ReportTests(unittest.TestCase):
    def test_truncation_and_window_cap(self):
        sequences = [['a b', 'c d e f f'], ['x'] * 5]
        report = token_length_report(sequences, FakeBert(), FakeLlama(), max_content_len=5, max_seq_len=3)
        bert = report['bert_message_tokens']
        # Only the first three messages of the five-message window are kept: 2 + 3 messages.
        self.assertEqual((bert['count'], bert['max'], bert['over_limit']), (5, 7, 1))
        self.assertEqual(report['bert_message_tokens_after_truncation']['max'], 5)
        self.assertEqual(report['messages_per_window']['over_limit'], 1)
        overhead = report['llama_prompt_overhead']['total']
        llama = report['llama_input_tokens_per_window']
        self.assertEqual((llama['min'], llama['max']), (overhead + 2, overhead + 3))

    def test_bert_tokens_per_window_use_truncated_lengths(self):
        report = token_length_report([['a b', 'c d e f f']], FakeBert(), FakeLlama(), max_content_len=5, max_seq_len=10)
        self.assertEqual(report['bert_tokens_per_window']['max'], 4 + 5)

    def test_no_sequences_is_rejected(self):
        with self.assertRaises(ValueError):
            token_length_report([], FakeBert(), FakeLlama())


class CliTests(unittest.TestCase):
    def write_csv(self, tmp):
        path = os.path.join(tmp, 'train.csv')
        with open(path, 'w') as stream:
            stream.write('Content,Label,item_Label,session_length\n')
            stream.write('a b ;-; c d e,0,"[0, 0]",2\n')
            stream.write('x y z w ;-; q,1,"[0, 1]",2\n')
        return path

    def test_read_sequences_splits_normalizes_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_csv(tmp)
            self.assertEqual(read_sequences(path), [['a b', 'c d e'], ['x y z w', 'q']])
            self.assertEqual(read_sequences(path, normalize=str.upper, limit=1), [['A B', 'C D E']])

    def test_main_writes_report_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_csv(tmp)
            report = main(['--csv', path, '--bert-path', 'b', '--llama-path', 'l', '--no-normalize',
                           '--max-content-len', '4', '--max-seq-len', '2'],
                          load_tokenizers=lambda bert_path, llama_path: (FakeBert(), FakeLlama()))
            with open(os.path.join(tmp, 'train_token_stats.json')) as stream:
                saved = json.load(stream)
            self.assertEqual(saved, report)
            self.assertEqual(saved['source'], {'csv': path, 'normalized': False, 'windows': 2})
            self.assertEqual(saved['bert_message_tokens']['count'], 4)
            # Token counts are 4, 5, 6 and 3 with [CLS]/[SEP]; two exceed the limit of 4.
            self.assertEqual(saved['bert_message_tokens']['over_limit'], 2)

    def test_invalid_limits_are_rejected(self):
        with self.assertRaises(SystemExit):
            main(['--csv', 'x', '--bert-path', 'b', '--llama-path', 'l', '--max-seq-len', '0'])


if __name__ == '__main__':
    unittest.main()
