"""Token-length statistics for the BERT and Llama stages, to choose window and length limits.

BERT tokenizes every message and truncates it to max_content_len. Llama does not see message
tokens: each message becomes one projected vector, so a window's Llama input is a fixed prompt
(prefix, question, answer prefix) plus at most max_seq_len message vectors. The prompt pieces
below mirror LogAnomalyModel in model.py.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SEQUENCE_SEPARATOR = ' ;-; '
PREFIX_TEXT = 'Below is a sequence of system log messages:'
QUESTION_TEXT = '. Is this sequence normal or anomalous? \\n'
ANSWER_PREFIX_TEXT = 'The sequence is'
PERCENTILES = (50, 90, 95, 99)
TOKENIZE_CHUNK = 1024


def describe(values, limit=None):
    """Summarize integer lengths; with a limit, also count how many exceed it."""
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError('no values to describe')
    summary = {'count': int(values.size), 'mean': float(values.mean()), 'min': int(values.min()),
               **{f'p{p}': float(np.percentile(values, p)) for p in PERCENTILES},
               'max': int(values.max())}
    if limit is not None:
        over = int((values > limit).sum())
        summary.update({'limit': limit, 'over_limit': over, 'over_limit_fraction': over / values.size})
    return summary


def message_token_lengths(messages, tokenizer):
    """Untruncated token count (including special tokens) for every message."""
    lengths = []
    for start in range(0, len(messages), TOKENIZE_CHUNK):
        chunk = list(messages[start:start + TOKENIZE_CHUNK])
        encoded = tokenizer(chunk, add_special_tokens=True, truncation=False)['input_ids']
        lengths.extend(len(ids) for ids in encoded)
    return lengths


def llama_prompt_overhead(tokenizer):
    """Token counts of the fixed Llama prompt pieces, following LogAnomalyModel.forward."""
    prefix = tokenizer(PREFIX_TEXT)['input_ids']
    question = tokenizer(QUESTION_TEXT)['input_ids']
    bos = getattr(tokenizer, 'bos_token_id', None)
    if bos is not None and question and question[0] == bos:
        question = question[1:]
    answer_prefix = tokenizer(ANSWER_PREFIX_TEXT, add_special_tokens=False)['input_ids']
    pieces = {'prefix': len(prefix), 'question': len(question), 'answer_prefix': len(answer_prefix)}
    pieces['total'] = sum(pieces.values())
    return pieces


def token_length_report(sequences, bert_tokenizer, llama_tokenizer, max_content_len=100, max_seq_len=128):
    """Summarize BERT message lengths and Llama window lengths for a list of message lists."""
    if len(sequences) == 0:
        raise ValueError('no sequences')
    flat = [message for messages in sequences for message in messages[:max_seq_len]]
    message_lengths = np.asarray(message_token_lengths(flat, bert_tokenizer))
    truncated = np.minimum(message_lengths, max_content_len)

    kept = [min(len(messages), max_seq_len) for messages in sequences]
    boundaries = np.cumsum([0] + kept)
    bert_per_window = [int(truncated[boundaries[i]:boundaries[i + 1]].sum()) for i in range(len(sequences))]
    overhead = llama_prompt_overhead(llama_tokenizer)
    llama_lengths = [overhead['total'] + count for count in kept]

    return {
        'settings': {'max_content_len': max_content_len, 'max_seq_len': max_seq_len},
        'bert_message_tokens': describe(message_lengths, limit=max_content_len),
        'bert_message_tokens_after_truncation': describe(truncated),
        'messages_per_window': describe([len(messages) for messages in sequences], limit=max_seq_len),
        'bert_tokens_per_window': describe(bert_per_window),
        'llama_prompt_overhead': overhead,
        'llama_input_tokens_per_window': describe(llama_lengths),
    }
