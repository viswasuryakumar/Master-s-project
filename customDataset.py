import os
import re
from itertools import accumulate, chain
from numbers import Integral

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

patterns = [
    r'True',
    r'true',
    r'False',
    r'false',
    r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b',
    r'\b(Mon|Monday|Tue|Tuesday|Wed|Wednesday|Thu|Thursday|Fri|Friday|Sat|Saturday|Sun|Sunday)\b',
    r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+\b',
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?',
    r'([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}',
    r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}',
    r'[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*',
    r'\b[0-9a-fA-F]{8}\b',
    r'\b[0-9a-fA-F]{10}\b',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)',
    r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*',
]

combined_pattern = '|'.join(patterns)


def normalize_log_message(text):
    compact_text = re.sub(r'[.]{3,}', '.. ', text)
    return re.sub(combined_pattern, '<*>', compact_text)


class LogSequenceDataset(Dataset):
    def __init__(self, file_path, drop_duplicates=False, max_samples=None):
        records = pd.read_csv(file_path, nrows=max_samples)
        self._report_counts(records['Label'].values, 'in original dataset')
        records['Content'] = records['Content'].map(normalize_log_message)
        if drop_duplicates:
            records = records.drop_duplicates(subset='Content', keep='first')
        self.sequences = np.array(
            [message.split(' ;-; ') for message in records['Content']], dtype=object)
        self.labels = records['Label'].values
        if drop_duplicates:
            self._report_counts(self.labels, 'after dropping duplicates')

    @staticmethod
    def _report_counts(labels, description):
        for value, category in ((0, 'normal'), (1, 'anomalous')):
            print(f'Number of {category} samples {description}: {(labels == value).sum()}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def get_label(self):
        return self.labels


def flatten_log_sequences(data):
    sequences = list(data)
    boundaries = list(accumulate(map(len, sequences), initial=0))[:-1]
    return list(chain.from_iterable(sequences)), boundaries


class ClassBalancedSampler(Sampler):
    def __init__(self, dataset, target_ratio=0.3, max_samples=None, min_samples=50000):
        self.labels = dataset.get_label()
        self.dataset = dataset
        self.target_ratio = target_ratio
        self.max_samples = max_samples
        self.min_samples = min_samples

        self.normal_indices = np.where(self.labels == 0)[0]
        self.anomalous_indices = np.where(self.labels == 1)[0]

        self.minority_indices = (
            self.anomalous_indices if len(self.anomalous_indices) < len(self.normal_indices)
            else self.normal_indices
        )
        self.majority_indices = (
            self.normal_indices if self.minority_indices is self.anomalous_indices
            else self.anomalous_indices
        )

        self.minority_count = max(int((self.target_ratio * len(self.majority_indices)) / (1 - self.target_ratio)), len(self.minority_indices))
        self.total_size = self.minority_count + len(self.majority_indices)

        if self.max_samples is not None:
            if self.max_samples > self.total_size:
                raise ValueError(
            f"The hyperparameter 'max_samples' should smaller than the samples in the dataset.")
            self.total_size = self.max_samples

        elif self.total_size < self.min_samples:
            self.total_size = self.min_samples


    def __iter__(self):
        minority_repeats = np.tile(self.minority_indices, int(self.minority_count / len(self.minority_indices)))
        minority_remainder = np.random.choice(
            self.minority_indices,
            self.minority_count - len(minority_repeats),
            replace=False
        )
        sample_indices = np.concatenate([self.majority_indices, minority_repeats, minority_remainder])
        if len(sample_indices) > self.total_size:
            sample_indices = np.random.choice(
                sample_indices,
                self.total_size,
                replace=False
            )
        else:
            sample_indices = np.tile(sample_indices, int(self.total_size/len(sample_indices)))
            batch_remainder = np.random.choice(
                sample_indices,
                self.total_size-len(sample_indices),
                replace=False
            )
            sample_indices = np.concatenate([sample_indices, batch_remainder])
            np.random.shuffle(sample_indices)
        return iter(sample_indices)

    def __len__(self):
        return self.total_size


class LogBatchCollator:
    def __init__(self, tokenizer, max_seq_len=128, max_content_len=100):
        for name, value in (("max_seq_len", max_seq_len), ("max_content_len", max_content_len)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_content_len = max_content_len

    def __call__(self, batch):
        if not batch:
            raise ValueError("Cannot prepare an empty batch")
        message_groups, reference_labels = zip(*batch)
        for position, messages in enumerate(message_groups):
            if len(messages) == 0:
                raise ValueError(f"Sequence {position} is empty; every sequence needs a log message")
        bounded_groups = [messages[:self.max_seq_len] for messages in message_groups]
        message_rows, starts = flatten_log_sequences(bounded_groups)
        encoded_rows = self.tokenizer(
            message_rows, return_tensors="pt", max_length=self.max_content_len,
            padding=True, truncation=True)
        label_names = np.array(reference_labels).astype(object)
        label_names[label_names == 0] = 'normal'
        label_names[label_names == 1] = 'anomalous'
        return {
            "inputs": encoded_rows,
            "seq_positions": torch.tensor(starts[1:], dtype=torch.long),
            "labels": label_names,
        }



replace_patterns = normalize_log_message
merge_data = flatten_log_sequences
CustomDataset = LogSequenceDataset
BalancedSampler = ClassBalancedSampler
CustomCollator = LogBatchCollator
