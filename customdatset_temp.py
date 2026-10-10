"""Temporary copy of selected components from customDataset.py.

The existing module supplies normalization; the pipeline continues to use it.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from customDataset import replace_patterns


class CustomDataset(Dataset):
    def __init__(self, file_path, drop_duplicates=False, max_samples=None):
        # Bound the CSV read before expensive normalization for local smoke runs.
        df = pd.read_csv(file_path, nrows=max_samples)
        print('Number of normal samples in original dataset: {}'.format((df['Label'].values==0).sum()))
        print('Number of anomalous samples in original dataset: {}'.format((df['Label'].values==1).sum()))
        df['Content'] = df['Content'].apply(replace_patterns)
        if drop_duplicates:
            df = df.drop_duplicates(subset='Content', keep='first')
        contents = df['Content'].values
        self.sequences = np.array([content.split(' ;-; ') for content in contents], dtype=object)
        self.labels = df['Label'].values
        if drop_duplicates:
            print('Number of normal samples after dropping duplicates: {}'.format((self.labels==0).sum()))
            print('Number of anomalous samples after dropping duplicates: {}'.format((self.labels==1).sum()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def get_label(self):
        return self.labels


def merge_data(data):
    merged_data = []

    # 记录每个子列表的开始位置
    start_positions = []

    current_position = 0
    for sublist in data:
        start_positions.append(current_position)
        merged_data.extend(sublist)
        current_position += len(sublist)

    return merged_data, start_positions


class CustomCollator:
    def __init__(self, tokenizer, max_seq_len=128, max_content_len=100):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_content_len = max_content_len

    def __call__(self, batch):
        sequences_, labels = zip(*batch)

        # 截断每个子序列的长度
        sequences = [seq[:self.max_seq_len] for seq in sequences_]

        data, seq_positions = merge_data(sequences)
        seq_positions = seq_positions[1:]  # 去掉第一个0位置，用于后续分界处理

        # 将合并后的 data 编码
        inputs = self.tokenizer(
            data,
            return_tensors="pt",
            max_length=self.max_content_len,
            padding=True,
            truncation=True
        )

        # 构建 label tensor
        # labels_tensor = torch.tensor(labels, dtype=torch.long)

        labels = np.array(labels).astype(object)

        labels[labels == 0] = 'normal'
        labels[labels == 1] = 'anomalous'

        return {
            "inputs": inputs,
            "seq_positions": torch.tensor(seq_positions, dtype=torch.long),
            # "labels": labels_tensor
            "labels": labels
        }
