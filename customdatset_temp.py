"""Temporary copy of selected components from customDataset.py.

The existing module supplies normalization; the pipeline continues to use it.
"""

import numpy as np
import pandas as pd
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
