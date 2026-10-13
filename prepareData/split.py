"""Chronological train/test split with a leakage check."""


def chronological_split(df, train_ratio=0.8):
    """Split by row order (logs are time ordered); never shuffle.

    :return: (train_df, test_df); test_df has its index reset
    """
    if not 0 < train_ratio < 1:
        raise ValueError('train_ratio must be between 0 and 1 (exclusive)')
    train_len = int(train_ratio * len(df))
    return df[:train_len], df[train_len:].reset_index(drop=True)


def assert_no_overlap(train_windows, test_windows, content_col='Content'):
    """Raise if any window (as a tuple of messages) appears in both sets.

    Overlapping windows inside one split are fine; sharing across the split is leakage.
    """
    to_keys = lambda df: {tuple(seq) for seq in df[content_col]}
    shared = to_keys(train_windows) & to_keys(test_windows)
    if shared:
        raise ValueError('%d window(s) appear in both train and test' % len(shared))
