"""Summary statistics for windowed log data, to compare window settings."""
import pandas as pd


def summarize_windows(df, content_col='Content', label_col='Label'):
    """Return length and label statistics for a dataframe of windows.

    :param df: dataframe with a sequence column (list/array per row) and a 0/1 window label
    :return: dict with num_windows, mean/max window length, and anomalous ratio
    """
    if len(df) == 0:
        return {'num_windows': 0, 'mean_length': 0.0, 'max_length': 0, 'anomalous_ratio': 0.0}
    lengths = df[content_col].apply(len)
    return {
        'num_windows': int(len(df)),
        'mean_length': float(lengths.mean()),
        'max_length': int(lengths.max()),
        'anomalous_ratio': float((df[label_col] == 1).mean()),
    }


def compare_settings(raw_data, settings, window_fn):
    """Run window_fn(raw_data, window_size, step_size) per setting and summarize each.

    :param settings: iterable of (window_size, step_size) pairs
    :return: dataframe with one row per setting
    """
    rows = []
    for window_size, step_size in settings:
        stats = summarize_windows(window_fn(raw_data, window_size, step_size))
        rows.append({'window_size': window_size, 'step_size': step_size, **stats})
    return pd.DataFrame(rows)
