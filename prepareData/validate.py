"""Validation of structured log dataframes before windowing."""
import pandas as pd

REQUIRED_COLUMNS = ('Label', 'Content')


class LogValidationError(ValueError):
    pass


def validate_logs(df, required=REQUIRED_COLUMNS, allow_empty_content=False):
    """Raise LogValidationError if df is not safe to window.

    Checks: required columns exist, df is non-empty, Label contains only 0/1,
    and (unless allowed) Content has no null or blank messages.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise LogValidationError('missing columns: %s' % ', '.join(missing))
    if len(df) == 0:
        raise LogValidationError('dataframe is empty')
    bad = set(pd.unique(df['Label'])) - {0, 1}
    if bad:
        raise LogValidationError('Label must be 0 or 1, found: %s' % sorted(map(str, bad)))
    if not allow_empty_content:
        content = df['Content']
        n_bad = int((content.isna() | (content.fillna('').astype(str).str.strip() == '')).sum())
        if n_bad:
            raise LogValidationError('Content has %d null/blank messages' % n_bad)
    return df
