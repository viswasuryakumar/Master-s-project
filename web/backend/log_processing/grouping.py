import re
from collections import OrderedDict
from numbers import Integral

from .records import LogSequence


def window_records(records, size=100, step=None):
    step = size if step is None else step
    for name, value in (('size', size), ('step', step)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    rows = list(records)
    return [LogSequence(f'window-{index + 1:06}', tuple(rows[offset:offset + size]))
            for index, offset in enumerate(range(0, len(rows), step))]


def group_by_attribute(records, key, missing='error'):
    if missing not in ('error', 'skip'):
        raise ValueError('missing must be error or skip')
    if not isinstance(key, str) or not key:
        raise ValueError('key must contain text')
    groups = OrderedDict()
    for index, record in enumerate(records, 1):
        identifier = record.attributes.get(key)
        if not identifier or not identifier.strip():
            if missing == 'skip':
                continue
            raise ValueError(f'record {index} has no {key} attribute')
        groups.setdefault(identifier, []).append(record)
    return [LogSequence(identifier, tuple(rows)) for identifier, rows in groups.items()]


def group_hdfs_blocks(records, missing='error'):
    if missing not in ('error', 'skip'):
        raise ValueError('missing must be error or skip')
    groups = OrderedDict()
    for index, record in enumerate(records, 1):
        identifiers = sorted(set(re.findall(r'\bblk_-?\d+\b', record.message)))
        if not identifiers and missing == 'error':
            raise ValueError(f'record {index} has no HDFS block identifier')
        for identifier in identifiers:
            groups.setdefault(identifier, []).append(record)
    return [LogSequence(identifier, tuple(rows)) for identifier, rows in groups.items()]
