import csv
import json
from pathlib import Path

from .records import LogRecord
from .validation import validate_log_mapping


class LogReadError(ValueError):
    pass


def record_from_mapping(value, line_number):
    if isinstance(value, dict) and isinstance(value.get('attributes'), str):
        value = dict(value)
        try:
            value['attributes'] = json.loads(value['attributes']) if value['attributes'] else {}
        except json.JSONDecodeError as error:
            raise LogReadError(f'line {line_number}: invalid attributes JSON') from error
    issues = validate_log_mapping(value, line_number)
    if issues:
        raise LogReadError('; '.join(f'line {issue.row}: {issue.field}: {issue.message}' for issue in issues))
    return LogRecord(value['message'], value.get('source', ''), value.get('severity', 'UNKNOWN'),
                     attributes=value.get('attributes', {}))


def read_records(path, format=None, max_records=None, max_bytes=20_000_000):
    path = Path(path)
    for name, limit in (('max_bytes', max_bytes), ('max_records', max_records)):
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError(f'{name} must be a positive integer')
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise LogReadError('input exceeds byte limit')
    selected = format or {'.jsonl': 'jsonl', '.csv': 'csv', '.log': 'text', '.txt': 'text'}.get(path.suffix.lower())
    if selected not in ('text', 'jsonl', 'csv'):
        raise LogReadError('select text, jsonl, or csv format')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        if selected == 'csv':
            reader = csv.DictReader(stream)
            if not reader.fieldnames or 'message' not in reader.fieldnames:
                raise LogReadError('CSV requires a message column')
            rows = ((reader.line_num, row) for row in reader)
        else:
            rows = enumerate(stream, 1)
        count = 0
        for line_number, raw in rows:
            if selected != 'csv' and not raw.strip():
                continue
            if max_records is not None and count >= max_records:
                raise LogReadError('input exceeds record limit')
            if selected == 'jsonl':
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise LogReadError(f'line {line_number}: invalid JSON') from error
            elif selected == 'text':
                value = {'message': raw.rstrip('\r\n')}
            else:
                value = raw
            yield record_from_mapping(value, line_number)
            count += 1
