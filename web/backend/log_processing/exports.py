import csv
import io
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


COLUMNS = ('message', 'source', 'severity', 'timestamp', 'attributes')


def spreadsheet_safe(value):
    text = str(value)
    return "'" + text if text.lstrip().startswith(('=', '+', '-', '@')) or text.startswith(('\t', '\r')) else text


def records_to_csv(records):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=COLUMNS)
    writer.writeheader()
    for record in records:
        row = record.to_dict()
        row['timestamp'] = row['timestamp'] or ''
        row['attributes'] = json.dumps(row['attributes'], ensure_ascii=False, sort_keys=True)
        writer.writerow({key: spreadsheet_safe(value) for key, value in row.items()})
    return output.getvalue()


def records_to_jsonl(records):
    return ''.join(json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False) + '\n' for record in records)


def export_records(records, path, format='jsonl', overwrite=False):
    if format not in ('jsonl', 'csv'):
        raise ValueError('format must be jsonl or csv')
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=path.parent,
                                prefix='.' + path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            if format == 'csv':
                stream.write(records_to_csv(records))
            else:
                for record in records:
                    stream.write(json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
