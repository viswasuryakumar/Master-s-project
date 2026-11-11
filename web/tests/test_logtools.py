import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from web.logtools import LogRecord, LogSequence


def test_records_preserve_text_and_copy_metadata():
    attributes = {'host': 'worker'}
    record = LogRecord('  original text  ', attributes=attributes)
    attributes['host'] = 'other'
    assert record.message == '  original text  '
    assert record.attributes == {'host': 'worker'}
    assert LogSequence('batch', [record]).to_dict()['records'][0]['message'] == record.message


@pytest.mark.parametrize('kwargs', [{'message': ''}, {'message': 3}, {'message': 'x', 'timestamp': datetime(2026, 1, 1)}, {'message': 'x', 'attributes': {'n': 2}}])
def test_record_validation(kwargs):
    with pytest.raises(ValueError):
        LogRecord(**kwargs)


def test_sequence_requires_records_and_identifier():
    with pytest.raises(ValueError):
        LogSequence('empty', [])
    with pytest.raises(ValueError):
        LogSequence('', [LogRecord('x')])

from web.logtools.validation import validate_log_mapping, validate_log_rows


def test_validation_collects_row_and_field_errors():
    issues = validate_log_rows([{'message': 'ok'}, {'message': '', 'source': 1}, None])
    assert [(issue.row, issue.field) for issue in issues] == [(2, 'message'), (2, 'source'), (3, 'record')]
    assert issues[0].to_dict()['row'] == 2


def test_validation_enforces_length_nulls_and_metadata():
    assert validate_log_mapping({'message': 'long'}, max_message_length=3)
    assert validate_log_mapping({'message': 'x\x00'})
    assert validate_log_mapping({'message': 'ok', 'attributes': {'count': 1}})
    assert not validate_log_mapping({'message': ' ok ', 'attributes': {'count': '1'}})
    with pytest.raises(ValueError):
        validate_log_mapping({}, max_message_length=0)

from web.logtools.readers import read_records, LogReadError


def test_text_reader_handles_bom_and_blank_lines(tmp_path):
    path = tmp_path / 'input.log'
    path.write_bytes(b'\xef\xbb\xbf first \r\n\r\nsecond\n')
    assert [row.message for row in read_records(path)] == [' first ', 'second']
    with pytest.raises(LogReadError, match='record limit'):
        list(read_records(path, max_records=1))
    with pytest.raises(LogReadError, match='byte limit'):
        list(read_records(path, max_bytes=1))


def test_jsonl_reader_reports_line_numbers(tmp_path):
    path = tmp_path / 'input.jsonl'
    path.write_text('{"message":"ok","source":"host"}\n\nnot-json\n')
    iterator = read_records(path)
    assert next(iterator).source == 'host'
    with pytest.raises(LogReadError, match='line 3'):
        next(iterator)


def test_csv_reader_supports_multiline_messages(tmp_path):
    path = tmp_path / 'input.csv'
    path.write_text('message,source\n"first\nsecond",server\n')
    assert list(read_records(path))[0].message == 'first\nsecond'
    path.write_text('other\nvalue\n')
    with pytest.raises(LogReadError, match='message column'):
        list(read_records(path))
