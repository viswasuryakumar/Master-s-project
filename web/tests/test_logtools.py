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
