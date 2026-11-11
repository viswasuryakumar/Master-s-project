import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from web.backend.log_processing import LogRecord, LogSequence


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

from web.backend.log_processing.validation import validate_log_mapping, validate_log_rows


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

from web.backend.log_processing.readers import read_records, LogReadError


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

from web.backend.log_processing.timestamps import parse_timestamp, time_bounds


def test_timestamp_offsets_epoch_and_explicit_local_formats():
    expected = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert parse_timestamp('2026-01-01T02:00:00+02:00') == expected
    assert parse_timestamp(expected.timestamp()) == expected
    assert parse_timestamp('2026-01-01-00.00.00.000000', timezone.utc) == expected
    assert parse_timestamp('260101 000000', timezone.utc) == expected
    assert time_bounds([LogRecord('a'), LogRecord('b', timestamp=expected)]) == (expected, expected)


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), 'bad', '2026-01-01'])
def test_invalid_or_ambiguous_timestamps(value):
    with pytest.raises(ValueError):
        parse_timestamp(value)


def test_reader_retains_timestamp(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_text('{"message":"ok","timestamp":"2026-01-01T00:00:00Z"}\n')
    assert list(read_records(path))[0].timestamp == parse_timestamp('2026-01-01T00:00:00Z')

from web.backend.log_processing.grouping import window_records, group_by_attribute, group_hdfs_blocks


def test_overlapping_windows_and_tail():
    rows = [LogRecord(str(i)) for i in range(5)]
    assert [[r.message for r in s.records] for s in window_records(rows, 3, 2)] == [['0','1','2'],['2','3','4'],['4']]
    assert window_records([]) == []
    with pytest.raises(ValueError):
        window_records(rows, size=True)


def test_attribute_grouping_and_missing_policy():
    rows = [LogRecord('a', attributes={'request': 'x'}), LogRecord('b'), LogRecord('c', attributes={'request': 'x'})]
    assert [r.message for r in group_by_attribute(rows, 'request', 'skip')[0].records] == ['a','c']
    with pytest.raises(ValueError, match='record 2'):
        group_by_attribute(rows, 'request')


def test_hdfs_grouping_deduplicates_and_keeps_order():
    rows = [LogRecord('blk_1 blk_1 blk_-2'), LogRecord('blk_1')]
    groups = group_hdfs_blocks(rows)
    assert [s.identifier for s in groups] == ['blk_-2','blk_1']
    assert len(groups[1].records) == 2
    assert group_hdfs_blocks([LogRecord('ordinary')], missing='skip') == []

from web.backend.log_processing.filters import filter_records


def test_filters_combine_source_text_severity_and_time():
    stamp = parse_timestamp('2026-01-01T00:00:00Z')
    records = [LogRecord('Timeout', 'a', 'WARNING', stamp), LogRecord('timeout', 'b', 'ERROR', stamp), LogRecord('timeout', 'a')]
    assert list(filter_records(records, source='a', contains='TIME', minimum_severity='warn', start=stamp, end=stamp)) == records[:1]
    assert list(filter_records(records, contains='TIME', case_sensitive=True)) == []
    with pytest.raises(ValueError):
        list(filter_records(records, minimum_severity='bad'))
    with pytest.raises(ValueError):
        list(filter_records(records, start='2026-02-01T00:00:00Z', end=stamp))

from web.backend.log_processing.statistics import summarize_records, summarize_sequences


def test_statistics_counts_duplicates_and_missing_times():
    records = [LogRecord('same', 'b', 'WARNING'), LogRecord('same', 'a', 'WARN'), LogRecord('unique', timestamp=parse_timestamp(0))]
    summary = summarize_records(records)
    assert summary['records'] == 3
    assert summary['duplicate_messages'] == 1
    assert summary['severity_counts']['WARN'] == 2
    assert summary['missing_timestamps'] == 2
    assert summary['records_per_second'] is None
    assert summary['repeated_messages'] == [{'message': 'same', 'count': 2}]


def test_empty_and_sequence_statistics():
    assert summarize_records([])['first_timestamp'] is None
    assert summarize_sequences([])['mean_length'] is None
    records = [LogRecord(str(i)) for i in range(5)]
    summary = summarize_sequences(window_records(records, 3))
    assert summary['record_memberships'] == 5
    assert summary['median_length'] == 2.5

from web.backend.log_processing.redaction import redact_text, redact_records


def test_redaction_covers_assignments_bearer_email_and_ipv4():
    result = redact_text('password="example value" Bearer example-token user@example.org 10.2.3.4 999.2.3.4')
    assert 'example value' not in result.text
    assert 'example-token' not in result.text
    assert '[EMAIL]' in result.text and '[IP]' in result.text
    assert '999.2.3.4' in result.text
    assert result.counts == {'secret': 1, 'bearer': 1, 'email': 1, 'address': 1}


def test_redaction_leaves_original_record_and_masks_metadata():
    record = LogRecord('api_key=placeholder', source='10.0.0.1', attributes={'password': 'example'})
    rows, counts = redact_records([record])
    assert record.message == 'api_key=placeholder'
    assert record.attributes['password'] == 'example'
    assert rows[0].attributes['password'] == '[REDACTED]'
    assert counts['secret'] == 2
    assert redact_text('10.0.0.1', mask_addresses=False).text == '10.0.0.1'

import csv
import io
from web.backend.log_processing.exports import records_to_csv, records_to_jsonl, export_records


def test_csv_export_protects_formula_cells_and_quotes_messages():
    rows = [LogRecord('=SUM(1,2)', source=' +command'), LogRecord('line\n"quoted"')]
    parsed = list(csv.DictReader(io.StringIO(records_to_csv(rows))))
    assert parsed[0]['message'].startswith("'=")
    assert parsed[0]['source'].startswith("' +")
    assert parsed[1]['message'] == 'line\n"quoted"'


def test_jsonl_export_round_trip_and_no_clobber(tmp_path):
    rows = [LogRecord('original', timestamp=parse_timestamp(0), attributes={'key': 'value'})]
    path = tmp_path / 'records.jsonl'
    export_records(rows, path)
    assert list(read_records(path)) == rows
    with pytest.raises(FileExistsError):
        export_records([LogRecord('replacement')], path)
    assert list(read_records(path)) == rows
    export_records([LogRecord('replacement')], path, overwrite=True)
    assert list(read_records(path))[0].message == 'replacement'
    assert not list(tmp_path.glob('.records*'))


def test_failed_export_does_not_leave_partial_output(tmp_path):
    def broken():
        yield LogRecord('first')
        raise ValueError('interrupted')
    with pytest.raises(ValueError):
        export_records(broken(), tmp_path / 'out.jsonl')
    assert not list(tmp_path.iterdir())
