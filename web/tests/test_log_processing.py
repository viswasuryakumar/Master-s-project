import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.log_records import read_log_lines
from backend.sequence_grouping import build_message_windows, group_block_messages


def test_read_lines_retains_message_whitespace_and_order():
    assert read_log_lines('  first  \r\n\t\r\nsecond\n\n') == ['  first  ', 'second']
    assert read_log_lines(' \n\t') == []


def test_windows_retain_partial_tail_and_stable_identifiers():
    messages = [str(index) for index in range(201)]
    windows = build_message_windows(messages)
    assert [row['external_id'] for row in windows] == ['window-000001', 'window-000002', 'window-000003']
    assert [len(row['logs']) for row in windows] == [100, 100, 1]
    assert [line for row in windows for line in row['logs']] == messages
    assert build_message_windows([]) == []


@pytest.mark.parametrize('size', [0, -1, True, 1.5])
def test_invalid_window_size(size):
    with pytest.raises(ValueError, match='positive integer'):
        build_message_windows(['message'], size)


def test_block_grouping_retains_order_and_deduplicates_references():
    messages = ['open blk_2 blk_-1 blk_2', 'close blk_2']
    groups = group_block_messages(messages)
    assert [row['block_id'] for row in groups] == ['blk_-1', 'blk_2']
    assert groups[0]['logs'] == [messages[0]]
    assert groups[1]['logs'] == messages
    assert group_block_messages([]) == []


def test_block_grouping_rejects_missing_identifier():
    with pytest.raises(ValueError, match='Every HDFS raw message'):
        group_block_messages(['open blk_2', 'missing identifier'])
