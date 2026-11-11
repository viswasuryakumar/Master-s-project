import json

import pytest
from backend.data_quality import summarize_sequences
from test_comparison import client, finished, new_run, seeded


def sequence(identifier, logs, truth=None, score=None):
    return {'external_id': identifier, 'logs': logs, 'truth': truth, 'score': score}


def test_quality_counts_labels_messages_and_lengths():
    report = summarize_sequences([
        sequence('one', ['start', 'stop'], 'normal', 0.8),
        sequence('two', ['error'], 'anomalous', 0.9),
        sequence('three', ['a', 'b', 'c', 'd']),
    ])
    assert report['total_sequences'] == 3
    assert report['total_messages'] == 7
    assert report['label_counts'] == {'normal': 1, 'anomalous': 1, 'unlabeled': 1, 'invalid': 0}
    assert report['label_coverage'] == pytest.approx(2 / 3)
    assert report['lengths'] == {'minimum': 1, 'maximum': 4, 'mean': pytest.approx(7 / 3), 'median': 2, 'p95': 4}
    assert report['scores'] == {'missing': 1, 'invalid': 0}
    assert [warning['code'] for warning in report['warnings']] == ['missing_labels']


def test_duplicate_labels_and_examples_are_reported():
    report = summarize_sequences([
        sequence('a', ['same', 'messages'], 'normal'),
        sequence('b', ['same', 'messages'], 'anomalous'),
        sequence('c', ['different'], 'normal'),
    ])
    assert report['duplicates'] == {
        'groups': 1, 'extra_sequences': 1, 'conflicting_groups': 1,
        'examples': [{'sequence_ids': ['a', 'b'], 'count': 2, 'labels': ['anomalous', 'normal']}],
    }
    assert {warning['code'] for warning in report['warnings']} == {'duplicate_sequences', 'conflicting_labels'}


def test_sequence_boundaries_and_message_order_are_preserved():
    report = summarize_sequences([
        sequence('joined', ['a\nb'], 'normal'),
        sequence('split', ['a', 'b'], 'normal'),
        sequence('reversed', ['b', 'a'], 'normal'),
    ])
    assert report['duplicates']['groups'] == 0
    assert report['status'] == 'ready'


def test_json_logs_and_unicode_are_supported():
    report = summarize_sequences([
        sequence('one', json.dumps(['日本語', 'café']), 'normal'),
        sequence('two', ['日本語', 'café'], 'normal'),
    ])
    assert report['duplicates']['groups'] == 1
    assert report['total_messages'] == 4


def test_empty_collection_has_finite_summary():
    report = summarize_sequences([])
    assert report['total_sequences'] == 0
    assert report['label_coverage'] == 0
    assert report['lengths'] == {'minimum': 0, 'maximum': 0, 'mean': 0, 'median': 0, 'p95': 0}
    assert report['warnings'][0]['code'] == 'empty_collection'
    json.dumps(report, allow_nan=False)


def test_quality_exposes_bad_existing_records_without_mutating_them():
    rows = [sequence('empty', [], 'unknown', float('nan')), sequence('blank', ['   '], score=2)]
    before = json.dumps(rows)
    report = summarize_sequences(rows)
    assert report['scores']['invalid'] == 2
    assert {'invalid_labels', 'empty_sequences', 'blank_messages', 'invalid_scores'} <= {
        warning['code'] for warning in report['warnings']
    }
    assert json.dumps(rows) == before


def test_example_lists_remain_bounded():
    report = summarize_sequences(sequence(str(index), ['same'], 'normal') for index in range(30))
    assert report['duplicates']['extra_sequences'] == 29
    assert len(report['duplicates']['examples'][0]['sequence_ids']) == 5


def test_quality_api_matches_collection_and_leaves_predictions_unchanged(client):
    run = new_run(client, [
        {'sequence_id': 'normal', 'logs': ['routine'], 'ground_truth': 'normal'},
        {'sequence_id': 'unknown', 'logs': ['connection refused']},
    ])
    finished(run)
    before = client.get(f'/api/runs/{run}/comparison/export?format=json').json()
    response = client.get(f'/api/runs/{run}/quality')
    assert response.status_code == 200
    report = response.json()
    assert report['run_id'] == run
    assert report['total_sequences'] == 2
    assert report['labeled_sequences'] == 1
    collection = client.get(f"/api/collections/{report['collection_id']}/quality").json()
    assert collection == {key: value for key, value in report.items() if key != 'run_id'}
    assert client.get(f'/api/runs/{run}/comparison/export?format=json').json() == before
    assert 'routine' not in response.text


@pytest.mark.parametrize('path', ['/api/runs/missing/quality', '/api/collections/missing/quality'])
def test_quality_api_missing_records_return_404(client, path):
    assert client.get(path).status_code == 404
