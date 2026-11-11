import hashlib
import json
import math
from collections import Counter
from statistics import median

from fastapi import HTTPException


def summarize_sequences(rows):
    labels = Counter(normal=0, anomalous=0, unlabeled=0, invalid=0)
    lengths = []
    groups = {}
    empty_sequences = 0
    blank_messages = 0
    missing_scores = 0
    invalid_scores = 0

    for row in rows:
        logs = json.loads(row['logs']) if isinstance(row['logs'], str) else row['logs']
        lengths.append(len(logs))
        empty_sequences += not logs
        blank_messages += sum(not message.strip() for message in logs)
        truth = row.get('truth')
        label = 'unlabeled' if truth is None else truth if truth in ('normal', 'anomalous') else 'invalid'
        labels[label] += 1
        score = row.get('score')
        if score is None:
            missing_scores += 1
        elif not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            invalid_scores += 1
        fingerprint = hashlib.sha256(
            json.dumps(logs, ensure_ascii=False, separators=(',', ':')).encode('utf-8'),
        ).hexdigest()
        group = groups.setdefault(fingerprint, {'count': 0, 'ids': [], 'labels': set()})
        group['count'] += 1
        if len(group['ids']) < 5:
            group['ids'].append(row['external_id'])
        if truth in ('normal', 'anomalous'):
            group['labels'].add(truth)

    total = len(lengths)
    labeled = labels['normal'] + labels['anomalous']
    repeated = [group for group in groups.values() if group['count'] > 1]
    conflicting = [group for group in repeated if len(group['labels']) > 1]
    warnings = []

    def warn(code, count, message):
        if count:
            warnings.append({'code': code, 'count': count, 'message': message})

    warn('empty_collection', int(total == 0), 'The collection contains no sequences.')
    warn('missing_labels', labels['unlabeled'], 'Reference labels are missing; those sequences cannot contribute to evaluation metrics.')
    warn('invalid_labels', labels['invalid'], 'Some reference labels are neither normal nor anomalous.')
    warn('empty_sequences', empty_sequences, 'Some sequences contain no log messages.')
    warn('blank_messages', blank_messages, 'Some log messages contain only whitespace.')
    warn('duplicate_sequences', sum(group['count'] - 1 for group in repeated), 'Identical message sequences occur more than once and may overweight repeated evidence.')
    warn('conflicting_labels', len(conflicting), 'Identical sequences have different reference labels.')
    warn('invalid_scores', invalid_scores, 'Some supplied scores are not finite values between zero and one.')
    ordered = sorted(lengths)
    return {
        'total_sequences': total,
        'total_messages': sum(lengths),
        'labeled_sequences': labeled,
        'label_coverage': labeled / total if total else 0,
        'label_counts': dict(labels),
        'lengths': {
            'minimum': min(lengths) if total else 0,
            'maximum': max(lengths) if total else 0,
            'mean': sum(lengths) / total if total else 0,
            'median': median(lengths) if total else 0,
            'p95': ordered[math.ceil(total * 0.95) - 1] if total else 0,
        },
        'duplicates': {
            'groups': len(repeated),
            'extra_sequences': sum(group['count'] - 1 for group in repeated),
            'conflicting_groups': len(conflicting),
            'examples': [
                {'sequence_ids': group['ids'], 'count': group['count'], 'labels': sorted(group['labels'])}
                for group in repeated[:5]
            ],
        },
        'scores': {'missing': missing_scores, 'invalid': invalid_scores},
        'status': 'needs_review' if warnings else 'ready',
        'warnings': warnings,
    }


def collection_report(connection, collection_id):
    collection = connection.execute(
        'SELECT id, name, profile FROM v2_collections WHERE id=?', (collection_id,),
    ).fetchone()
    if collection is None:
        raise HTTPException(404, 'Collection not found')
    rows = connection.execute(
        'SELECT external_id, logs, truth, score FROM v2_seq WHERE collection_id=? ORDER BY ordinal',
        (collection_id,),
    )
    report = summarize_sequences(dict(row) for row in rows)
    return {**report, 'collection_id': collection['id'], 'name': collection['name'], 'profile': collection['profile']}


def install(app, db):
    @app.get('/api/collections/{collection_id}/quality')
    def collection_quality(collection_id: str):
        with db() as connection:
            return collection_report(connection, collection_id)

    @app.get('/api/runs/{run_id}/quality')
    def run_quality(run_id: str):
        with db() as connection:
            run = connection.execute('SELECT collection_id FROM v2_runs WHERE id=?', (run_id,)).fetchone()
            if run is None:
                raise HTTPException(404, 'Run not found')
            return {**collection_report(connection, run['collection_id']), 'run_id': run_id}
