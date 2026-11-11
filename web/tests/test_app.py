"""API tests for the log observatory: seeding, uploads, streaming, pagination, chat and archives."""
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import app as module
from backend.engine import classify, generate

PROFILES = ['BGL', 'HDFS', 'Liberty', 'Thunderbird']
SEEDED_MESSAGES_PER_PROFILE = 12000
SEEDED_MESSAGES_PER_SERVER = 10000
HDFS_SEQUENCE_LENGTHS = {20, 30, 40, 50, 60}
MAX_TICKS = 50


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'DB_PATH', tmp_path / 'test.sqlite3')
    monkeypatch.setenv('OBSERVATORY_NO_WORKERS', '1')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with TestClient(module.app) as test_client:
        yield test_client


def run_status(client, run):
    return client.get(f'/api/runs/{run}').json()['status']


def upload(client, **fields):
    """Create a collection, analyze it to completion, and return (run id, result rows)."""
    created = client.post('/api/collections', json={'name': 'Test', 'profile': 'BGL', **fields})
    assert created.status_code == 200, created.text
    run = client.post(f"/api/collections/{created.json()['id']}/analyze").json()['id']
    assert run_status(client, run) == 'queued'
    module.tick()
    assert run_status(client, run) == 'processing'
    for _ in range(MAX_TICKS):
        module.tick()
        if run_status(client, run) == 'completed':
            break
    assert run_status(client, run) == 'completed'
    rows = client.get('/api/results', params={'run_id': run, 'size': 100}).json()['items']
    return run, rows


def first_result(client, **params):
    return client.get('/api/results', params={'size': 1, **params}).json()['items'][0]


def count(connection, sql):
    return connection.execute(sql).fetchone()[0]


def test_seed_volumes_and_formats(client):
    state = client.get('/api/overview').json()
    assert len(state['profiles']) == len(PROFILES)
    assert all(p['messages'] == SEEDED_MESSAGES_PER_PROFILE for p in state['profiles'])
    assert all(s['messages'] == SEEDED_MESSAGES_PER_SERVER for s in state['servers'])
    assert state['chat_provider'] == 'local'
    with module.db() as c:
        lengths = {r[0] for r in c.execute("SELECT DISTINCT length FROM v2_seq WHERE profile='HDFS'")}
        assert lengths == HDFS_SEQUENCE_LENGTHS
        assert count(c, "SELECT COUNT(*) FROM v2_seq WHERE profile='HDFS' AND block_id IS NULL") == 0


@pytest.mark.parametrize('profile', PROFILES)
def test_raw_grouping_and_progress(client, profile):
    sequences = [generate(profile, i) for i in range(4)]
    raw = '\n'.join(line for seq in sequences for line in seq['logs'])
    run, rows = upload(client, profile=profile, raw_text=raw)
    assert len(rows) == 4
    assert sum(r['prediction'] == 'anomalous' for r in rows) == 1
    exported = client.get(f'/api/runs/{run}/export?format=json').json()
    assert exported[0]['logs'] == sequences[0]['logs']
    assert all(r['ground_truth'] is None for r in exported)
    assert all(r['processing_origin'] == 'profile_simulation' for r in exported)


def test_truth_is_not_prediction_and_imported_precedence(client):
    run, rows = upload(client, sequences=[
        {'sequence_id': 'a', 'logs': ['routine activity'], 'ground_truth': 'anomalous'},
        {'sequence_id': 'b', 'logs': ['connection refused'], 'prediction': 'normal',
         'ground_truth': 'anomalous', 'score': 0.1},
        {'sequence_id': 'c', 'logs': ['connection refused'], 'ground_truth': 'normal'},
    ])
    by_id = {r['external_id']: r for r in rows}
    assert by_id['a']['prediction'] == 'normal'
    assert by_id['b']['prediction'] == 'normal' and by_id['b']['origin'] == 'imported_prediction'
    assert by_id['c']['prediction'] == 'anomalous'
    assert client.get(f'/api/runs/{run}').json()['metrics']['fn'] == 2
    assert 'processing_origin' in client.get(f'/api/runs/{run}/export').text

    # Benchmark label annotations must not become classifier features.
    labeled = generate('BGL', 0)['logs'][0].replace('- ', 'UNCORRECTABLE ', 1)
    assert classify('BGL', [labeled])[0] == 'normal'


def test_stream_restart_no_duplicates_and_independent_controls(client):
    stopped = client.get('/api/overview').json()['servers'][0]
    control = f"/api/servers/{stopped['id']}/control"
    client.post(control, json={'action': 'stop'})
    module.tick()
    module.initialize()
    module.tick()

    servers = client.get('/api/overview').json()['servers']
    assert servers[0]['messages'] == SEEDED_MESSAGES_PER_SERVER
    assert all(s['messages'] == SEEDED_MESSAGES_PER_SERVER + 20 for s in servers[1:])
    with module.db() as c:
        assert count(c, 'SELECT COUNT(*) FROM v2_events') == 30040
        distinct = 'SELECT COUNT(*) FROM (SELECT run_id,seq_id,line_index FROM v2_events GROUP BY run_id,seq_id,line_index)'
        assert count(c, distinct) == 30040

    client.post(control, json={'action': 'start'})
    module.tick()
    assert client.get('/api/overview').json()['servers'][0]['messages'] == SEEDED_MESSAGES_PER_SERVER + 10


def test_cursor_filters_and_results_pagination(client):
    first = client.get('/api/logs?limit=20').json()
    older = client.get('/api/logs', params={'before_id': first['next_before'], 'limit': 20}).json()
    assert not {r['id'] for r in first['items']} & {r['id'] for r in older['items']}

    module.tick()
    newer = client.get('/api/logs', params={'after_id': first['next_after'], 'limit': 20}).json()
    newest = client.get('/api/logs', params={'after_id': newer['next_after'], 'limit': 20}).json()
    assert len(newer['items']) + len(newest['items']) == 30

    filtered = client.get('/api/logs?severity=ERROR&prediction=anomalous&search=failed').json()['items']
    assert filtered and all(r['severity'] == 'ERROR' and r['prediction'] == 'anomalous' for r in filtered)

    page_one = client.get('/api/results?prediction=anomalous&size=3').json()
    page_two = client.get('/api/results?prediction=anomalous&size=3&page=2').json()
    assert page_one['total'] > 3
    assert not {r['id'] for r in page_one['items']} & {r['id'] for r in page_two['items']}


def test_chat_evidence_persistence_and_unknown_questions(client):
    path = f"/api/results/{first_result(client, prediction='anomalous')['id']}"
    initial = client.post(path + '/chat', json={'initial': True})
    assert initial.status_code == 200 and '[L' in initial.json()['answer']

    client.post(path + '/chat', json={'initial': True})
    assert len(client.get(path).json()['messages']) == 1

    unsupported = client.post(path + '/chat', json={'message': 'What is the price of gold?'})
    # A specific unsupported subject should not invent information.
    answer = unsupported.json()['answer']
    assert 'cannot' in answer or 'not provide enough evidence' in answer

    module.initialize()
    assert len(client.get(path).json()['messages']) == 3


def test_invalid_imports_and_cross_origin(client):
    not_hdfs = {'name': 'bad', 'profile': 'HDFS', 'raw_text': 'not an hdfs log'}
    duplicate_ids = {'name': 'bad', 'profile': 'BGL',
                     'sequences': [{'sequence_id': 'x', 'logs': ['ok']}, {'sequence_id': 'x', 'logs': ['ok']}]}
    assert client.post('/api/collections', json=not_hdfs).status_code == 422
    assert client.post('/api/collections', json=duplicate_ids).status_code == 422
    assert client.post('/api/servers', json={}, headers={'Origin': 'https://other.example'}).status_code == 403


def test_add_server(client):
    body = {'name': 'application-04', 'address': '10.1.2.3', 'environment': 'staging', 'profile': 'Liberty'}
    assert client.post('/api/servers', json=body).status_code == 200
    assert client.post('/api/servers', json=body).status_code == 409
    server = next(s for s in client.get('/api/overview').json()['servers'] if s['name'] == body['name'])
    assert server['messages'] == SEEDED_MESSAGES_PER_SERVER and server['profile'] == 'Liberty'


# Minimal legacy schema: enough to verify archives are preserved without remapping old results.
LEGACY_SCHEMA = """
CREATE TABLE sources(id TEXT,name TEXT,status TEXT);
CREATE TABLE datasets(id TEXT,name TEXT);
CREATE TABLE sessions(id TEXT,source_id TEXT,dataset_id TEXT,started REAL,status TEXT);
CREATE TABLE sequences(id TEXT,external_id TEXT,logs TEXT,prediction TEXT,score REAL,truth TEXT);
CREATE TABLE results(id INTEGER,session_id TEXT,sequence_id TEXT,completed REAL);
INSERT INTO sources VALUES('old','Legacy source','running');
INSERT INTO datasets VALUES('d','Old collection');
INSERT INTO sessions VALUES('s','old','d',1,'running');
INSERT INTO sequences VALUES('q','original-id','["original message"]','normal',0.8,'anomalous');
INSERT INTO results VALUES(1,'s','q',2);
"""


class OfflineClient:
    """Stands in for httpx.AsyncClient and fails every request as if the provider were unreachable."""

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, *args, **kwargs):
        raise httpx.ConnectError('offline')


def test_archives_and_provider_failure(client, monkeypatch):
    with module.db() as c:
        c.executescript(LEGACY_SCHEMA)
    module.initialize()
    assert client.get('/api/runs?archive=true').json()['items'][0]['status'] == 'paused'
    assert client.get('/api/archive/s/results').json()[0]['sequence_id'] == 'original-id'

    result = first_result(client, prediction='anomalous')
    monkeypatch.setenv('OPENAI_API_KEY', 'not-a-real-key')
    monkeypatch.setenv('OPENAI_MODEL', 'test')
    monkeypatch.setattr(module.httpx, 'AsyncClient', OfflineClient)
    path = f"/api/results/{result['id']}"
    assert client.post(path + '/chat', json={'initial': True}).status_code == 502
    assert client.get(path).json()['messages'] == []
