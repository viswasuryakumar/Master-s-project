from sample_logs.generate_bgl import generate_sample
from test_comparison import seeded, client, finished, module


def test_sample_metrics_and_label_independence(client, tmp_path):
    generate_sample(tmp_path)
    raw = (tmp_path / 'bgl_12000.log').read_text()
    assert len(raw.splitlines()) == 12000
    outcomes = []
    for labeled in (True, False):
        response = client.post('/api/collections', json={
            'name': 'BGL sample', 'profile': 'BGL', 'raw_text': raw,
            'use_bgl_labels': labeled,
        })
        assert response.status_code == 200, response.text
        collection = response.json()
        assert collection['messages'] == 12000 and collection['sequences'] == 120
        run = client.post('/api/collections/'+collection['id']+'/analyze').json()['id']
        summary = finished(run)
        assert summary['coverage']['evaluated'] == (120 if labeled else 0)
        for detector in summary['detectors']:
            if labeled:
                assert all(detector['metrics'][key] is not None for key in ('precision','recall','f1','accuracy'))
            else:
                assert detector['metrics'] is None
        with module.db() as c:
            rows = c.execute('SELECT * FROM v2_seq WHERE collection_id=? ORDER BY ordinal', (collection['id'],)).fetchall()
            assert sum(r['truth'] == 'anomalous' for r in rows) == (36 if labeled else 0)
            assert all(r['length'] == 100 for r in rows)
            outcomes.append([tuple(r) for r in c.execute('SELECT s.ordinal,p.detector_id,p.prediction FROM comparison_predictions p JOIN v2_seq s ON s.id=p.seq_id WHERE s.collection_id=? ORDER BY s.ordinal,p.detector_id', (collection['id'],))])
    assert outcomes[0] == outcomes[1]

def test_label_option_rejects_prepared_sequences(client):
    response = client.post('/api/collections', json={
        'name':'invalid', 'profile':'BGL', 'use_bgl_labels':True,
        'sequences':[{'sequence_id':'one','logs':['ordinary message']}],
    })
    assert response.status_code == 422
