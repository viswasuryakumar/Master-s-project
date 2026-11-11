import pytest
from test_comparison import client,seeded,new_run,finished
from backend.recommendations import recommend
from backend.engine import generate

@pytest.mark.parametrize('message,category',[
 ('ERROR connection refused 10.0.0.1:443','connection'),
 ('ERROR bind: address already in use','port'),
 ('ERROR temporary failure in name resolution','dns'),
 ('ERROR certificate verify failed','tls'),
 ('ERROR permission denied','permissions'),
 ('ERROR failed to parse config','configuration'),
 ('ERROR no space left on device','disk_capacity'),
 ('ERROR out of memory','memory_capacity'),
 ('ERROR uncorrectable memory parity failure','memory'),
 ('ERROR checksum mismatch blk_123','storage'),
 ('ERROR too many open files','capacity'),
])
def test_evidence_and_checklists(message,category):
    action=recommend('BGL',['INFO routine',message],True)
    assert action['issues'][0]['category']==category
    assert action['issues'][0]['evidence']==[{'line':2,'text':message}]
    assert action['issues'][0]['checks']
    assert 'not confirmed' in action['basis']

def test_no_fault_invention_and_no_labels_as_evidence():
    assert recommend('BGL',['connection refused'],False)['issues']==[]
    unknown=recommend('BGL',['unusual sequence'],True)
    assert unknown['issues']==[] and 'do not establish' in unknown['basis']
    line=generate('BGL',0)['logs'][0].replace('- ','NXDOMAIN ',1)
    assert recommend('BGL',[line],True)['issues']==[]

def test_original_and_comparison_recommendations(client):
    run=new_run(client,[{'sequence_id':'network','logs':['connection refused 10.0.0.1:443']},
                        {'sequence_id':'port','logs':['address already in use'],'prediction':'anomalous'},
                        {'sequence_id':'ordinary','logs':['routine heartbeat']}])
    finished(run)
    original=client.get('/api/results',params={'run_id':run}).json()['items']
    by_id={r['external_id']:r for r in original}
    assert by_id['network']['recommended_action']['issues'][0]['category']=='connection'
    assert by_id['port']['recommended_action']['issues'][0]['category']=='port'
    assert by_id['ordinary']['recommended_action'] is None
    comparison=client.get(f'/api/runs/{run}/comparison/results').json()['items']
    network=next(r for r in comparison if r['external_id']=='network')
    assert network['recommended_action']['issues'][0]['category']=='connection'
    assert next(r for r in comparison if r['external_id']=='ordinary')['recommended_action'] is None
    assert 'logs' not in network
    detail=client.get('/api/results/'+str(by_id['network']['id'])).json()
    assert detail['recommended_action']['issues'][0]['evidence'][0]['line']==1
    server=client.get('/api/overview').json()['servers'][0]
    for classification in ('normal','anomalous'):
        live=client.get('/api/results',params={'run_id':server['run_id'],'prediction':classification}).json()['items']
        assert live
        for row in live:
            assert bool(row['recommended_action']) == (classification=='anomalous')
        detail=client.get('/api/results/'+str(live[0]['id'])).json()
        assert bool(detail['recommended_action']) == (classification=='anomalous')
