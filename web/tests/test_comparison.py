import json
import shutil
import pytest
from test_app import module
from backend import comparison
from backend.engine import generate
from fastapi.testclient import TestClient

@pytest.fixture(scope='module')
def seeded(tmp_path_factory):
    original=module.DB_PATH
    path=tmp_path_factory.mktemp('comparison_seed')/'seed.sqlite3'
    module.DB_PATH=path
    module.initialize()
    with module.db() as c:c.execute("UPDATE v2_servers SET status='stopped'")
    module.DB_PATH=original
    return path

@pytest.fixture
def client(seeded,tmp_path,monkeypatch):
    path=tmp_path/'test.sqlite3';shutil.copyfile(seeded,path)
    monkeypatch.setattr(module,'DB_PATH',path)
    monkeypatch.setenv('OBSERVATORY_NO_WORKERS','1')
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    with TestClient(module.app) as client:yield client

def finished(run):
    for _ in range(60):
        module.tick()
        with module.db() as c:
            j=comparison.summary(c,run)
            if j and j['status'] in ('completed','completed_with_errors'):return j
    raise AssertionError('Comparison never completed')

def new_run(client,seqs,compare=True):
    collection=client.post('/api/collections',json={'name':'comparison-test','profile':'BGL','sequences':seqs}).json()['id']
    return client.post(f'/api/collections/{collection}/analyze',json={'compare':compare}).json()['id']

def test_metrics_known_and_undefined():
    m=comparison.metric([('anomalous','anomalous'),('normal','anomalous'),('anomalous','normal'),('normal','normal')])
    assert (m['precision'],m['recall'],m['f1'],m['accuracy'])==(.5,.5,.5,.5)
    assert (m['tp'],m['fp'],m['fn'],m['tn'])==(1,1,1,1)
    assert comparison.metric([]) is None
    m=comparison.metric([('normal','normal')]);assert m['precision'] is None and m['recall'] is None and m['f1'] is None

def test_all_profiles_48000_messages_and_restart(client):
    profiles=client.get('/api/overview').json()['profiles']
    assert sum(p['messages'] for p in profiles)==48000
    runs=[client.post('/api/collections/'+p['collection_id']+'/analyze').json()['id'] for p in profiles]
    module.tick();module.tick();module.initialize()
    for p,run in zip(profiles,runs):
        summary=finished(run)
        assert summary['coverage']['evaluated']==p['sequences']
        assert all(d['completed']==p['sequences'] for d in summary['detectors'])
        first=client.post(f'/api/runs/{run}/comparison').json()
        second=client.post(f'/api/runs/{run}/comparison').json()
        assert first['id']==second['id']
        with module.db() as c:
            assert c.execute('SELECT COUNT(*) FROM comparison_predictions WHERE job_id=?',(first['id'],)).fetchone()[0]==4*p['sequences']
            assert c.execute('SELECT COUNT(*) FROM comparison_predictions p JOIN v2_seq s ON s.id=p.seq_id WHERE p.job_id=? AND s.collection_id!=?',(first['id'],p['collection_id'])).fetchone()[0]==0

def test_labels_imports_pagination_filters_and_exports(client):
    seqs=[{'sequence_id':f'seq-{i}','logs':['ERROR failed to connect: connection refused'],'ground_truth':'anomalous','prediction':'normal'} for i in range(30)]
    run=new_run(client,seqs);summary=finished(run)
    assert summary['coverage']['evaluated']==30
    base=f'/api/runs/{run}/comparison'
    page=client.get(base+'/results?filter=disagreement&size=10').json()
    assert page['total']==30 and len(page['items'])==10
    second=client.get(base+'/results?filter=flagged&size=10&page=2').json()
    assert not {r['seq_id'] for r in page['items']}&{r['seq_id'] for r in second['items']}
    assert client.get(base+'/results?search=seq-29').json()['total']==1
    assert client.get(base+'/export?format=json').json()['summary']['coverage']['evaluated']==30
    assert 'metrics' in client.get(base+'/export?format=csv').text
    result=client.get('/api/results/'+str(page['items'][0]['result_id'])).json()
    assert result['prediction']=='normal' and len(result['comparisons'])==4
    assert {p['prediction'] for p in result['comparisons']}=={'normal','anomalous'}
    with module.db() as c:
        before=[tuple(r) for r in c.execute('SELECT * FROM comparison_predictions ORDER BY seq_id,detector_id')]
        c.execute("UPDATE v2_seq SET truth='normal' WHERE collection_id=(SELECT collection_id FROM v2_runs WHERE id=?)",(run,))
    module.tick()
    with module.db() as c:
        after=[tuple(r) for r in c.execute('SELECT * FROM comparison_predictions ORDER BY seq_id,detector_id')]
    assert before==after
    assert client.get(base).json()['detectors'][0]['metrics']['fp']==30
    answer=client.post('/api/results/'+str(result['id'])+'/chat',json={'initial':True}).json()['answer']
    assert 'Simulated detector comparison' in answer and 'deeplog' in answer

def test_missing_labels_and_partial_failures(client,monkeypatch):
    original=comparison.predict
    def fail(d,p,logs):
        if d.id=='logbert' and 'break' in logs[0]:raise RuntimeError('test')
        return original(d,p,logs)
    monkeypatch.setattr(comparison,'predict',fail)
    run=new_run(client,[{'sequence_id':'a','logs':['routine'],'ground_truth':'normal'},{'sequence_id':'b','logs':['break'],'ground_truth':'anomalous'},{'sequence_id':'c','logs':['routine']}])
    summary=finished(run)
    assert summary['status']=='completed_with_errors'
    assert summary['coverage']=={'total':3,'all_completed':2,'labeled':2,'evaluated':1,'excluded':2,'unlabeled':1,'labeled_incomplete':1}
    assert all(d['metrics']['evaluated']==1 for d in summary['detectors'])
    assert summary['detectors'][2]['failed']==1
    run=new_run(client,[{'sequence_id':'x','logs':['connection refused']}]);summary=finished(run)
    assert all(d['metrics'] is None for d in summary['detectors'])

def test_opt_out_historical_run_and_live_isolation(client):
    run=new_run(client,[{'sequence_id':'x','logs':['routine']}],False)
    module.tick();module.tick()
    assert client.get(f'/api/runs/{run}/comparison').json() is None
    before=client.get('/api/results?run_id='+run).json()
    client.post(f'/api/runs/{run}/comparison');finished(run)
    assert before==client.get('/api/results?run_id='+run).json()
    server=client.get('/api/overview').json()['servers'][0]
    assert client.post(f"/api/runs/{server['run_id']}/comparison").status_code==422
    live=client.get('/api/results?run_id='+server['run_id']).json()['items'][0]
    assert client.get('/api/results/'+str(live['id'])).json()['comparisons']==[]

def test_policy_repeatability_and_order():
    logs=['service recovered','ERROR connection refused']
    d=comparison.DETECTORS[1]
    assert comparison.predict(d,'BGL',logs)==comparison.predict(d,'BGL',logs)
    assert comparison.predict(d,'BGL',logs)['prediction']=='anomalous'
    assert comparison.predict(d,'BGL',list(reversed(logs)))['prediction']=='normal'

def test_hosted_chat_receives_comparison_and_evidence(client,monkeypatch):
    import httpx
    run=new_run(client,[{'sequence_id':'context-seq','logs':['service recovered','ERROR connection refused'],'ground_truth':'anomalous'}]);finished(run)
    result=client.get(f'/api/runs/{run}/comparison/results').json()['items'][0]
    captured=[]
    class Provider:
        def __init__(self,**kw):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def post(self,url,headers,json):
            captured.append(json)
            return httpx.Response(200,request=httpx.Request('POST',url),json={'output':[{'content':[{'type':'output_text','text':'Illustrative policies differ at [L2].'}]}]})
    monkeypatch.setattr(module.httpx,'AsyncClient',Provider)
    monkeypatch.setenv('OPENAI_API_KEY','test-not-real');monkeypatch.setenv('OPENAI_MODEL','test-model')
    assert client.post(f"/api/results/{result['result_id']}/chat",json={'message':'Why do the detectors disagree?'}).status_code==200
    evidence=json.loads(captured[0]['input'][0]['content'].split(': ',1)[1])
    assert len(evidence['detector_comparison'])==4
    assert all(p['origin']=='comparison_simulation' for p in evidence['detector_comparison'])
    assert evidence['logs']==['L1: service recovered','L2: ERROR connection refused']
