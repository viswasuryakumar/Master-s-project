import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend import app as module
from backend.engine import generate, classify

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'DB_PATH',tmp_path/'test.sqlite3')
    monkeypatch.setenv('OBSERVATORY_NO_WORKERS','1')
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    with TestClient(module.app) as client:yield client

def upload(client,**kw):
    response=client.post('/api/collections',json={'name':'Test','profile':'BGL',**kw})
    assert response.status_code==200,response.text
    run=client.post('/api/collections/'+response.json()['id']+'/analyze').json()['id']
    assert client.get('/api/runs/'+run).json()['status']=='queued'
    module.tick()
    assert client.get('/api/runs/'+run).json()['status']=='processing'
    for _ in range(50):
        module.tick()
        if client.get('/api/runs/'+run).json()['status']=='completed':break
    assert client.get('/api/runs/'+run).json()['status']=='completed'
    return run,client.get('/api/results',params={'run_id':run,'size':100}).json()['items']

def test_seed_volumes_and_formats(client):
    state=client.get('/api/overview').json()
    assert len(state['profiles'])==4
    assert all(p['messages']==12000 for p in state['profiles'])
    assert all(s['messages']==10000 for s in state['servers'])
    assert state['chat_provider']=='local'
    with module.db() as c:
        assert {r[0] for r in c.execute("SELECT DISTINCT length FROM v2_seq WHERE profile='HDFS'")}=={20,30,40,50,60}
        assert c.execute("SELECT COUNT(*) FROM v2_seq WHERE profile='HDFS' AND block_id IS NULL").fetchone()[0]==0

@pytest.mark.parametrize('profile',['BGL','HDFS','Liberty','Thunderbird'])
def test_raw_grouping_and_progress(client,profile):
    sequences=[generate(profile,i) for i in range(4)]
    raw='\n'.join(line for seq in sequences for line in seq['logs'])
    run,rows=upload(client,profile=profile,raw_text=raw)
    assert len(rows)==4
    assert sum(r['prediction']=='anomalous' for r in rows)==1
    exported=client.get(f'/api/runs/{run}/export?format=json').json()
    assert exported[0]['logs']==sequences[0]['logs']
    assert all(r['ground_truth'] is None for r in exported)
    assert all(r['processing_origin']=='profile_simulation' for r in exported)

def test_truth_is_not_prediction_and_imported_precedence(client):
    run,rows=upload(client,sequences=[{'sequence_id':'a','logs':['routine activity'],'ground_truth':'anomalous'},{'sequence_id':'b','logs':['connection refused'],'prediction':'normal','ground_truth':'anomalous','score':0.1},{'sequence_id':'c','logs':['connection refused'],'ground_truth':'normal'}])
    rows={r['external_id']:r for r in rows}
    assert rows['a']['prediction']=='normal'
    assert rows['b']['prediction']=='normal' and rows['b']['origin']=='imported_prediction'
    assert rows['c']['prediction']=='anomalous'
    assert client.get('/api/runs/'+run).json()['metrics']['fn']==2
    assert 'processing_origin' in client.get('/api/runs/'+run+'/export').text

    line=generate('BGL',0)['logs'][0].replace('- ', 'UNCORRECTABLE ',1)
    assert classify('BGL',[line])[0]=='normal'

def test_stream_restart_no_duplicates_and_independent_controls(client):
    servers=client.get('/api/overview').json()['servers'];stopped=servers[0]
    client.post(f"/api/servers/{stopped['id']}/control",json={'action':'stop'})
    module.tick();module.initialize();module.tick()
    current=client.get('/api/overview').json()['servers']
    assert current[0]['messages']==10000
    assert all(s['messages']==10020 for s in current[1:])
    with module.db() as c:
        assert c.execute('SELECT COUNT(*) FROM v2_events').fetchone()[0]==30040
        assert c.execute('SELECT COUNT(*) FROM (SELECT run_id,seq_id,line_index FROM v2_events GROUP BY run_id,seq_id,line_index)').fetchone()[0]==30040
    client.post(f"/api/servers/{stopped['id']}/control",json={'action':'start'})
    module.tick()
    assert client.get('/api/overview').json()['servers'][0]['messages']==10010

def test_cursor_filters_and_results_pagination(client):
    first=client.get('/api/logs?limit=20').json()
    older=client.get('/api/logs',params={'before_id':first['next_before'],'limit':20}).json()
    assert not {r['id'] for r in first['items']}&{r['id'] for r in older['items']}
    module.tick()
    newer=client.get('/api/logs',params={'after_id':first['next_after'],'limit':20}).json()
    newer2=client.get('/api/logs',params={'after_id':newer['next_after'],'limit':20}).json()
    assert len(newer['items'])+len(newer2['items'])==30
    filtered=client.get('/api/logs?severity=ERROR&prediction=anomalous&search=failed').json()['items']
    assert filtered and all(r['severity']=='ERROR' and r['prediction']=='anomalous' for r in filtered)
    one=client.get('/api/results?prediction=anomalous&size=3').json()
    two=client.get('/api/results?prediction=anomalous&size=3&page=2').json()
    assert one['total']>3 and not {r['id'] for r in one['items']}&{r['id'] for r in two['items']}

def test_chat_evidence_persistence_and_unknown_questions(client):
    result=client.get('/api/results?prediction=anomalous&size=1').json()['items'][0]
    path=f"/api/results/{result['id']}"
    a=client.post(path+'/chat',json={'initial':True})
    assert a.status_code==200 and '[L' in a.json()['answer']
    client.post(path+'/chat',json={'initial':True})
    assert len(client.get(path).json()['messages'])==1
    answer=client.post(path+'/chat',json={'message':'What is the price of gold?'})

    assert 'cannot' in answer.json()['answer'] or 'not provide enough evidence' in answer.json()['answer']
    module.initialize()
    assert len(client.get(path).json()['messages'])==3

def test_invalid_imports_and_cross_origin(client):
    assert client.post('/api/collections',json={'name':'bad','profile':'HDFS','raw_text':'not an hdfs log'}).status_code==422
    assert client.post('/api/collections',json={'name':'bad','profile':'BGL','sequences':[{'sequence_id':'x','logs':['ok']},{'sequence_id':'x','logs':['ok']}]}).status_code==422
    assert client.post('/api/servers',json={},headers={'Origin':'https://other.example'}).status_code==403

def test_add_server(client):
    body={'name':'application-04','address':'10.1.2.3','environment':'staging','profile':'Liberty'}
    assert client.post('/api/servers',json=body).status_code==200
    assert client.post('/api/servers',json=body).status_code==409
    server=next(s for s in client.get('/api/overview').json()['servers'] if s['name']==body['name'])
    assert server['messages']==10000 and server['profile']=='Liberty'

def test_archives_and_provider_failure(client,monkeypatch):

    with module.db() as c:
        c.executescript("CREATE TABLE sources(id TEXT,name TEXT,status TEXT); CREATE TABLE datasets(id TEXT,name TEXT); CREATE TABLE sessions(id TEXT,source_id TEXT,dataset_id TEXT,started REAL,status TEXT); CREATE TABLE sequences(id TEXT,external_id TEXT,logs TEXT,prediction TEXT,score REAL,truth TEXT); CREATE TABLE results(id INTEGER,session_id TEXT,sequence_id TEXT,completed REAL); INSERT INTO sources VALUES('old','Legacy source','running'); INSERT INTO datasets VALUES('d','Old collection'); INSERT INTO sessions VALUES('s','old','d',1,'running'); INSERT INTO sequences VALUES('q','original-id','[\"original message\"]','normal',0.8,'anomalous'); INSERT INTO results VALUES(1,'s','q',2);")
    module.initialize()
    assert client.get('/api/runs?archive=true').json()['items'][0]['status']=='paused'
    assert client.get('/api/archive/s/results').json()[0]['sequence_id']=='original-id'
    result=client.get('/api/results?prediction=anomalous&size=1').json()['items'][0]
    import httpx
    monkeypatch.setenv('OPENAI_API_KEY','not-a-real-key');monkeypatch.setenv('OPENAI_MODEL','test')
    class Failure:
        def __init__(self,**kw):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def post(self,*args,**kw):raise httpx.ConnectError('offline')
    monkeypatch.setattr(module.httpx,'AsyncClient',Failure)
    path=f"/api/results/{result['id']}"
    assert client.post(path+'/chat',json={'initial':True}).status_code==502
    assert client.get(path).json()['messages']==[]
