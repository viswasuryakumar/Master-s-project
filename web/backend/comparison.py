
import csv
import io
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Literal
from fastapi import HTTPException, Query
from fastapi.responses import Response
from .engine import parse, classify
from .recommendations import recommend

ORIGIN = 'comparison_simulation'
VERSION = 'simulation-v1'

@dataclass(frozen=True)
class Detector:
    id: str
    name: str
    policy: str
    version: str = VERSION
    origin: str = ORIGIN

DETECTORS = (
    Detector('bert_llama', 'BERT + Llama', 'Recognized failure phrases or explicit service-impact language.'),
    Detector('deeplog', 'DeepLog', 'Repeated warnings/failures or a failure following a recovery event.'),
    Detector('logbert', 'LogBERT', 'Critical phrases or multiple distinct failure events.'),
    Detector('rules', 'Rule-based baseline', 'Existing profile-specific failure-pattern rules.'),
)

def predict(detector, profile, logs):

    content=[parse(profile,line)['content'].lower() for line in logs]
    baseline=classify(profile,logs)[0]=='anomalous'
    failures=[i for i,line in enumerate(content) if re.search(r'error|fail|refused|timeout|timed out|mismatch|exhausted|uncorrectable|unavailable',line)]
    normalized=[re.sub(r'\d+', '#', content[i]) for i in failures]
    if detector.id=='bert_llama':
        flagged=baseline or any(re.search(r'service unavailable|lost connection|permission denied|data corruption|fatal',line) for line in content)
    elif detector.id=='deeplog':
        recovered=[i for i,line in enumerate(content) if 'recovery cycle completed' in line or 'recovered' in line]
        flagged=any(n>=3 for n in Counter(normalized).values()) or any(i<j for i in recovered for j in failures)
    elif detector.id=='logbert':
        flagged=any(re.search(r'uncorrectable|corrupt|out of memory|fatal|panic',line) for line in content) or len(set(normalized))>=2
    elif detector.id=='rules':flagged=baseline
    else:raise ValueError('Unknown detector adapter')
    return {'prediction':'anomalous' if flagged else 'normal', 'reason':detector.policy + (' Policy matched.' if flagged else ' Policy did not match.')}

def initialize(c):
    c.executescript('''
    CREATE TABLE IF NOT EXISTS comparison_jobs(
      id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT NOT NULL UNIQUE REFERENCES v2_runs(id),
      version TEXT NOT NULL,origin TEXT NOT NULL,status TEXT NOT NULL,total INTEGER NOT NULL,
      processed INTEGER NOT NULL DEFAULT 0,created REAL NOT NULL,ended REAL);
    CREATE TABLE IF NOT EXISTS comparison_predictions(
      job_id INTEGER NOT NULL REFERENCES comparison_jobs(id),seq_id TEXT NOT NULL REFERENCES v2_seq(id),
      detector_id TEXT NOT NULL,version TEXT NOT NULL,origin TEXT NOT NULL,
      status TEXT NOT NULL,prediction TEXT,reason TEXT,error TEXT,
      PRIMARY KEY(job_id,seq_id,detector_id));
    CREATE INDEX IF NOT EXISTS comparison_seq ON comparison_predictions(seq_id,job_id);
    CREATE INDEX IF NOT EXISTS comparison_status ON comparison_predictions(job_id,status,seq_id);
    ''')

def start(c,run_id):
    run=c.execute('SELECT * FROM v2_runs WHERE id=?',(run_id,)).fetchone()
    if not run:raise HTTPException(404,'Run not found')
    if run['kind']!='upload':raise HTTPException(422,'Comparisons are available only for upload analyses')
    c.execute('INSERT OR IGNORE INTO comparison_jobs(run_id,version,origin,status,total,created) VALUES(?,?,?,?,?,?)',
              (run_id,VERSION,ORIGIN,'queued',run['total'],time.time()))
    return dict(c.execute('SELECT * FROM comparison_jobs WHERE run_id=?',(run_id,)).fetchone())

def tick(c):
    changed=False
    for job in c.execute("SELECT j.*,r.collection_id,r.processed AS available FROM comparison_jobs j JOIN v2_runs r ON r.id=j.run_id WHERE j.status IN ('queued','processing')").fetchall():
        if job['status']=='queued':
            c.execute("UPDATE comparison_jobs SET status='processing' WHERE id=?",(job['id'],));changed=True;continue
        rows=c.execute('SELECT * FROM v2_seq WHERE collection_id=? AND ordinal>=? AND ordinal<? ORDER BY ordinal LIMIT 8',
                       (job['collection_id'],job['processed'],job['available'])).fetchall()
        for seq in rows:
            logs=json.loads(seq['logs'])
            for detector in DETECTORS:
                try:
                    result=predict(detector,seq['profile'],logs)
                    if result['prediction'] not in ('normal','anomalous'):raise ValueError('Invalid prediction')
                    status='completed';prediction=result['prediction'];reason=result['reason'];error=None
                except Exception:
                    status='failed';prediction=None;reason=None;error='Detector could not process this sequence.'
                c.execute('INSERT OR IGNORE INTO comparison_predictions VALUES(?,?,?,?,?,?,?,?,?)',
                          (job['id'],seq['id'],detector.id,detector.version,detector.origin,status,prediction,reason,error))
        if rows:
            processed=job['processed']+len(rows);status='processing'
            if processed==job['total']:
                failed=c.execute("SELECT COUNT(*) FROM comparison_predictions WHERE job_id=? AND status='failed'",(job['id'],)).fetchone()[0]
                status='completed_with_errors' if failed else 'completed'
            c.execute('UPDATE comparison_jobs SET processed=?,status=?,ended=? WHERE id=?',
                      (processed,status,time.time() if processed==job['total'] else None,job['id']));changed=True
    return changed

def metric(rows):
    if not rows:return None
    tp=sum(t=='anomalous' and p=='anomalous' for t,p in rows)
    fp=sum(t=='normal' and p=='anomalous' for t,p in rows)
    fn=sum(t=='anomalous' and p=='normal' for t,p in rows)
    tn=len(rows)-tp-fp-fn
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,evaluated=len(rows),precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,accuracy=(tp+tn)/len(rows))

def summary(c,run_id):
    job=c.execute('SELECT * FROM comparison_jobs WHERE run_id=?',(run_id,)).fetchone()
    if not job:return None

    cohort='''SELECT seq_id FROM comparison_predictions WHERE job_id=? AND status='completed' AND origin=? AND version=? GROUP BY seq_id HAVING COUNT(DISTINCT detector_id)=4'''
    common=[r[0] for r in c.execute(cohort,(job['id'],ORIGIN,job['version']))]
    data=c.execute('SELECT p.detector_id,p.prediction,q.truth FROM comparison_predictions p JOIN v2_seq q ON q.id=p.seq_id WHERE p.job_id=? AND p.seq_id IN ('+cohort+') AND q.truth IS NOT NULL',
                   (job['id'],job['id'],ORIGIN,job['version'])).fetchall()
    labeled=c.execute('SELECT COUNT(*) FROM v2_seq q JOIN v2_runs r ON r.collection_id=q.collection_id WHERE r.id=? AND q.truth IS NOT NULL',(run_id,)).fetchone()[0]
    detectors=[]
    for d in DETECTORS:
        counts={r['status']:r['n'] for r in c.execute('SELECT status,COUNT(*) AS n FROM comparison_predictions WHERE job_id=? AND detector_id=? GROUP BY status',(job['id'],d.id))}
        detectors.append({**d.__dict__,'completed':counts.get('completed',0),'failed':counts.get('failed',0),'metrics':metric([(r['truth'],r['prediction']) for r in data if r['detector_id']==d.id])})
    evaluated=detectors[0]['metrics']['evaluated'] if detectors[0]['metrics'] else 0
    return {**dict(job),'detectors':detectors,'coverage':{'total':job['total'],'all_completed':len(common),'labeled':labeled,'evaluated':evaluated,'excluded':job['total']-evaluated,'unlabeled':job['total']-labeled,'labeled_incomplete':labeled-evaluated}}

def predictions(c,run_id,seq_id):
    return [dict(r) for r in c.execute('SELECT p.detector_id,p.version,p.origin,p.status,p.prediction,p.reason,p.error FROM comparison_predictions p JOIN comparison_jobs j ON j.id=p.job_id WHERE j.run_id=? AND p.seq_id=? ORDER BY p.detector_id',(run_id,seq_id))]

def result_page(c,run_id,search='',filter='all',page=1,size=25):
    job=c.execute('SELECT id FROM comparison_jobs WHERE run_id=?',(run_id,)).fetchone()
    if not job:raise HTTPException(404,'Start a model comparison first')
    where=' WHERE r.run_id=?';args=[run_id]
    if search:where+=' AND (q.external_id LIKE ? OR q.logs LIKE ?)';args+=['%'+search+'%']*2
    sub='SELECT COUNT(DISTINCT p.prediction) FROM comparison_predictions p WHERE p.job_id=? AND p.seq_id=q.id AND p.status=\'completed\''
    if filter=='disagreement':where+=' AND ('+sub+')>1';args.append(job['id'])
    if filter=='flagged':
        where+=" AND EXISTS(SELECT 1 FROM comparison_predictions p WHERE p.job_id=? AND p.seq_id=q.id AND p.prediction='anomalous' AND p.status='completed')";args.append(job['id'])
    base=' FROM v2_results r JOIN v2_seq q ON q.id=r.seq_id'+where
    total=c.execute('SELECT COUNT(*)'+base,args).fetchone()[0]
    rows=[dict(r) for r in c.execute('SELECT r.id AS result_id,q.id AS seq_id,q.external_id,q.length,q.profile,q.logs,q.truth AS ground_truth'+base+' ORDER BY q.ordinal LIMIT ? OFFSET ?',args+[size,(page-1)*size])]
    for row in rows:
        row['predictions']=predictions(c,run_id,row['seq_id'])
        flagged=any(p['prediction']=='anomalous' for p in row['predictions'])
        logs=json.loads(row.pop('logs'))
        row['recommended_action']=recommend(row['profile'],logs,True) if flagged else None
    return {'items':rows,'total':total}

def install(app,db,bump):
    @app.post('/api/runs/{run_id}/comparison')
    def create(run_id:str):
        with db() as c:
            job=start(c,run_id);bump(c);return job

    @app.get('/api/runs/{run_id}/comparison')
    def get(run_id:str):
        with db() as c:return summary(c,run_id)

    @app.get('/api/runs/{run_id}/comparison/results')
    def results(run_id:str,search:str='',filter:Literal['all','disagreement','flagged']='all',page:int=Query(1,ge=1),size:int=Query(25,ge=1,le=100)):
        with db() as c:return result_page(c,run_id,search,filter,page,size)

    @app.get('/api/runs/{run_id}/comparison/export')
    def export(run_id:str,format:Literal['json','csv']='json'):
        with db() as c:
            info=summary(c,run_id)
            if not info:raise HTTPException(404,'Comparison not found')
            items=result_page(c,run_id,size=info['total'])['items']
        if format=='json':content=json.dumps({'summary':info,'sequences':items},indent=2);media='application/json'
        else:
            out=io.StringIO();writer=csv.writer(out)
            writer.writerow(['record_type','sequence_id','ground_truth','detector','version','origin','status','prediction','precision','recall','f1','accuracy','tp','fp','fn','tn','evaluated','excluded'])
            for d in info['detectors']:
                m=d['metrics'] or {};writer.writerow(['metrics','','',d['name'],d['version'],d['origin'],info['status'],'']+[m.get(k,'') for k in ('precision','recall','f1','accuracy','tp','fp','fn','tn','evaluated')]+[info['coverage']['excluded']])
            for item in items:
                identifier=item['external_id'];identifier="'"+identifier if identifier.startswith(('=','+','-','@','\t','\r')) else identifier
                for p in item['predictions']:writer.writerow(['prediction',identifier,item['ground_truth'],p['detector_id'],p['version'],p['origin'],p['status'],p['prediction']]+['']*10)
            content=out.getvalue();media='text/csv'
        return Response(content,media_type=media,headers={'Content-Disposition':f'attachment; filename="comparison-{run_id}.{format}"'})
