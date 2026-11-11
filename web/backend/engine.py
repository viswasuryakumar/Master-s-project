
from datetime import datetime, timezone
import re

PROFILES = {
    'BGL': {'description':'Supercomputer system events', 'format':'Label · timestamp · location · component · level · message', 'grouping':'100-message windows'},
    'HDFS': {'description':'Distributed file system events', 'format':'Date · time · process · level · component · block event', 'grouping':'Block-ID sessions'},
    'Liberty': {'description':'Cluster operating-system events', 'format':'Label · timestamp · host · service · message', 'grouping':'100-message windows'},
    'Thunderbird': {'description':'Cluster service and system events', 'format':'Label · timestamp · host · service · message', 'grouping':'100-message windows'},
}
SCENARIOS = {
    'memory': {'title':'Memory integrity errors','severity':'High','cause':'Repeated memory parity failures can indicate unstable hardware or an uncorrectable memory condition.','check':'Inspect hardware error counters and diagnostic reports; correlate the affected host with workload failures.','prevent':'Monitor corrected-error trends, schedule memory diagnostics, and replace failing hardware after confirmation.'},
    'storage': {'title':'Block storage failure','severity':'High','cause':'A block could not be read or replicated successfully. Disk health, replica availability, or a damaged block are possible causes.','check':'Check the block identifier, replica locations, disk health, and the surrounding storage-service logs.','prevent':'Track disk and replica health, verify replication policies, and test recovery from unavailable replicas.'},
    'connection': {'title':'Service connection failures','severity':'Medium','cause':'A remote service is unavailable or communication is failing. Logs alone cannot establish whether the cause is networking or the remote process.','check':'Check service health, name resolution, network reachability, connection limits, and recent deployments.','prevent':'Use bounded retries with backoff, timeouts, service health monitoring, and capacity alerts.'},
    'capacity': {'title':'Resource exhaustion','severity':'High','cause':'A process reports exhausted memory or filesystem capacity. Workload growth, leaks, or retention settings may contribute.','check':'Inspect memory and disk utilization, process limits, workload changes, and retention policies.','prevent':'Set resource budgets and early utilization alerts; review log rotation and capacity planning.'},
    'normal': {'title':'Routine operation','severity':'Informational','cause':'The available sequence shows routine activity without a recognized failure pattern. This is not proof that the system is healthy.','check':'Correlate with service metrics and neighboring sequences if a user-visible failure was observed.','prevent':'Maintain health checks, baseline monitoring, and regular review of service logs.'},
    'unknown': {'title':'Sequence requires investigation','severity':'Undetermined','cause':'The classification does not have enough supporting context to establish a specific failure type.','check':'Review adjacent sequences and system metrics. Confirm the affected host, component, and timing.','prevent':'Improve contextual logging and establish alerts based on confirmed failure indicators.'},
}
PATTERNS = {
    'memory':r'uncorrectable|parity failure|machine check',
    'storage':r'checksum mismatch|missing replica|block verification failed|disk i/o error',
    'connection':r'connection refused|connection timed out|failed to connect|network unreachable',
    'capacity':r'out of memory|no space left|resource exhausted|oom-killer',
}

def parse(profile, raw):
    content=raw
    stamp=None; host=None; component=None
    severity='INFO'
    match=re.search(r'\b(DEBUG|INFO|NOTICE|WARN(?:ING)?|ERROR|FATAL|CRITICAL|ALERT)\b',raw)
    if match: severity=match.group(1).replace('WARNING','WARN')
    if profile=='HDFS':
        match=re.match(r'^(\d{6}) (\d{6}) (\d+) (\w+) ([^:]+):\s*(.*)$',raw)
        if match:
            stamp=f'{match[1]} {match[2]}'; severity=match[4]; component=match[5];content=match[6]
    elif profile=='BGL':
        parts=raw.split(maxsplit=9)
        if len(parts)==10 and parts[1].isdigit() and re.fullmatch(r'\d{4}\.\d{2}\.\d{2}',parts[2]):
            stamp=parts[4];host=parts[3];component=f'{parts[6]}.{parts[7]}';severity=parts[8];content=parts[9]
    else:
        parts=raw.split(maxsplit=9)
        if len(parts)==10 and parts[1].isdigit() and re.fullmatch(r'\d{4}\.\d{2}\.\d{2}',parts[2]):
            stamp=f'{parts[2]} {parts[6]}';host=parts[3];component=parts[8].rstrip(':');content=parts[9]
    block=re.search(r'blk_-?\d+',content)
    return {'raw':raw,'content':content,'original_time':stamp,'host':host,'component':component,'severity':severity,'block_id':block.group() if block else None}

def classify(profile, messages):

    text='\n'.join(parse(profile,line)['content'] for line in messages).lower()
    priority={'HDFS':['storage','capacity','connection','memory'],'BGL':['memory','connection','capacity','storage']}.get(profile,['capacity','connection','memory','storage'])
    for scenario in priority:
        if re.search(PATTERNS[scenario],text):
            return 'anomalous',scenario,0.94
    return 'normal','normal',0.88

def generate(profile, number, count=None, epoch=1768474800, namespace=0):
    count=count or (20+(number%5)*10 if profile=='HDFS' else 100)
    scenario=['memory','storage','connection','capacity'][(number//7)%4] if number%7==3 else 'normal'
    if profile=='HDFS' and scenario=='memory': scenario='storage'
    block=f'blk_{1000000000000+namespace*1000000+number}'
    host=f'node{(number%48)+1:03}'
    lines=[]
    for i in range(count):
        dt=datetime.fromtimestamp(epoch+number*100+i,tz=timezone.utc)
        level='INFO'
        if profile=='HDFS':
            message=[f'Receiving block {block} src: /10.20.1.{number%200+1}:50010 dest: /10.20.2.8:50010',f'PacketResponder for block {block} completed',f'BLOCK* addStoredBlock: blockMap updated for {block} size 67108864'][i%3]
        else:
            message=['service heartbeat completed','background maintenance cycle finished','workload execution completed successfully','checkpoint committed successfully','network interface operational'][i%5]+f' operation={number*100+i}'
        if scenario!='normal' and i in (count//2,count//2+1,count//2+2):
            level='ERROR'
            message={'memory':'uncorrectable memory parity failure reported by hardware monitor','storage':f'block verification failed: checksum mismatch for {block}','connection':'failed to connect to service endpoint: connection refused','capacity':'resource exhausted: no space left on device'}[scenario]
        elif scenario!='normal' and i==count-2:
            level='WARN';message=f'recovery attempt scheduled for operation={number}'
        elif scenario!='normal' and i==count-1:
            message='recovery cycle completed; monitor subsequent operations'
        if profile=='HDFS':
            if block not in message:message+=f' {block}'
            line=f'{dt:%y%m%d %H%M%S} {100+number%500} {level} dfs.DataNode: {message}'
        elif profile=='BGL':
            label='ALERT' if level=='ERROR' else '-'
            location=f'R{number%32:02}-M0-N0-C:J01-U01'
            line=f'{label} {int(dt.timestamp())} {dt:%Y.%m.%d} {location} {dt:%Y-%m-%d-%H.%M.%S}.000000 {location} RAS KERNEL {level} {message}'
        else:
            label='ALERT' if level=='ERROR' else '-'
            line=f'{label} {int(dt.timestamp())} {dt:%Y.%m.%d} {host} {dt:%b} {dt.day} {dt:%H:%M:%S} {host}/{host} systemd[{100+number%500}]: {level} {message}'
        lines.append(line)
    return {'external_id':block if profile=='HDFS' else f'seq-{number+1:06}', 'logs':lines,'block_id':block if profile=='HDFS' else None,'original_time':parse(profile,lines[0])['original_time'],'truth':'anomalous' if scenario!='normal' else 'normal','scenario':scenario}

def explain(profile, logs, prediction, question, initial=False):
    _,scenario,_=classify(profile,logs)
    if prediction=='anomalous' and scenario=='normal':scenario='unknown'
    info=SCENARIOS[scenario]
    evidence=[(i+1,parse(profile,line)['content']) for i,line in enumerate(logs) if scenario in PATTERNS and re.search(PATTERNS[scenario],parse(profile,line)['content'],re.I)][:3]
    refs='\n'.join(f'[L{i}] {text}' for i,text in evidence) or 'No specific failure pattern is established by the available lines.'
    q=question.lower()
    if initial or any(x in q for x in ('happen','explain this','summarize this','anomal','what is wrong')):
        return f"{info['title']}\n\nEstimated severity: {info['severity']}\n\nEvidence\n{refs}\n\nPossible cause\n{info['cause']}\n\nWhat to check\n{info['check']}\n\nPrevention\n{info['prevent']}\n\nThese are investigation suggestions, not a confirmed root cause."
    if any(x in q for x in ('prevent','avoid','future','mitigat')):return f"Prevention\n{info['prevent']}\n\nValidate the cause first: {info['check']}\n\nEvidence\n{refs}"
    if any(x in q for x in ('why','cause','reason')):return f"Possible cause\n{info['cause']}\n\nEvidence\n{refs}\n\nThe sequence alone cannot confirm the cause. {info['check']}"
    if any(x in q for x in ('check','fix','next','recover','resolve')):return f"Recommended checks\n{info['check']}\n\nVerify service recovery before taking corrective action.\n\nEvidence\n{refs}"
    if any(x in q for x in ('sever','serious','impact','urgent')):return f"Estimated severity: {info['severity']}\n\n{info['cause']} Actual impact depends on redundancy and affected workloads. Confirm using metrics and surrounding logs.\n\nEvidence\n{refs}"
    return 'The available sequence does not provide enough evidence to answer that specific question. I can help examine its symptoms, possible causes, severity, checks, and prevention. Add surrounding logs or operational context for a deeper investigation.'
