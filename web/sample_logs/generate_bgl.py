
from datetime import datetime,timezone,timedelta
from pathlib import Path
import json,random

def generate_sample(output_dir):
    rng=random.Random(20261110)
    out=Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    normal=[
     'instruction cache parity scrub completed',
     'L3 cache initialization completed successfully',
     'torus link training completed; link state active',
     'tree network collective operation completed',
     'compute job checkpoint committed successfully',
     'ECC memory scrub completed without errors',
     'I/O request completed successfully',
     'RAS environmental monitor reports temperature within limits',
     'job scheduler allocated partition successfully',
     'control daemon heartbeat received',
     'network interface operational',
     'compute task exited with status 0',
    ]
    scenarios=[
     ('KERNMC','memory','uncorrectable memory parity failure at physical address 0x000000014a008000','machine check interrupt: memory controller status 0x81000000'),
     ('NETFAIL','connection','failed to connect to service endpoint 10.64.2.14:7001: connection refused','connection timed out while waiting for service endpoint 10.64.2.14:7001'),
     ('DISKERR','storage','disk I/O error on device /dev/sdb sector 44128','block verification failed: checksum mismatch for block 44128'),
     ('FSFULL','capacity','resource exhausted: no space left on device /var/log','write operation failed: no space left on device /var/log'),
     ('KERNMEM','memory_capacity','out of memory: allocation of 67108864 bytes failed','oom-killer selected process worker for termination'),
     ('NETBIND','port','bind failed for 10.64.3.12:7001: address already in use','service unavailable: listener initialization failed'),
     ('AUTHFAIL','permissions','authentication failed for service identity compute-agent','permission denied while accessing /etc/cluster/agent.conf'),
     ('CONFIG','configuration','invalid configuration: missing required controller_host setting','service unavailable: failed to parse config /etc/cluster/agent.conf'),
     ('DNSFAIL','dns','temporary failure in name resolution for controller.cluster.internal','failed to connect to controller.cluster.internal:7001'),
     ('CERTFAIL','tls','certificate verify failed for controller.cluster.internal','failed to connect: TLS handshake rejected by peer'),
    ]
    start=datetime(2026,11,10,0,0,0,tzinfo=timezone.utc)
    sequences=[]
    for seq in range(120):
        anomalous=seq%10 in (2,5,8)
        scenario=scenarios[(seq//10*3+seq%10)%len(scenarios)]
        location=f'R{seq%32:02X}-M{seq%2}-N{seq%16:02X}-C:J{seq%32:02d}-U{seq%2:02d}'
        logs=[]
        for i in range(100):
            dt=start+timedelta(seconds=seq*100+i)
            label='-';level='INFO';component='KERNEL'
            message=rng.choice(normal)+f' job={41000+seq} rank={seq%128}'
            if i in (15,16) and not anomalous:
                level='WARN';message='link retry completed successfully; transfer resumed'
            if anomalous and i in (43,44,45,62):
                label=scenario[0];level='ERROR';message=scenario[2 if i in (43,44,62) else 3]
                if scenario[1] in ('connection','port','dns','tls'):component='NETWORK'
            if anomalous and i==70:
                level='WARN';message=f'recovery attempt scheduled; retry delay 30 seconds job={41000+seq}'
            if anomalous and i==95:
                message='recovery cycle completed; subsequent service health verification required'
            logs.append(f'{label} {int(dt.timestamp())} {dt:%Y.%m.%d} {location} {dt:%Y-%m-%d-%H.%M.%S}.000000 {location} RAS {component} {level} {message}')
        sequences.append({'sequence_id':f'bgl-window-{seq+1:04d}','logs':logs,'ground_truth':'anomalous' if anomalous else 'normal','original_time':f'{start+timedelta(seconds=seq*100):%Y-%m-%d-%H.%M.%S}.000000'})
    (out/'bgl_12000.log').write_text('\n'.join(line for s in sequences for line in s['logs'])+'\n')
    (out/'bgl_12000_labeled.json').write_text(json.dumps(sequences,indent=2)+'\n')
    print(f'Generated {sum(len(s["logs"]) for s in sequences)} messages, {len(sequences)} windows, {sum(s["ground_truth"]=="anomalous" for s in sequences)} anomalous windows; no supplied predictions.')


if __name__ == "__main__":
    generate_sample(Path(__file__).parent)
