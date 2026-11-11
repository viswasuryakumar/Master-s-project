"""Deterministic synthetic log generation and classification; never model inference."""
import re
from datetime import datetime, timezone

PROFILES = {
    'BGL': {
        'description': 'Supercomputer system events',
        'format': 'Label · timestamp · location · component · level · message',
        'grouping': '100-message windows',
    },
    'HDFS': {
        'description': 'Distributed file system events',
        'format': 'Date · time · process · level · component · block event',
        'grouping': 'Block-ID sessions',
    },
    'Liberty': {
        'description': 'Cluster operating-system events',
        'format': 'Label · timestamp · host · service · message',
        'grouping': '100-message windows',
    },
    'Thunderbird': {
        'description': 'Cluster service and system events',
        'format': 'Label · timestamp · host · service · message',
        'grouping': '100-message windows',
    },
}

SCENARIOS = {
    'memory': {
        'title': 'Memory integrity errors',
        'severity': 'High',
        'cause': 'Repeated memory parity failures can indicate unstable hardware or an uncorrectable memory condition.',
        'check': 'Inspect hardware error counters and diagnostic reports; correlate the affected host with workload failures.',
        'prevent': 'Monitor corrected-error trends, schedule memory diagnostics, and replace failing hardware after confirmation.',
    },
    'storage': {
        'title': 'Block storage failure',
        'severity': 'High',
        'cause': 'A block could not be read or replicated successfully. Disk health, replica availability, or a damaged block are possible causes.',
        'check': 'Check the block identifier, replica locations, disk health, and the surrounding storage-service logs.',
        'prevent': 'Track disk and replica health, verify replication policies, and test recovery from unavailable replicas.',
    },
    'connection': {
        'title': 'Service connection failures',
        'severity': 'Medium',
        'cause': 'A remote service is unavailable or communication is failing. Logs alone cannot establish whether the cause is networking or the remote process.',
        'check': 'Check service health, name resolution, network reachability, connection limits, and recent deployments.',
        'prevent': 'Use bounded retries with backoff, timeouts, service health monitoring, and capacity alerts.',
    },
    'capacity': {
        'title': 'Resource exhaustion',
        'severity': 'High',
        'cause': 'A process reports exhausted memory or filesystem capacity. Workload growth, leaks, or retention settings may contribute.',
        'check': 'Inspect memory and disk utilization, process limits, workload changes, and retention policies.',
        'prevent': 'Set resource budgets and early utilization alerts; review log rotation and capacity planning.',
    },
    'normal': {
        'title': 'Routine operation',
        'severity': 'Informational',
        'cause': 'The available sequence shows routine activity without a recognized failure pattern. This is not proof that the system is healthy.',
        'check': 'Correlate with service metrics and neighboring sequences if a user-visible failure was observed.',
        'prevent': 'Maintain health checks, baseline monitoring, and regular review of service logs.',
    },
    'unknown': {
        'title': 'Sequence requires investigation',
        'severity': 'Undetermined',
        'cause': 'The classification does not have enough supporting context to establish a specific failure type.',
        'check': 'Review adjacent sequences and system metrics. Confirm the affected host, component, and timing.',
        'prevent': 'Improve contextual logging and establish alerts based on confirmed failure indicators.',
    },
}

PATTERNS = {
    'memory': r'uncorrectable|parity failure|machine check',
    'storage': r'checksum mismatch|missing replica|block verification failed|disk i/o error',
    'connection': r'connection refused|connection timed out|failed to connect|network unreachable',
    'capacity': r'out of memory|no space left|resource exhausted|oom-killer',
}

# Order in which scenarios are tried when classifying; the first match wins.
DEFAULT_PRIORITY = ['capacity', 'connection', 'memory', 'storage']
PRIORITY = {
    'HDFS': ['storage', 'capacity', 'connection', 'memory'],
    'BGL': ['memory', 'connection', 'capacity', 'storage'],
}

SEVERITY_PATTERN = re.compile(r'\b(DEBUG|INFO|NOTICE|WARN(?:ING)?|ERROR|FATAL|CRITICAL|ALERT)\b')
HDFS_PATTERN = re.compile(r'^(\d{6}) (\d{6}) (\d+) (\w+) ([^:]+):\s*(.*)$')
DATE_PATTERN = re.compile(r'\d{4}\.\d{2}\.\d{2}')
BLOCK_PATTERN = re.compile(r'blk_-?\d+')
RECORD_FIELDS = 10  # label, id, date, host, time/month, ..., message

# Scenarios cycled through by generate() for the anomalous sequences.
ANOMALY_SCENARIOS = ['memory', 'storage', 'connection', 'capacity']
NORMAL_MESSAGES = [
    'service heartbeat completed',
    'background maintenance cycle finished',
    'workload execution completed successfully',
    'checkpoint committed successfully',
    'network interface operational',
]
DEFAULT_EPOCH = 1768474800


def _split_record(raw):
    """Split a BGL/Liberty/Thunderbird record into its fields, or None if it is not one."""
    parts = raw.split(maxsplit=RECORD_FIELDS - 1)
    if len(parts) == RECORD_FIELDS and parts[1].isdigit() and DATE_PATTERN.fullmatch(parts[2]):
        return parts
    return None


def parse(profile, raw):
    """Extract timestamp, host, component, severity, content and block id from one raw line."""
    content = raw
    stamp = host = component = None
    severity = 'INFO'
    found = SEVERITY_PATTERN.search(raw)
    if found:
        severity = found.group(1).replace('WARNING', 'WARN')

    if profile == 'HDFS':
        match = HDFS_PATTERN.match(raw)
        if match:
            stamp = f'{match[1]} {match[2]}'
            severity, component, content = match[4], match[5], match[6]
    elif profile == 'BGL':
        parts = _split_record(raw)
        if parts:
            stamp, host = parts[4], parts[3]
            component = f'{parts[6]}.{parts[7]}'
            severity, content = parts[8], parts[9]
    else:
        parts = _split_record(raw)
        if parts:
            stamp = f'{parts[2]} {parts[6]}'
            host = parts[3]
            component = parts[8].rstrip(':')
            content = parts[9]

    block = BLOCK_PATTERN.search(content)
    return {
        'raw': raw,
        'content': content,
        'original_time': stamp,
        'host': host,
        'component': component,
        'severity': severity,
        'block_id': block.group() if block else None,
    }


def classify(profile, messages):
    """Return (prediction, scenario, score) from failure patterns in the message content."""
    # Strip benchmark annotations before matching content: labels are never features.
    text = '\n'.join(parse(profile, line)['content'] for line in messages).lower()
    for scenario in PRIORITY.get(profile, DEFAULT_PRIORITY):
        if re.search(PATTERNS[scenario], text):
            return 'anomalous', scenario, 0.94
    return 'normal', 'normal', 0.88


def _scenario_for(profile, number):
    """Every seventh sequence is anomalous; HDFS has no memory failures."""
    if number % 7 != 3:
        return 'normal'
    scenario = ANOMALY_SCENARIOS[(number // 7) % len(ANOMALY_SCENARIOS)]
    if profile == 'HDFS' and scenario == 'memory':
        return 'storage'
    return scenario


def _failure_message(scenario, block):
    return {
        'memory': 'uncorrectable memory parity failure reported by hardware monitor',
        'storage': f'block verification failed: checksum mismatch for {block}',
        'connection': 'failed to connect to service endpoint: connection refused',
        'capacity': 'resource exhausted: no space left on device',
    }[scenario]


def _routine_message(profile, number, i, block):
    if profile == 'HDFS':
        return [
            f'Receiving block {block} src: /10.20.1.{number % 200 + 1}:50010 dest: /10.20.2.8:50010',
            f'PacketResponder for block {block} completed',
            f'BLOCK* addStoredBlock: blockMap updated for {block} size 67108864',
        ][i % 3]
    return NORMAL_MESSAGES[i % len(NORMAL_MESSAGES)] + f' operation={number * 100 + i}'


def _level_and_message(scenario, number, i, count, block, routine):
    """Overlay the failure and recovery lines on the routine message for sequence position i."""
    if scenario == 'normal':
        return 'INFO', routine
    if i in (count // 2, count // 2 + 1, count // 2 + 2):
        return 'ERROR', _failure_message(scenario, block)
    if i == count - 2:
        return 'WARN', f'recovery attempt scheduled for operation={number}'
    if i == count - 1:
        return 'INFO', 'recovery cycle completed; monitor subsequent operations'
    return 'INFO', routine


def _format_line(profile, number, when, level, message, block, host):
    if profile == 'HDFS':
        if block not in message:
            message += f' {block}'
        return f'{when:%y%m%d %H%M%S} {100 + number % 500} {level} dfs.DataNode: {message}'
    label = 'ALERT' if level == 'ERROR' else '-'
    if profile == 'BGL':
        location = f'R{number % 32:02}-M0-N0-C:J01-U01'
        return (f'{label} {int(when.timestamp())} {when:%Y.%m.%d} {location} '
                f'{when:%Y-%m-%d-%H.%M.%S}.000000 {location} RAS KERNEL {level} {message}')
    return (f'{label} {int(when.timestamp())} {when:%Y.%m.%d} {host} {when:%b} {when.day} {when:%H:%M:%S} '
            f'{host}/{host} systemd[{100 + number % 500}]: {level} {message}')


def generate(profile, number, count=None, epoch=DEFAULT_EPOCH, namespace=0):
    """Build the number-th synthetic sequence for a profile, with its generated ground truth."""
    count = count or (20 + (number % 5) * 10 if profile == 'HDFS' else 100)
    scenario = _scenario_for(profile, number)
    block = f'blk_{1000000000000 + namespace * 1000000 + number}'
    host = f'node{(number % 48) + 1:03}'

    lines = []
    for i in range(count):
        when = datetime.fromtimestamp(epoch + number * 100 + i, tz=timezone.utc)
        routine = _routine_message(profile, number, i, block)
        level, message = _level_and_message(scenario, number, i, count, block, routine)
        lines.append(_format_line(profile, number, when, level, message, block, host))

    return {
        'external_id': block if profile == 'HDFS' else f'seq-{number + 1:06}',
        'logs': lines,
        'block_id': block if profile == 'HDFS' else None,
        'original_time': parse(profile, lines[0])['original_time'],
        'truth': 'anomalous' if scenario != 'normal' else 'normal',
        'scenario': scenario,
    }


# Question keywords, tried in order; the first group that matches picks the answer.
SUMMARY_WORDS = ('happen', 'explain this', 'summarize this', 'anomal', 'what is wrong')
PREVENT_WORDS = ('prevent', 'avoid', 'future', 'mitigat')
CAUSE_WORDS = ('why', 'cause', 'reason')
CHECK_WORDS = ('check', 'fix', 'next', 'recover', 'resolve')
SEVERITY_WORDS = ('sever', 'serious', 'impact', 'urgent')
NO_EVIDENCE_ANSWER = (
    'The available sequence does not provide enough evidence to answer that specific question. '
    'I can help examine its symptoms, possible causes, severity, checks, and prevention. '
    'Add surrounding logs or operational context for a deeper investigation.'
)


def _evidence_lines(profile, logs, scenario):
    """Up to three numbered lines whose content matches the scenario's failure pattern."""
    if scenario not in PATTERNS:
        return []
    found = []
    for number, line in enumerate(logs, start=1):
        content = parse(profile, line)['content']
        if re.search(PATTERNS[scenario], content, re.I):
            found.append((number, content))
    return found[:3]


def _mentions(question, words):
    return any(word in question for word in words)


def explain(profile, logs, prediction, question, initial=False):
    """Answer an investigation question from the sequence's own evidence, with numbered references."""
    _, scenario, _ = classify(profile, logs)
    if prediction == 'anomalous' and scenario == 'normal':
        scenario = 'unknown'
    info = SCENARIOS[scenario]
    evidence = _evidence_lines(profile, logs, scenario)
    refs = '\n'.join(f'[L{n}] {text}' for n, text in evidence) or \
        'No specific failure pattern is established by the available lines.'
    q = question.lower()

    if initial or _mentions(q, SUMMARY_WORDS):
        return (f"{info['title']}\n\nEstimated severity: {info['severity']}\n\nEvidence\n{refs}\n\n"
                f"Possible cause\n{info['cause']}\n\nWhat to check\n{info['check']}\n\n"
                f"Prevention\n{info['prevent']}\n\nThese are investigation suggestions, not a confirmed root cause.")
    if _mentions(q, PREVENT_WORDS):
        return f"Prevention\n{info['prevent']}\n\nValidate the cause first: {info['check']}\n\nEvidence\n{refs}"
    if _mentions(q, CAUSE_WORDS):
        return (f"Possible cause\n{info['cause']}\n\nEvidence\n{refs}\n\n"
                f"The sequence alone cannot confirm the cause. {info['check']}")
    if _mentions(q, CHECK_WORDS):
        return (f"Recommended checks\n{info['check']}\n\n"
                f"Verify service recovery before taking corrective action.\n\nEvidence\n{refs}")
    if _mentions(q, SEVERITY_WORDS):
        return (f"Estimated severity: {info['severity']}\n\n{info['cause']} Actual impact depends on redundancy "
                f"and affected workloads. Confirm using metrics and surrounding logs.\n\nEvidence\n{refs}")
    return NO_EVIDENCE_ANSWER
