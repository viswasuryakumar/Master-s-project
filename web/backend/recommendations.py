"""Evidence-based investigation checklists; no diagnosis or configuration changes."""
import re
from typing import NamedTuple

from .engine import parse

MAX_ISSUES = 3
MAX_EVIDENCE_PER_ISSUE = 2
EVIDENCE_TEXT_LIMIT = 300
ORIGIN = 'evidence_rules'


class Rule(NamedTuple):
    key: str
    title: str
    pattern: str
    summary: str
    checks: list


# Rules are checked in order and a message can match several; the first
# MAX_ISSUES matching rules are reported, so order expresses priority.
RULES = [
    Rule('port', 'Port already in use',
         r'address already in use|eaddrinuse|port .*already.*use',
         'Check which process owns the port and compare it with the configured listener.',
         ['Confirm the listener address and port in the service configuration.',
          'Identify the process holding that port and check for duplicate service instances.',
          'Resolve the confirmed conflict through the normal deployment process; do not terminate an unknown process.']),
    Rule('dns', 'Name resolution issue',
         r'could not resolve|name or service not known|temporary failure in name resolution|nxdomain|unknown host',
         'Verify the configured hostname and DNS resolution from the affected host.',
         ['Check the service endpoint hostname for spelling or environment mismatches.',
          'Check the affected host’s DNS resolver and relevant DNS records.',
          'Compare resolution from the affected host with a healthy host.']),
    Rule('connection', 'Connection failure',
         r'connection refused|failed to connect|connection timed out|network unreachable|no route to host|connection timeout',
         'Check the service listener, endpoint configuration, and network path.',
         ['Verify that the target service is running and listening on the intended address and port.',
          'Compare the client’s configured hostname and port with the server configuration.',
          'Check routing and narrowly scoped firewall or security-group rules between the affected endpoints; do not open all ports.',
          'Correlate timeouts with service health and connection limits before changing network settings.']),
    Rule('tls', 'Certificate or TLS issue',
         r'certificate verify failed|certificate has expired|ssl handshake|tls handshake|certificate mismatch',
         'Check certificate validity, hostname matching, trust chain, and system time.',
         ['Compare the endpoint hostname with the certificate names.',
          'Check expiry, issuer chain, client trust configuration, and system time.',
          'Renew or correct a confirmed certificate configuration problem; keep certificate verification enabled.']),
    Rule('permissions', 'Access or authentication failure',
         r'permission denied|access denied|authentication failed|unauthorized',
         'Check the service identity and the permissions required for this operation.',
         ['Identify the account or service identity used by the failed request.',
          'Check credential validity and the specific file, resource, or role permissions.',
          'Grant only the missing permission after confirming the expected access; avoid broad administrator access.']),
    Rule('configuration', 'Configuration validation failure',
         r'invalid config|configuration error|unknown configuration|missing required.*config|failed to parse.*config|invalid.*configuration',
         'Validate the relevant configuration and compare it with the last working version.',
         ['Locate the named configuration file, setting, or environment variable in the evidence.',
          'Validate its syntax, required values, and compatibility with the running version.',
          'Review recent deployment changes and test a correction before applying it.']),
    Rule('disk_capacity', 'Disk capacity issue',
         r'no space left|disk full|quota exceeded',
         'Check disk space, inode availability, quotas, and log retention.',
         ['Identify the affected volume and check free space, inodes, and quotas.',
          'Check log rotation, retention, and unexpected file growth.',
          'Use approved retention or capacity changes; do not delete application data blindly.']),
    Rule('memory_capacity', 'Memory pressure',
         r'out of memory|oom-killer|cannot allocate memory',
         'Check memory usage, process limits, and recent workload changes.',
         ['Check host and container memory usage and configured limits.',
          'Correlate process restarts with workload growth or a possible leak.',
          'Validate capacity or application changes before increasing resource limits.']),
    Rule('memory', 'Memory integrity error',
         r'uncorrectable|parity failure|machine check|correctable.*memory|memory.*error',
         'Inspect hardware error counters and diagnostics on the affected host.',
         ['Correlate the memory errors with workload failures and affected hardware.',
          'Review hardware diagnostics and corrected-error trends.',
          'Schedule maintenance if hardware failure is confirmed; a port change would not address this evidence.']),
    Rule('storage', 'Storage integrity or replica issue',
         r'checksum mismatch|missing replica|block verification failed|disk i/o error|corrupt',
         'Check the affected block or disk and verify replica health.',
         ['Identify the affected block or storage device from the evidence.',
          'Check disk health, replica availability, and surrounding storage-service logs.',
          'Follow the storage system’s verified recovery procedure after confirming the fault.']),
    Rule('capacity', 'Resource limit reached',
         r'resource exhausted|too many open files|connection limit',
         'Check resource usage against the service’s configured limits.',
         ['Identify whether the limit concerns files, connections, memory, or another resource.',
          'Review resource usage, leaks, workload changes, and per-process limits.',
          'Test a targeted capacity or configuration change after identifying the constrained resource.']),
]


def _no_anomaly():
    return {
        'title': 'Continue monitoring',
        'summary': 'No anomaly was flagged in this view. No corrective change is suggested.',
        'basis': 'Normal classification does not prove that the system is healthy.',
        'issues': [],
        'origin': ORIGIN,
    }


def _unexplained_anomaly():
    return {
        'title': 'Investigate the sequence',
        'summary': 'Review surrounding logs and system metrics to identify the cause before changing settings.',
        'basis': 'A detector flagged this sequence, but the messages do not establish a specific configuration, network, or port fault.',
        'issues': [],
        'origin': ORIGIN,
    }


def _matching_issues(messages):
    """Issues for the first MAX_ISSUES rules that match any message, each with its evidence lines."""
    issues = []
    for rule in RULES:
        evidence = [{'line': number, 'text': text[:EVIDENCE_TEXT_LIMIT]}
                    for number, text in enumerate(messages, start=1)
                    if re.search(rule.pattern, text, re.I)][:MAX_EVIDENCE_PER_ISSUE]
        if evidence:
            issues.append({'category': rule.key, 'title': rule.title, 'summary': rule.summary,
                           'checks': rule.checks, 'evidence': evidence})
        if len(issues) == MAX_ISSUES:
            break
    return issues


def recommend(profile, logs, flagged):
    """Return an investigation checklist for a flagged sequence, or a no-action note otherwise."""
    if not flagged:
        return _no_anomaly()
    issues = _matching_issues([parse(profile, line)['content'] for line in logs])
    if not issues:
        return _unexplained_anomaly()
    return {
        'title': issues[0]['title'],
        'summary': issues[0]['summary'],
        'basis': 'Suggested checks based on log evidence; the root cause is not confirmed. No server settings have been inspected or changed.',
        'issues': issues,
        'origin': ORIGIN,
    }
