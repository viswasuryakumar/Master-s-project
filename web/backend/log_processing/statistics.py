from collections import Counter
from statistics import mean, median

from .filters import normalize_severity


def summarize_records(records, top_limit=5):
    if isinstance(top_limit, bool) or not isinstance(top_limit, int) or top_limit < 0:
        raise ValueError('top_limit must be a nonnegative integer')
    severities, sources, messages = Counter(), Counter(), Counter()
    total = timestamped = characters = 0
    earliest = latest = None
    for record in records:
        total += 1
        characters += len(record.message)
        severities[normalize_severity(record.severity)] += 1
        sources[record.source or '(unspecified)'] += 1
        messages[record.message] += 1
        if record.timestamp is not None:
            timestamped += 1
            earliest = min(earliest, record.timestamp) if earliest else record.timestamp
            latest = max(latest, record.timestamp) if latest else record.timestamp
    duration = (latest - earliest).total_seconds() if earliest else None
    repeated = sorted(((text, count) for text, count in messages.items() if count > 1),
                      key=lambda item: (-item[1], item[0]))
    return {
        'records': total, 'timestamped': timestamped,
        'missing_timestamps': total - timestamped,
        'characters': characters, 'unique_messages': len(messages),
        'duplicate_messages': total - len(messages),
        'severity_counts': dict(sorted(severities.items())),
        'source_counts': dict(sorted(sources.items())),
        'first_timestamp': earliest.isoformat() if earliest else None,
        'last_timestamp': latest.isoformat() if latest else None,
        'duration_seconds': duration,
        'records_per_second': timestamped / duration if duration else None,
        'repeated_messages': [{'message': text, 'count': count} for text, count in repeated[:top_limit]],
    }


def summarize_sequences(sequences):
    lengths = [len(sequence.records) for sequence in sequences]
    return {
        'sequences': len(lengths), 'record_memberships': sum(lengths),
        'minimum_length': min(lengths) if lengths else None,
        'maximum_length': max(lengths) if lengths else None,
        'mean_length': mean(lengths) if lengths else None,
        'median_length': median(lengths) if lengths else None,
    }
