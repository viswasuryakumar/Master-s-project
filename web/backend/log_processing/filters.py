from .timestamps import parse_timestamp


SEVERITY_ORDER = {'TRACE': 0, 'DEBUG': 1, 'INFO': 2, 'WARN': 3, 'ERROR': 4, 'CRITICAL': 5}
SEVERITY_ALIASES = {'WARNING': 'WARN', 'FATAL': 'CRITICAL', 'ERR': 'ERROR'}


def normalize_severity(value):
    name = value.strip().upper()
    return SEVERITY_ALIASES.get(name, name)


def filter_records(records, source=None, contains=None, minimum_severity=None,
                   start=None, end=None, case_sensitive=False):
    lower = parse_timestamp(start) if start is not None else None
    upper = parse_timestamp(end) if end is not None else None
    if lower and upper and lower > upper:
        raise ValueError('start must not be after end')
    threshold = None
    if minimum_severity is not None:
        threshold = SEVERITY_ORDER.get(normalize_severity(minimum_severity))
        if threshold is None:
            raise ValueError('unknown minimum severity')
    needle = contains if case_sensitive or contains is None else contains.casefold()
    for record in records:
        if source is not None and record.source != source:
            continue
        if threshold is not None and SEVERITY_ORDER.get(normalize_severity(record.severity), -1) < threshold:
            continue
        text = record.message if case_sensitive else record.message.casefold()
        if needle is not None and needle not in text:
            continue
        if lower or upper:
            if record.timestamp is None:
                continue
            if lower and record.timestamp < lower or upper and record.timestamp > upper:
                continue
        yield record
