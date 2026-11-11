from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping


@dataclass(frozen=True)
class LogRecord:
    message: str
    source: str = ''
    severity: str = 'UNKNOWN'
    timestamp: datetime | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError('message must contain text')
        if not isinstance(self.source, str) or not isinstance(self.severity, str):
            raise ValueError('source and severity must be strings')
        if self.timestamp is not None:
            if not isinstance(self.timestamp, datetime) or self.timestamp.utcoffset() is None:
                raise ValueError('timestamp must include a timezone')
        if not isinstance(self.attributes, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.attributes.items()
        ):
            raise ValueError('attributes must map strings to strings')
        object.__setattr__(self, 'attributes', dict(self.attributes))

    def to_dict(self):
        return {
            'message': self.message,
            'source': self.source,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'attributes': dict(self.attributes),
        }


@dataclass(frozen=True)
class LogSequence:
    identifier: str
    records: tuple[LogRecord, ...]

    def __post_init__(self):
        if not isinstance(self.identifier, str) or not self.identifier.strip():
            raise ValueError('identifier must contain text')
        records = tuple(self.records)
        if not records or any(not isinstance(record, LogRecord) for record in records):
            raise ValueError('sequence must contain LogRecord instances')
        object.__setattr__(self, 'records', records)

    def to_dict(self):
        return {'identifier': self.identifier, 'records': [record.to_dict() for record in self.records]}
