from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True)
class ValidationIssue:
    row: int
    field: str
    message: str

    def to_dict(self):
        return {'row': self.row, 'field': self.field, 'message': self.message}


def validate_log_mapping(value, row=1, max_message_length=1_000_000):
    if isinstance(max_message_length, bool) or not isinstance(max_message_length, int) or max_message_length < 1:
        raise ValueError('max_message_length must be a positive integer')
    if not isinstance(value, Mapping):
        return [ValidationIssue(row, 'record', 'expected an object')]
    issues = []
    message = value.get('message')
    if not isinstance(message, str) or not message.strip():
        issues.append(ValidationIssue(row, 'message', 'expected nonempty text'))
    elif len(message) > max_message_length:
        issues.append(ValidationIssue(row, 'message', 'message exceeds length limit'))
    elif '\x00' in message:
        issues.append(ValidationIssue(row, 'message', 'message contains a null byte'))
    for key in ('source', 'severity'):
        if key in value and not isinstance(value[key], str):
            issues.append(ValidationIssue(row, key, 'expected text'))
    attributes = value.get('attributes', {})
    if not isinstance(attributes, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in attributes.items()
    ):
        issues.append(ValidationIssue(row, 'attributes', 'expected string keys and values'))
    return issues


def validate_log_rows(rows, max_message_length=1_000_000):
    return [issue for index, row in enumerate(rows, 1)
            for issue in validate_log_mapping(row, index, max_message_length)]
