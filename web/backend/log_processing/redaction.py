import ipaddress
import re
from dataclasses import dataclass, replace
from collections import Counter


@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: dict[str, int]


SECRET_ASSIGNMENT = re.compile(r'(?i)\b(password|passwd|api[_-]?key|access[_-]?token|secret)\s*([=:])\s*(?:"[^"\r\n]*"|\x27[^\x27\r\n]*\x27|[^\s,;]+)')
BEARER = re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+')
EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
IPV4 = re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])')


def redact_text(text, mask_addresses=True):
    counts = Counter()

    def secret(match):
        counts['secret'] += 1
        return match[1] + match[2] + '[REDACTED]'

    text = SECRET_ASSIGNMENT.sub(secret, text)
    text, count = BEARER.subn('Bearer [REDACTED]', text)
    counts['bearer'] += count
    text, count = EMAIL.subn('[EMAIL]', text)
    counts['email'] += count
    if mask_addresses:
        def address(match):
            try:
                ipaddress.IPv4Address(match[0])
            except ipaddress.AddressValueError:
                return match[0]
            counts['address'] += 1
            return '[IP]'
        text = IPV4.sub(address, text)
    return RedactionResult(text, {key: count for key, count in counts.items() if count})


def redact_records(records, mask_addresses=True):
    output, totals = [], Counter()
    for record in records:
        message = redact_text(record.message, mask_addresses)
        source = redact_text(record.source, mask_addresses)
        totals.update(message.counts)
        totals.update(source.counts)
        attributes = {}
        for key, value in record.attributes.items():
            if key.lower().replace('-', '_') in ('password', 'passwd', 'api_key', 'apikey', 'access_token', 'secret'):
                attributes[key] = '[REDACTED]'
                totals['secret'] += 1
            else:
                result = redact_text(value, mask_addresses)
                attributes[key] = result.text
                totals.update(result.counts)
        output.append(replace(record, message=message.text, source=source.text, attributes=attributes))
    return output, dict(totals)
