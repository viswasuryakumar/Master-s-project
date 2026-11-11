from datetime import datetime, timezone, tzinfo
import math


def parse_timestamp(value, assume_timezone=None):
    if assume_timezone is not None and not isinstance(assume_timezone, tzinfo):
        raise ValueError('assume_timezone must be a timezone')
    if isinstance(value, bool):
        raise ValueError('boolean timestamps are invalid')
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError('timestamp must be finite')
        try:
            parsed = datetime.fromtimestamp(value, timezone.utc)
        except (OverflowError, OSError, ValueError) as error:
            raise ValueError('timestamp is outside the supported range') from error
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
        except ValueError:
            parsed = None
            for pattern in ('%Y-%m-%d-%H.%M.%S.%f', '%y%m%d %H%M%S', '%Y.%m.%d %H:%M:%S'):
                try:
                    parsed = datetime.strptime(text, pattern)
                    break
                except ValueError:
                    continue
            if parsed is None:
                raise ValueError('unsupported timestamp format')
    else:
        raise ValueError('timestamp must be text, epoch seconds, or datetime')
    if parsed.utcoffset() is None:
        if assume_timezone is None:
            raise ValueError('timestamp has no timezone; provide assume_timezone')
        parsed = parsed.replace(tzinfo=assume_timezone)
    return parsed.astimezone(timezone.utc)


def time_bounds(records):
    stamps = [record.timestamp for record in records if record.timestamp is not None]
    return (min(stamps), max(stamps)) if stamps else (None, None)
