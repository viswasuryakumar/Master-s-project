import re
from numbers import Integral


def build_message_windows(messages, window_size=100):
    if isinstance(window_size, bool) or not isinstance(window_size, Integral) or window_size < 1:
        raise ValueError('window_size must be a positive integer')
    return [
        {'external_id': f'window-{offset // window_size + 1:06}',
         'logs': messages[offset:offset + window_size]}
        for offset in range(0, len(messages), window_size)
    ]


def group_block_messages(messages):
    grouped = {}
    for message in messages:
        block_ids = sorted(set(re.findall(r'blk_-?\d+', message)))
        if not block_ids:
            raise ValueError('Every HDFS raw message must contain a block ID; use prepared sequences for other records')
        for block_id in block_ids:
            grouped.setdefault(block_id, []).append(message)
    return [
        {'external_id': block_id, 'block_id': block_id, 'logs': records}
        for block_id, records in grouped.items()
    ]
