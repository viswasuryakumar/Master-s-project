"""Benchmark and audit the log-message masking done by customDataset.normalize_log_message.

customDataset.py imports torch, so the patterns are read from its source with ast instead of
importing it. Every variant here must give the same output as the original; benchmark() checks that.
"""
import ast
import functools
import os
import re

SEQUENCE_SEPARATOR = ' ;-; '
MASK = '<*>'
ELLIPSIS_PATTERN = r'[.]{3,}'
ELLIPSIS_REPLACEMENT = '.. '
DEFAULT_SOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'customDataset.py')
CACHE_SIZE = 65536


def load_patterns(source_path=DEFAULT_SOURCE):
    """Return the `patterns` list assigned at module level in customDataset.py, without importing it."""
    with open(source_path) as stream:
        tree = ast.parse(stream.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'patterns' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('no module-level `patterns` list in ' + source_path)


class Normalizer:
    """Callable with cache reset and statistics hooks so variants can be timed uniformly."""
    name = 'normalizer'

    def __call__(self, text):
        raise NotImplementedError

    def reset(self):
        pass

    def cache_info(self):
        return None


class Baseline(Normalizer):
    """The original: re.sub with the joined pattern string on every call."""
    name = 'baseline'

    def __init__(self, patterns):
        self.combined = '|'.join(patterns)

    def __call__(self, text):
        compact = re.sub(ELLIPSIS_PATTERN, ELLIPSIS_REPLACEMENT, text)
        return re.sub(self.combined, MASK, compact)


class Precompiled(Normalizer):
    """Same regexes compiled once, skipping re's per-call pattern-cache lookup."""
    name = 'precompiled'

    def __init__(self, patterns):
        self.ellipsis = re.compile(ELLIPSIS_PATTERN)
        self.mask = re.compile('|'.join(patterns))

    def __call__(self, text):
        return self.mask.sub(MASK, self.ellipsis.sub(ELLIPSIS_REPLACEMENT, text))


class CachedRows(Precompiled):
    """Precompiled plus a cache keyed on the whole input string."""
    name = 'cached_rows'

    def __init__(self, patterns, cache_size=CACHE_SIZE):
        super().__init__(patterns)
        self.cached = functools.lru_cache(maxsize=cache_size)(super().__call__)

    def __call__(self, text):
        return self.cached(text)

    def reset(self):
        self.cached.cache_clear()

    def cache_info(self):
        info = self.cached.cache_info()
        return {'hits': info.hits, 'misses': info.misses}


class CachedMessages(Precompiled):
    """Split a window into messages, mask each through a cache, and rejoin.

    Log messages repeat far more than whole windows do. This matches the original only if no
    pattern matches across the separator, which benchmark() verifies on the data it is given.
    """
    name = 'cached_messages'

    def __init__(self, patterns, cache_size=CACHE_SIZE):
        super().__init__(patterns)
        self.cached = functools.lru_cache(maxsize=cache_size)(super().__call__)

    def __call__(self, text):
        return SEQUENCE_SEPARATOR.join(self.cached(message) for message in text.split(SEQUENCE_SEPARATOR))

    def reset(self):
        self.cached.cache_clear()

    def cache_info(self):
        info = self.cached.cache_info()
        return {'hits': info.hits, 'misses': info.misses}


VARIANTS = (Baseline, Precompiled, CachedRows, CachedMessages)
