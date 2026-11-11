"""Benchmark and audit the log-message masking done by customDataset.normalize_log_message.

customDataset.py imports torch, so the patterns are read from its source with ast instead of
importing it. Every variant here must give the same output as the original; benchmark() checks that.
"""
import ast
import functools
import os
import re
import time
from collections import Counter

SEQUENCE_SEPARATOR = ' ;-; '
MASK = '<*>'
ELLIPSIS_PATTERN = r'[.]{3,}'
ELLIPSIS_REPLACEMENT = '.. '
DEFAULT_SOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'customDataset.py')
CACHE_SIZE = 65536
PUNCTUATION = '.,;:()[]{}"\'='
HEX_WORD = re.compile(r'[0-9a-fA-F]{6,}')
LONG_TOKEN_LENGTH = 20
TOP_TOKENS = 20


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


def benchmark(rows, patterns, repeat=3, variants=VARIANTS):
    """Time each variant over rows (best of `repeat` runs, caches cleared before each run).

    Every variant's output is compared with the baseline's, so a faster variant that changes the
    masked text shows up as identical=False instead of looking like a win.
    """
    if len(rows) == 0:
        raise ValueError('no rows to benchmark')
    if repeat < 1:
        raise ValueError('repeat must be at least 1')
    expected = [Baseline(patterns)(row) for row in rows]
    results = []
    for variant in variants:
        normalize = variant(patterns)
        best = None
        for _ in range(repeat):
            normalize.reset()
            start = time.perf_counter()
            outputs = [normalize(row) for row in rows]
            elapsed = time.perf_counter() - start
            best = elapsed if best is None else min(best, elapsed)
        wrong = [index for index, (got, want) in enumerate(zip(outputs, expected)) if got != want]
        entry = {'name': normalize.name, 'seconds': best,
                 'rows_per_second': len(rows) / best if best > 0 else None,
                 'identical': not wrong, 'mismatches': len(wrong), 'first_mismatch': None,
                 'cache': normalize.cache_info()}
        if wrong:
            entry['first_mismatch'] = {'row': wrong[0], 'got': outputs[wrong[0]][:200],
                                       'expected': expected[wrong[0]][:200]}
        results.append(entry)
    base = next((entry['seconds'] for entry in results if entry['name'] == Baseline.name), None)
    for entry in results:
        entry['speedup_vs_baseline'] = base / entry['seconds'] if base and entry['seconds'] > 0 else None
    return results


def profile_patterns(rows, patterns, repeat=3):
    """Time each pattern scanning the ellipsis-collapsed rows on its own, slowest first.

    This shows which patterns are relatively expensive; it is not the cost inside the combined
    alternation, which stops trying alternatives at a position once one matches.
    """
    if repeat < 1:
        raise ValueError('repeat must be at least 1')
    ellipsis = re.compile(ELLIPSIS_PATTERN)
    collapsed = [ellipsis.sub(ELLIPSIS_REPLACEMENT, row) for row in rows]
    entries = []
    for index, pattern in enumerate(patterns):
        compiled = re.compile(pattern)
        best = None
        for _ in range(repeat):
            matches = 0
            start = time.perf_counter()
            for text in collapsed:
                for _match in compiled.finditer(text):
                    matches += 1
            elapsed = time.perf_counter() - start
            best = elapsed if best is None else min(best, elapsed)
        entries.append({'index': index, 'pattern': pattern, 'seconds': best, 'matches': matches})
    total = sum(entry['seconds'] for entry in entries)
    for entry in entries:
        entry['share'] = entry['seconds'] / total if total > 0 else 0.0
    return sorted(entries, key=lambda entry: entry['seconds'], reverse=True)


def classify_residual(token):
    """Reasons a token left after masking still looks like a variable value (empty if it does not).

    has_digit should never occur: the last mask pattern matches any token containing a digit, so a
    hit means a masking gap. hex_like and long are heuristics for identifiers the patterns miss.
    """
    core = token.replace(MASK, '').strip(PUNCTUATION)
    reasons = []
    if any(char.isdigit() for char in core):
        reasons.append('has_digit')
    if HEX_WORD.fullmatch(core):
        reasons.append('hex_like')
    if len(core) >= LONG_TOKEN_LENGTH:
        reasons.append('long')
    return reasons


def audit_masking(messages, normalize, top=TOP_TOKENS):
    """Report what is left after masking: unmasked variable-looking tokens and rarely seen tokens.

    singleton tokens (seen once in the whole input) are a proxy for variable fields that leaked
    through masking; they also include genuinely rare words, so read the samples before acting.
    """
    if len(messages) == 0:
        raise ValueError('no messages to audit')
    masked = [normalize(message) for message in messages]
    residual = Counter()
    reasons_by_token = {}
    vocabulary = Counter()
    mask_total = 0
    with_residual = 0
    for text in masked:
        mask_total += text.count(MASK)
        found = False
        for token in text.split():
            if token != MASK:
                vocabulary[token] += 1
            reasons = classify_residual(token)
            if reasons:
                residual[token] += 1
                reasons_by_token[token] = reasons
                found = True
        with_residual += found
    by_reason = Counter()
    for token, count in residual.items():
        for reason in reasons_by_token[token]:
            by_reason[reason] += count
    singletons = [token for token, count in vocabulary.items() if count == 1]
    return {
        'messages': len(messages),
        'distinct_raw_messages': len(set(messages)),
        'distinct_masked_messages': len(set(masked)),
        'masks_per_message': mask_total / len(messages),
        'messages_with_unmasked_tokens': with_residual,
        'fraction_with_unmasked_tokens': with_residual / len(messages),
        'unmasked_occurrences_by_reason': dict(by_reason),
        'distinct_unmasked_tokens': len(residual),
        'top_unmasked_tokens': [{'token': token, 'count': count, 'reasons': reasons_by_token[token]}
                                for token, count in residual.most_common(top)],
        'distinct_tokens': len(vocabulary),
        'singleton_tokens': len(singletons),
        'singleton_fraction': len(singletons) / len(vocabulary) if vocabulary else 0.0,
        'singleton_samples': singletons[:top],
    }
