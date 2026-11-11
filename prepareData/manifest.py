"""Windowing manifest: record what produced a train/test split so it can be reproduced."""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

MANIFEST_NAME = 'manifest.json'
CHUNK_BYTES = 1 << 20


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fin:
        for chunk in iter(lambda: fin.read(CHUNK_BYTES), b''):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path, hash_file=True):
    """Return size and (optionally) sha256 of path. Hashing a very large log can take minutes."""
    info = {'bytes': os.path.getsize(path)}
    if hash_file:
        info['sha256'] = file_sha256(path)
    return info


def git_commit(cwd=None):
    """Return the current git commit hash, or None when not in a git checkout."""
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=cwd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def build_manifest(script, params, inputs, outputs, seed=None, stats=None, hash_files=True):
    """Build a manifest dict. inputs/outputs are lists of file paths; params and stats are plain dicts."""
    return {
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'script': script,
        'params': params,
        'seed': seed,
        'git_commit': git_commit(),
        'python': sys.version.split()[0],
        'pandas': pd.__version__,
        'inputs': {path: describe_file(path, hash_files) for path in inputs},
        'outputs': {path: describe_file(path, hash_files) for path in outputs},
        'stats': stats or {},
    }


def write_manifest(output_dir, manifest, name=MANIFEST_NAME):
    path = os.path.join(output_dir, name)
    with open(path, 'w') as fout:
        json.dump(manifest, fout, indent=2, default=str)
    return path
