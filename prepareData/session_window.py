import argparse
import os
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

if __package__:
    from .helper import export_sequence_split, write_structured_log, structure_log
    from .manifest import build_manifest, write_manifest
else:
    from helper import export_sequence_split, write_structured_log, structure_log
    from manifest import build_manifest, write_manifest

DEFAULT_DATA_DIR = r'/mnt/public/gw/SyslogData/HDFS_v1'
DEFAULT_LOG_NAME = 'HDFS.log'
DEFAULT_SEED = 42
TRAIN_RATIO = 0.8


def collect_block_sessions(records, block_labels):
    messages_by_block = defaultdict(list)
    for message in tqdm(records['Content'], total=len(records)):
        for block_id in set(re.findall(r'(blk_-?\d+)', message)):
            messages_by_block[block_id].append(message)
    sessions = pd.DataFrame(messages_by_block.items(), columns=['BlockId', 'Content'])
    labels_by_block = {
        block_id: int(label == 'Anomaly')
        for block_id, label in zip(block_labels['BlockId'], block_labels['Label'])
    }

    sessions['Label'] = sessions['BlockId'].map(labels_by_block.get)
    return sessions


def partition_block_sessions(sessions, train_ratio=TRAIN_RATIO, seed=DEFAULT_SEED):
    split_position = int(train_ratio * len(sessions))
    shuffled = sessions.sample(frac=1, random_state=seed).reset_index(drop=True)
    return (shuffled.iloc[:split_position].copy(),
            shuffled.iloc[split_position:].reset_index(drop=True).copy())


def main(argv=None):
    arguments = argparse.ArgumentParser(description='Build session-based train/test data for HDFS logs.')
    arguments.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    arguments.add_argument('--log-name', default=DEFAULT_LOG_NAME)
    arguments.add_argument('--use-cache', action='store_true', help='reuse an up-to-date structured csv')
    arguments.add_argument('--seed', type=int, default=DEFAULT_SEED, help='seed for the session shuffle before the split')
    arguments.add_argument('--skip-hash', action='store_true', help='skip sha256 of large files in manifest.json')
    options = arguments.parse_args(argv)
    layout = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
    write_structured_log(options.data_dir, options.data_dir, options.log_name, layout,
                         use_cache=options.use_cache)
    structured_path = os.path.join(options.data_dir, options.log_name + '_structured.csv')
    records = pd.read_csv(structured_path, engine='c', na_filter=False,
                          memory_map=True, dtype={'Date': object, 'Time': object})
    print(f'number of messages in {structured_path} is {len(records)}')
    label_path = os.path.join(options.data_dir, 'anomaly_label.csv')
    block_labels = pd.read_csv(label_path)
    training, testing = partition_block_sessions(collect_block_sessions(records, block_labels), seed=options.seed)
    exported = {name: export_sequence_split(sequences, options.data_dir, name)
                for name, sequences in (('train', training), ('test', testing))}
    manifest = build_manifest(
        'prepareData/session_window.py',
        {'log_name': options.log_name, 'log_format': layout, 'window_type': 'session_by_block_id',
         'train_ratio': TRAIN_RATIO, 'split': 'seeded random shuffle'},
        inputs=[os.path.join(options.data_dir, options.log_name), label_path],
        outputs=[structured_path] + [os.path.join(options.data_dir, name + '.csv') for name in exported],
        seed=options.seed,
        stats={f'{name}_sessions': len(rows) for name, rows in exported.items()}
              | {f'{name}_anomalous': int(rows['Label'].sum()) for name, rows in exported.items()},
        hash_files=not options.skip_hash)
    print('Wrote ' + write_manifest(options.data_dir, manifest))


if __name__ == '__main__':
    main()
