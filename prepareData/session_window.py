import argparse
import os
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

if __package__:
    from .helper import export_sequence_split, write_structured_log, structure_log
else:
    from helper import export_sequence_split, write_structured_log, structure_log

DEFAULT_DATA_DIR = r'/mnt/public/gw/SyslogData/HDFS_v1'
DEFAULT_LOG_NAME = 'HDFS.log'


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


def partition_block_sessions(sessions, train_ratio=0.8):
    split_position = int(train_ratio * len(sessions))
    shuffled = sessions.sample(frac=1).reset_index(drop=True)
    return (shuffled.iloc[:split_position].copy(),
            shuffled.iloc[split_position:].reset_index(drop=True).copy())


def main(argv=None):
    arguments = argparse.ArgumentParser(description='Build session-based train/test data for HDFS logs.')
    arguments.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    arguments.add_argument('--log-name', default=DEFAULT_LOG_NAME)
    arguments.add_argument('--use-cache', action='store_true', help='reuse an up-to-date structured csv')
    options = arguments.parse_args(argv)
    layout = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
    write_structured_log(options.data_dir, options.data_dir, options.log_name, layout,
                         use_cache=options.use_cache)
    structured_path = os.path.join(options.data_dir, options.log_name + '_structured.csv')
    records = pd.read_csv(structured_path, engine='c', na_filter=False,
                          memory_map=True, dtype={'Date': object, 'Time': object})
    print(f'number of messages in {structured_path} is {len(records)}')
    block_labels = pd.read_csv(os.path.join(options.data_dir, 'anomaly_label.csv'))
    training, testing = partition_block_sessions(collect_block_sessions(records, block_labels))
    for split_name, sequences in (('train', training), ('test', testing)):
        export_sequence_split(sequences, options.data_dir, split_name)


if __name__ == '__main__':
    main()
