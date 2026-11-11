import argparse
import os.path

import pandas as pd

if __package__:
    from .helper import (build_fixed_windows, export_sequence_split, write_structured_log,
                         sliding_window, fixedSize_window, structure_log)
    from .manifest import build_manifest, write_manifest
else:
    from helper import (build_fixed_windows, export_sequence_split, write_structured_log,
                        sliding_window, fixedSize_window, structure_log)
    from manifest import build_manifest, write_manifest

DEFAULT_DATA_DIR = r'/mnt/public/gw/SyslogData/BGL'
DEFAULT_LOG_NAME = 'BGL.log'
WINDOW_SIZE = 100
STEP_SIZE = 100
TRAIN_RATIO = 0.8
start_line = 0
end_line = None


def select_log_format(log_name):
    filename = log_name.lower()
    if any(source in filename for source in ('thunderbird', 'spirit', 'liberty')):
        return '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'
    if 'bgl' in filename:
        return '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'
    raise Exception('missing valid log format')


def prepare_window_splits(records, window_size=WINDOW_SIZE, step_size=STEP_SIZE, train_ratio=TRAIN_RATIO):
    labeled_records = records.copy()
    labeled_records['Label'] = labeled_records['Label'].map(lambda label: int(label != '-'))
    split_position = int(train_ratio * len(labeled_records))
    partitions = (labeled_records.iloc[:split_position], labeled_records.iloc[split_position:])
    return tuple(build_fixed_windows(partition[['Content', 'Label']], window_size, step_size)
                 for partition in partitions)


def main(argv=None):
    arguments = argparse.ArgumentParser(description='Build train/test windows for BGL, Liberty or Thunderbird logs.')
    arguments.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    arguments.add_argument('--log-name', default=DEFAULT_LOG_NAME)
    arguments.add_argument('--use-cache', action='store_true', help='reuse an up-to-date structured csv')
    arguments.add_argument('--skip-hash', action='store_true', help='skip sha256 of large files in manifest.json')
    options = arguments.parse_args(argv)
    layout = select_log_format(options.log_name)
    print(f'Auto log_format: {layout}')
    write_structured_log(options.data_dir, options.data_dir, options.log_name, layout,
                         start_line=start_line, end_line=end_line, use_cache=options.use_cache)
    print(f'window_size: {WINDOW_SIZE}; step_size: {STEP_SIZE}')
    records = pd.read_csv(os.path.join(options.data_dir, options.log_name + '_structured.csv'))
    print(len(records))
    print('Start grouping.')
    training, testing = prepare_window_splits(records)
    exported = {name: export_sequence_split(sequences, options.data_dir, name)
                for name, sequences in (('train', training), ('test', testing))}
    structured_path = os.path.join(options.data_dir, options.log_name + '_structured.csv')
    manifest = build_manifest(
        'prepareData/sliding_window.py',
        {'log_name': options.log_name, 'log_format': layout, 'window_type': 'fixed_size',
         'window_size': WINDOW_SIZE, 'step_size': STEP_SIZE, 'train_ratio': TRAIN_RATIO,
         'start_line': start_line, 'end_line': end_line, 'split': 'chronological'},
        inputs=[os.path.join(options.data_dir, options.log_name)],
        outputs=[structured_path] + [os.path.join(options.data_dir, name + '.csv') for name in exported],
        stats={f'{name}_windows': len(rows) for name, rows in exported.items()}
              | {f'{name}_anomalous': int(rows['Label'].sum()) for name, rows in exported.items()},
        hash_files=not options.skip_hash)
    print('Wrote ' + write_manifest(options.data_dir, manifest))


if __name__ == '__main__':
    main()
