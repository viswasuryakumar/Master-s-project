"""Prepare ordered fixed-size sequences using the existing LogLLM formats."""
import argparse
import os.path

import pandas as pd

if __package__:
    from .helper import (build_fixed_windows, export_sequence_split, write_structured_log,
                         sliding_window, fixedSize_window, structure_log)
else:
    from helper import (build_fixed_windows, export_sequence_split, write_structured_log,
                        sliding_window, fixedSize_window, structure_log)

DEFAULT_DATA_DIR = r'/mnt/public/gw/SyslogData/BGL'
DEFAULT_LOG_NAME = 'BGL.log'
start_line = 0
end_line = None


def select_log_format(log_name):
    filename = log_name.lower()
    if any(source in filename for source in ('thunderbird', 'spirit', 'liberty')):
        return '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'
    if 'bgl' in filename:
        return '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'
    raise Exception('missing valid log format')


def prepare_window_splits(records, window_size=100, step_size=100, train_ratio=0.8):
    """Split before windowing; preserve the input order within each partition."""
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
    options = arguments.parse_args(argv)
    layout = select_log_format(options.log_name)
    print(f'Auto log_format: {layout}')
    write_structured_log(options.data_dir, options.data_dir, options.log_name, layout,
                         start_line=start_line, end_line=end_line, use_cache=options.use_cache)
    print('window_size: 100; step_size: 100')
    records = pd.read_csv(os.path.join(options.data_dir, options.log_name + '_structured.csv'))
    print(len(records))
    print('Start grouping.')
    training, testing = prepare_window_splits(records)
    for split_name, sequences in (('train', training), ('test', testing)):
        export_sequence_split(sequences, options.data_dir, split_name)


if __name__ == '__main__':
    main()
