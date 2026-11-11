"""Log parsing and sequence helpers derived from the existing LogLLM pipeline.

Descriptive entry points coexist with the original public names below.
"""
import os
from numbers import Integral
import re
from datetime import datetime

import pandas as pd


def build_fixed_windows(raw_data, window_size, step_size):
    """Group rows in input order, retaining a short final window.

    Labels are reference answers: any anomalous event labels its window.
    """
    for name, value in (("window_size", window_size), ("step_size", step_size)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    missing = {"Content", "Label"} - set(raw_data.columns)
    if missing:
        raise ValueError(f"Missing window input columns: {sorted(missing)}")
    windows = []
    for offset in range(0, len(raw_data), step_size):
        section = raw_data.iloc[offset:offset + window_size]
        event_labels = section['Label']
        windows.append((section['Content'].values, max(event_labels), event_labels.tolist()))
    return pd.DataFrame(windows, columns=['Content', 'Label', 'item_Label'])



def build_time_windows(raw_data, para):
    log_size = len(raw_data)
    if not log_size:
        return pd.DataFrame(columns=list(raw_data.columns) + ['item_Label'])
    label_data, time_data = raw_data.iloc[:, 1], raw_data.iloc[:, 0]
    deltaT_data, content = raw_data.iloc[:, 2], raw_data.iloc[:, 3]
    new_data, pairs = [], set()
    start_time, end_time = time_data.iloc[0], time_data.iloc[0] + para['window_size']
    start_index = end_index = 0
    for cur_time in time_data:
        if cur_time < end_time:
            end_index += 1
        else:
            break
    pairs.add((start_index, end_index))
    while end_index < log_size:
        start_time += para['step_size']
        end_time = start_time + para['window_size']
        start_index = next((i for i in range(start_index, log_size) if time_data.iloc[i] >= start_time), log_size)
        end_index = next((i for i in range(end_index, log_size) if time_data.iloc[i] >= end_time), log_size)
        if start_index != end_index:
            pairs.add((start_index, end_index))
    for start_index, end_index in pairs:
        dt = deltaT_data.iloc[start_index:end_index].to_numpy(copy=True)
        if len(dt):
            dt[0] = 0
        new_data.append([
            time_data.iloc[start_index:end_index].values,
            max(label_data.iloc[start_index:end_index]),
            dt,
            content.iloc[start_index:end_index].values,
            label_data.iloc[start_index:end_index].values.tolist(),
        ])
    return pd.DataFrame(new_data, columns=list(raw_data.columns) + ['item_Label'])


def read_log_records(log_file, regex, headers, start_line=0, end_line=None):
    log_messages = []
    with open(log_file, 'r', encoding='latin-1') as fin:
        for line_number, line in enumerate(fin):
            if line_number < start_line:
                continue
            if end_line is not None and line_number >= end_line:
                break
            match = regex.search(line.strip())
            if match:
                log_messages.append([match.group(header) for header in headers])
    print(f'Total size is {len(log_messages)}; Total size after encoding is {len(log_messages)}')
    return pd.DataFrame(log_messages, columns=headers)


def compile_log_format(logformat):
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for index, splitter in enumerate(splitters):
        if index % 2 == 0:
            regex += re.sub(' +', lambda _match: r'\s+', splitter)
        else:
            header = splitter.strip('<').strip('>')
            regex += f'(?P<{header}>.*?)'
            headers.append(header)
    return headers, re.compile('^' + regex + '$')


def write_structured_log(input_dir, output_dir, log_name, log_format, start_line=0, end_line=None, use_cache=False):
    source_path = os.path.join(input_dir, log_name)
    destination_path = os.path.join(output_dir, log_name + "_structured.csv")
    # Only whole-file parses can reuse a CSV without range metadata.
    if (use_cache and start_line == 0 and end_line is None
            and os.path.isfile(destination_path)
            and os.path.getmtime(destination_path) >= os.path.getmtime(source_path)):
        return
    print('Structuring file: ' + os.path.join(input_dir, log_name))
    start_time = datetime.now()
    headers, regex = compile_log_format(log_format)
    df_log = read_log_records(os.path.join(input_dir, log_name), regex, headers, start_line, end_line)
    os.makedirs(output_dir, exist_ok=True)
    df_log.to_csv(os.path.join(output_dir, log_name + '_structured.csv'), index=False)
    print('Structuring done. [Time taken: {!s}]'.format(datetime.now() - start_time))


def export_sequence_split(sequences, output_dir, split_name):
    """Write the existing CSV schema and summary text for one prepared split."""
    serialized = sequences.copy()
    serialized['session_length'] = serialized['Content'].map(len)
    serialized['Content'] = serialized['Content'].map(' ;-; '.join)
    lengths = serialized['session_length']
    anomalies = serialized['Label'].sum()
    summary = (
        f"max session length: {lengths.max()}; mean session length: {lengths.mean()}\n"
        f"number of anomalous sessions: {anomalies}; "
        f"number of normal sessions: {len(serialized) - anomalies}; "
        f"number of total sessions: {len(serialized)}\n"
    )
    serialized.to_csv(os.path.join(output_dir, split_name + '.csv'), index=False)
    with open(os.path.join(output_dir, split_name + '_info.txt'), 'w') as stream:
        stream.write(summary)
    print(f'{split_name.title()} dataset info:')
    print(summary)
    return serialized


# Compatibility exports retain the original positional and keyword signatures.
fixedSize_window = build_fixed_windows
sliding_window = build_time_windows
log_to_dataframe = read_log_records
generate_logformat_regex = compile_log_format
structure_log = write_structured_log
