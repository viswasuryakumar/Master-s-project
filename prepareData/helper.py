import os
import re
from datetime import datetime

import pandas as pd


def fixedSize_window(raw_data, window_size, step_size):
    aggregated = [
        [
            raw_data['Content'].iloc[i:i + window_size].values,
            max(raw_data['Label'].iloc[i:i + window_size]),
            raw_data['Label'].iloc[i:i + window_size].values.tolist(),
        ]
        for i in range(0, len(raw_data), step_size)
    ]
    return pd.DataFrame(aggregated, columns=list(raw_data.columns) + ['item_Label'])


def sliding_window(raw_data, para):
    log_size = raw_data.shape[0]
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
        dt = deltaT_data.iloc[start_index:end_index].values
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


def log_to_dataframe(log_file, regex, headers, start_line=0, end_line=None):
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


def generate_logformat_regex(logformat):
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


def structure_log(input_dir, output_dir, log_name, log_format, start_line=0, end_line=None):
    print('Structuring file: ' + os.path.join(input_dir, log_name))
    start_time = datetime.now()
    headers, regex = generate_logformat_regex(log_format)
    df_log = log_to_dataframe(os.path.join(input_dir, log_name), regex, headers, start_line, end_line)
    os.makedirs(output_dir, exist_ok=True)
    df_log.to_csv(os.path.join(output_dir, log_name + '_structured.csv'), index=False)
    print('Structuring done. [Time taken: {!s}]'.format(datetime.now() - start_time))
