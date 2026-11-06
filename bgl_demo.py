from pathlib import Path
import pandas as pd

from prepareData.helper import fixedSize_window, generate_logformat_regex, log_to_dataframe


BGL_FORMAT = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'


def parse_bgl_text(text):
    path = Path('_bgl_input.tmp')
    path.write_text(text, encoding='latin-1')
    try:
        headers, regex = generate_logformat_regex(BGL_FORMAT)
        frame = log_to_dataframe(str(path), regex, headers)
    finally:
        path.unlink(missing_ok=True)
    if frame.empty:
        raise ValueError('No valid BGL records were found.')
    frame['Anomaly'] = frame['Label'].ne('-').astype(int)
    frame['Timestamp'] = pd.to_datetime(frame['Time'], format='%Y-%m-%d-%H.%M.%S.%f', errors='coerce')
    frame['Message'] = frame['Content'].astype(str)
    return frame


def analyze_text(text, window_size=2, step_size=2, threshold=1):
    records = parse_bgl_text(text)
    terms = ('error', 'fail', 'fatal', 'critical', 'exception', 'corrupt', 'fault', 'unable', 'panic', 'timeout', 'denied')
    reasons = records['Message'].str.lower().apply(lambda message: ', '.join(term for term in terms if term in message))
    records['Score'] = reasons.map(lambda reason: len([x for x in reason.split(', ') if x]))
    records['Predicted anomaly'] = (records['Score'] >= threshold).astype(int)
    records['Reason'] = reasons.where(records['Score'].gt(0), 'no anomaly term')
    grouped = fixedSize_window(records[['Content', 'Anomaly']].rename(columns={'Anomaly': 'Label'}), window_size, step_size)
    grouped = grouped.rename(columns={'Label': 'Sequence anomaly', 'item_Label': 'Record labels'})
    grouped['Sequence'] = grouped['Content'].apply(lambda values: ' | '.join(values))
    grouped['Sequence anomaly'] = grouped['Sequence anomaly'].astype(int)
    split = max(1, int(len(records) * 0.8)) if len(records) > 1 else len(records)
    records['Split'] = ['train' if i < split else 'test' for i in range(len(records))]
    grouped['Split'] = ['train' if i < max(1, int(len(grouped) * 0.8)) else 'test' for i in range(len(grouped))]
    return records, grouped
