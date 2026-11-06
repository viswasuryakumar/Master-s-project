from pathlib import Path

import streamlit as st

from bgl_demo import analyze_text


st.set_page_config(page_title='BGL Signal Lab', page_icon='B', layout='wide', initial_sidebar_state='expanded')

st.markdown(
    '''
    <style>
    :root { --ink:#17202a; --muted:#64748b; --line:#dbe4ee; --blue:#2166f3; }
    .stApp { background: #f6f8fb; color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #101a2b; }
    [data-testid="stSidebar"] * { color: #edf4ff; }
    [data-testid="stSidebar"] .stCaption { color: #aebbd0; }
    .hero { padding: 1.6rem 0 1.2rem; }
    .hero h1 { margin: 0; font-size: 2.25rem; letter-spacing: -0.03em; color: #17202a; }
    .hero p { margin: .45rem 0 0; color: #64748b; max-width: 62rem; font-size: 1.03rem; }
    .baseline { margin: .8rem 0 1.2rem; padding: .75rem 1rem; border: 1px solid #c8d8f7; border-radius: 12px; background: #edf4ff; color: #24436f; }
    .section-title { margin: 1.3rem 0 .55rem; color: #17202a; font-size: 1.2rem; font-weight: 700; }
    div[data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 12px; padding: .8rem 1rem; }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    .stButton > button[kind="primary"] { background: var(--blue); border-color: var(--blue); }
    </style>
    ''',
    unsafe_allow_html=True,
)


def metric_values(records):
    y_true, y_pred = records['Anomaly'], records['Predicted anomaly']
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(records) if len(records) else 0.0
    return accuracy, f1, precision, recall, [[tn, fp], [fn, tp]]


sample = Path('data/BGL/raw-excerpt.log').read_text(encoding='latin-1')

with st.sidebar:
    st.markdown('## BGL Signal Lab')
    st.caption('A transparent log review workspace')
    st.divider()
    source = st.radio('Log source', ['Bundled sample', 'Upload a file', 'Paste records'])
    st.divider()
    st.markdown('### Sequence settings')
    window_size = st.number_input('Window size', min_value=1, value=2, step=1, help='Records per grouped sequence.')
    step_size = st.number_input('Step size', min_value=1, value=2, step=1, help='Records to advance between groups.')
    threshold = st.number_input('Anomaly threshold', min_value=1, value=1, step=1, help='Keyword hits required for a baseline anomaly prediction.')
    st.divider()
    st.caption('Baseline terms: error, fail, fatal, critical, exception, corrupt, fault, unable, panic, timeout, denied')

if source == 'Bundled sample':
    text = sample
    source_note = 'Using the bundled six-record sample.'
elif source == 'Upload a file':
    uploaded = st.file_uploader('Choose a BGL log file', type=['log', 'txt'])
    text = uploaded.getvalue().decode('latin-1') if uploaded else ''
    source_note = uploaded.name if uploaded else 'Choose a file to begin.'
else:
    text = st.text_area('Paste BGL records', value=sample, height=190)
    source_note = 'Pasted input stays in this session and is not written to the dataset.'

st.markdown('<div class="hero"><h1>Find the signal in your logs.</h1><p>Parse BGL records, group them into fixed-size sequences, and review transparent baseline anomaly predictions.</p></div>', unsafe_allow_html=True)
st.markdown('<div class="baseline"><strong>Baseline prediction method</strong><br>This CPU-friendly keyword scorer is for demonstration only. It is not trained model inference.</div>', unsafe_allow_html=True)

top_left, top_right = st.columns([4, 1])
with top_left:
    st.caption(source_note)
with top_right:
    analyze = st.button('Analyze logs', type='primary', use_container_width=True)

if analyze:
    try:
        st.session_state['analysis'] = analyze_text(text, window_size, step_size, threshold)
    except Exception as exc:
        st.error(f'Could not analyze these logs: {exc}')

if 'analysis' not in st.session_state:
    st.info('Choose a source and select Analyze logs to see parsed records and evaluation results.')
else:
    records, grouped = st.session_state['analysis']
    accuracy, f1, precision, recall, matrix = metric_values(records)
    anomaly_count = int(records['Predicted anomaly'].sum())
    normal_count = len(records) - anomaly_count

    st.markdown('<div class="section-title">Run summary</div>', unsafe_allow_html=True)
    summary = st.columns(4)
    summary[0].metric('Records parsed', len(records))
    summary[1].metric('Predicted anomalies', anomaly_count, delta=f'{anomaly_count} flagged')
    summary[2].metric('Normal predictions', normal_count)
    summary[3].metric('Sequences', len(grouped))

    st.markdown('<div class="section-title">Parsed records</div>', unsafe_allow_html=True)
    anomaly_only = st.checkbox('Show anomaly predictions only')
    visible = records[records['Predicted anomaly'].eq(1)] if anomaly_only else records
    st.dataframe(visible[['Label', 'Id', 'Date', 'Time', 'Level', 'Message', 'Predicted anomaly', 'Reason', 'Split']], use_container_width=True, hide_index=True)

    records_tab, sequences_tab, evaluation_tab = st.tabs(['Records', 'Grouped sequences', 'Evaluation'])
    with records_tab:
        st.caption('Ground-truth labels come from the BGL label field: “-” is normal; any other label is anomalous.')
        st.dataframe(visible[['Id', 'Label', 'Message', 'Score', 'Reason']], use_container_width=True, hide_index=True)
    with sequences_tab:
        st.dataframe(grouped[['Split', 'Sequence anomaly', 'Record labels', 'Sequence']], use_container_width=True, hide_index=True)
        st.download_button('Download grouped sequences CSV', grouped.to_csv(index=False), 'bgl_sequences.csv', 'text/csv', use_container_width=True)
    with evaluation_tab:
        st.write(f"Training records: {(records['Split'] == 'train').sum()}  ·  Testing records: {(records['Split'] == 'test').sum()}  ·  Training sequences: {(grouped['Split'] == 'train').sum()}  ·  Testing sequences: {(grouped['Split'] == 'test').sum()}")
        if records['Label'].notna().all():
            metric_cols = st.columns(4)
            for col, label, value in zip(metric_cols, ['Accuracy', 'F1 score', 'Precision', 'Recall'], [accuracy, f1, precision, recall]):
                col.metric(label, f'{value:.3f}')
            st.caption('Confusion matrix · rows = actual, columns = predicted')
            st.table({'Predicted normal': [matrix[0][0], matrix[1][0]], 'Predicted anomalous': [matrix[0][1], matrix[1][1]]})
        else:
            st.info('Evaluation metrics are unavailable because these records do not contain ground-truth labels.')

    st.download_button('Download predictions CSV', records.to_csv(index=False), 'bgl_predictions.csv', 'text/csv', use_container_width=True)

with st.expander('How it works'):
    st.write('The parser reads the fixed BGL fields, maps “-” to normal and every other label to anomalous, then applies the existing fixed-size grouping helper. Predictions are based on configured anomaly terms. Evaluation metrics use the BGL ground-truth labels when present.')
