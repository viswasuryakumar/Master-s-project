"""First UI-only slice for the LogLLM workbench."""

import streamlit as st


st.set_page_config(page_title="LogLLM Workbench", page_icon="LL", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
:root { --ink: #17202a; --muted: #66717d; --line: #dce4e8; --paper: #f7faf9; --teal: #0f766e; }
.stApp { background: var(--paper); color: var(--ink); }
[data-testid="stSidebar"] { background: #102b2b; }
[data-testid="stSidebar"] * { color: #e8f2ee !important; }
[data-testid="stSidebar"] label { color: #b9cec7 !important; }
h1, h2, h3, p, label, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { font-family: 'DM Sans', sans-serif; }
h1 { letter-spacing: -0.045em; font-weight: 700; font-size: 3rem; margin-bottom: .2rem; }
h2 { letter-spacing: -0.025em; }
[data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; }
[data-testid="stMetricValue"] { color: var(--ink); }
[data-testid="stMetricLabel"] { color: var(--muted); }
.lede { color: var(--muted); font-size: 1.05rem; max-width: 760px; margin-bottom: 1.6rem; }
.status-pill { display: inline-block; border-radius: 999px; padding: 5px 10px; font-size: .78rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; background: #d9f1e5; color: #17613d; }
.mono { font-family: 'Space Mono', monospace; font-size: .82rem; }
.note { background: #edf5f3; border: 1px solid #c9dfd9; border-radius: 12px; padding: 14px 16px; color: #24413c; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## LogLLM\n**Anomaly detection workbench**")
    st.caption("A focused workspace for exploring system log health.")
    st.divider()
    st.markdown("### Analysis controls")
    input_source = st.radio("Input source", ["Upload a log file", "Paste log records"])
    uploaded_file = None
    pasted_records = ""
    if input_source == "Upload a log file":
        uploaded_file = st.file_uploader("Choose a .log or .txt file", type=["log", "txt"])
    else:
        pasted_records = st.text_area("Paste one record per line", height=150, placeholder="Paste BGL-format records here")
    st.slider("Sequence window", 1, 20, 2, disabled=True)
    st.slider("Sequence step", 1, 20, 2, disabled=True)
    st.divider()
    st.caption("Engine")
    st.markdown('<span class="status-pill">Ready for analysis</span>', unsafe_allow_html=True)

st.title("LogLLM Workbench")
st.markdown('<p class="lede">Upload, inspect, and evaluate log anomaly predictions in one focused workspace.</p>', unsafe_allow_html=True)
selected_source = uploaded_file.name if uploaded_file else ("Pasted records" if pasted_records.strip() else "Waiting for input")
st.markdown(f'<span class="status-pill">Source: {selected_source}</span>', unsafe_allow_html=True)
st.write("")

metrics = st.columns(5)
metrics[0].metric("Records", "--")
metrics[1].metric("Predicted anomalies", "--")
metrics[2].metric("F1 score", "--")
metrics[3].metric("Precision", "--")
metrics[4].metric("Recall", "--")

st.info("Your input is selected and ready for the analysis pipeline. Prediction and evaluation modules will be connected in a later commit.")

tab_overview, tab_predictions, tab_sequences, tab_evaluation, tab_guide = st.tabs(["Overview", "Predictions", "Sequences", "Evaluation", "How it works"])
with tab_overview:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Prediction distribution")
        st.write("An overview chart will appear after a log file is analyzed.")
    with right:
        st.subheader("Run summary")
        st.markdown(f'<div class="note"><div class="mono">SELECTED SOURCE</div><b>{selected_source}</b><br><br>Input controls are ready. Analysis will be connected next.</div>', unsafe_allow_html=True)
with tab_predictions:
    st.subheader("Record-level predictions")
    st.write("Predictions will be listed here after analysis.")
with tab_sequences:
    st.subheader("Grouped sequences")
    st.write("Grouped log sequences will be listed here after analysis.")
with tab_evaluation:
    st.subheader("Evaluation metrics")
    st.write("Accuracy, precision, recall, F1, and the confusion matrix will appear here.")
with tab_guide:
    st.subheader("What this workspace will do")
    st.write("The workbench will parse log records, group them into sequences, produce anomaly predictions, and evaluate labeled results.")
