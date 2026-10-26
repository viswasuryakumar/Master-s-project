"""LogLLM anomaly detection workbench."""

import pandas as pd
import streamlit as st

from demo_pipeline import analyze_text, run_demo


st.set_page_config(page_title="LogLLM Workbench", page_icon="LL", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
:root { --ink: #17202a; --muted: #66717d; --line: #dce4e8; --paper: #f7faf9; --teal: #0f766e; }
.stApp { background: var(--paper); color: var(--ink); }
[data-testid="stSidebar"] { background: #102b2b; }
[data-testid="stSidebar"] * { color: #e8f2ee !important; }
[data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stSlider label { color: #b9cec7 !important; }
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
button[kind="primary"] { background: var(--teal); border-color: var(--teal); }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## LogLLM\n**Anomaly detection workbench**")
    st.caption("Turn raw BGL records into inspectable predictions.")
    mode = st.radio("Input source", ["Bundled BGL sample", "Upload BGL log", "Paste BGL records"])
    raw_text = None
    source_name = "Bundled sample"
    if mode == "Upload BGL log":
        uploaded = st.file_uploader("Choose a .log or .txt file", type=["log", "txt"])
        if uploaded:
            raw_text = uploaded.getvalue().decode("latin-1")
            source_name = uploaded.name
        else:
            source_name = "No file selected"
    elif mode == "Paste BGL records":
        raw_text = st.text_area("Paste one BGL record per line", height=180, placeholder="- 1117838571 2005.06.03 ... message")
        source_name = "Pasted records"
    st.divider()
    st.markdown("### Analysis controls")
    window_size = st.slider("Sequence window", 1, 20, 2)
    step_size = st.slider("Sequence step", 1, 20, 2)
    threshold = st.slider("Anomaly threshold", 1, 6, 1, help="Higher values require more anomaly evidence in a message.")
    st.divider()
    st.caption("Engine")
    st.markdown('<span class="status-pill">CPU baseline active</span>', unsafe_allow_html=True)
    st.caption("Transparent keyword scoring is active. Full LogLLM model inference can be connected later.")


try:
    if mode == "Upload BGL log" and raw_text is None:
        raise ValueError("Choose a BGL log file before running analysis.")
    result = run_demo(window_size=window_size, step_size=step_size, threshold=threshold) if raw_text is None else analyze_text(raw_text, window_size, step_size, threshold)
    error = None
except ValueError as exc:
    result = None
    error = str(exc)

st.title("LogLLM Workbench")
st.markdown('<p class="lede">Upload, inspect, and evaluate log anomaly predictions in one focused workspace.</p>', unsafe_allow_html=True)

if error:
    st.error(error)
    st.stop()

parsed = result["parsed"]
metrics = result["metrics"]
st.markdown(f'<span class="status-pill">Analyzed: {source_name}</span>', unsafe_allow_html=True)
st.write("")

metric_cols = st.columns(5)
metric_cols[0].metric("Records", len(parsed))
metric_cols[1].metric("Predicted anomalies", int(parsed["Prediction"].sum()))
metric_cols[2].metric("F1 score", f"{metrics['f1']:.3f}")
metric_cols[3].metric("Precision", f"{metrics['precision']:.3f}")
metric_cols[4].metric("Recall", f"{metrics['recall']:.3f}")

st.info("Predictions use a transparent CPU baseline that scores terms such as error, fault, fatal, and timeout. F1 is calculated because the BGL input includes ground-truth labels.")

tab_overview, tab_predictions, tab_sequences, tab_evaluation, tab_guide = st.tabs(["Overview", "Predictions", "Sequences", "Evaluation", "How it works"])

with tab_overview:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Prediction distribution")
        distribution = pd.DataFrame({"count": [int((parsed["Prediction"] == 0).sum()), int((parsed["Prediction"] == 1).sum())]}, index=["Normal", "Anomalous"])
        st.bar_chart(distribution, color="#0f766e")
    with right:
        st.subheader("Run summary")
        st.markdown(f"""
        <div class="note">
        <div class="mono">SOURCE</div><b>{source_name}</b><br><br>
        <div class="mono">WINDOW / STEP</div><b>{window_size} / {step_size}</b><br><br>
        <div class="mono">THRESHOLD</div><b>{threshold}</b><br><br>
        The current result is a prediction pass only. No model training or dataset files were written.
        </div>
        """, unsafe_allow_html=True)

with tab_predictions:
    st.subheader("Record-level predictions")
    view = parsed[["Truth", "Prediction", "Score", "Reason", "Id", "Time", "Level", "Content"]].copy()
    view["Truth"] = view["Truth"].map({0: "Normal", 1: "Anomalous"})
    view["Prediction"] = view["Prediction"].map({0: "Normal", 1: "Anomalous"})
    anomaly_only = st.checkbox("Show anomalous records only", help="Focus the table on records predicted as anomalous.")
    visible_view = view[view["Prediction"] == "Anomalous"] if anomaly_only else view
    st.caption(f"Showing {len(visible_view)} of {len(view)} records")
    st.dataframe(visible_view, width="stretch", hide_index=True, height=460)
    st.download_button(
        "Download prediction report",
        data=visible_view.to_csv(index=False).encode("utf-8"),
        file_name="logllm_predictions.csv",
        mime="text/csv",
        help="Save the visible record-level predictions as a CSV file.",
    )

with tab_sequences:
    st.subheader("Grouped sequences")
    st.caption("A sequence is anomalous when at least one record in that window is predicted anomalous.")
    sequence_view = pd.concat([result["train_groups"].assign(Split="Train"), result["test_groups"].assign(Split="Test")], ignore_index=True)
    st.dataframe(sequence_view[["Split", "Prediction", "Score", "Label", "item_Label", "Sequence"]], width="stretch", hide_index=True, height=420)
    st.download_button(
        "Download sequence report",
        data=sequence_view.to_csv(index=False).encode("utf-8"),
        file_name="logllm_sequences.csv",
        mime="text/csv",
        help="Save grouped train and test sequences as a CSV file.",
    )

with tab_evaluation:
    st.subheader("Evaluation against BGL labels")
    st.caption("These metrics compare the baseline prediction with the label embedded in each BGL record.")
    eval_cols = st.columns(4)
    eval_cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    eval_cols[1].metric("True positives", metrics["tp"])
    eval_cols[2].metric("False positives", metrics["fp"])
    eval_cols[3].metric("False negatives", metrics["fn"])
    matrix = pd.DataFrame([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], index=["Actual normal", "Actual anomalous"], columns=["Predicted normal", "Predicted anomalous"])
    st.dataframe(matrix, width="stretch")
    with st.expander("How to read these metrics"):
        metric_guide = pd.DataFrame({
            "Metric": ["F1 score", "Precision", "Recall", "Accuracy"],
            "Meaning": [
                "Balance between precision and recall.",
                "Of predicted anomalies, how many were truly anomalous.",
                "Of true anomalies, how many the detector found.",
                "Share of all records classified correctly.",
            ],
        })
        st.dataframe(metric_guide, width="stretch", hide_index=True)
    st.warning("For unlabeled production logs, predictions are available but precision, recall, and F1 cannot be validated until labels are supplied.")

with tab_guide:
    st.subheader("What this version does")
    st.markdown("""
    1. Parses BGL-format records using the original LogLLM parsing helper.
    2. Scores anomaly evidence in each message using a transparent CPU baseline.
    3. Groups records into fixed-size sequences using the original grouping helper.
    4. Shows predictions, evidence, sequence labels, and evaluation metrics.

    The full LogLLM fine-tuned model is not loaded by this laptop-friendly UI. It can be connected later on a machine with the required model files and runtime.
    """)
