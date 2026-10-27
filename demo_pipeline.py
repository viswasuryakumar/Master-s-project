"""CPU-friendly parsing, grouping, baseline prediction, and evaluation helpers."""

from io import StringIO
from pathlib import Path
import re

import pandas as pd

from prepareData.helper import fixedSize_window, generate_logformat_regex


BGL_FORMAT = "<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>"
SEPARATOR = " ;-; "
ANOMALY_TERMS = {
    "error": 2, "fail": 2, "fatal": 3, "critical": 3, "exception": 2,
    "corrupt": 2, "fault": 2, "unable": 1, "panic": 3, "timeout": 1, "denied": 1,
}


def run_demo(log_path=None, window_size=2, step_size=2, threshold=1):
    """Run the bundled sample without writing generated dataset files."""
    root = Path(__file__).resolve().parent
    log_path = Path(log_path or root / "data" / "BGL" / "raw-excerpt.log")
    result = analyze_text(log_path.read_text(encoding="latin-1"), window_size, step_size, threshold)
    result["log_path"] = str(log_path)
    return result


def analyze_text(text, window_size=2, step_size=2, threshold=1):
    """Parse BGL text, predict anomalies, group sequences, and score them."""
    headers, regex = generate_logformat_regex(BGL_FORMAT)
    parsed = _parse_text(text, regex, headers)
    if parsed.empty:
        raise ValueError("No valid BGL records were found. Check the expected 10-column format.")

    parsed["Truth"] = parsed["Label"].apply(lambda value: int(str(value).strip() != "-"))
    parsed["Content"] = parsed["Content"].astype(str)
    predictions = parsed["Content"].apply(lambda value: score_content(value, threshold))
    parsed["Prediction"] = predictions.apply(lambda item: item[0])
    parsed["Score"] = predictions.apply(lambda item: item[1])
    parsed["Reason"] = predictions.apply(lambda item: item[2])

    split_at = max(1, int(len(parsed) * 0.8))
    train_rows = parsed.iloc[:split_at].reset_index(drop=True)
    test_rows = parsed.iloc[split_at:].reset_index(drop=True)
    train_groups = _group(train_rows, window_size, step_size)
    test_groups = _group(test_rows, window_size, step_size)
    metrics = calculate_metrics(parsed["Truth"].tolist(), parsed["Prediction"].tolist())
    return {
        "parsed": parsed, "train_rows": train_rows, "test_rows": test_rows,
        "train_groups": train_groups, "test_groups": test_groups, "metrics": metrics,
        "window_size": window_size, "step_size": step_size, "threshold": threshold,
    }


def _parse_text(text, regex, headers):
    records = []
    for line in StringIO(text):
        line = line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=9)
        looks_like_full_bgl = len(fields) >= 10 and fields[1].isdigit() and re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", fields[2])
        if looks_like_full_bgl:
            match = regex.search(line)
            if match:
                records.append([match.group(header) for header in headers])
            continue

        # Also accept the compact format used by the UI examples:
        # LABEL DATE TIME CODE NODE COMPONENT: MESSAGE
        compact_fields = line.split(maxsplit=6)
        if len(compact_fields) >= 7:
            label, date, time, code, node, component, content = compact_fields
            values = {
                "Label": label,
                "Id": date,
                "Date": date,
                "Code1": code,
                "Time": time,
                "Code2": node,
                "Component1": component.rstrip(":"),
                "Component2": component.rstrip(":"),
                "Level": component.rstrip(":"),
                "Content": content,
            }
            records.append([values.get(header, "") for header in headers])
    return pd.DataFrame(records, columns=headers)


def score_content(content, threshold=1):
    lowered = content.lower()
    hits = [(term, weight) for term, weight in ANOMALY_TERMS.items() if re.search(rf"\b{re.escape(term)}\b", lowered)]
    score = sum(weight for _, weight in hits)
    prediction = int(score >= threshold)
    reason = ", ".join(term for term, _ in hits) if hits else "no anomaly terms"
    return prediction, score, reason


def calculate_metrics(truth, prediction):
    tp = sum(actual == 1 and predicted == 1 for actual, predicted in zip(truth, prediction))
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in zip(truth, prediction))
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in zip(truth, prediction))
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in zip(truth, prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(truth) if truth else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _group(rows, window_size, step_size):
    source = rows[["Content", "Truth"]].rename(columns={"Truth": "Label"})
    groups = fixedSize_window(source, window_size, step_size)
    prediction_groups = [rows["Prediction"].iloc[i:i + window_size].values.tolist() for i in range(0, len(rows), step_size)]
    score_groups = [rows["Score"].iloc[i:i + window_size].values.tolist() for i in range(0, len(rows), step_size)]
    groups["Sequence"] = groups["Content"].apply(lambda values: SEPARATOR.join(values))
    groups["Prediction"] = [int(max(values)) for values in prediction_groups]
    groups["Score"] = [int(max(values)) for values in score_groups]
    return groups[["Sequence", "Label", "item_Label", "Prediction", "Score"]]

