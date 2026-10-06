"""Evaluate existing LogLLM checkpoints, including on Apple Silicon."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from customDataset import CustomDataset, CustomCollator

ROOT_DIR = Path(__file__).resolve().parent


def parse_prediction(text):
    match = re.search(r"\b(normal|anomalous)\b", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Model did not return normal/anomalous: {text!r}")
    return int(match.group(1).lower() == "anomalous")


def evalModel(model, dataloader, device=None, output_dir=None, trace_output=None):
    device = model.device if device is None else device
    model.eval()
    preds, responses = [], []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            trace = None
            if trace_output is not None and not preds:
                from pipeline_trace import PipelineTrace
                trace = PipelineTrace(model, batch)
            try:
                outputs_ids = model(batch['inputs'].to(device), batch['seq_positions'])
                outputs = model.Llama_tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
                if trace is not None:
                    trace.finish(outputs_ids, outputs)
            finally:
                if trace is not None:
                    trace.close()
            for text in outputs:
                # Never silently count an unparseable response as normal.
                preds.append(parse_prediction(text))
                responses.append(text)
            if trace is not None:
                trace.save(trace_output)
    gt = np.asarray(dataloader.dataset.get_label())
    if len(preds) != len(gt) or not len(gt):
        raise ValueError("Expected one prediction for every sequence in a nonempty dataset.")
    metrics = {
        "samples": len(preds),
        "accuracy": float(accuracy_score(gt, preds)),
        "precision": float(precision_score(gt, preds, zero_division=0)),
        "recall": float(recall_score(gt, preds, zero_division=0)),
        "f1": float(f1_score(gt, preds, zero_division=0)),
    }
    print(json.dumps(metrics, indent=2))
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        with (output_dir / "predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sequence_index", "label", "prediction", "response"])
            writer.writerows((i, int(label), pred, response)
                             for i, (label, pred, response) in enumerate(zip(gt, preds, responses)))
        print(f"Saved results to {output_dir}")
    return metrics

