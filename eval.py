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


def main():
    parser = argparse.ArgumentParser(description="Evaluate log anomaly checkpoints.")
    parser.add_argument("--dataset", choices=["BGL", "HDFS", "Liberty", "Thunderbird"], default="BGL")
    parser.add_argument("--data", type=Path, help="Prepared CSV with Content and binary Label columns")
    parser.add_argument("--bert-path", type=Path, default=ROOT_DIR / "models/bert-base-uncased")
    parser.add_argument("--llama-path", type=Path, default=ROOT_DIR / "models/Meta-Llama-3-8B")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to ft_model_<dataset>")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=20, help="First N sequences; 0 evaluates the entire file")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-content-len", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, help="Defaults to outputs/<dataset>-smoke")
    parser.add_argument("--check-only", action="store_true", help="Validate local files without loading models")
    parser.add_argument("--trace-output", type=Path, help="Save actual first-batch tensor shapes and excerpts for the walkthrough")
    args = parser.parse_args()
    if min(args.batch_size, args.max_seq_len, args.max_content_len) < 1 or args.limit < 0:
        parser.error("Batch size and length limits must be positive; --limit must be nonnegative.")
    data_path = args.data or ROOT_DIR / "data" / args.dataset / "test.csv"
    checkpoint = args.checkpoint or ROOT_DIR / f"ft_model_{args.dataset}"
    output_dir = args.output_dir or ROOT_DIR / "outputs" / f"{args.dataset}-smoke"
    required = [data_path, args.bert_path / "config.json", args.bert_path / "model.safetensors",
                args.bert_path / "vocab.txt", args.llama_path / "config.json",
                args.llama_path / "tokenizer.json", args.llama_path / "tokenizer_config.json",
                args.llama_path / "model.safetensors.index.json", checkpoint / "projector.pt"]
    for name in ("Bert_ft", "Llama_ft"):
        required.extend([checkpoint / name / "adapter_config.json", checkpoint / name / "adapter_model.safetensors"])
    index_path = args.llama_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        required.extend(args.llama_path / name for name in set(index["weight_map"].values()))
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        parser.exit(1, "Missing required files:\n  " + "\n  ".join(missing) + "\nSee RUN_LOCAL.md for downloads.\n")
    dataset = CustomDataset(data_path, max_samples=args.limit or None)
    if not len(dataset) or not np.isin(dataset.labels, [0, 1]).all():
        parser.error("The selected dataset must be nonempty and have labels 0 or 1.")
    if args.check_only:
        print(f"PASS: required local files exist; {len(dataset)} sequences selected.")
        return
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable. Run from your Mac terminal outside a restricted sandbox.")
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable on this machine.")
    device = torch.device(device_name)
    print(f"Dataset: {args.dataset}; device: {device}; sequences: {len(dataset)}; batch size: {args.batch_size}", flush=True)
    from model import LogAnomalyModel
    model = LogAnomalyModel(str(args.bert_path), str(args.llama_path), ft_path=str(checkpoint),
                   is_train_mode=False, device=device,
                   max_content_len=args.max_content_len, max_seq_len=args.max_seq_len)
    collator = CustomCollator(model.Bert_tokenizer, max_seq_len=args.max_seq_len,
                              max_content_len=args.max_content_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator,
                            num_workers=0, shuffle=False, drop_last=False)
    evalModel(model, dataloader, device=device, output_dir=output_dir, trace_output=args.trace_output)
    metadata = {"dataset": args.dataset, "data": str(data_path.resolve()),
                "checkpoint": str(checkpoint.resolve()), "device": str(device),
                "batch_size": args.batch_size, "limit": args.limit,
                "max_seq_len": args.max_seq_len, "max_content_len": args.max_content_len}
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
