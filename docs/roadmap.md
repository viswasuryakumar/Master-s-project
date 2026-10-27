# Preprocessing Roadmap (Oct 2026 – Jan 2027)

Scope: the data-preprocessing layer (`prepareData/`, `customDataset.py`) that feeds the model.
Status values: planned / in progress / done. Update this file as work lands.

## Current state (as of 2026-10-05)
- Fixed-size, time-based sliding and HDFS session windowing exist (`helper.py`, `sliding_window.py`, `session_window.py`).
- `CustomDataset`, `BalancedSampler` and `CustomCollator` handle masking, balancing and batching.
- Unit tests cover `helper`, `customDataset` and `window_stats`.
- `window_stats` summarizes windows to compare window and step sizes.

## Month 1 – Data quality and safety (Oct)
| Item | Status |
|---|---|
| Input validation for structured logs (required columns, empty content, label values) | done |
| Chronological train/test split helper with a leakage check, extracted from `sliding_window.py` | done |
| Run the full test suite in an environment with real torch; add CI | planned |

## Month 2 – Windowing experiments (Nov)
| Item | Status |
|---|---|
| Sweep window and step sizes per dataset using `window_stats`; record results in `docs/` | planned |
| Compare fixed-size vs time-based windows on BGL (window count, length, anomaly ratio) | planned |
| Fix inconsistent imports (`from helper import` vs `from prepareData.helper import`) so scripts run from the repo root | done |

## Month 3 – Robustness and performance (Dec)
| Item | Status |
|---|---|
| Replace hard-coded data paths in `sliding_window.py` / `session_window.py` with CLI arguments | done |
| Cache structured logs so reruns skip parsing (`--use-cache`) | done |
| Profile `replace_patterns` masking and benchmark alternatives | planned |
| Edge cases: empty windows, very long sequences, unparseable lines (tests) | done |

## Month 4 – Integration and write-up (Jan)
| Item | Status |
|---|---|
| Hand-off notes for the model stage: data formats, columns, window settings used | planned |
| Final documentation pass on `prepareData/README.md` | planned |
| Retrospective: what worked, what did not, measured results | planned |

## Working rules
- One small, tested change per commit; commit with the real date.
- Experiments are recorded with their actual results, including negative ones.

## Weekly plan (Oct 27 – Dec 15, 2026)
One tested commit per Tuesday, committed with the real date. Update status as each lands.

| Week | Date | Item | Status |
|---|---|---|---|
| 1 | Oct 27 | Parse-quality report for raw logs (`prepareData/parse_report.py`) | done |
| 2 | Nov 3 | Windowing manifest: config, dataset hash and seed saved beside each output | planned |
| 3 | Nov 10 | Token-length stats for BERT messages and Llama windows | planned |
| 4 | Nov 17 | Benchmark `replace_patterns` masking and add a masking audit of unmasked tokens | planned |
| 5 | Nov 24 | Window-size sweep and fixed vs time-based comparison on BGL, recorded in `docs/` | planned |
| 6 | Dec 1 | Parser registry (replace per-script format constants) and dataset fingerprint for `--use-cache` | planned |
| 7 | Dec 8 | CI smoke test on `data/BGL/raw-excerpt.log` with a stub model, prompt snapshot test, keyword baseline | planned |
| 8 | Dec 15 | Repo hygiene (stale `customdatset_temp.py`, README CLI usage), hand-off notes | planned |
