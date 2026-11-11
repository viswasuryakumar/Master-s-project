# Data preparation and dataset adapter refactor

This contribution reorganizes the existing LogLLM-derived implementation in
`customDataset.py` and the three preparation modules. It retains attribution to
the original pipeline; the contribution is the refactoring, compatibility layer,
integration repairs, and regression coverage.

## Public names

New code can use `LogSequenceDataset`, `ClassBalancedSampler`, `LogBatchCollator`,
`normalize_log_message`, and `flatten_log_sequences`. The old names
`CustomDataset`, `BalancedSampler`, `CustomCollator`, `replace_patterns`, and
`merge_data` remain aliases with the same call signatures and return structures.

The helper names are now `build_fixed_windows`, `build_time_windows`,
`read_log_records`, `compile_log_format`, and `write_structured_log`. Their old
names remain aliases, including support for existing keyword arguments.

Both preparation scripts expose `main(argv=None)` and separate the transformation
steps from command-line execution. They share `export_sequence_split` to avoid
repeating CSV serialization and summary calculations.

## Preserved contracts

- CSV columns, ` ;-; ` delimiter, output filenames, and summary-file contents.
- Default 100-message windows and steps, 80/20 splits, and partial tail windows.
- Ordered fixed-window splits and randomized HDFS session splits.
- HDFS message deduplication per block, missing-label behavior, and block labels.
- Masking patterns, tokenization arguments, truncation, labels, tensor boundaries,
  and the existing sampler's random-number call order.
- Dataset attributes and PyTorch special-method names used by callers.

Intentional repairs: the existing preparation commands passed `use_cache` to a
helper that did not accept it; the helper now accepts it and reuses whole-file
CSV results when at least as recent as their source log. Cache freshness is based
on modification time, not file content or format hashes; disable the cache if
changing parsing formats. Time-window deltas are copied before setting their
first value to zero, avoiding read-only pandas arrays and input mutation. Empty
time-window input now yields an empty frame.

## Validation

All 64 tests pass, including seven new regression tests. The CLI regression test
runs both preparation scripts as modules and direct scripts, with and without
cache reuse, on small raw-log fixtures. The seven failures previously present
on main are resolved by the integration repairs.

An additional comparison against the pre-refactor commit checked 100 seeds for
fixed-window contents, normalization, flattening, collator output, and three
sampler configurations with identical results. No neural weights or GPU inference
were needed; this validates data preparation contracts, not model accuracy.
