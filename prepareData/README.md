# Preprocessing: preparing training and testing data


- Download BGL/HDFS_v1/Thunderbird dataset from [here](https://github.com/logpai/loghub). Download Liberty dataset
  from [here](http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/liberty2.gz).
- For **BGL**, **Thunderbird** and **Liberty**, set the following variations in **sliding_window.py** under *
  *prepareData**
  directory:
   ```
   data_dir =  # i.e. r'/mnt/public/gw/SyslogData/BGL'
   log_name =  # i.e. 'BGL.log'
   ```

  For  **Liberty**, you should activate
  ```
  start_line = 40000000
  end_line = 45000000
  ```

  For  **Thunderbird**, you should activate
  ```
  start_line = 160000000
  end_line = 170000000
  ```

  Run ```python prepareData.sliding_window.py```  from the root directory to generate training and testing data.
  Training and testing data will be saved in {data_dir}.

- For **HDFS**, set the following directories in **session_window.py** under **prepareData**
  directory:
   ```
   data_dir =  # i.e. r'/mnt/public/gw/SyslogData/HDFS_v1'
   log_name =  # i.e. 'HDFS.log'
   ```
  Run ```python prepareData.session_window.py```  from the root directory to generate training and testing data.
  Training and testing data will be saved in {data_dir}


## Parse-quality report

Before windowing, check how much of a raw log actually parses:

```
python -m prepareData.parse_report --data-dir <dir> --log-name BGL.log
```

This writes `<log_name>_parse_report.json` to the data directory with counts of blank, unparseable and empty-content
lines, duplicate messages, the parse rate and a few unparseable examples. Pass `--log-format` for log names that do not
contain `bgl`, `thunderbird`, `spirit` or `liberty`, and `--start-line`/`--end-line` to match a sliced run.

## Windowing manifest

`sliding_window.py` and `session_window.py` write `manifest.json` next to `train.csv` and `test.csv`. It records the
script, window and split parameters, seed, git commit, Python and pandas versions, and the size and sha256 of the input
log and each output file, plus window and anomaly counts. Pass `--skip-hash` to skip hashing very large logs.

`session_window.py` shuffles sessions before the split and takes `--seed` (default 42), so the same seed gives the same
train/test split. It was unseeded before, and every run produced a different split. The sliding-window split is
chronological and has no seed.

## Token-length statistics

Check whether the window and length limits fit the model before a training or evaluation run:

```
python -m prepareData.token_stats --csv <data_dir>/train.csv --bert-path <bert> --llama-path <llama>
```

It writes `train_token_stats.json` and prints a summary. BERT sees every message, truncated to `--max-content-len`
(default 100), so the report gives message token percentiles and how many exceed the limit. Llama does not see message
tokens: each message becomes one projected vector, so a window's Llama input is a fixed prompt plus at most
`--max-seq-len` (default 128) message vectors. The report gives the prompt size, messages per window, the share of
windows over `--max-seq-len`, and BERT tokens per window. Rows are normalized like `LogSequenceDataset`, which needs
`torch`; pass `--no-normalize` to skip that. Use `--limit N` for a quick look at the first N windows.

## Masking benchmark and audit

`customDataset.normalize_log_message` (`replace_patterns`) masks numbers, addresses, hex values and paths with `<*>`.
To time it against alternatives and see what it leaves behind:

```
python -m prepareData.masking --csv <data_dir>/train.csv [--limit 2000] [--repeat 3]
```

It writes `train_masking_report.json` and prints a summary with three parts:

- **benchmark**: the original, precompiled regexes, a whole-window cache and a per-message cache. Each variant's output is
  compared with the original, so a variant that changes the masked text shows `identical=False` and is not a valid
  replacement. The per-message cache only matches the original if no pattern spans the `;-;` separator, which the
  comparison checks on your data.
- **pattern profile**: each pattern timed on its own, slowest first. This shows relative cost, not the cost inside the
  combined alternation.
- **audit**: masks per message, distinct messages before and after masking, tokens that still look like variable values
  (hex-like or very long), and tokens seen only once. `has_digit` should be zero because the last pattern masks any
  token with a digit; a non-zero value is a masking gap. Singleton tokens include rare real words, so read the samples
  before changing a pattern.

The patterns are read from `customDataset.py` with `ast`, so the module runs without `torch` and always uses the
pipeline's current patterns. Timings depend on the data and machine; record results from a real BGL run in `docs/`.
