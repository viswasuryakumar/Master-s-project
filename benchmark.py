"""Benchmark a fine-tuned log anomaly checkpoint against a labelled test split."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from customDataset import CustomCollator, CustomDataset

HERE = Path(__file__).resolve().parent

# Keywords the decoder is trained to emit; anything else counts as "benign".
LABEL_WORDS = {"anomalous": 1, "normal": 0}



# --------------------------------------------------------------------------- #
# Confusion-matrix based scoring
# --------------------------------------------------------------------------- #
class Tally(NamedTuple):
    tp: int
    fp: int
    tn: int
    fn: int

    @classmethod
    def build(cls, truth: np.ndarray, guess: np.ndarray) -> "Tally":
        truth = np.asarray(truth).astype(bool)
        guess = np.asarray(guess).astype(bool)
        if truth.shape != guess.shape:
            raise ValueError(f"{truth.size} labels but {guess.size} predictions")
        return cls(
            tp=int(np.sum(truth & guess)),
            fp=int(np.sum(~truth & guess)),
            tn=int(np.sum(~truth & ~guess)),
            fn=int(np.sum(truth & ~guess)),
        )

    @staticmethod
    def _ratio(num: int, den: int) -> float:
        return num / den if den else 0.0

    @property
    def precision(self) -> float:
        return self._ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return self._ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        return self._ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    @property
    def accuracy(self) -> float:
        return self._ratio(self.tp + self.tn, sum(self))

    def as_dict(self) -> dict:
        return {
            **self._asdict(),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }


# --------------------------------------------------------------------------- #
# Decoding the model's free-text answers
# --------------------------------------------------------------------------- #
def first_label_word(answer: str) -> str | None:
    """Return the earliest label keyword found in `answer`, ignoring case."""
    lowered = answer.lower()
    hits = [(lowered.find(word), word) for word in LABEL_WORDS if word in lowered]
    return min(hits)[1] if hits else None


def answers_to_flags(answers: Iterable[str], unreadable: list[str]) -> list[int]:
    flags = []
    for answer in answers:
        word = first_label_word(answer)
        if word is None:
            unreadable.append(answer)
        flags.append(LABEL_WORDS.get(word, 0))
    return flags


# --------------------------------------------------------------------------- #
# Evaluation driver
# --------------------------------------------------------------------------- #
def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


class CheckpointBenchmark:
    def __init__(self, opts: argparse.Namespace):
        self.opts = opts
        self.gpu = pick_device(opts.gpu)
        self.csv_file = Path(opts.data or HERE / "data" / opts.dataset / "test.csv")
        self.adapter = Path(opts.checkpoint_dir or HERE / f"ft_model_{opts.dataset}")
        self.unreadable: list[str] = []

    def _load_network(self):
        from model import LogAnomalyModel  # heavy import, deferred until needed

        return LogAnomalyModel(
            str(self.opts.encoder_dir),
            str(self.opts.decoder_dir),
            ft_path=str(self.adapter),
            is_train_mode=False,
            device=self.gpu,
            max_content_len=self.opts.content_limit,
            max_seq_len=self.opts.sequence_limit,
            max_new_tokens=self.opts.new_tokens,
        )

    def _make_loader(self, split, network) -> DataLoader:
        packer = CustomCollator(
            network.Bert_tokenizer,
            max_seq_len=self.opts.sequence_limit,
            max_content_len=self.opts.content_limit,
        )
        return DataLoader(split, batch_size=self.opts.batch_size, collate_fn=packer,
                          num_workers=self.opts.workers, shuffle=False, drop_last=False)

    @torch.inference_mode()
    def _predict(self, network, loader: DataLoader) -> np.ndarray:
        network.eval()
        flags: list[int] = []
        for group in tqdm(loader, desc=f"{self.opts.dataset} inference", unit="batch"):
            token_out = network(group["inputs"].to(self.gpu), group["seq_positions"])
            texts = network.Llama_tokenizer.batch_decode(token_out, skip_special_tokens=True)
            flags += answers_to_flags(texts, self.unreadable)
        return np.fromiter(flags, dtype=np.int64)

    def execute(self) -> Tally:
        self._banner()
        split = CustomDataset(self.csv_file, max_samples=self.opts.limit or None)
        network = self._load_network()
        guesses = self._predict(network, self._make_loader(split, network))
        tally = Tally.build(split.get_label(), guesses)
        self._summarise(tally)
        return tally

    def _banner(self) -> None:
        rows = [
            ("test file", self.csv_file),
            ("checkpoint", self.adapter),
            ("batch size", self.opts.batch_size),
            ("content / sequence limit", f"{self.opts.content_limit} / {self.opts.sequence_limit}"),
            ("device", self.gpu),
        ]
        print(f"== log anomaly benchmark: {self.opts.dataset} ==")
        for key, val in rows:
            print(f"  {key:<26}{val}")

    def _summarise(self, t: Tally) -> None:
        if self.unreadable:
            print(f"\n[warn] {len(self.unreadable)} answer(s) had no label word; "
                  f"treated as normal. First one: {self.unreadable[0]!r}")
        print("\n-- confusion matrix (rows = truth) --")
        print(f"{'':>12}{'pred anom':>12}{'pred norm':>12}")
        print(f"{'anomalous':>12}{t.tp:>12}{t.fn:>12}")
        print(f"{'normal':>12}{t.fp:>12}{t.tn:>12}")
        print("\n-- scores --")
        for name in ("precision", "recall", "f1", "accuracy"):
            print(f"  {name:<10}{getattr(t, name):.4f}")
