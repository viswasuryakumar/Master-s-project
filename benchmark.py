"""Benchmark a fine-tuned log anomaly checkpoint against a labelled test split."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


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


