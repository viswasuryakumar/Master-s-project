"""Benchmark a fine-tuned log anomaly checkpoint against a labelled test split."""
from __future__ import annotations

from typing import Iterable, NamedTuple

import numpy as np

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


