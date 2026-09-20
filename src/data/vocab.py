from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path


class Vocabulary:
    """Vocabulary class for managing token-to-index and index-to-token mappings."""

    PAD_TOKEN: str = "<pad>"
    UNK_TOKEN: str = "<unk>"
    PAD_IDX: int = 0
    UNK_IDX: int = 1

    def __init__(self, token2idx: dict[str, int] | None = None) -> None:
        if token2idx is not None:
            self.token2idx: dict[str, int] = token2idx
            self.idx2token: dict[int, str] = {idx: token for token, idx in token2idx.items()}
        else:
            self.token2idx = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
            self.idx2token = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}

    def __len__(self) -> int:
        return len(self.token2idx)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize input text using regex word and punctuation matching."""
        return re.findall(r"\w+|[^\w\s]", text.lower())

    @classmethod
    def build_vocab(
        cls,
        texts: Sequence[str],
        max_size: int = 25000,
        min_freq: int = 1,
    ) -> Vocabulary:
        """Build vocabulary from a collection of text strings."""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(cls.tokenize(text))

        vocab = cls()
        most_common = counter.most_common()
        for token, freq in most_common:
            if freq < min_freq:
                continue
            if len(vocab.token2idx) >= max_size:
                break
            if token not in vocab.token2idx:
                idx = len(vocab.token2idx)
                vocab.token2idx[token] = idx
                vocab.idx2token[idx] = token

        return vocab

    def encode(self, text: str) -> list[int]:
        """Convert raw text to a list of token indices."""
        tokens = self.tokenize(text)
        return [self.token2idx.get(token, self.UNK_IDX) for token in tokens]

    def decode(self, indices: Sequence[int]) -> list[str]:
        """Convert a list of token indices back to tokens."""
        return [self.idx2token.get(idx, self.UNK_TOKEN) for idx in indices]

    def save(self, filepath: str | Path) -> None:
        """Serialize vocabulary to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> Vocabulary:
        """Load vocabulary from a JSON file."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            token2idx: dict[str, int] = json.load(f)
        return cls(token2idx=token2idx)
