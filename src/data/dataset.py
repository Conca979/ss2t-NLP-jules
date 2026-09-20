from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.vocab import Vocabulary


class SST2Dataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset for SST-2 binary sentiment classification."""

    def __init__(
        self,
        data_path: str | Path,
        vocab: Vocabulary,
        max_samples: int | None = None,
    ) -> None:
        self.vocab = vocab
        path = Path(data_path)

        if not path.exists():
            # Check for alternative extensions if given path doesn't exist directly
            possible_paths = [
                path,
                Path(f"{path}.parquet"),
                Path(f"{path}.tsv"),
                Path(f"{path}.csv"),
            ]
            found_path = None
            for p in possible_paths:
                if p.exists():
                    found_path = p
                    break
            if found_path is None:
                raise FileNotFoundError(f"Data file not found at {data_path} or with .parquet/.tsv/.csv extensions.")
            path = found_path

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".tsv":
            df = pd.read_csv(path, sep="\t")
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            try:
                df = pd.read_parquet(path)
            except (ValueError, TypeError, OSError):
                df = pd.read_csv(path, sep=None, engine="python")

        if max_samples is not None and max_samples > 0:
            df = df.iloc[:max_samples]

        self.sentences: list[str] = df["sentence"].astype(str).tolist()
        self.labels: list[int] = df["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sentence = self.sentences[idx]
        label = self.labels[idx]
        input_ids = self.vocab.encode(sentence)
        return {
            "sentence": sentence,
            "input_ids": input_ids,
            "label": label,
        }


def collate_fn(
    batch: Sequence[dict[str, Any]],
    pad_idx: int = Vocabulary.PAD_IDX,
    max_len: int = 64,
) -> dict[str, torch.Tensor]:
    """Collate function applying dynamic padding up to batch max length (capped at max_len)."""
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    # Truncate sequences to max_len
    truncated_ids = [item["input_ids"][:max_len] for item in batch]

    # Calculate max length in this batch (minimum 1 to handle empty edge case)
    batch_max_len = max(max(len(ids) for ids in truncated_ids), 1)

    batch_size = len(batch)
    padded_input_ids = torch.full((batch_size, batch_max_len), pad_idx, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, ids in enumerate(truncated_ids):
        seq_len = len(ids)
        if seq_len > 0:
            padded_input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            lengths[i] = seq_len
        else:
            lengths[i] = 1  # avoid 0 length sequence

    return {
        "input_ids": padded_input_ids,
        "lengths": lengths,
        "labels": labels,
    }
