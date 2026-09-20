from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import SST2Dataset, collate_fn
from src.data.vocab import Vocabulary
from src.models.bilstm_attention import BiLSTMAttentionClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BiLSTM Luong Attention Classifier on SST-2 dataset.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (default: 2)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and validation (default: 64)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on ('cpu' or 'cuda', default: 'cpu')")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to load for quick testing")
    parser.add_argument("--train-path", type=str, default="dataset/train", help="Path to training data")
    parser.add_argument("--val-path", type=str, default="dataset/validation", help="Path to validation data")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory to save model checkpoints and vocab")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension (default: 128)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension (default: 128)")
    parser.add_argument("--max-vocab-size", type=int, default=25000, help="Maximum vocabulary size (default: 25000)")
    parser.add_argument("--max-len", type=int, default=64, help="Maximum sequence length (default: 64)")
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits, _ = model(input_ids, lengths)
        logits = logits.squeeze(-1)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += input_ids.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(input_ids, lengths)
            logits = logits.squeeze(-1)

            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += input_ids.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Building vocabulary from training data...")
    # Load raw text sentences for building vocabulary
    raw_train_ds = SST2Dataset(args.train_path, vocab=Vocabulary(), max_samples=args.max_samples)
    vocab = Vocabulary.build_vocab(raw_train_ds.sentences, max_size=args.max_vocab_size)
    vocab_path = artifacts_dir / "vocab.json"
    vocab.save(vocab_path)
    print(f"Vocabulary built (size: {len(vocab)}) and saved to {vocab_path}")

    train_ds = SST2Dataset(args.train_path, vocab=vocab, max_samples=args.max_samples)
    val_ds = SST2Dataset(args.val_path, vocab=vocab, max_samples=args.max_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx=Vocabulary.PAD_IDX, max_len=args.max_len),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx=Vocabulary.PAD_IDX, max_len=args.max_len),
    )

    model = BiLSTMAttentionClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        dropout=0.2,
        padding_idx=Vocabulary.PAD_IDX,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model architecture initialized with {total_params:,} trainable parameters.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_time = time.time()
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_dur = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_dur:.2f}s"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = artifacts_dir / "model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "vocab_size": len(vocab),
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                },
                checkpoint_path,
            )
            print(f"Saved new best model checkpoint to {checkpoint_path}")

    total_duration = time.time() - start_time
    print(f"Training completed in {total_duration:.2f} seconds. Best Validation Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
