from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field

from src.data.vocab import Vocabulary
from src.models.bilstm_attention import BiLSTMAttentionClassifier


class SentimentResponse(BaseModel):
    """Pydantic v2 schema for sentiment classification prediction."""

    model_config = ConfigDict(strict=True)

    label: int = Field(description="Binary sentiment label (0: negative, 1: positive)")
    sentiment: str = Field(description="Sentiment string representation ('negative' or 'positive')")
    confidence: float = Field(description="Prediction confidence score between 0.0 and 1.0")
    attention_weights: list[float] = Field(description="List of attention weight values for input tokens")


def predict_sentiment(
    text: str,
    model: BiLSTMAttentionClassifier,
    vocab: Vocabulary,
    device: torch.device,
    max_len: int = 64,
) -> SentimentResponse:
    """Predict sentiment and return Pydantic v2 SentimentResponse object."""
    model.eval()
    tokens = Vocabulary.tokenize(text)[:max_len]
    if not tokens:
        tokens = [Vocabulary.UNK_TOKEN]

    input_ids_list = [vocab.token2idx.get(t, Vocabulary.UNK_IDX) for t in tokens]
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    lengths = torch.tensor([len(input_ids_list)], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, attn_weights = model(input_ids, lengths)
        prob = torch.sigmoid(logits.squeeze(-1)).item()
        attn = attn_weights.squeeze(0).cpu().tolist()

    label = 1 if prob >= 0.5 else 0
    sentiment = "positive" if label == 1 else "negative"
    confidence = prob if label == 1 else 1.0 - prob

    return SentimentResponse(
        label=label,
        sentiment=sentiment,
        confidence=float(confidence),
        attention_weights=[float(w) for w in attn],
    )


def load_model_and_vocab(
    model_path: str | Path = "artifacts/model.pt",
    vocab_path: str | Path = "artifacts/vocab.json",
    device: torch.device | None = None,
) -> tuple[BiLSTMAttentionClassifier, Vocabulary]:
    """Load model weights and vocabulary from artifacts."""
    if device is None:
        device = torch.device("cpu")

    vocab = Vocabulary.load(vocab_path)

    checkpoint = torch.load(model_path, map_location=device)
    embed_dim = checkpoint.get("embed_dim", 128)
    hidden_dim = checkpoint.get("hidden_dim", 128)

    model = BiLSTMAttentionClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.2,
        padding_idx=Vocabulary.PAD_IDX,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sentiment for a given text string.")
    parser.add_argument("--text", type=str, required=True, help="Raw text string to classify")
    parser.add_argument("--model-path", type=str, default="artifacts/model.pt", help="Path to model checkpoint")
    parser.add_argument("--vocab-path", type=str, default="artifacts/vocab.json", help="Path to vocabulary JSON file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    model, vocab = load_model_and_vocab(args.model_path, args.vocab_path, device)
    response = predict_sentiment(args.text, model, vocab, device)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
