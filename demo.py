from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.predict import load_model_and_vocab, predict_sentiment


def run_interactive_demo(
    model_path: str | Path = "artifacts/model.pt",
    vocab_path: str | Path = "artifacts/vocab.json",
    device_name: str = "cpu",
) -> None:
    """Run interactive terminal classification loop."""
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    if not Path(model_path).exists() or not Path(vocab_path).exists():
        print(f"Error: Model file '{model_path}' or vocab file '{vocab_path}' not found.")
        print("Please run training first: uv run python -m src.train")
        sys.exit(1)

    print("=" * 70)
    print(" SST-2 BiLSTM + Luong Attention Live Sentiment Classifier ")
    print("=" * 70)
    print(f"Loading model checkpoint from '{model_path}'...")
    model, vocab = load_model_and_vocab(model_path, vocab_path, device)
    print("Model loaded successfully!")
    print("Type your sentence and press Enter. (Type 'exit' or 'quit' to stop)\n")

    while True:
        try:
            user_input = input("Enter text > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting live demo. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Exiting live demo. Goodbye!")
            break

        response = predict_sentiment(user_input, model, vocab, device)

        sentiment_str = response.sentiment.upper()
        emoji = "POSITIVE" if response.label == 1 else "NEGATIVE"
        print("\n--- Prediction Result ---")
        print(f"Sentiment:  [{sentiment_str}] ({emoji})")
        print(f"Confidence: {response.confidence * 100:.2f}%")
        print("Token Attention Weights:")

        tokens = vocab.tokenize(user_input)[:64]
        if not tokens:
            tokens = ["<unk>"]

        for token, weight in zip(tokens, response.attention_weights, strict=False):
            bar = "#" * int(weight * 50)
            print(f"  {token:<15} : {weight:.4f} | {bar}")
        print("-" * 50 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Sentiment Classification Demo")
    parser.add_argument("--model-path", type=str, default="artifacts/model.pt", help="Path to saved model weights")
    parser.add_argument("--vocab-path", type=str, default="artifacts/vocab.json", help="Path to saved vocabulary JSON")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on ('cpu' or 'cuda')")
    parser.add_argument("--text", type=str, default=None, help="Single text input for non-interactive demo run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.text:
        device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        model, vocab = load_model_and_vocab(args.model_path, args.vocab_path, device)
        response = predict_sentiment(args.text, model, vocab, device)
        print(response.model_dump_json(indent=2))
    else:
        run_interactive_demo(args.model_path, args.vocab_path, args.device)


if __name__ == "__main__":
    main()
