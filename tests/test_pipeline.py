import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import SST2Dataset, collate_fn
from src.data.vocab import Vocabulary
from src.models.bilstm_attention import BiLSTMAttentionClassifier
from src.predict import predict_sentiment
from src.train import train_epoch


def test_pipeline_smoke_run(tmp_path) -> None:
    """Smoke test running 1 epoch of training on a 32-sample synthetic batch."""
    # 1. Create synthetic dataset file
    sentences = [f"This is sample text line number {i} for sentiment classification" for i in range(32)]
    labels = [i % 2 for i in range(32)]

    data_file = tmp_path / "synthetic.csv"
    with open(data_file, "w", encoding="utf-8") as f:
        f.write("idx,sentence,label\n")
        f.writelines(f'{i},"{s}",{l}\n' for i, (s, l) in enumerate(zip(sentences, labels, strict=True)))

    # 2. Build vocab
    vocab = Vocabulary.build_vocab(sentences, max_size=100)

    # 3. Create dataset & dataloader
    dataset = SST2Dataset(data_file, vocab=vocab)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx=Vocabulary.PAD_IDX, max_len=64),
    )

    # 4. Initialize model, loss, optimizer
    model = BiLSTMAttentionClassifier(
        vocab_size=len(vocab),
        embed_dim=32,
        hidden_dim=32,
        num_layers=2,
        dropout=0.2,
        padding_idx=Vocabulary.PAD_IDX,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cpu")

    # 5. Run 1 epoch
    loss, accuracy = train_epoch(model, dataloader, optimizer, criterion, device)

    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0

    # 6. Test inference function with trained model
    response = predict_sentiment("this is sample text", model, vocab, device)
    assert response.label in (0, 1)
    assert response.sentiment in ("negative", "positive")
    assert 0.0 <= response.confidence <= 1.0
    assert len(response.attention_weights) > 0
    assert abs(sum(response.attention_weights) - 1.0) < 1e-4
