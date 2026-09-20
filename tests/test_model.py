import torch

from src.models.bilstm_attention import BiLSTMAttentionClassifier, LuongAttention


def test_luong_attention_shape_and_sum() -> None:
    batch_size = 4
    seq_len = 10
    hidden_dim = 128

    attention = LuongAttention(hidden_dim=hidden_dim)
    s = torch.randn(batch_size, 2 * hidden_dim)
    H = torch.randn(batch_size, seq_len, 2 * hidden_dim)

    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # Mask out last 3 tokens for sequence 0
    mask[0, 7:] = False

    context, attn_weights = attention(s, H, mask=mask)

    assert context.shape == (batch_size, 2 * hidden_dim)
    assert attn_weights.shape == (batch_size, seq_len)

    # Check attention weights sum to 1.0 across sequence length
    attn_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5)

    # Check masked positions have 0 attention weight
    assert torch.all(attn_weights[0, 7:] == 0.0)


def test_model_output_shapes() -> None:
    vocab_size = 500
    embed_dim = 128
    hidden_dim = 128
    batch_size = 8
    seq_len = 16

    model = BiLSTMAttentionClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.2,
        padding_idx=0,
    )

    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

    logits, attn_weights = model(input_ids, lengths)

    assert logits.shape == (batch_size, 1)
    assert attn_weights.shape == (batch_size, seq_len)


def test_padding_mask_invariance() -> None:
    """Verify padding tokens do not alter the output for non-padding prefix tokens."""
    vocab_size = 500
    embed_dim = 64
    hidden_dim = 64

    model = BiLSTMAttentionClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.0,
        padding_idx=0,
    )
    model.eval()

    # Create short sequence [2, 3, 4] and padded version [2, 3, 4, 0, 0, 0]
    seq_unpadded = torch.tensor([[2, 3, 4]], dtype=torch.long)
    lens_unpadded = torch.tensor([3], dtype=torch.long)

    seq_padded = torch.tensor([[2, 3, 4, 0, 0, 0]], dtype=torch.long)
    lens_padded = torch.tensor([3], dtype=torch.long)

    with torch.no_grad():
        logits_unpadded, attn_unpadded = model(seq_unpadded, lens_unpadded)
        logits_padded, attn_padded = model(seq_padded, lens_padded)

    # Logits should be identical (or virtually identical up to float precision)
    assert torch.allclose(logits_unpadded, logits_padded, atol=1e-5)

    # Attention weights for unpadded tokens should match, padding tokens get 0
    assert torch.allclose(attn_unpadded[0], attn_padded[0, :3], atol=1e-5)
    assert torch.all(attn_padded[0, 3:] == 0.0)
