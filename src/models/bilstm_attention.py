from __future__ import annotations

import torch
from torch import nn


class LuongAttention(nn.Module):
    """Luong Attention Mechanism using General Score function: score(s, H_i) = s W_a H_i^T."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # Query s has size 2 * hidden_dim, Keys/Values H have size 2 * hidden_dim
        # W_a maps from 2 * hidden_dim to 2 * hidden_dim
        self.wa = nn.Linear(2 * hidden_dim, 2 * hidden_dim, bias=False)

    def forward(
        self,
        s: torch.Tensor,
        H: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: Query vector of shape [Batch, 2 * hidden_dim]
            H: Sequence of hidden states of shape [Batch, Length, 2 * hidden_dim]
            mask: Optional boolean tensor of shape [Batch, Length], True for valid tokens, False for padding.

        Returns:
            context: Context vector of shape [Batch, 2 * hidden_dim]
            attn_weights: Alignment vector alpha of shape [Batch, Length]
        """
        # s_wa: [B, 2h]
        s_wa = self.wa(s)

        # score(s, H_i) = s W_a H_i^T -> [B, L]
        scores = torch.sum(s_wa.unsqueeze(1) * H, dim=-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)

        # Handle edge case where a sequence is completely masked
        if torch.isnan(attn_weights).any():
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Context vector c = sum(alpha_i * H_i) -> [B, 2h]
        context = torch.sum(attn_weights.unsqueeze(-1) * H, dim=1)

        return context, attn_weights


class BiLSTMAttentionClassifier(nn.Module):
    """Stacked Bidirectional LSTM with Luong Attention for Binary Sentiment Classification."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = LuongAttention(hidden_dim=hidden_dim)

        self.wc = nn.Linear(4 * hidden_dim, 2 * hidden_dim)
        self.fc = nn.Linear(2 * hidden_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: LongTensor of shape [Batch, Sequence_Length]
            lengths: Optional LongTensor of shape [Batch] containing actual sequence lengths

        Returns:
            logits: Unnormalized logits of shape [Batch, 1]
            attn_weights: Attention weights of shape [Batch, Sequence_Length]
        """
        mask = input_ids != self.padding_idx

        embedded = self.embedding(input_ids)

        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_output, (h_n, _) = self.lstm(packed_embedded)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=input_ids.size(1)
            )
        else:
            output, (h_n, _) = self.lstm(embedded)

        # Top-layer forward and backward final hidden states: [B, 2 * hidden_dim]
        s = torch.cat([h_n[-2], h_n[-1]], dim=-1)

        # Luong Attention
        context, attn_weights = self.attention(s, output, mask=mask)

        # Concatenate context c and final state s -> [B, 4 * hidden_dim]
        combined = torch.cat([context, s], dim=-1)

        # h_tilde = tanh(W_c [c; s]) -> [B, 2 * hidden_dim]
        h_tilde = torch.tanh(self.wc(combined))

        # Output raw logits -> [B, 1]
        logits = self.fc(h_tilde)

        return logits, attn_weights
