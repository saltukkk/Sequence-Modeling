"""Transformer-based communication models for Part 2."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from parameters import CommunicationModelParams


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        """Create a fixed positional-encoding table.

        Args:
            d_model: Embedding dimension.
            max_len: Maximum supported sequence length.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to the input sequence.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Positionally encoded tensor with the same shape as ``x``.
        """
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with self-attention and feed-forward layers."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        """Initialize one transformer block.

        Args:
            d_model: Hidden dimension.
            num_heads: Number of attention heads.
            ffn_dim: Feed-forward inner dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and a feed-forward sub-layer.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Transformed tensor with the same shape as ``x``.
        """
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerStack(nn.Module):
    """A positional-encoding layer followed by stacked transformer blocks."""

    def __init__(
        self,
        params: CommunicationModelParams,
        max_len: int,
    ) -> None:
        """Initialize the transformer stack.

        Args:
            params: Shared communication-model hyper-parameters.
            max_len: Maximum supported sequence length.
        """
        super().__init__()
        self.pos_encoding = PositionalEncoding(params.d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    params.d_model,
                    params.num_heads,
                    params.ffn_dim,
                    params.dropout,
                )
                for _ in range(params.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer stack.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Encoded tensor with the same shape as ``x``.
        """
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x


def apply_power_constraint(x: torch.Tensor, power_limit: float) -> torch.Tensor:
    """Scale coded symbols to satisfy a per-vector power constraint.

    Args:
        x: Coded symbols of shape ``(batch, symbols, d_channel)``.
        power_limit: Maximum allowed squared norm.

    Returns:
        Power-normalized coded symbols.
    """
    power = (x ** 2).sum(dim=-1, keepdim=True)
    scale = torch.rsqrt(power.clamp(min=power_limit))
    return x * scale


class TxEncoder(nn.Module):
    """Transmitter encoder used once per communication round."""

    def __init__(
        self,
        params: CommunicationModelParams,
        alphabet_size: int = 8,
        message_length: int = 4,
    ) -> None:
        """Initialize the transmitter encoder.

        Args:
            params: Communication-model hyper-parameters.
            alphabet_size: Number of symbols in the message alphabet.
            message_length: Number of symbols per message.
        """
        super().__init__()
        self.params = params
        self.message_length = message_length

        self.symbol_embed = nn.Embedding(alphabet_size, params.d_model // 2)
        self.round_embed = nn.Embedding(params.num_rounds, params.d_model // 2)

        history_dim = 2 * params.num_rounds * params.d_channel
        pre_input_dim = params.d_model + history_dim
        self.pre_mlp = nn.Sequential(
            nn.Linear(pre_input_dim, params.d_model),
            nn.GELU(),
            nn.Dropout(params.dropout),
            nn.Linear(params.d_model, params.d_model),
        )
        self.transformer = TransformerStack(params, max_len=self.message_length)
        self.post_mlp = nn.Sequential(
            nn.Linear(params.d_model, params.d_model),
            nn.GELU(),
            nn.Dropout(params.dropout),
            nn.Linear(params.d_model, params.d_channel),
        )

    def _build_round_input(
        self,
        message: torch.Tensor,
        tx_history: list[torch.Tensor],
        rx_history: list[torch.Tensor],
        round_idx: int,
    ) -> torch.Tensor:
        """Build transformer tokens for one communication round.

        Args:
            message: Integer message tensor of shape ``(batch, 4)``.
            tx_history: Previously transmitted coded symbols.
            rx_history: Previously received noisy symbols used as feedback.
            round_idx: Zero-based communication round index.

        Returns:
            Token tensor of shape ``(batch, 4, d_model)``.
        """
        batch_size = message.size(0)
        device = message.device

        symbol_emb = self.symbol_embed(message)
        round_emb = self.round_embed(
            torch.full((batch_size, self.message_length), round_idx, device=device, dtype=torch.long)
        )

        if round_idx == 0:
            hist = torch.zeros(
                batch_size,
                self.message_length,
                2 * self.params.num_rounds * self.params.d_channel,
                device=device,
            )
        else:
            tx_pad = torch.zeros(
                batch_size,
                self.message_length,
                self.params.num_rounds,
                self.params.d_channel,
                device=device,
            )
            rx_pad = torch.zeros_like(tx_pad)
            for step, tx_step in enumerate(tx_history):
                tx_pad[:, :, step, :] = tx_step
            for step, rx_step in enumerate(rx_history):
                rx_pad[:, :, step, :] = rx_step
            hist = torch.cat([tx_pad, rx_pad], dim=2).reshape(batch_size, self.message_length, -1)

        pre_input = torch.cat([symbol_emb, round_emb, hist], dim=-1)
        return self.pre_mlp(pre_input)

    def forward_round(
        self,
        message: torch.Tensor,
        tx_history: list[torch.Tensor],
        rx_history: list[torch.Tensor],
        round_idx: int,
    ) -> torch.Tensor:
        """Encode one communication round.

        Args:
            message: Integer message tensor of shape ``(batch, 4)``.
            tx_history: Previously transmitted coded symbols.
            rx_history: Previously received noisy symbols.
            round_idx: Zero-based communication round index.

        Returns:
            Coded symbols of shape ``(batch, 4, d_channel)``.
        """
        tokens = self._build_round_input(message, tx_history, rx_history, round_idx)
        hidden = self.transformer(tokens)
        coded = self.post_mlp(hidden)
        return apply_power_constraint(coded, self.params.power_limit)


class RxDecoder(nn.Module):
    """Receiver decoder that reconstructs the message after all rounds."""

    def __init__(self, params: CommunicationModelParams, alphabet_size: int = 8, message_length: int = 4) -> None:
        """Initialize the receiver decoder.

        Args:
            params: Communication-model hyper-parameters.
            alphabet_size: Number of symbols in the message alphabet.
            message_length: Number of symbols per message.
        """
        super().__init__()
        self.params = params
        self.alphabet_size = alphabet_size
        self.message_length = message_length
        seq_len = params.num_rounds * message_length

        self.input_proj = nn.Linear(params.d_channel, params.d_model)
        self.transformer = TransformerStack(params, max_len=seq_len)
        self.classifier = nn.Linear(params.d_model, alphabet_size)

    def forward(self, received: torch.Tensor) -> torch.Tensor:
        """Decode all collected noisy symbols.

        Args:
            received: Noisy symbols of shape ``(batch, T, 4, d_channel)``.

        Returns:
            Class logits of shape ``(batch, 4, alphabet_size)``.
        """
        batch_size = received.size(0)
        seq = received.reshape(batch_size, self.params.num_rounds * self.message_length, -1)
        hidden = self.input_proj(seq)
        hidden = self.transformer(hidden)

        last_round_start = (self.params.num_rounds - 1) * self.message_length
        symbol_hidden = hidden[:, last_round_start : last_round_start + self.message_length, :]
        return self.classifier(symbol_hidden)


class CommunicationSystem(nn.Module):
    """End-to-end interactive AWGN communication protocol."""

    def __init__(self, params: CommunicationModelParams, alphabet_size: int = 8, message_length: int = 4) -> None:
        """Initialize the full communication system.

        Args:
            params: Communication-model and channel hyper-parameters.
            alphabet_size: Number of symbols in the message alphabet.
            message_length: Number of symbols per message.
        """
        super().__init__()
        self.params = params
        self.alphabet_size = alphabet_size
        self.message_length = message_length
        self.encoder = TxEncoder(
            params,
            alphabet_size=alphabet_size,
            message_length=message_length,
        )
        self.decoder = RxDecoder(
            params,
            alphabet_size=alphabet_size,
            message_length=message_length,
        )

    def forward(self, message: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate the full T-round communication process.

        Args:
            message: Integer message tensor of shape ``(batch, 4)``.

        Returns:
            A tuple containing:
                - Class logits of shape ``(batch, 4, alphabet_size)``
                - Received noisy symbols of shape ``(batch, T, 4, d_channel)``
        """
        tx_history: list[torch.Tensor] = []
        rx_history: list[torch.Tensor] = []
        received_rounds: list[torch.Tensor] = []

        for round_idx in range(self.params.num_rounds):
            coded = self.encoder.forward_round(message, tx_history, rx_history, round_idx)
            tx_history.append(coded)

            noise = torch.randn_like(coded) * self.params.noise_std
            received = coded + noise
            rx_history.append(received)
            received_rounds.append(received)

        all_received = torch.stack(received_rounds, dim=1)
        logits = self.decoder(all_received)
        return logits, all_received

    @torch.no_grad()
    def decode_message(self, message: torch.Tensor) -> torch.Tensor:
        """Predict the most likely message symbols.

        Args:
            message: Integer message tensor used to drive the protocol.

        Returns:
            Predicted symbol indices with shape ``(batch, 4)``.
        """
        self.eval()
        logits, _ = self.forward(message)
        return logits.argmax(dim=-1)

    def symbol_accuracy(self, logits: torch.Tensor, message: torch.Tensor) -> float:
        """Compute per-symbol classification accuracy.

        Args:
            logits: Decoder logits of shape ``(batch, 4, alphabet_size)``.
            message: Ground-truth message symbols.

        Returns:
            Fraction of correctly predicted symbols.
        """
        preds = logits.argmax(dim=-1)
        return (preds == message).float().mean().item()

    def message_accuracy(self, logits: torch.Tensor, message: torch.Tensor) -> float:
        """Compute whole-message reconstruction accuracy.

        Args:
            logits: Decoder logits of shape ``(batch, 4, alphabet_size)``.
            message: Ground-truth message symbols.

        Returns:
            Fraction of messages reconstructed exactly.
        """
        preds = logits.argmax(dim=-1)
        return (preds == message).all(dim=-1).float().mean().item()
