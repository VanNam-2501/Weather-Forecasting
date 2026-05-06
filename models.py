"""
models.py  –  Model definitions for Weather Forecasting.

Models
------
* LSTMSeq2Seq      – Encoder-Decoder LSTM with scheduled sampling
* LSTMAttention    – Bahdanau-style attention over encoder outputs
* TransformerModel – Vanilla Transformer (encoder-decoder) with causal masking
"""

import math
import random

import torch
import torch.nn as nn


# ─────────────────────────── LSTM Seq2Seq ───────────────────────────────────

class LSTMSeq2Seq(nn.Module):
    """
    Classic Encoder-Decoder LSTM.

    During training, scheduled sampling (teacher forcing) is applied:
    at each decoder step the ground-truth target is fed with probability
    `teacher_forcing_ratio`; otherwise the model's own prediction is used.
    At inference time pass `future_target=None`.
    """

    def __init__(self, n_features: int, forecast_horizon: int, target_idx: int,
                 d_model: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.target_idx       = target_idx
        self.forecast_horizon = forecast_horizon

        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=d_model,
            num_layers=num_layers, dropout=dropout, batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=1, hidden_size=d_model,
            num_layers=num_layers, dropout=dropout, batch_first=True,
        )
        self.output_layer = nn.Linear(d_model, 1)

    def forward(
        self,
        past_features: torch.Tensor,                # (B, W, F)
        future_target: torch.Tensor | None = None,  # (B, H, 1)  training only
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:                              # (B, H, 1)

        _, (h, c) = self.encoder(past_features)

        curr_input = (
            past_features[:, -1, self.target_idx]
            .unsqueeze(1).unsqueeze(2)              # (B, 1, 1)
        )
        outputs = []

        for t in range(self.forecast_horizon):
            dec_out, (h, c) = self.decoder(curr_input, (h, c))
            step_out = self.output_layer(dec_out.squeeze(1))   # (B, 1)
            outputs.append(step_out.unsqueeze(1))

            if future_target is not None and random.random() < teacher_forcing_ratio:
                curr_input = future_target[:, t, :].unsqueeze(1)
            else:
                curr_input = step_out.unsqueeze(1)

        return torch.cat(outputs, dim=1)            # (B, H, 1)


# ─────────────────────────── LSTM + Attention ───────────────────────────────

class LSTMAttention(nn.Module):
    """
    Encoder-Decoder LSTM with Bahdanau-style (additive) attention.

    At each decoder step, an attention score over all encoder hidden states
    produces a context vector that is concatenated with the decoder input.
    """

    def __init__(self, n_features: int, forecast_horizon: int, target_idx: int,
                 d_model: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.target_idx       = target_idx
        self.forecast_horizon = forecast_horizon
        self.d_model          = d_model

        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=d_model,
            num_layers=num_layers, dropout=dropout, batch_first=True,
        )

        # Bahdanau attention
        self.attn = nn.Linear(d_model * 2, d_model)
        self.v    = nn.Linear(d_model, 1, bias=False)

        self.decoder = nn.LSTM(
            input_size=1 + d_model,   # target value + context
            hidden_size=d_model,
            num_layers=num_layers, dropout=dropout, batch_first=True,
        )
        self.output_layer = nn.Linear(d_model, 1)

    def _attend(
        self,
        h_last: torch.Tensor,   # (B, d_model)  – last-layer hidden state
        enc_out: torch.Tensor,  # (B, W, d_model)
    ) -> torch.Tensor:          # (B, 1, d_model)  – context vector
        h_exp   = h_last.unsqueeze(1).expand_as(enc_out)          # (B, W, d)
        energy  = torch.tanh(self.attn(torch.cat([h_exp, enc_out], dim=2)))
        weights = torch.softmax(self.v(energy), dim=1)             # (B, W, 1)
        return (weights * enc_out).sum(dim=1, keepdim=True)        # (B, 1, d)

    def forward(
        self,
        past_features: torch.Tensor,
        future_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:

        enc_out, (h, c) = self.encoder(past_features)

        curr_input = (
            past_features[:, -1, self.target_idx]
            .unsqueeze(1).unsqueeze(2)
        )
        outputs = []

        for t in range(self.forecast_horizon):
            context   = self._attend(h[-1], enc_out)              # (B, 1, d)
            rnn_input = torch.cat([curr_input, context], dim=2)    # (B, 1, 1+d)
            dec_out, (h, c) = self.decoder(rnn_input, (h, c))
            step_out  = self.output_layer(dec_out.squeeze(1))
            outputs.append(step_out.unsqueeze(1))

            if future_target is not None and random.random() < teacher_forcing_ratio:
                curr_input = future_target[:, t, :].unsqueeze(1)
            else:
                curr_input = step_out.unsqueeze(1)

        return torch.cat(outputs, dim=1)


# ─────────────────────────── Transformer ────────────────────────────────────

class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerModel(nn.Module):
    """
    Encoder-Decoder Transformer for multi-step forecasting.

    Training : teacher-forcing (full sequence fed to decoder at once).
    Inference : autoregressive generation step-by-step.
    """

    def __init__(self, n_features: int, forecast_horizon: int, target_idx: int,
                 d_model: int, num_layers: int, dropout: float, nhead: int) -> None:
        super().__init__()
        self.target_idx       = target_idx
        self.forecast_horizon = forecast_horizon

        self.src_embed = nn.Linear(n_features, d_model)
        self.tgt_embed = nn.Linear(1, d_model)
        self.pos_enc   = _PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(d_model, 1)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask so decoder cannot attend to future tokens."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask.float().masked_fill(mask, float("-inf")).masked_fill(~mask, 0.0)

    def forward(
        self,
        past_features: torch.Tensor,
        future_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,   # not used; kept for API consistency
    ) -> torch.Tensor:

        device  = past_features.device
        src_emb = self.pos_enc(self.src_embed(past_features))

        if future_target is not None:
            # Teacher-forcing: feed shifted ground truth
            start = past_features[:, -1, self.target_idx].view(-1, 1, 1)
            tgt_in  = torch.cat([start, future_target[:, :-1, :]], dim=1)
            tgt_emb = self.pos_enc(self.tgt_embed(tgt_in))
            tgt_mask = self._causal_mask(tgt_emb.size(1), device)
            out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
            return self.output_layer(out)

        # Autoregressive inference
        memory   = self.transformer.encoder(src_emb)
        tgt_inp  = past_features[:, -1, self.target_idx].view(-1, 1, 1)
        outputs  = []

        for _ in range(self.forecast_horizon):
            tgt_emb  = self.pos_enc(self.tgt_embed(tgt_inp))
            tgt_mask = self._causal_mask(tgt_emb.size(1), device)
            out      = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            next_val = self.output_layer(out[:, -1:, :])   # (B, 1, 1)
            outputs.append(next_val)
            tgt_inp  = torch.cat([tgt_inp, next_val], dim=1)

        return torch.cat(outputs, dim=1)                   # (B, H, 1)


# ─────────────────────────── Registry ───────────────────────────────────────

MODEL_REGISTRY: dict[str, type] = {
    "LSTM":        LSTMSeq2Seq,
    "LSTM_Attn":   LSTMAttention,
    "Transformer": TransformerModel,
}


def build_model(name: str, cfg, n_features: int, target_idx: int) -> nn.Module:
    """Instantiate a model by name using the global Config."""
    common = dict(
        n_features=n_features,
        forecast_horizon=cfg.FORECAST_HORIZON,
        target_idx=target_idx,
        d_model=cfg.D_MODEL,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
    )
    if name == "Transformer":
        common["nhead"] = cfg.NHEAD

    cls = MODEL_REGISTRY[name]
    return cls(**common).to(cfg.DEVICE)


def model_summary(model: nn.Module, name: str) -> None:
    """Print a concise architecture + parameter-count summary."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'─'*60}")
    print(f"  {name.upper()} MODEL SUMMARY")
    print(f"{'─'*60}")
    print(f"  Total parameters     : {total:>12,}")
    print(f"  Trainable parameters : {trainable:>12,}")
    print(f"{'─'*60}")
    print(f"  {'Module':<18}  {'Type'}")
    print(f"  {'──────':<18}  {'────'}")
    for mname, module in model.named_children():
        if isinstance(module, nn.LSTM):
            desc = (
                f"LSTM(in={module.input_size}, hid={module.hidden_size}, "
                f"layers={module.num_layers}, drop={module.dropout})"
            )
        elif isinstance(module, nn.Linear):
            desc = f"Linear(in={module.in_features}, out={module.out_features})"
        elif isinstance(module, nn.Transformer):
            desc = (
                f"Transformer(d={module.d_model}, heads={module.nhead}, "
                f"enc={module.encoder.num_layers}, dec={module.decoder.num_layers})"
            )
        else:
            desc = module.__class__.__name__

        params = sum(p.numel() for p in module.parameters())
        print(f"  {mname:<18}  {desc}")
        print(f"  {'':18}  ↳ params: {params:,}")
    print(f"{'─'*60}\n")
