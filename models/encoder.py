import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ── Utilities ─────────────────────────────────────────────────────────────────
class L2Norm(nn.Module):
    """L2-normalise along dim=1 (feature dimension).
    eps=1e-8 prevents 0/0 when a zero-norm vector arrives (e.g. from
    zero-padded missing EEG channels through LayerNorm).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=1, eps=1e-8)


# ── Transformer EEG Encoder ───────────────────────────────────────────────────
class TransformerEEGEncoder(nn.Module):
    """
    Encodes (B, C, T) EEG → (B, OUT_DIM) L2-normalised embedding.

    Architecture:
        1. Project channel-wise: (B, T, C) → (B, T, embed_dim)
        2. Add learned positional embedding
        3. N × Pre-Norm Transformer encoder layers
        4. Pool (mean or cls token)
        5. Linear projection → LayerNorm → L2Norm
    """
    def __init__(
        self,
        n_channels: int = config.N_CHANNELS,
        seq_len   : int = config.SEQ_LEN,
        embed_dim : int = config.EMBED_DIM,
        n_heads   : int = config.N_HEADS,
        n_layers  : int = config.N_LAYERS,
        ff_dim    : int = config.FF_DIM,
        dropout   : float = config.DROPOUT,
        out_dim   : int = config.OUT_DIM,
        pooling   : str = "mean",
    ):
        super().__init__()
        self.pooling     = pooling
        self.embed_dim   = embed_dim

        # Channel-to-embedding projection (applied per time-step)
        self.input_proj  = nn.Linear(n_channels, embed_dim)

        # Learnable positional embedding
        self.pos_embed   = nn.Parameter(
            torch.zeros(1, seq_len + 1, embed_dim))   # +1 for optional [CLS]
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # [CLS] token (only used when pooling == "cls")
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model      = embed_dim,
            nhead        = n_heads,
            dim_feedforward = ff_dim,
            dropout      = dropout,
            activation   = "gelu",
            batch_first  = True,
            norm_first   = True,   # Pre-LN: more stable gradients
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            L2Norm(),
        )
        self._init_weights()

    def _init_weights(self):
        # Conservative init for the input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)  or  (B, T, C)  –– shape is auto-detected.
        Returns:
            embedding: (B, out_dim)  L2-normalised.
        """
        # Guarantee (B, T, C) for the linear projection
        if x.dim() == 2:
            x = x.unsqueeze(0)                         # (1, C, T)
        if x.shape[-1] != self.input_proj.in_features:
            if x.shape[1] == self.input_proj.in_features:
                x = x.permute(0, 2, 1)                 # (B, C, T) → (B, T, C)
        # x is now (B, T, C)
        x = self.input_proj(x)                         # (B, T, embed_dim)

        B, T, _ = x.shape
        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            x   = torch.cat([cls, x], dim=1)           # (B, 1+T, embed_dim)
            x   = x + self.pos_embed[:, :T+1, :]
        else:
            x = x + self.pos_embed[:, :T, :]

        x = self.transformer(x)                        # (B, T(+1), embed_dim)

        if self.pooling == "cls":
            pooled = x[:, 0]                           # [CLS] token
        else:
            pooled = x.mean(dim=1)                     # mean over time

        return self.output_proj(pooled)                # (B, out_dim) L2-normed


# ── LSTM EEG Encoder ──────────────────────────────────────────────────────────
class LSTMEEGEncoder(nn.Module):
    """
    Baseline LSTM encoder: (B, C, T) → (B, OUT_DIM) L2-normalised.

    Architecture:
        1. Bi-LSTM over time steps, with channel as feature dimension
        2. Concatenate [last-forward | last-backward] states
        3. Linear projection → LayerNorm → L2Norm
    """
    def __init__(
        self,
        n_channels: int = config.N_CHANNELS,
        hidden_dim: int = config.EMBED_DIM,
        n_layers  : int = 2,
        dropout   : float = config.DROPOUT,
        out_dim   : int = config.OUT_DIM,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = n_channels,
            hidden_size   = hidden_dim,
            num_layers    = n_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if n_layers > 1 else 0.0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            L2Norm(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            embedding: (B, out_dim)  L2-normalised.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        # LSTM expects (B, T, C)
        if x.shape[1] != self.lstm.input_size:
            x = x.permute(0, 2, 1)                    # (B, C, T) → (B, T, C)
        _, (h_n, _) = self.lstm(x)                    # h_n: (2*n_layers, B, hidden)
        # Concat last forward and backward hidden states
        fwd = h_n[-2]                                  # (B, hidden)
        bwd = h_n[-1]                                  # (B, hidden)
        pooled = torch.cat([fwd, bwd], dim=-1)
        return self.output_proj(pooled)
