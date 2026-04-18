import math
import torch
import torch.nn as nn
import config

class TransformerEEGEncoder(nn.Module):
    def __init__(
        self,
        n_channels : int = config.N_CHANNELS,
        seq_len    : int = config.SEQ_LEN,
        embed_dim  : int = config.EMBED_DIM,
        n_heads    : int = config.N_HEADS,
        n_layers   : int = config.N_LAYERS,
        ff_dim     : int = config.FF_DIM,
        dropout    : float = config.DROPOUT,
        out_dim    : int = config.OUT_DIM,
        pooling    : str = "mean",
    ):
        super().__init__()
        assert embed_dim % n_heads == 0
        assert pooling in ("mean", "cls")
        self.pooling   = pooling
        self.embed_dim = embed_dim
        self.seq_len   = seq_len
        self.input_proj = nn.Linear(n_channels, embed_dim)
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = n_heads,
            dim_feedforward = ff_dim,
            dropout         = dropout,
            activation      = "relu",
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.pooling == "cls":
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            x   = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        if self.pooling == "cls":
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)
        return self.output_proj(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class LSTMEEGEncoder(nn.Module):
    def __init__(
        self,
        n_channels : int = config.N_CHANNELS,
        hidden_dim : int = 128,
        out_dim    : int = config.OUT_DIM,
        num_layers : int = 1,
        dropout    : float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = n_channels,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        return self.output_proj(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
