import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1, eps=1e-8)

class TransformerEEGEncoder(nn.Module):
    def __init__(
        self,
        n_channels=config.N_CHANNELS,
        seq_len=config.SEQ_LEN,
        embed_dim=config.EMBED_DIM,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        ff_dim=config.FF_DIM,
        dropout=config.DROPOUT,
        out_dim=config.OUT_DIM,
        pooling="mean",
    ):
        super().__init__()
        self.pooling = pooling
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(n_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            L2Norm(),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.input_proj.in_features:
            if x.shape[1] == self.input_proj.in_features:
                x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        B, T, _ = x.shape
        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.pos_embed[:, :T+1, :]
        else:
            x = x + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        if self.pooling == "cls":
            pooled = x[:, 0]
        else:
            pooled = x.mean(dim=1)
        return self.output_proj(pooled)

class LSTMEEGEncoder(nn.Module):
    def __init__(
        self,
        n_channels=config.N_CHANNELS,
        hidden_dim=config.EMBED_DIM,
        n_layers=2,
        dropout=config.DROPOUT,
        out_dim=config.OUT_DIM,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            L2Norm(),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.lstm.input_size:
            x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        fwd = h_n[-2]
        bwd = h_n[-1]
        pooled = torch.cat([fwd, bwd], dim=-1)
        return self.output_proj(pooled)