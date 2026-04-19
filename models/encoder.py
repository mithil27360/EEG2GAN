import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

class TransformerEEGEncoder(nn.Module):
    def __init__(
        self,
        n_channels= config.N_CHANNELS,
        seq_len= config.SEQ_LEN,
        embed_dim= config.EMBED_DIM,
        n_heads= config.N_HEADS,
        n_layers= config.N_LAYERS,
        ff_dim= config.FF_DIM,
        dropout= config.DROPOUT,
        out_dim= config.OUT_DIM,
        pooling= "mean",
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
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # No ReLU at end — unit-normalized embeddings are all directions
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            L2Norm(),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.pooling == "cls":
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x):
        # Ensure x is (Batch, Time, Channels)
        if x.shape[-1] != self.input_proj.in_features:
            if x.shape[1] == self.input_proj.in_features:
                x = x.permute(0, 2, 1)

        B, T, C = x.shape
        x = self.input_proj(x)
        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            x   = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
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
        n_channels= config.N_CHANNELS,
        hidden_dim= 256,
        out_dim= config.OUT_DIM,
        num_layers= 2,
        dropout= 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = n_channels,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0,
            bidirectional= True,
        )
        # Bidirectional doubles hidden size
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            L2Norm(),
        )

    def forward(self, x):
        # Ensure x is (Batch, Time, Channels)
        if x.shape[-1] != self.lstm.input_size:
            if x.shape[1] == self.lstm.input_size:
                x = x.permute(0, 2, 1)

        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers*2, B, hidden) for bidirectional
        # Concatenate last forward and backward hidden states
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.output_proj(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
