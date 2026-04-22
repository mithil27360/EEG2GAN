import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from models.encoder import TransformerEEGEncoder, LSTMEEGEncoder

DEFAULT_CKPT = config.CHECKPOINT_DIR
DEFAULT_DATA = os.path.join(config.DATA_DIR, "mindbigdata_imagenet")

class SimpleDCGANGenerator(nn.Module):
    def __init__(self, z_dim=228, nf=32, nc=3):
        super().__init__()
        self.fc = nn.Linear(z_dim, 4 * 4 * 512)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf), nn.ReLU(True),
            nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias=False),
        )
    def forward(self, z):
        h = self.fc(z).view(-1, 512, 4, 4)
        return torch.tanh(self.conv_layers(h))

def _detect_gan_type(state_dict):
    keys = list(state_dict.keys())
    if any("conv_layers" in k for k in keys):
        return "simple"
    return "resnet"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_type", choices=["transformer", "lstm"], default="transformer")
    p.add_argument("--encoder_ckpt", type=str, default="")
    p.add_argument("--gan_ckpt", type=str, default="")
    p.add_argument("--eeg_npy", type=str, default="")
    p.add_argument("--labels_npy", type=str, default="")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--random", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", default="imagenet", choices=["imagenet","objects","characters"])
    p.add_argument("--output", type=str, default="generated_grid.png")
    return p.parse_args()

def _auto_ckpt(ckpt_dir, enc_type, dataset, kind):
    tag = "main" if enc_type == "transformer" else "baseline"
    name = f"{kind}_{enc_type}_{dataset}_{tag}.pth"
    path = os.path.join(ckpt_dir, name)
    return path if os.path.isfile(path) else ""

@torch.no_grad()
def generate_grid(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_ckpt = args.encoder_ckpt or _auto_ckpt(DEFAULT_CKPT, args.encoder_type, args.dataset, "encoder")
    gan_ckpt = args.gan_ckpt or _auto_ckpt(DEFAULT_CKPT, args.encoder_type, args.dataset, "gan")
    n_channels = 5 if args.dataset == "imagenet" else 14
    enc_raw, enc_args = None, {}
    if enc_ckpt and os.path.isfile(enc_ckpt):
        enc_raw = torch.load(enc_ckpt, map_location=device, weights_only=False)
        enc_args = enc_raw.get("args", {})
    if args.encoder_type == "transformer":
        if enc_raw:
            sd = enc_raw["encoder_state"]
            embed_dim = enc_args.get("embed_dim", sd["input_proj.weight"].shape[0])
            n_heads = enc_args.get("n_heads", 4)
            n_layers = enc_args.get("n_layers", 2)
            ff_dim = sd["transformer.layers.0.linear1.weight"].shape[0]
            out_dim = sd["output_proj.0.weight"].shape[0]
            seq_len = sd["pos_embed"].shape[1]
            pooling = enc_args.get("pooling", "mean")
        else:
            embed_dim, n_heads, n_layers = config.EMBED_DIM, config.N_HEADS, config.N_LAYERS
            ff_dim, out_dim, seq_len, pooling = config.FF_DIM, config.OUT_DIM, config.SEQ_LEN, "mean"
        encoder = TransformerEEGEncoder(
            n_channels=n_channels, seq_len=seq_len,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers,
            ff_dim=ff_dim, out_dim=out_dim, pooling=pooling,
            dropout=enc_args.get("dropout", config.DROPOUT),
        ).to(device)
        encoder.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim, device=device))
    else:
        encoder = LSTMEEGEncoder(n_channels=n_channels).to(device)
    if enc_raw:
        encoder.load_state_dict(enc_raw["encoder_state"], strict=False)
    encoder.eval()
    gan_raw = None
    if gan_ckpt and os.path.isfile(gan_ckpt):
        gan_raw = torch.load(gan_ckpt, map_location=device, weights_only=False)
    if gan_raw:
        z_dim = gan_raw["G_state"]["fc.weight"].shape[1]
        gan_type = _detect_gan_type(gan_raw["G_state"])
    else:
        z_dim, gan_type = config.Z_DIM, "resnet"
    noise_dim = config.NOISE_DIM
    eeg_feat_dim = z_dim - noise_dim
    if gan_type == "simple":
        netG = SimpleDCGANGenerator(z_dim=z_dim).to(device)
    else:
        from models.gan import Generator
        netG = Generator(z_dim=z_dim, cond_dim=eeg_feat_dim).to(device)
    if gan_raw:
        netG.load_state_dict(gan_raw["G_state"])
    netG.eval()
    labels = None
    if args.random:
        eeg = torch.randn(args.n, n_channels, config.SEQ_LEN, device=device)
    else:
        eeg_path = args.eeg_npy or os.path.join(DEFAULT_DATA, "eeg_signals.npy")
        lbl_path = args.labels_npy or os.path.join(DEFAULT_DATA, "labels.npy")
        if not os.path.isfile(eeg_path):
            sys.exit(1)
        eeg_np = np.load(eeg_path).astype(np.float32)
        labels = np.load(lbl_path) if os.path.isfile(lbl_path) else None
        idx = np.linspace(0, len(eeg_np) - 1, args.n, dtype=int)
        eeg_np = eeg_np[idx]
        if labels is not None:
            labels = labels[idx]
        eeg = torch.from_numpy(eeg_np).to(device)
    feat = encoder(eeg)
    if feat.shape[1] != eeg_feat_dim:
        feat = feat[:, :eeg_feat_dim]
    noise = torch.randn(args.n, noise_dim, device=device)
    z = torch.cat([noise, feat], dim=1)
    fake = netG(z)
    fake = ((fake + 1) / 2.0).clamp(0, 1)
    fake = fake.cpu().permute(0, 2, 3, 1).numpy()
    cols = int(np.ceil(args.n / args.rows))
    fig, axes = plt.subplots(args.rows, cols, figsize=(cols * 2.4, args.rows * 2.4))
    fig.patch.set_facecolor("#111111")
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < args.n:
            ax.imshow(fake[i])
            if labels is not None:
                ax.set_title(f"cls {labels[i]}", color="#aaaaaa", fontsize=7)
            ax.axis("off")
        else:
            ax.set_visible(False)
    plt.tight_layout(pad=0.3)
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    generate_grid(args)
