import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from models.encoder import TransformerEEGEncoder, LSTMEEGEncoder

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = config.CHECKPOINT_DIR
DEFAULT_DATA = config.DATA_DIR

class SimpleDCGANGenerator(nn.Module):
    def __init__(self, z_dim=228, nc=3):
        super().__init__()
        self.fc = nn.Linear(z_dim, 4 * 4 * 512)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  4, 2, 1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d(64,  32,  4, 2, 1, bias=False),
            nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.ConvTranspose2d(32,  nc,  4, 2, 1, bias=False),
        )
    def forward(self, z):
        return torch.tanh(self.conv_layers(self.fc(z).view(-1, 512, 4, 4)))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--encoder_type", choices=["transformer","lstm"], default="transformer")
    p.add_argument("--encoder_ckpt", type=str, default="")
    p.add_argument("--gan_ckpt", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="comparison.png")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = "main" if args.encoder_type == "transformer" else "baseline"
    enc_ckpt = args.encoder_ckpt or os.path.join(DEFAULT_CKPT, f"encoder_{args.encoder_type}_imagenet_{tag}.pth")
    gan_ckpt = args.gan_ckpt or os.path.join(DEFAULT_CKPT, f"gan_{args.encoder_type}_imagenet_{tag}.pth")
    enc_raw = torch.load(enc_ckpt, map_location=device, weights_only=False)
    enc_args = enc_raw.get("args", {})
    sd = enc_raw["encoder_state"]
    embed_dim = enc_args.get("embed_dim", sd["input_proj.weight"].shape[0])
    n_heads = enc_args.get("n_heads", 4)
    n_layers = enc_args.get("n_layers", 2)
    ff_dim = sd["transformer.layers.0.linear1.weight"].shape[0]
    out_dim = sd["output_proj.0.weight"].shape[0]
    seq_len = sd["pos_embed"].shape[1]
    pooling = enc_args.get("pooling", "mean")
    encoder = TransformerEEGEncoder(
        n_channels=5, seq_len=seq_len,
        embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers,
        ff_dim=ff_dim, out_dim=out_dim, pooling=pooling,
        dropout=enc_args.get("dropout", 0.1),
    ).to(device)
    encoder.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim, device=device))
    encoder.load_state_dict(sd, strict=False)
    encoder.eval()
    gan_raw = torch.load(gan_ckpt, map_location=device, weights_only=False)
    z_dim = gan_raw["G_state"]["fc.weight"].shape[1]
    noise_dim = config.NOISE_DIM
    eeg_feat_dim = z_dim - noise_dim
    netG = SimpleDCGANGenerator(z_dim=z_dim).to(device)
    netG.load_state_dict(gan_raw["G_state"])
    netG.eval()
    eeg_np = np.load(os.path.join(DEFAULT_DATA, "eeg_signals.npy")).astype(np.float32)
    imgs_np = np.load(os.path.join(DEFAULT_DATA, "images.npy"), mmap_mode='r')
    lbls_np = np.load(os.path.join(DEFAULT_DATA, "labels.npy"))
    idx = np.linspace(0, len(eeg_np) - 1, args.n, dtype=int)
    eeg_sel = torch.from_numpy(eeg_np[idx]).to(device)
    real_sel = imgs_np[idx]
    lbls_sel = lbls_np[idx]
    feat = encoder(eeg_sel)
    if feat.shape[1] != eeg_feat_dim:
        feat = feat[:, :eeg_feat_dim]
    noise = torch.randn(args.n, noise_dim, device=device)
    z = torch.cat([noise, feat], dim=1)
    fake = netG(z)
    fake = ((fake + 1) / 2.0).clamp(0, 1)
    fake = fake.cpu().permute(0, 2, 3, 1).numpy()
    fig = plt.figure(figsize=(7, args.n * 2.0 + 1.2), facecolor="#0e0e0e")
    header_ax = fig.add_axes([0.02, 0.97, 0.96, 0.025])
    header_ax.set_facecolor("#0e0e0e")
    header_ax.axis("off")
    header_ax.text(0.25, 0.5, "📷  Original ImageNet Photo",
                   color="#6bcfff", fontsize=11, fontweight="bold",
                   ha="center", va="center", transform=header_ax.transAxes)
    header_ax.text(0.75, 0.5, "🧠  Brain-Generated Image",
                   color="#ff8c6b", fontsize=11, fontweight="bold",
                   ha="center", va="center", transform=header_ax.transAxes)
    cell_h = 0.96 / args.n
    for i in range(args.n):
        y_pos = 0.97 - (i + 1) * cell_h
        ax_real = fig.add_axes([0.04, y_pos, 0.43, cell_h * 0.88])
        ax_real.imshow(real_sel[i])
        ax_real.axis("off")
        ax_real.set_title(f"class {lbls_sel[i]}",
                          color="#aaaaaa", fontsize=7.5, pad=2)
        for spine in ax_real.spines.values():
            spine.set_edgecolor("#6bcfff")
            spine.set_linewidth(1.5)
            spine.set_visible(True)
        ax_arr = fig.add_axes([0.48, y_pos, 0.06, cell_h * 0.88])
        ax_arr.set_facecolor("#0e0e0e")
        ax_arr.axis("off")
        ax_arr.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#888888",
                                        lw=1.8, mutation_scale=14))
        ax_arr.text(0.5, 0.15, "EEG", color="#888888", fontsize=6.5,
                    ha="center", va="center", transform=ax_arr.transAxes)
        ax_fake = fig.add_axes([0.53, y_pos, 0.43, cell_h * 0.88])
        ax_fake.imshow(fake[i])
        ax_fake.axis("off")
        for spine in ax_fake.spines.values():
            spine.set_edgecolor("#ff8c6b")
            spine.set_linewidth(1.5)
            spine.set_visible(True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    main()
