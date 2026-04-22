import os, sys, argparse, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from models.encoder import TransformerEEGEncoder

DEFAULT_CKPT = config.CHECKPOINT_DIR
DEFAULT_DATA = config.DATA_DIR

def _build_name_fn(meta_path):
    meta = json.load(open(meta_path))
    id_to_synset = {v: k for k, v in meta["synset_to_id"].items()}
    try:
        import nltk
        nltk.download('wordnet', quiet=True)
        from nltk.corpus import wordnet as wn
        def get_name(label_id):
            syn = id_to_synset.get(int(label_id), "")
            if not syn: return f"class {label_id}"
            try:
                ss = wn.synset_from_pos_and_offset('n', int(syn[1:]))
                name = ss.lemma_names()[0].replace('_', ' ')
                return name.title()
            except: return syn
    except:
        def get_name(label_id):
            return id_to_synset.get(int(label_id), f"class {label_id}")
    return get_name

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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="eeg_comparison.png")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    get_name = _build_name_fn(os.path.join(DEFAULT_DATA, "metadata.json"))
    enc_path = os.path.join(DEFAULT_CKPT, "encoder_transformer_imagenet_main.pth")
    enc_raw = torch.load(enc_path, map_location=device, weights_only=False)
    enc_args = enc_raw.get("args", {})
    sd = enc_raw["encoder_state"]
    embed_dim = enc_args.get("embed_dim", 64)
    seq_len = sd["pos_embed"].shape[1]
    out_dim = sd["output_proj.0.weight"].shape[0]
    encoder = TransformerEEGEncoder(
        n_channels=5, seq_len=seq_len,
        embed_dim=embed_dim,
        n_heads=enc_args.get("n_heads", 4),
        n_layers=enc_args.get("n_layers", 2),
        ff_dim=sd["transformer.layers.0.linear1.weight"].shape[0],
        out_dim=out_dim,
        pooling=enc_args.get("pooling", "mean"),
        dropout=0.0,
    ).to(device)
    encoder.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
    encoder.load_state_dict(sd, strict=False)
    encoder.eval()
    gan_path = os.path.join(DEFAULT_CKPT, "gan_transformer_imagenet_main.pth")
    gan_raw = torch.load(gan_path, map_location=device, weights_only=False)
    z_dim = gan_raw["G_state"]["fc.weight"].shape[1]
    netG = SimpleDCGANGenerator(z_dim=z_dim).to(device)
    netG.load_state_dict(gan_raw["G_state"])
    netG.eval()
    noise_dim = config.NOISE_DIM
    eeg_feat_dim = z_dim - noise_dim
    eeg_np = np.load(os.path.join(DEFAULT_DATA, "eeg_signals.npy")).astype(np.float32)
    imgs_np = np.load(os.path.join(DEFAULT_DATA, "images.npy"), mmap_mode='r')
    lbls_np = np.load(os.path.join(DEFAULT_DATA, "labels.npy"))
    idx = np.linspace(0, len(eeg_np) - 1, args.n, dtype=int)
    eeg_sel = torch.from_numpy(eeg_np[idx])
    real_sel = np.array(imgs_np[idx])
    lbls_sel = lbls_np[idx]
    feat = encoder(eeg_sel)
    if feat.shape[1] != eeg_feat_dim:
        feat = feat[:, :eeg_feat_dim]
    noise = torch.randn(args.n, noise_dim)
    z = torch.cat([noise, feat], dim=1)
    fake = netG(z)
    fake = ((fake + 1) / 2.0).clamp(0, 1)
    fake = fake.cpu().permute(0, 2, 3, 1).numpy()
    BLUE = "#1a6fbf"
    ORG = "#cc3d00"
    GRAY = "#444444"
    fig = plt.figure(figsize=(11, args.n * 2.6 + 1.0), facecolor="white")
    outer = gridspec.GridSpec(
        args.n + 1, 3, figure=fig,
        hspace=0.08, wspace=0.05,
        left=0.02, right=0.98,
        top=0.97, bottom=0.01,
        height_ratios=[0.35] + [1.0] * args.n,
    )
    for col, (label, color) in enumerate([
        ("EEG Brain Signal", GRAY),
        ("Original ImageNet Photo", BLUE),
        ("Brain-Generated Image", ORG),
    ]):
        ax = fig.add_subplot(outer[0, col])
        ax.set_facecolor("white")
        ax.axis("off")
        ax.text(0.5, 0.4, label, color=color,
                fontsize=11, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes)
    CH_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    CH_NAMES = ["AF3", "AF4", "T7", "T8", "Pz"]
    for i in range(args.n):
        ax_eeg = fig.add_subplot(outer[i + 1, 0])
        ax_eeg.set_facecolor("#f5f5f5")
        eeg_sample = eeg_np[idx[i]]
        eeg_z = (eeg_sample - eeg_sample.mean(axis=1, keepdims=True)) / \
                (eeg_sample.std(axis=1, keepdims=True) + 1e-6)
        t = np.arange(eeg_z.shape[1])
        spacing = 3.5
        for ch in range(eeg_z.shape[0]):
            offset = (eeg_z.shape[0] - 1 - ch) * spacing
            ax_eeg.plot(t, eeg_z[ch] + offset,
                        color=CH_COLORS[ch], linewidth=0.8, alpha=0.9)
            ax_eeg.text(-4, offset, CH_NAMES[ch],
                        color=CH_COLORS[ch], fontsize=5.5, va='center', ha='right')
        ax_eeg.set_xlim(-8, len(t))
        ax_eeg.set_ylim(-spacing, eeg_z.shape[0] * spacing)
        ax_eeg.axis("off")
        real_name = get_name(lbls_sel[i])
        ax_real = fig.add_subplot(outer[i + 1, 1])
        ax_real.imshow(real_sel[i])
        ax_real.axis("off")
        ax_real.set_title(real_name, color=BLUE, fontsize=8.5, fontweight="bold", pad=3)
        ax_fake = fig.add_subplot(outer[i + 1, 2])
        ax_fake.imshow(fake[i])
        ax_fake.axis("off")
        ax_fake.set_title(f"Generated ({real_name})", color=ORG, fontsize=7.5, pad=3)
    plt.savefig(args.output, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    main()
