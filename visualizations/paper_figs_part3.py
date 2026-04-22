import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch
import nltk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from models.encoder import TransformerEEGEncoder

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = config.DATA_DIR
DEFAULT_CKPT = config.CHECKPOINT_DIR

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

meta = json.load(open(os.path.join(DEFAULT_DATA, "metadata.json")))
id_to_synset = {v: k for k, v in meta["synset_to_id"].items()}
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn

def get_name(lid):
    syn = id_to_synset.get(int(lid), "")
    try:
        ss = wn.synset_from_pos_and_offset('n', int(syn[1:]))
        return ss.lemma_names()[0].replace('_', ' ').title()
    except:
        return f"Class {lid}"

def fig6_spectral():
    eeg_np = np.load(os.path.join(DEFAULT_DATA, "eeg_signals.npy")).astype(np.float32)
    lbls_np = np.load(os.path.join(DEFAULT_DATA, "labels.npy"))
    fs = 128
    bands = {'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 45)}
    COLORS = ["#1a6fbf", "#cc3d00", "#107c10"]
    from collections import Counter
    top_3 = [lid for lid, _ in Counter(lbls_np).most_common(3)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor='white')
    for i, cid in enumerate(top_3):
        idxs = np.where(lbls_np == cid)[0]
        class_eeg = eeg_np[idxs]
        all_psds = []
        for s in range(len(class_eeg)):
            sample_psds = []
            for ch in range(5):
                f, pxx = welch(class_eeg[s, ch], fs, nperseg=64)
                sample_psds.append(pxx)
            all_psds.append(np.mean(sample_psds, axis=0))
        mean_psd = np.mean(all_psds, axis=0)
        axes[i].plot(f, 10 * np.log10(mean_psd), color='#444444', linewidth=1.5)
        for b_name, (low, high) in bands.items():
            mask = (f >= low) & (f <= high)
            color = COLORS[list(bands.keys()).index(b_name)]
            axes[i].fill_between(f, 10 * np.log10(mean_psd), -100, 
                                 where=mask, color=color, alpha=0.3, label=b_name)
        axes[i].set_title(f"EEG Power Band: {get_name(cid)}", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Frequency (Hz)")
        if i == 0: axes[i].set_ylabel("Power Spectral Density (dB/Hz)")
        axes[i].set_xlim(1, 60)
        axes[i].set_ylim(10, 80)
        axes[i].grid(True, linestyle='--', alpha=0.5)
        if i == 0: axes[i].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("fig_eeg_spectra.png", dpi=150, bbox_inches='tight')
    plt.close()

@torch.no_grad()
def fig7_large_generation():
    device = torch.device("cpu")
    enc_path = os.path.join(DEFAULT_CKPT, "encoder_transformer_imagenet_main.pth")
    enc_raw = torch.load(enc_path, map_location=device, weights_only=False)
    enc_args = enc_raw.get("args", {})
    sd = enc_raw["encoder_state"]
    encoder = TransformerEEGEncoder(
        n_channels=5, seq_len=sd["pos_embed"].shape[1],
        embed_dim=enc_args.get("embed_dim", 64),
        n_heads=enc_args.get("n_heads", 4),
        n_layers=enc_args.get("n_layers", 2),
        ff_dim=sd["transformer.layers.0.linear1.weight"].shape[0],
        out_dim=sd["output_proj.0.weight"].shape[0],
        pooling=enc_args.get("pooling", "mean"),
    ).to(device)
    encoder.pos_embed = nn.Parameter(torch.zeros(1, sd["pos_embed"].shape[1], enc_args.get("embed_dim", 64)))
    encoder.load_state_dict(sd, strict=False)
    encoder.eval()
    gan_path = os.path.join(DEFAULT_CKPT, "gan_transformer_imagenet_main.pth")
    gan_raw = torch.load(gan_path, map_location=device, weights_only=False)
    z_dim = gan_raw["G_state"]["fc.weight"].shape[1]
    netG = SimpleDCGANGenerator(z_dim=z_dim).to(device)
    netG.load_state_dict(gan_raw["G_state"])
    netG.eval()
    eeg_np = np.load(os.path.join(DEFAULT_DATA, "eeg_signals.npy")).astype(np.float32)
    imgs_np = np.load(os.path.join(DEFAULT_DATA, "images.npy"), mmap_mode='r')
    lbls_np = np.load(os.path.join(DEFAULT_DATA, "labels.npy"))
    n_rows, n_cols = 16, 8
    total = n_rows * n_cols
    idx = np.linspace(0, len(eeg_np)-1, total, dtype=int)
    fig = plt.figure(figsize=(20, 30), facecolor='white')
    outer = gridspec.GridSpec(n_rows, n_cols, hspace=0.3, wspace=0.1)
    noise_dim = config.NOISE_DIM
    eeg_feat_dim = z_dim - noise_dim
    for i in range(total):
        s_idx = idx[i]
        real_img = imgs_np[s_idx]
        real_name = get_name(lbls_np[s_idx])
        eeg_in = torch.from_numpy(eeg_np[s_idx:s_idx+1]).to(device)
        feat = encoder(eeg_in)
        if feat.shape[1] != eeg_feat_dim:
            feat = feat[:, :eeg_feat_dim]
        z = torch.cat([torch.randn(1, noise_dim), feat], dim=1)
        fake = netG(z)
        fake = ((fake + 1) / 2.0).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()[0]
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.02)
        ax_real = plt.Subplot(fig, inner[0])
        ax_real.imshow(real_img); ax_real.axis('off')
        fig.add_subplot(ax_real)
        ax_fake = plt.Subplot(fig, inner[1])
        ax_fake.imshow(fake); ax_fake.axis('off')
        fig.add_subplot(ax_fake)
        ax_title = fig.add_subplot(outer[i])
        ax_title.text(0.5, -0.15, f"{real_name[:12]}", ha='center', fontsize=7, color='#444', transform=ax_title.transAxes)
        ax_title.axis('off')
    plt.savefig("cherry_pick_reference.png", dpi=120, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    fig6_spectral()
    fig7_large_generation()
