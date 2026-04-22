import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nltk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import config
from models.encoder import TransformerEEGEncoder

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(config.DATA_DIR, "mindbigdata_imagenet")
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

@torch.no_grad()
def load_models():
    device = torch.device("cpu")
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
    return encoder, netG, z_dim

def fig4_grids(encoder, netG, z_dim):
    device = torch.device("cpu")
    eeg_np = np.load(os.path.join(DEFAULT_DATA, "eeg_signals.npy")).astype(np.float32)
    lbls_np = np.load(os.path.join(DEFAULT_DATA, "labels.npy"))
    classes_to_show = [0, 47, 192, 381]
    fig = plt.figure(figsize=(12, 12), facecolor='white')
    outer = gridspec.GridSpec(2, 2, hspace=0.2, wspace=0.1)
    noise_dim = config.NOISE_DIM
    eeg_feat_dim = z_dim - noise_dim
    for i, cid in enumerate(classes_to_show):
        class_name = get_name(cid)
        idxs = np.where(lbls_np == cid)[0]
        if len(idxs) < 16:
            n_samples = len(idxs)
        else:
            n_samples = 16
            idxs = idxs[:16]
        eeg_batch = torch.from_numpy(eeg_np[idxs]).to(device)
        feat = encoder(eeg_batch)
        if feat.shape[1] != eeg_feat_dim:
            feat = feat[:, :eeg_feat_dim]
        noise = torch.randn(len(idxs), noise_dim)
        z = torch.cat([noise, feat], dim=1)
        fake = netG(z)
        fake = ((fake + 1) / 2.0).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
        inner = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer[i], hspace=0.05, wspace=0.05)
        for j in range(n_samples):
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(fake[j])
            ax.axis('off')
            fig.add_subplot(ax)
        ax_title = fig.add_subplot(outer[i])
        ax_title.set_title(class_name, color='#1a6fbf', fontsize=11, fontweight='bold', pad=10)
        ax_title.axis('off')
    plt.savefig("fig_per_class_grids.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    enc, g, z_d = load_models()
    fig4_grids(enc, g, z_d)
