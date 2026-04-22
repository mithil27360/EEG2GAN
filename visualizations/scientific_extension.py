import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import config
from models.encoder import TransformerEEGEncoder
from visualizations.generate_images import SimpleDCGANGenerator, _detect_gan_type, _auto_ckpt

def load_models(device):
    DEFAULT_CKPT = config.CHECKPOINT_DIR
    enc_ckpt = _auto_ckpt(DEFAULT_CKPT, "transformer", "imagenet", "encoder")
    gan_ckpt = _auto_ckpt(DEFAULT_CKPT, "transformer", "imagenet", "gan")
    enc_raw = torch.load(enc_ckpt, map_location=device, weights_only=False)
    sd = enc_raw["encoder_state"]
    enc_args = enc_raw.get("args", {})
    embed_dim = enc_args.get("embed_dim", sd["input_proj.weight"].shape[0])
    n_heads = enc_args.get("n_heads", 4)
    n_layers = enc_args.get("n_layers", 2)
    ff_dim = sd["transformer.layers.0.linear1.weight"].shape[0]
    out_dim = sd["output_proj.0.weight"].shape[0]
    seq_len = sd["pos_embed"].shape[1]
    encoder = TransformerEEGEncoder(
        n_channels=5, seq_len=seq_len, embed_dim=embed_dim, 
        n_heads=n_heads, n_layers=n_layers, ff_dim=ff_dim, 
        out_dim=out_dim, pooling=enc_args.get("pooling", "mean")
    ).to(device)
    encoder.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim, device=device))
    encoder.load_state_dict(sd, strict=False)
    encoder.eval()
    gan_raw = torch.load(gan_ckpt, map_location=device, weights_only=False)
    z_dim = gan_raw["G_state"]["fc.weight"].shape[1]
    gan_type = _detect_gan_type(gan_raw["G_state"])
    if gan_type == "simple":
        netG = SimpleDCGANGenerator(z_dim=z_dim).to(device)
    else:
        from models.gan import Generator
        netG = Generator(z_dim=z_dim, cond_dim=z_dim-100).to(device)
    netG.load_state_dict(gan_raw["G_state"])
    netG.eval()
    return encoder, netG, z_dim

def generate_interpolation(encoder, netG, device, netG_z_dim):
    data_dir = os.path.join(config.DATA_DIR, "mindbigdata_imagenet")
    eeg_np = np.load(os.path.join(data_dir, "eeg_signals.npy")).astype(np.float32)
    idx1, idx2 = 0, 47
    eeg1 = torch.from_numpy(eeg_np[idx1:idx1+1]).to(device)
    eeg2 = torch.from_numpy(eeg_np[idx2:idx2+1]).to(device)
    with torch.no_grad():
        emb1 = encoder(eeg1)
        emb2 = encoder(eeg2)
        eeg_feat_dim = netG_z_dim - 100
        emb1 = emb1[:, :eeg_feat_dim]
        emb2 = emb2[:, :eeg_feat_dim]
        steps = 8
        noise = torch.randn(1, 100, device=device)
        imgs = []
        for alpha in np.linspace(0, 1, steps):
            interp_emb = (1 - alpha) * emb1 + alpha * emb2
            z = torch.cat([noise, interp_emb], dim=1)
            fake = netG(z)
            fake = ((fake + 1) / 2.0).clamp(0, 1)
            imgs.append(fake.cpu().squeeze(0).permute(1, 2, 0).numpy())
    fig, axes = plt.subplots(1, steps, figsize=(steps * 2, 2.5))
    for i, img in enumerate(imgs):
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.savefig("results/figures/fig_interpolation.png", bbox_inches='tight', dpi=150)
    plt.close()

def generate_memorization_check(encoder, netG, device, netG_z_dim):
    data_dir = os.path.join(config.DATA_DIR, "mindbigdata_imagenet")
    real_imgs = np.load(os.path.join(data_dir, "images.npy"))
    num_checks = 4
    eeg_feat_dim = netG_z_dim - 100
    with torch.no_grad():
        noise = torch.randn(num_checks, 100, device=device)
        feat = torch.randn(num_checks, eeg_feat_dim, device=device)
        z = torch.cat([noise, feat], dim=1)
        fakes = netG(z)
        fakes = ((fakes + 1) / 2.0).clamp(0, 1)
        fakes_np = fakes.cpu().permute(0, 2, 3, 1).numpy()
    subset_size = min(2000, len(real_imgs))
    real_subset = real_imgs[:subset_size].reshape(subset_size, -1)
    fig, axes = plt.subplots(num_checks, 2, figsize=(6, num_checks * 3))
    for i in range(num_checks):
        fake_flat = fakes_np[i].reshape(1, -1)
        dists = np.mean(np.abs(real_subset - fake_flat), axis=1)
        nn_idx = np.argmin(dists)
        axes[i, 0].imshow(fakes_np[i])
        axes[i, 0].axis('off')
        axes[i, 1].imshow(real_imgs[nn_idx])
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.savefig("results/figures/fig_memorization_check.png", bbox_inches='tight', dpi=150)
    plt.close()

def generate_hyperparameter_table():
    data = [
        ["Hardware", "Value"],
        ["EEG Device", "Emotiv Insight (5-ch)"],
        ["Sample Rate", f"{config.EEG_SAMPLING_RATE} Hz"],
        ["Bandpass Filter", f"{config.EEG_BANDPASS_FREQ[0]}-{config.EEG_BANDPASS_FREQ[1]} Hz"],
        ["Transformer Layers", f"{config.N_LAYERS}"],
        ["Transformer Heads", f"{config.N_HEADS}"],
        ["Encoder LR", f"{config.ENC_LR}"],
        ["GAN Device", "DCGAN / ResNet"],
        ["GAN G/D LR", f"{config.GAN_LR_G} / {config.GAN_LR_D}"],
        ["Batch Size", f"{config.GAN_BATCH_SIZE}"],
        ["DiffAugment", "Enabled (Color/Transl/Cutout)"],
        ["Mode-Seeking Lambda", f"{config.LAMBDA_MS}"]
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data, loc='center', cellLoc='center', colWidths=[0.4, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
        else:
            cell.set_facecolor('#f2f2f2' if row % 2 == 0 else 'white')
    plt.savefig("results/figures/tab_hyperparameters.png", bbox_inches='tight', dpi=200)
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, netG, z_dim = load_models(device)
    generate_interpolation(encoder, netG, device, z_dim)
    generate_memorization_check(encoder, netG, device, z_dim)
    generate_hyperparameter_table()
