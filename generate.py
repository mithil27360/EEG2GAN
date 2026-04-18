import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import config
from models.encoder import TransformerEEGEncoder, LSTMEEGEncoder
from models.gan import Generator

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_channels = 5 if args.dataset == "imagenet" else 14
    if args.encoder_type == "transformer":
        encoder = TransformerEEGEncoder(n_channels=n_channels).to(device)
    else:
        encoder = LSTMEEGEncoder(n_channels=n_channels).to(device)
        
    if os.path.isfile(args.encoder_ckpt):
        ckpt = torch.load(args.encoder_ckpt, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    netG = Generator().to(device)
    if os.path.isfile(args.gan_ckpt):
        ckpt = torch.load(args.gan_ckpt, map_location=device)
        netG.load_state_dict(ckpt["G_state"])
    netG.eval()

    if args.input_npy and os.path.isfile(args.input_npy):
        eeg = np.load(args.input_npy).astype(np.float32)
        if eeg.ndim == 2: eeg = eeg[np.newaxis, ...]
        eeg = torch.from_numpy(eeg).to(device)
    else:
        print("Using random EEG input for demo...")
        eeg = torch.randn(1, n_channels, config.SEQ_LEN, device=device)

    with torch.no_grad():
        feat = encoder(eeg)
        noise = torch.randn(eeg.size(0), config.NOISE_DIM, device=device)
        z = torch.cat([noise, feat], dim=1).view(eeg.size(0), config.Z_DIM, 1, 1)
        fake = netG(z)
        
        fake = (fake + 1) / 2.0
        fake = fake.cpu().permute(0, 2, 3, 1).numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(fake[0])
    plt.axis("off")
    plt.title(f"Brain-Generated Image ({args.dataset})")
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Success! Image saved to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_ckpt", type=str, required=True)
    p.add_argument("--gan_ckpt",     type=str, required=True)
    p.add_argument("--encoder_type", choices=["transformer", "lstm"], default="transformer")
    p.add_argument("--dataset",      choices=["objects", "characters", "mindbigdata", "imagenet"], default="objects")
    p.add_argument("--input_npy",    type=str, default="")
    p.add_argument("--output",       type=str, default="generated_brain_image.png")
    args = p.parse_args()
    generate(args)
