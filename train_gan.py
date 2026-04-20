"""train_gan.py — Conditional GAN Training

Trains the Generator and Discriminator conditioned on frozen EEG embeddings.
Supports both pre-saved images.npy and on-the-fly ImageNet loading.

Key design decisions:
  • Hinge loss + R1 gradient penalty (γ=10, every 4 steps, first D-step only)
  • Mode-seeking loss to encourage output diversity
  • TTUR: D learns at 2× G's LR for better stability
  • DiffAugment on real AND fake images for data efficiency
  • Encoder is fully frozen (no gradients) during GAN training
"""
import os
import sys
import json
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

import config
from dataset import (get_eeg_image_loaders, get_eeg_image_loaders_otf,
                     DummyEEGImageDataset)
from models.encoder import TransformerEEGEncoder, LSTMEEGEncoder
from models.gan import Generator, Discriminator, weights_init
from utils.diffaugment import DiffAugment


# ── Losses ────────────────────────────────────────────────────────────────────
def hinge_loss_d(real_scores: torch.Tensor,
                 fake_scores: torch.Tensor) -> torch.Tensor:
    return (F.relu(1.0 - real_scores) + F.relu(1.0 + fake_scores)).mean()


def hinge_loss_g(fake_scores: torch.Tensor) -> torch.Tensor:
    return -fake_scores.mean()


def mode_seeking_loss(f1: torch.Tensor, f2: torch.Tensor,
                      z1: torch.Tensor, z2: torch.Tensor,
                      eps: float = 1e-5) -> torch.Tensor:
    """Encourage diversity: penalise when similar z produce similar images."""
    img_dist  = (f1 - f2).view(f1.size(0), -1).norm(dim=1).mean()
    z_dist    = (z1 - z2).view(z1.size(0), -1).norm(dim=1).mean()
    return 1.0 / (img_dist / (z_dist + eps) + eps)


def r1_gradient_penalty(netD, real_imgs, eeg_feat, gamma: float = 10.0):
    """R1 weight regularisation (Mescheder et al., 2018)."""
    real_imgs = real_imgs.detach().requires_grad_(True)
    scores    = netD(real_imgs, eeg_feat)
    grads     = torch.autograd.grad(
        outputs     = scores.sum(),
        inputs      = real_imgs,
        create_graph= True,
        retain_graph= True,
    )[0]
    penalty = grads.pow(2).view(grads.size(0), -1).sum(1).mean()
    return (gamma / 2.0) * penalty


def parse_args():
    p = argparse.ArgumentParser(description="Train conditional GAN")
    p.add_argument("--encoder_ckpt",  type=str, required=True)
    p.add_argument("--encoder_type",  choices=["transformer", "lstm"],
                   default="transformer")
    p.add_argument("--dataset",       choices=["objects","characters",
                                               "mindbigdata","imagenet"],
                   default="objects")
    p.add_argument("--dummy",         action="store_true")
    p.add_argument("--batch_size",    type=int,   default=config.GAN_BATCH_SIZE)
    p.add_argument("--epochs",        type=int,   default=config.GAN_EPOCHS)
    p.add_argument("--lr_g",          type=float, default=config.GAN_LR_G)
    p.add_argument("--lr_d",          type=float, default=config.GAN_LR_D)
    p.add_argument("--tag",           type=str,   default="")
    p.add_argument("--no_diffaug",    action="store_true")
    return p.parse_args()


def _build_loader(args) -> DataLoader:
    """Return a DataLoader of (eeg, image, label) tuples."""
    if args.dummy:
        ds = DummyEEGImageDataset(n_samples=256, n_classes=10,
                                  n_channels=config.IMAGENET_CHANNELS
                                  if args.dataset == "imagenet"
                                  else config.N_CHANNELS)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=True, num_workers=0)

    DS_MAP = {
        "objects"    : (config.THOUGHTVIZ_EEG_OBJECTS,   config.THOUGHTVIZ_LABELS_OBJECTS,
                        config.THOUGHTVIZ_IMAGES_OBJECTS),
        "characters" : (config.THOUGHTVIZ_EEG_CHARS,     config.THOUGHTVIZ_LABELS_CHARS,
                        config.THOUGHTVIZ_IMAGES_CHARS),
        "mindbigdata": (config.MINDBIGDATA_EEG,          config.MINDBIGDATA_LABELS,
                        config.MINDBIGDATA_IMAGES),
        "imagenet"   : (config.MINDBIGDATA_IMAGENET_EEG, config.MINDBIGDATA_IMAGENET_LABELS,
                        config.MINDBIGDATA_IMAGENET_IMAGES),
    }
    eeg_p, lbl_p, img_p = DS_MAP[args.dataset]

    if not os.path.exists(eeg_p):
        print(f"Error: EEG file not found: {eeg_p}\n"
              f"       Run process_mindbigdata.py first.")
        sys.exit(1)

    if os.path.exists(img_p):
        # Prefer pre-saved images.npy when available (small datasets)
        tr_ld, _ = get_eeg_image_loaders(eeg_p, lbl_p, img_p,
                                          batch_size=args.batch_size)
        return tr_ld

    # On-the-fly loading for ImageNet (images.npy would be ~GB)
    meta_p = config.MINDBIGDATA_IMAGENET_META
    if not os.path.exists(meta_p):
        print(f"Error: metadata.json not found at {meta_p}.")
        sys.exit(1)

    img_dir = config.IMAGENET_DIR
    if not os.path.isdir(img_dir):
        print(f"Warning: IMAGENET_DIR '{img_dir}' not found — "
              f"images will be random noise.")
        img_dir = None

    print(f"On-the-fly image loading from: {img_dir}")
    tr_ld, _ = get_eeg_image_loaders_otf(
        eeg_p, lbl_p, meta_p, img_dir,
        batch_size=args.batch_size,
    )
    return tr_ld


def _build_encoder(args, device):
    n_ch = config.IMAGENET_CHANNELS if args.dataset == "imagenet" else config.N_CHANNELS
    if args.encoder_type == "transformer":
        enc = TransformerEEGEncoder(n_channels=n_ch).to(device)
    else:
        enc = LSTMEEGEncoder(n_channels=n_ch).to(device)

    if os.path.isfile(args.encoder_ckpt):
        ckpt = torch.load(args.encoder_ckpt, map_location=device,
                          weights_only=False)
        enc.load_state_dict(ckpt["encoder_state"])
        print(f"Loaded encoder from {args.encoder_ckpt}")
    else:
        print(f"Warning: encoder checkpoint not found at {args.encoder_ckpt}. "
              f"Using random weights.")

    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def train(args):
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader = _build_loader(args)
    encoder      = _build_encoder(args, device)

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # TTUR: D uses 2× LR — well-established trick for stable GAN convergence
    optimizerG = optim.Adam(netG.parameters(),
                            lr=args.lr_g, betas=(config.BETA1, config.BETA2))
    optimizerD = optim.Adam(netD.parameters(),
                            lr=args.lr_d * 2.0,
                            betas=(config.BETA1, config.BETA2))
    schedulerG = CosineAnnealingLR(optimizerG, T_max=args.epochs,
                                   eta_min=args.lr_g * 0.05)
    schedulerD = CosineAnnealingLR(optimizerD, T_max=args.epochs,
                                   eta_min=args.lr_d * 0.1)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR,     exist_ok=True)
    tag_str   = f"_{args.tag}" if args.tag else ""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR,
                             f"gan_{args.encoder_type}_{args.dataset}{tag_str}.pth")

    history = {"G_loss": [], "D_loss": []}
    t0      = time.time()

    for epoch in range(args.epochs):
        netG.train()
        netD.train()
        epoch_g = epoch_d = 0.0
        n_batches = 0

        for i, (eeg, real_imgs, _) in enumerate(train_loader):
            B         = eeg.size(0)
            eeg       = eeg.to(device)
            real_imgs = real_imgs.to(device)

            with torch.no_grad():
                eeg_feat = encoder(eeg)   # (B, EEG_FEAT_DIM)

            # ── Discriminator (2 inner steps per outer iter) ────────────────────
            for d_step in range(2):
                optimizerD.zero_grad()

                real_in = (DiffAugment(real_imgs, policy=config.DIFFAUG_POLICY)
                           if not args.no_diffaug else real_imgs)
                real_s   = netD(real_in, eeg_feat.detach())

                noise   = torch.randn(B, config.NOISE_DIM, device=device)
                z       = torch.cat([noise, eeg_feat.detach()], dim=1)
                fakes   = netG(z).detach()
                fake_in = (DiffAugment(fakes, policy=config.DIFFAUG_POLICY)
                           if not args.no_diffaug else fakes)
                fake_s  = netD(fake_in, eeg_feat.detach())

                lossD = hinge_loss_d(real_s, fake_s)

                # R1 every 4 outer steps — ONLY on first D-step to avoid
                # doubling the effective regularisation strength
                if i % 4 == 0 and d_step == 0:
                    lossD = lossD + r1_gradient_penalty(
                        netD,
                        real_imgs.detach().requires_grad_(True),
                        eeg_feat.detach(),
                    )

                lossD.backward()
                torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
                optimizerD.step()

            # ── Generator ────────────────────────────────────────────────────────
            optimizerG.zero_grad()
            noise1 = torch.randn(B, config.NOISE_DIM, device=device)
            noise2 = torch.randn(B, config.NOISE_DIM, device=device)
            z1     = torch.cat([noise1, eeg_feat], dim=1)
            z2     = torch.cat([noise2, eeg_feat], dim=1)
            f1     = netG(z1)
            f2     = netG(z2)

            fi1    = (DiffAugment(f1, policy=config.DIFFAUG_POLICY)
                      if not args.no_diffaug else f1)
            lossG  = hinge_loss_g(netD(fi1, eeg_feat))
            lossMS = mode_seeking_loss(f1, f2, z1, z2)
            total_G = lossG + config.LAMBDA_MS * lossMS
            total_G.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            optimizerG.step()

            epoch_g   += total_G.item()
            epoch_d   += lossD.item()
            n_batches += 1

        schedulerG.step()
        schedulerD.step()
        avg_g = epoch_g / max(n_batches, 1)
        avg_d = epoch_d / max(n_batches, 1)
        history["G_loss"].append(avg_g)
        history["D_loss"].append(avg_d)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:4d}/{args.epochs} | "
              f"G={avg_g:.4f}  D={avg_d:.4f} | t={elapsed:.0f}s", flush=True)

    # ── Save checkpoint and loss curves ─────────────────────────────────────────
    torch.save({
        "epoch"           : args.epochs,
        "G_state"         : netG.state_dict(),
        "D_state"         : netD.state_dict(),
        "optimizerG_state": optimizerG.state_dict(),
        "optimizerD_state": optimizerD.state_dict(),
        "args"            : vars(args),
    }, ckpt_path)
    print(f"GAN checkpoint saved to {ckpt_path}")

    for key, arr in history.items():
        np.save(os.path.join(config.OUTPUT_DIR,
                             f"{key}_{args.encoder_type}_{args.dataset}{tag_str}.npy"),
                np.array(arr))
    return ckpt_path


if __name__ == "__main__":
    args = parse_args()
    train(args)
