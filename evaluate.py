"""evaluate.py — Compute IS, FID, EISC, and K-Means for a trained GAN."""
import os
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import config
from models.encoder import TransformerEEGEncoder, LSTMEEGEncoder
from models.gan import Generator
from utils.metrics import (InceptionScoreCalculator, kmeans_accuracy,
                           EISCCalculator, FIDCalculator, tensor_to_pil_list)
from dataset import DummyEEGImageDataset, EEGImageDataset, EEGDataset


class _EEGOnlyWrapper(Dataset):
    """Wraps an EEGDataset to yield (eeg, dummy_img, label).
    Used when images.npy is not available for ImageNet.
    """
    _dummy: torch.Tensor   # zero image

    def __init__(self, eeg_path, label_path):
        self.ds = EEGDataset(
            eeg_path, label_path,
            window_size = config.EEG_WINDOW_SIZE,
            stride      = config.EEG_WINDOW_SIZE,  # no overlap for eval
        )
        self._dummy = torch.zeros(config.NC, config.IMAGE_SIZE, config.IMAGE_SIZE)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        eeg, label = self.ds[idx]
        return eeg, self._dummy.clone(), label


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained EEG-to-Image GAN")
    p.add_argument("--encoder_ckpt",  type=str,   default="")
    p.add_argument("--gan_ckpt",      type=str,   default="")
    p.add_argument("--encoder_type",  choices=["transformer", "lstm"],
                   default="transformer")
    p.add_argument("--dataset",       choices=["objects","characters",
                                               "mindbigdata","imagenet"],
                   default="objects")
    p.add_argument("--dummy",         action="store_true")
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--n_gen",         type=int,   default=config.IS_N_SAMPLES)
    p.add_argument("--no_eisc",       action="store_true")
    p.add_argument("--output_csv",    type=str,   default="")
    return p.parse_args()


@torch.no_grad()
def generate_images(encoder, netG, dataloader, device, n_gen=2048):
    """Run encoder + generator over loader, return PIL image lists + embeddings."""
    encoder.eval()
    netG.eval()
    gen_imgs, real_imgs_out = [], []
    all_embs, all_labels    = [], []
    has_real = True

    for eeg, real_img, label in dataloader:
        if len(gen_imgs) >= n_gen:
            break
        eeg_feat = encoder(eeg.to(device))
        noise    = torch.randn(eeg.size(0), config.NOISE_DIM, device=device)
        z        = torch.cat([noise, eeg_feat], dim=1)
        fake     = netG(z)

        gen_imgs.extend(tensor_to_pil_list(fake))

        if real_img.abs().sum() > 0:
            real_imgs_out.extend(tensor_to_pil_list(real_img))
        else:
            has_real = False

        all_embs.append(eeg_feat.cpu().numpy())
        all_labels.append(label.numpy())

    embs   = np.concatenate(all_embs  )[:n_gen]
    labels = np.concatenate(all_labels)[:n_gen]
    gen_imgs = gen_imgs[:n_gen]
    # If no real images available, FID compares generated vs generated
    # (gives lower-bound score; note this in the CSV)
    real_imgs_out = real_imgs_out[:n_gen] if has_real else gen_imgs
    return gen_imgs, real_imgs_out, embs, labels, has_real


def _build_dataset(args):
    if args.dummy:
        n_ch = (config.IMAGENET_CHANNELS if args.dataset == "imagenet"
                else config.N_CHANNELS)
        return DummyEEGImageDataset(n_samples=256, n_classes=10,
                                    n_channels=n_ch)
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
        print(f"Error: EEG file not found: {eeg_p}")
        return None
    if os.path.exists(img_p):
        return EEGImageDataset(eeg_p, lbl_p, img_p)
    print(f"Note: images.npy not found — using EEG-only mode.")
    return _EEGOnlyWrapper(eeg_p, lbl_p)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = _build_dataset(args)
    if ds is None:
        return {}
    n_gen  = min(args.n_gen, len(ds))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    n_ch = config.IMAGENET_CHANNELS if args.dataset == "imagenet" else config.N_CHANNELS
    if args.encoder_type == "transformer":
        encoder = TransformerEEGEncoder(n_channels=n_ch).to(device)
    else:
        encoder = LSTMEEGEncoder(n_channels=n_ch).to(device)

    if args.encoder_ckpt and os.path.isfile(args.encoder_ckpt):
        ckpt = torch.load(args.encoder_ckpt, map_location=device,
                          weights_only=False)
        encoder.load_state_dict(ckpt["encoder_state"])
    else:
        print("Warning: No encoder checkpoint found. Using random weights.")

    netG = Generator().to(device)
    if args.gan_ckpt and os.path.isfile(args.gan_ckpt):
        ckpt = torch.load(args.gan_ckpt, map_location=device, weights_only=False)
        netG.load_state_dict(ckpt["G_state"])
    else:
        print("Warning: No GAN checkpoint found. Using random weights.")

    print(f"Generating {n_gen} images...")
    gen_imgs, real_imgs, embs, labels, has_real = generate_images(
        encoder, netG, loader, device, n_gen)

    print("K-Means accuracy...")
    km_acc = kmeans_accuracy(embs, labels)

    print("Inception Score...")
    is_calc        = InceptionScoreCalculator(device=device)
    is_mean, is_std = is_calc.compute(gen_imgs)

    eisc_score = None
    if not args.no_eisc and has_real:
        print("EISC...")
        eisc_score = EISCCalculator(device=device).compute(gen_imgs, real_imgs)
    elif not has_real:
        print("Skipping EISC (no real images available).")

    print("FID...")
    fid_score = FIDCalculator(device=device).compute(gen_imgs, real_imgs)

    eisc_str = f", EISC={eisc_score:.4f}" if eisc_score is not None else ""
    print(f"\n── Results ──────────────────────────────────────────────")
    print(f"  IS  = {is_mean:.4f} ± {is_std:.4f}")
    print(f"  FID = {fid_score:.4f}")
    print(f"  K-Means Acc = {km_acc:.4f}")
    if eisc_score is not None:
        print(f"  EISC = {eisc_score:.4f}")
    print(f"─────────────────────────────────────────────────────────")

    if args.output_csv:
        row = {
            "method"  : args.encoder_type,
            "dataset" : args.dataset,
            "IS_mean" : f"{is_mean:.4f}",
            "IS_std"  : f"{is_std:.4f}",
            "FID"     : f"{fid_score:.4f}",
            "kmeans"  : f"{km_acc:.4f}",
            "EISC"    : f"{eisc_score:.4f}" if eisc_score is not None else "N/A",
            "real_imgs": "yes" if has_real else "no",
        }
        fieldnames = list(row.keys())
        write_hdr  = not os.path.isfile(args.output_csv)
        with open(args.output_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_hdr:
                w.writeheader()
            w.writerow(row)
        print(f"Results appended to {args.output_csv}")

    return {"IS": is_mean, "IS_std": is_std, "FID": fid_score,
            "kmeans": km_acc, "EISC": eisc_score}


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
