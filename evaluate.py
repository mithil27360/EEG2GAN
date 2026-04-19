import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import config
from models.encoder import TransformerEEGEncoder, LSTMEEGEncoder
from models.gan import Generator
from utils.metrics import InceptionScoreCalculator, kmeans_accuracy, EISCCalculator, FIDCalculator, tensor_to_pil_list
from dataset import DummyEEGImageDataset, EEGImageDataset, EEGDataset

class _EEGOnlyWrapper(Dataset):
    """
    Wraps EEGDataset to return (eeg, dummy_img, label) tuples,
    used when images.npy is not available (e.g. large ImageNet sets).
    """
    def __init__(self, eeg_path, label_path):
        self.ds = EEGDataset(eeg_path, label_path,
                             window_size=config.EEG_WINDOW_SIZE,
                             stride=config.EEG_WINDOW_SIZE)  # no stride overlap for eval
        # Dummy black image placeholder
        self._dummy = torch.zeros(config.NC, config.IMAGE_SIZE, config.IMAGE_SIZE)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        eeg, label = self.ds[idx]
        return eeg, self._dummy, label

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_ckpt", type=str, default="")
    p.add_argument("--gan_ckpt",     type=str, default="")
    p.add_argument("--encoder_type", choices=["transformer", "lstm"], default="transformer")
    p.add_argument("--dataset",      choices=["objects", "characters", "mindbigdata", "imagenet"], default="objects")
    p.add_argument("--dummy",        action="store_true")
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--n_gen",        type=int, default=getattr(config, "IS_N_SAMPLES", 2048))
    p.add_argument("--no_eisc",      action="store_true")
    p.add_argument("--output_csv",   type=str, default="")
    return p.parse_args()

@torch.no_grad()
def generate_images(encoder, netG, dataloader, device, n_gen=2048):
    encoder.eval()
    netG.eval()
    gen_images  = []
    real_images = []
    all_embs    = []
    all_labels  = []
    has_real    = True   # track if real images are meaningful

    for eeg, real_img, label in dataloader:
        if len(gen_images) >= n_gen:
            break
        eeg      = eeg.to(device)
        eeg_feat = encoder(eeg)
        noise    = torch.randn(eeg.size(0), config.NOISE_DIM, device=device)
        z        = torch.cat([noise, eeg_feat], dim=1)
        fake     = netG(z)
        gen_images.extend(tensor_to_pil_list(fake))

        # Detect if real_img is meaningful (not a dummy black tensor)
        if real_img.abs().sum() > 0:
            real_images.extend(tensor_to_pil_list(real_img))
        else:
            has_real = False

        all_embs.append(eeg_feat.cpu().numpy())
        all_labels.append(label.numpy())

    all_embs   = np.concatenate(all_embs)[:n_gen]
    all_labels = np.concatenate(all_labels)[:n_gen]
    gen_images  = gen_images[:n_gen]
    real_images = real_images[:n_gen] if has_real else gen_images  # FID vs self if no real
    return gen_images, real_images, all_embs, all_labels, has_real

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dummy:
        ds = DummyEEGImageDataset(n_samples=230, n_classes=10)
    else:
        ds_map = {
            "objects"    : (config.THOUGHTVIZ_EEG_OBJECTS,   config.THOUGHTVIZ_LABELS_OBJECTS,   config.THOUGHTVIZ_IMAGES_OBJECTS),
            "characters" : (config.THOUGHTVIZ_EEG_CHARS,     config.THOUGHTVIZ_LABELS_CHARS,     config.THOUGHTVIZ_IMAGES_CHARS),
            "mindbigdata": (config.MINDBIGDATA_EEG,          config.MINDBIGDATA_LABELS,          config.MINDBIGDATA_IMAGES),
            "imagenet"   : (config.MINDBIGDATA_IMAGENET_EEG, config.MINDBIGDATA_IMAGENET_LABELS, config.MINDBIGDATA_IMAGENET_IMAGES),
        }
        eeg_p, lbl_p, img_p = ds_map[args.dataset]

        # Verify EEG file exists
        if not os.path.exists(eeg_p):
            print(f"Error: EEG file not found: {eeg_p}")
            print("Did process_mindbigdata.py complete successfully?")
            return {}

        if os.path.exists(img_p):
            ds = EEGImageDataset(eeg_p, lbl_p, img_p)
        else:
            # images.npy not saved (common for ImageNet — file would be multi-GB).
            # Fall back to EEG-only; FID/EISC will compare two sets of generated images.
            print(f"Note: images.npy not found ({img_p}). Using EEG-only mode for evaluation.")
            ds = _EEGOnlyWrapper(eeg_p, lbl_p)

    # Cap n_gen to dataset size
    n_gen = min(args.n_gen, len(ds))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    n_channels = 5 if args.dataset == "imagenet" else 14
    if args.encoder_type == "transformer":
        encoder = TransformerEEGEncoder(n_channels=n_channels).to(device)
    else:
        encoder = LSTMEEGEncoder(n_channels=n_channels).to(device)
    if args.encoder_ckpt and os.path.isfile(args.encoder_ckpt):
        ckpt = torch.load(args.encoder_ckpt, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state"])
    else:
        print("Warning: No encoder checkpoint found. Using random weights.")

    netG = Generator().to(device)
    if args.gan_ckpt and os.path.isfile(args.gan_ckpt):
        ckpt = torch.load(args.gan_ckpt, map_location=device)
        netG.load_state_dict(ckpt["G_state"])
    else:
        print("Warning: No GAN checkpoint found. Using random weights.")

    print(f"Generating {n_gen} images for evaluation...")
    gen_imgs, real_imgs, embs, labels, has_real = generate_images(
        encoder, netG, loader, device, n_gen
    )

    print("Calculating K-Means accuracy...")
    km_acc = kmeans_accuracy(embs, labels)

    print("Calculating Inception Score...")
    is_calc = InceptionScoreCalculator(device=device)
    is_mean, is_std = is_calc.compute(gen_imgs)

    eisc_score = None
    if not args.no_eisc and has_real:
        print("Calculating EISC...")
        eisc_calc = EISCCalculator(device=device)
        eisc_score = eisc_calc.compute(gen_imgs, real_imgs)
    elif not has_real:
        print("Skipping EISC (no real images available).")

    print("Calculating FID...")
    fid_calc  = FIDCalculator(device=device)
    fid_score = fid_calc.compute(gen_imgs, real_imgs)

    eisc_str = f", EISC={eisc_score:.4f}" if eisc_score is not None else ""
    print(f"Results: IS={is_mean:.4f} ±{is_std:.4f}, KM={km_acc:.4f}, FID={fid_score:.4f}{eisc_str}")

    if args.output_csv:
        import csv
        fieldnames = ["method", "dataset", "IS_mean", "IS_std", "kmeans", "EISC", "FID"]
        row = {
            "method" : args.encoder_type,
            "dataset": args.dataset,
            "IS_mean": f"{is_mean:.4f}",
            "IS_std" : f"{is_std:.4f}",
            "kmeans" : f"{km_acc:.4f}",
            "EISC"   : f"{eisc_score:.4f}" if eisc_score is not None else "N/A",
            "FID"    : f"{fid_score:.4f}",
        }
        write_header = not os.path.isfile(args.output_csv)
        with open(args.output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    return {"IS": is_mean, "IS_std": is_std, "kmeans": km_acc, "EISC": eisc_score, "FID": fid_score}

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
