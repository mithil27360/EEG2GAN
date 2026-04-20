import os
import json
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import config
from dataset import get_eeg_image_loaders, DummyEEGImageDataset
from models.encoder import TransformerEEGEncoder, LSTMEEGEncoder
from models.gan import Generator, Discriminator, weights_init, hinge_loss_g, hinge_loss_d, mode_seeking_loss
from utils.diffaugment import DiffAugment


class EEGImageOnTheFlyDataset(Dataset):
    """
    Loads EEG + images on-the-fly from disk when images.npy is absent (e.g. ImageNet).
    Uses synset metadata JSON and the original image_dir to fetch images at runtime.
    Falls back to a random crop of a solid-colour image if a file is missing.
    """
    def __init__(self, eeg_path, label_path, metadata_path, image_dir):
        self.eeg    = np.load(eeg_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.int64)

        # Fix shape: ensure (N, C, T)
        if self.eeg.ndim == 3 and self.eeg.shape[1] == config.SEQ_LEN:
            self.eeg = self.eeg.transpose(0, 2, 1)

        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        # id_to_synset: reverse of synset_to_id
        self.id_to_synset = {v: k for k, v in meta["synset_to_id"].items()}
        self.image_dir    = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def _load_image(self, label_id):
        synset = self.id_to_synset.get(int(label_id))
        if synset and self.image_dir:
            synset_dir = os.path.join(self.image_dir, synset)
            if os.path.isdir(synset_dir):
                files = [f for f in os.listdir(synset_dir)
                         if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
                if files:
                    path = os.path.join(synset_dir, random.choice(files))
                    try:
                        return Image.open(path).convert('RGB')
                    except Exception:
                        pass
        # Fallback: random noise image
        arr = np.random.randint(0, 256, (config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg   = torch.from_numpy(self.eeg[idx])
        if config.EEG_NORMALIZE:
            eeg = (eeg - eeg.mean(dim=-1, keepdim=True)) / (eeg.std(dim=-1, keepdim=True) + 1e-6)
        img   = self._load_image(self.labels[idx])
        img   = self.transform(img)
        label = torch.tensor(self.labels[idx])
        return eeg, img, label

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_ckpt", type=str, required=True)
    p.add_argument("--encoder_type", choices=["transformer", "lstm"], default="transformer")
    p.add_argument("--dataset",      choices=["objects", "characters", "mindbigdata", "imagenet"], default="objects")
    p.add_argument("--dummy",        action="store_true")
    p.add_argument("--batch_size",   type=int,   default=config.GAN_BATCH_SIZE)
    p.add_argument("--epochs",       type=int,   default=config.GAN_EPOCHS)
    p.add_argument("--lr_g",         type=float, default=config.GAN_LR_G)
    p.add_argument("--lr_d",         type=float, default=config.GAN_LR_D)
    p.add_argument("--tag",          type=str,   default="")
    p.add_argument("--no_diffaug",   action="store_true")
    return p.parse_args()

def r1_gradient_penalty(netD, real_imgs, eeg_feat, gamma=10.0):
    """R1 gradient penalty for training stability."""
    real_imgs.requires_grad_(True)
    real_scores = netD(real_imgs, eeg_feat)
    grads = torch.autograd.grad(
        outputs=real_scores.sum(),
        inputs=real_imgs,
        create_graph=True,
        retain_graph=True,
    )[0]
    penalty = grads.pow(2).view(grads.size(0), -1).sum(1).mean()
    return gamma / 2 * penalty

def train(args):
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dummy:
        train_ds = DummyEEGImageDataset(n_samples=230, n_classes=10)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    else:
        ds_map = {
            "objects"    : (config.THOUGHTVIZ_EEG_OBJECTS, config.THOUGHTVIZ_LABELS_OBJECTS, config.THOUGHTVIZ_IMAGES_OBJECTS),
            "characters" : (config.THOUGHTVIZ_EEG_CHARS,   config.THOUGHTVIZ_LABELS_CHARS,   config.THOUGHTVIZ_IMAGES_CHARS),
            "mindbigdata": (config.MINDBIGDATA_EEG, config.MINDBIGDATA_LABELS, config.MINDBIGDATA_IMAGES),
            "imagenet"   : (config.MINDBIGDATA_IMAGENET_EEG, config.MINDBIGDATA_IMAGENET_LABELS, config.MINDBIGDATA_IMAGENET_IMAGES),
        }
        eeg_p, lbl_p, img_p = ds_map[args.dataset]
        if not os.path.exists(eeg_p):
            print(f"Error: EEG file for {args.dataset} not found: {eeg_p}")
            print("Did process_mindbigdata.py complete successfully?")
            sys.exit(1)
        if os.path.exists(img_p):
            train_loader, _ = get_eeg_image_loaders(eeg_p, lbl_p, img_p, batch_size=args.batch_size)
        else:
            # images.npy not saved (typical for ImageNet — would be ~600 MB+).
            # Use on-the-fly loader: reads images from the original ImageNet directory.
            meta_p = img_p.replace("images.npy", "metadata.json")
            if not os.path.exists(meta_p):
                print(f"Error: metadata.json not found at {meta_p}. Cannot use on-the-fly loader.")
                sys.exit(1)
            img_dir = config.IMAGENET_DIR
            if not os.path.isdir(img_dir):
                print(f"Warning: IMAGENET_DIR '{img_dir}' not found. Images will be random noise.")
                img_dir = None
            print(f"Note: images.npy not found — using on-the-fly loader from {img_dir}")
            otf_ds = EEGImageOnTheFlyDataset(eeg_p, lbl_p, meta_p, img_dir)
            train_loader = DataLoader(otf_ds, batch_size=args.batch_size, shuffle=True,
                                      drop_last=True, num_workers=2, pin_memory=True)

    n_channels = 5 if args.dataset == "imagenet" else 14
    if args.encoder_type == "transformer":
        encoder = TransformerEEGEncoder(n_channels=n_channels).to(device)
    else:
        encoder = LSTMEEGEncoder(n_channels=n_channels).to(device)

    if os.path.isfile(args.encoder_ckpt):
        ckpt = torch.load(args.encoder_ckpt, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state"])
    else:
        print(f"Warning: encoder checkpoint not found at {args.encoder_ckpt}. Using random weights.")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # TTUR: slower G, faster D — improves convergence
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.0, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d * 2, betas=(0.0, 0.999))
    schedulerG = CosineAnnealingLR(optimizerG, T_max=args.epochs, eta_min=args.lr_g * 0.1)
    schedulerD = CosineAnnealingLR(optimizerD, T_max=args.epochs, eta_min=args.lr_d * 0.2)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    tag_str     = f"_{args.tag}" if args.tag else ""
    ckpt_path_g = os.path.join(config.CHECKPOINT_DIR, f"gan_{args.encoder_type}_{args.dataset}{tag_str}.pth")

    history = {"G_loss": [], "D_loss": []}
    t0 = time.time()

    for epoch in range(args.epochs):
        netG.train()
        netD.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        n_batches    = 0

        for i, (eeg, real_imgs, _) in enumerate(train_loader):
            B = eeg.size(0)
            eeg, real_imgs = eeg.to(device), real_imgs.to(device)

            with torch.no_grad():
                eeg_feat = encoder(eeg)

            # -------- Discriminator update (2 steps per G step) --------
            for _ in range(2):
                optimizerD.zero_grad()
                real_input = DiffAugment(real_imgs, policy=config.DIFFAUG_POLICY) if not args.no_diffaug else real_imgs
                real_scores = netD(real_input, eeg_feat.detach())

                noise  = torch.randn(B, config.NOISE_DIM, device=device)
                z      = torch.cat([noise, eeg_feat.detach()], dim=1)
                fakes  = netG(z).detach()
                fake_input = DiffAugment(fakes, policy=config.DIFFAUG_POLICY) if not args.no_diffaug else fakes
                fake_scores = netD(fake_input, eeg_feat.detach())

                lossD_hinge = hinge_loss_d(real_scores, fake_scores)
                # R1 penalty every 4 steps for efficiency
                if i % 4 == 0:
                    r1 = r1_gradient_penalty(netD, real_imgs.detach().requires_grad_(True), eeg_feat.detach())
                    lossD = lossD_hinge + r1
                else:
                    lossD = lossD_hinge
                lossD.backward()
                torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
                optimizerD.step()

            # -------- Generator update --------
            optimizerG.zero_grad()
            noise1 = torch.randn(B, config.NOISE_DIM, device=device)
            noise2 = torch.randn(B, config.NOISE_DIM, device=device)
            z1  = torch.cat([noise1, eeg_feat], dim=1)
            z2  = torch.cat([noise2, eeg_feat], dim=1)
            f1  = netG(z1)
            f2  = netG(z2)
            fi1 = DiffAugment(f1, policy=config.DIFFAUG_POLICY) if not args.no_diffaug else f1
            fake_scores = netD(fi1, eeg_feat)
            lossG  = hinge_loss_g(fake_scores)
            lossMS = mode_seeking_loss(f1, f2, z1, z2)
            lossG_total = lossG + config.LAMBDA_MS * lossMS
            lossG_total.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            optimizerG.step()

            epoch_g_loss += lossG_total.item()
            epoch_d_loss += lossD.item()
            n_batches    += 1

        schedulerG.step()
        schedulerD.step()
        avg_g = epoch_g_loss / max(n_batches, 1)
        avg_d = epoch_d_loss / max(n_batches, 1)
        history["G_loss"].append(avg_g)
        history["D_loss"].append(avg_d)
        print(f"Epoch {epoch+1}/{args.epochs} - G_Loss: {avg_g:.4f} - D_Loss: {avg_d:.4f}", flush=True)

    torch.save({
        "epoch": args.epochs,
        "G_state": netG.state_dict(),
        "D_state": netD.state_dict(),
        "optimizerG_state": optimizerG.state_dict(),
        "optimizerD_state": optimizerD.state_dict(),
        "args": vars(args),
    }, ckpt_path_g)
    np.save(os.path.join(config.OUTPUT_DIR, f"G_losses_{args.encoder_type}_{args.dataset}{tag_str}.npy"),
            np.array(history["G_loss"]))
    np.save(os.path.join(config.OUTPUT_DIR, f"D_losses_{args.encoder_type}_{args.dataset}{tag_str}.npy"),
            np.array(history["D_loss"]))

if __name__ == "__main__":
    args = parse_args()
    train(args)
