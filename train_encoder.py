"""train_encoder.py — EEG Encoder Pretraining

Trains a Transformer or LSTM encoder to produce discriminative EEG embeddings
using supervised CrossEntropy with label smoothing.

Key design decisions:
  • Pure CE loss (no SupConLoss) — stable with 500+ class imagenet
  • BALANCED_SAMPLING=False for imagenet — random shuffle covers classes better
  • EEG_NORMALIZE=False — data is pre-Z-scored by process_mindbigdata.py
  • Gradient clipping (max_norm=1.0) for training stability
  • CosineAnnealingLR covers full epoch range
  • Early stopping based on val K-Means accuracy with configurable patience
"""
import os
import sys
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from dataset import get_eeg_loaders, DummyEEGDataset, EEGTransform
from models.encoder import TransformerEEGEncoder, LSTMEEGEncoder
from utils.metrics import kmeans_accuracy


def parse_args():
    p = argparse.ArgumentParser(description="Train EEG encoder")
    p.add_argument("--encoder",    choices=["transformer", "lstm"], default="transformer")
    p.add_argument("--dataset",    choices=["objects","characters","mindbigdata","imagenet"],
                   default="objects")
    p.add_argument("--dummy",      action="store_true",
                   help="Use random data for a quick smoke-test")
    p.add_argument("--n_layers",   type=int,   default=config.N_LAYERS)
    p.add_argument("--n_heads",    type=int,   default=config.N_HEADS)
    p.add_argument("--pooling",    choices=["mean", "cls"], default="mean")
    p.add_argument("--embed_dim",  type=int,   default=config.EMBED_DIM)
    p.add_argument("--dropout",    type=float, default=config.DROPOUT)
    p.add_argument("--lr",         type=float, default=config.ENC_LR)
    p.add_argument("--epochs",     type=int,   default=config.ENC_EPOCHS)
    p.add_argument("--batch_size", type=int,   default=config.ENC_BATCH_SIZE)
    p.add_argument("--patience",   type=int,   default=config.ENC_PATIENCE)
    p.add_argument("--tag",        type=str,   default="")
    p.add_argument("--resume",     type=str,   default="",
                   help="Path to encoder checkpoint to resume from")
    p.add_argument("--no_diffaug", action="store_true")
    return p.parse_args()


@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    """Return (embeddings, labels) numpy arrays over the full loader."""
    encoder.eval()
    embs, lbls = [], []
    for batch in loader:
        eeg, label = batch[0].to(device), batch[1]
        feat = encoder(eeg)
        feat = F.normalize(feat, p=2, dim=1, eps=1e-8)
        embs.append(feat.cpu().numpy())
        lbls.append(label.numpy())
    return np.concatenate(embs), np.concatenate(lbls)


def _build_loaders(args):
    """Return (train_loader, val_loader, n_channels, n_classes)."""
    EEG_MAP = {
        "objects"    : (config.THOUGHTVIZ_EEG_OBJECTS,   config.THOUGHTVIZ_LABELS_OBJECTS),
        "characters" : (config.THOUGHTVIZ_EEG_CHARS,     config.THOUGHTVIZ_LABELS_CHARS),
        "mindbigdata": (config.MINDBIGDATA_EEG,          config.MINDBIGDATA_LABELS),
        "imagenet"   : (config.MINDBIGDATA_IMAGENET_EEG, config.MINDBIGDATA_IMAGENET_LABELS),
    }
    N_CLASSES_MAP = {
        "objects":     config.N_CLASSES_OBJECTS,
        "characters":  config.N_CLASSES_CHARS,
        "mindbigdata": 10,
        "imagenet":    None,   # detected from labels.npy at runtime
    }
    N_CHANNELS_MAP = {"imagenet": config.IMAGENET_CHANNELS}

    if args.dummy:
        n_channels = N_CHANNELS_MAP.get(args.dataset, config.N_CHANNELS)
        n_classes  = N_CLASSES_MAP.get(args.dataset) or 10
        full_ds    = DummyEEGDataset(n_samples=256, n_classes=n_classes,
                                     n_channels=n_channels)
        from torch.utils.data import DataLoader, random_split
        n_val  = max(1, len(full_ds) // 5)
        n_train = len(full_ds) - n_val
        tr, vl = random_split(full_ds, [n_train, n_val],
                              generator=torch.Generator().manual_seed(config.SEED))
        tr_ld = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
        vl_ld = DataLoader(vl, batch_size=args.batch_size, shuffle=False)
        return tr_ld, vl_ld, n_channels, n_classes

    eeg_path, label_path = EEG_MAP[args.dataset]
    if not os.path.exists(eeg_path):
        print(f"Error: EEG file not found: {eeg_path}\n"
              f"       Run process_mindbigdata.py first.")
        sys.exit(1)

    n_channels = N_CHANNELS_MAP.get(args.dataset, config.N_CHANNELS)
    n_classes  = N_CLASSES_MAP.get(args.dataset)
    if n_classes is None:
        lbls      = np.load(label_path)
        n_classes = int(np.unique(lbls).size)
        print(f"Detected {n_classes} unique classes from {label_path}")

    transform = EEGTransform(
        noise_std = config.EEG_AUG_NOISE_STD,
        shift_max = config.EEG_AUG_SHIFT_MAX,
        mask_len  = config.EEG_AUG_MASK_LEN,
    )
    tr_ld, vl_ld = get_eeg_loaders(
        eeg_path, label_path,
        batch_size = args.batch_size,
        transform  = transform,
    )
    return tr_ld, vl_ld, n_channels, n_classes


def train(args):
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, n_channels, n_classes = _build_loaders(args)

    # ── Encoder ────────────────────────────────────────────────────────────────
    if args.encoder == "transformer":
        encoder = TransformerEEGEncoder(
            n_channels = n_channels,
            seq_len    = config.EEG_WINDOW_SIZE,
            embed_dim  = args.embed_dim,
            n_heads    = args.n_heads,
            n_layers   = args.n_layers,
            ff_dim     = config.FF_DIM,
            dropout    = args.dropout,
            out_dim    = config.OUT_DIM,
            pooling    = args.pooling,
        ).to(device)
    else:
        encoder = LSTMEEGEncoder(
            n_channels = n_channels,
            hidden_dim = args.embed_dim,
            out_dim    = config.OUT_DIM,
        ).to(device)

    # ── CE classification head (discarded after training) ──────────────────────
    ce_head = nn.Linear(config.OUT_DIM, n_classes).to(device)
    with torch.no_grad():
        # Scale down initial weights so logits start small (prevents early overflow)
        ce_head.weight.mul_(0.1)
        ce_head.bias.zero_()

    # Pure CrossEntropy with label smoothing — numerically stable at any class count.
    # label_smoothing=0.1 prevents log(0) from one-hot targets.
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    all_params = list(encoder.parameters()) + list(ce_head.parameters())
    optimizer  = optim.AdamW(all_params, lr=args.lr,
                             weight_decay=config.ENC_WEIGHT_DECAY)
    scheduler  = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Optional resume ────────────────────────────────────────────────────────
    start_epoch = 0
    best_kmeans = 0.0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    tag_str   = f"_{args.tag}" if args.tag else ""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR,
                             f"encoder_{args.encoder}_{args.dataset}{tag_str}.pth")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_kmeans = ckpt.get("best_kmeans", 0.0)
        print(f"Resumed from epoch {start_epoch}, best K-Means={best_kmeans:.4f}")

    patience_ctr = 0
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        encoder.train()
        ce_head.train()
        epoch_loss = 0.0
        n_batches  = 0

        for i, (eeg, labels) in enumerate(train_loader):
            eeg    = eeg.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            feat   = encoder(eeg)           # (B, OUT_DIM) L2-normalised
            logits = ce_head(feat)          # (B, n_classes)
            loss   = ce_loss_fn(logits, labels)

            # First-batch diagnostic on first epoch only
            if epoch == start_epoch and i == 0:
                nan_feat = torch.isnan(feat).any().item()
                nan_loss = torch.isnan(loss).item()
                print(f"  [diag] feat range [{feat.min():.3f}, {feat.max():.3f}] "
                      f"nan={nan_feat} | loss={loss.item():.4f} nan={nan_loss}")

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss at epoch {epoch+1} "
                      f"batch {i} ({loss.item()}) — skipping")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed  = time.time() - t0
        print(f"Epoch {epoch+1:4d}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"t={elapsed:.0f}s", end="")

        # ── Validation every 10 epochs (and epoch 1) ─────────────────────────
        if (epoch + 1) % 10 == 0 or epoch == start_epoch:
            embs, lbls = extract_embeddings(encoder, val_loader, device)
            val_km = kmeans_accuracy(embs, lbls)
            print(f" | Val K-Means: {val_km:.4f}", end="")

            if val_km > best_kmeans + 1e-4:   # genuine improvement only
                best_kmeans  = val_km
                patience_ctr = 0
                torch.save({
                    "epoch"         : epoch + 1,
                    "encoder_state" : encoder.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_kmeans"   : best_kmeans,
                    "args"          : vars(args),
                }, ckpt_path)
            else:
                patience_ctr += 1

            if patience_ctr >= (args.patience // 10):
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(patience_ctr={patience_ctr})")
                break

        print("", flush=True)

    # ── Save final embeddings for downstream use ────────────────────────────────
    embs, lbls = extract_embeddings(encoder, val_loader, device)
    emb_path   = os.path.join(config.CHECKPOINT_DIR,
                              f"embeddings_{args.encoder}_{args.dataset}{tag_str}.npy")
    lbl_path   = os.path.join(config.CHECKPOINT_DIR,
                              f"labels_{args.dataset}.npy")
    np.save(emb_path, embs)
    np.save(lbl_path, lbls)
    print(f"\nTraining finished. Best Val K-Means: {best_kmeans:.4f}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Embeddings : {emb_path}")
    return ckpt_path, best_kmeans


if __name__ == "__main__":
    args = parse_args()
    train(args)
