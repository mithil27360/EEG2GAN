import os
import sys
import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import config
from dataset import get_eeg_loaders, DummyEEGDataset
from models.encoder import TransformerEEGEncoder, LSTMEEGEncoder
from utils.triplet_loss import SupConLoss
from utils.metrics import kmeans_accuracy

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder",  choices=["transformer", "lstm"], default="transformer")
    p.add_argument("--dataset",  choices=["objects", "characters", "mindbigdata", "imagenet"], default="objects")
    p.add_argument("--dummy",    action="store_true")
    p.add_argument("--n_layers", type=int,   default=config.N_LAYERS)
    p.add_argument("--n_heads",  type=int,   default=config.N_HEADS)
    p.add_argument("--pooling",  choices=["mean", "cls"], default="mean")
    p.add_argument("--embed_dim",type=int,   default=config.EMBED_DIM)
    p.add_argument("--dropout",  type=float, default=config.DROPOUT)
    p.add_argument("--margin",   type=float, default=config.MARGIN)
    p.add_argument("--lr",       type=float, default=config.ENC_LR)
    p.add_argument("--epochs",   type=int,   default=config.ENC_EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.ENC_BATCH_SIZE)
    p.add_argument("--patience", type=int,   default=config.ENC_PATIENCE)
    p.add_argument("--tag",      type=str,   default="")
    p.add_argument("--resume",   type=str,   default="")
    p.add_argument("--no_diffaug", action="store_true")
    return p.parse_args()

@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    encoder.eval()
    embs, lbls = [], []
    for batch in loader:
        eeg, label = batch[0], batch[1]
        eeg = eeg.to(device)
        feat = encoder(eeg)
        # Encoder already L2-normalizes output, but re-normalize to be safe
        feat = F.normalize(feat, p=2, dim=1)
        embs.append(feat.cpu().numpy())
        lbls.append(label.numpy())
    return np.concatenate(embs), np.concatenate(lbls)

def train(args):
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dummy:
        from torch.utils.data import DataLoader, random_split
        full_ds = DummyEEGDataset(n_samples=230, n_classes=10)
        train_ds, val_ds = random_split(full_ds, [184, 46],
                                        generator=torch.Generator().manual_seed(config.SEED))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    else:
        eeg_map = {
            "objects"    : (config.THOUGHTVIZ_EEG_OBJECTS,   config.THOUGHTVIZ_LABELS_OBJECTS),
            "characters" : (config.THOUGHTVIZ_EEG_CHARS,     config.THOUGHTVIZ_LABELS_CHARS),
            "mindbigdata": (config.MINDBIGDATA_EEG,          config.MINDBIGDATA_LABELS),
            "imagenet"   : (config.MINDBIGDATA_IMAGENET_EEG, config.MINDBIGDATA_IMAGENET_LABELS),
        }
        eeg_path, label_path = eeg_map[args.dataset]
        if not os.path.exists(eeg_path):
            print(f"Error: EEG file {eeg_path} not found. Did you run the preprocessing script?")
            sys.exit(1)

        from dataset import EEGTransform
        transform = EEGTransform(
            noise_std=config.EEG_AUG_NOISE_STD,
            shift_max=config.EEG_AUG_SHIFT_MAX,
            mask_len=config.EEG_AUG_MASK_LEN
        )
        train_loader, val_loader = get_eeg_loaders(
            eeg_path, label_path,
            batch_size=args.batch_size,
            transform=transform
        )

    n_channels = 5 if args.dataset == "imagenet" else 14
    n_classes_map = {
        "objects":     config.N_CLASSES_OBJECTS,
        "characters":  config.N_CLASSES_CHARS,
        "mindbigdata": 10,
        "imagenet":    None,   # determined from data below
    }
    n_classes = n_classes_map.get(args.dataset, 10)

    # For imagenet, count unique classes from the saved label file
    if n_classes is None:
        import numpy as _np
        _lbl_path = eeg_map[args.dataset][1] if not args.dummy else None
        if _lbl_path and os.path.exists(_lbl_path):
            _lbls = _np.load(_lbl_path)
            n_classes = int(_lbls.max()) + 1  # IDs are 0-indexed sequential
            print(f"Detected {n_classes} unique classes from labels.npy")
        else:
            n_classes = 200   # safe fallback larger than any expected count

    if args.encoder == "transformer":
        encoder = TransformerEEGEncoder(
            n_channels = n_channels,
            seq_len   = config.EEG_WINDOW_SIZE,
            embed_dim = args.embed_dim,
            n_heads   = args.n_heads,
            n_layers  = args.n_layers,
            ff_dim    = config.FF_DIM,
            dropout   = args.dropout,
            out_dim   = config.OUT_DIM,
            pooling   = args.pooling,
        ).to(device)
    else:
        encoder = LSTMEEGEncoder(n_channels=n_channels).to(device)

    # Lightweight classification head — provides strong auxiliary gradient signal.
    # Discarded after training; only the encoder weights are saved.
    ce_head = nn.Linear(config.OUT_DIM, n_classes).to(device)
    # Scale down init weights so logits start small — prevents log_softmax underflow on NaN
    with torch.no_grad():
        ce_head.weight.mul_(0.1)
        ce_head.bias.zero_()

    supcon_loss_fn = SupConLoss(temperature=0.1)  # 0.07 too sharp for 500+ class noisy EEG
    ce_loss_fn     = nn.CrossEntropyLoss(label_smoothing=0.1)  # smoothing prevents -inf logits

    all_params = list(encoder.parameters()) + list(ce_head.parameters())
    optimizer  = optim.AdamW(all_params, lr=args.lr, weight_decay=config.ENC_WEIGHT_DECAY)
    scheduler  = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch  = 0
    best_kmeans  = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_kmeans = ckpt.get("best_kmeans", 0.0)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    tag_str   = f"_{args.tag}" if args.tag else ""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"encoder_{args.encoder}_{args.dataset}{tag_str}.pth")

    patience_ctr = 0
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        encoder.train()
        ce_head.train()
        epoch_loss = 0.0
        n_batches  = 0
        for batch in train_loader:
            eeg, labels = batch[0], batch[1]
            eeg, labels = eeg.to(device), labels.to(device)
            optimizer.zero_grad()

            feat = encoder(eeg)              # L2-normalized (B, OUT_DIM)
            logits = ce_head(feat)           # (B, n_classes)

            loss_supcon = supcon_loss_fn(feat, labels)
            loss_ce     = ce_loss_fn(logits, labels)
            loss = 0.7 * loss_supcon + 0.3 * loss_ce

            # Guard against NaN (can happen early in training with noisy EEG)
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}", end="")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            embs, lbls = extract_embeddings(encoder, val_loader, device)
            val_km = kmeans_accuracy(embs, lbls)
            print(f" - Val K-Means: {val_km:.4f}", end="")
            if val_km > best_kmeans:
                best_kmeans = val_km
                patience_ctr = 0
                torch.save({
                    "epoch": epoch + 1,
                    "encoder_state": encoder.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_kmeans": best_kmeans,
                    "args": vars(args),
                }, ckpt_path)
            else:
                patience_ctr += 1
            if patience_ctr >= (args.patience // 10):
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        print("", flush=True)

    embs, lbls = extract_embeddings(encoder, val_loader, device)
    np.save(os.path.join(config.CHECKPOINT_DIR, f"embeddings_{args.encoder}_{args.dataset}{tag_str}.npy"), embs)
    np.save(os.path.join(config.CHECKPOINT_DIR, f"labels_{args.dataset}.npy"), lbls)
    print(f"Training finished. Best Val K-Means: {best_kmeans:.4f}")
    return ckpt_path, best_kmeans

if __name__ == "__main__":
    args = parse_args()
    train(args)
