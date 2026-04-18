import os
import sys
import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import config
from dataset import get_eeg_loaders, DummyEEGDataset
from models.encoder import TransformerEEGEncoder, LSTMEEGEncoder
from utils.triplet_loss import batch_semi_hard_triplet_loss
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
    return p.parse_args()

@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    encoder.eval()
    embs, lbls = [], []
    for eeg, label in loader:
        eeg = eeg.to(device)
        feat = encoder(eeg)
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
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    else:
        eeg_map = {
            "objects"    : (config.THOUGHTVIZ_EEG_OBJECTS,   config.THOUGHTVIZ_LABELS_OBJECTS),
            "characters" : (config.THOUGHTVIZ_EEG_CHARS,     config.THOUGHTVIZ_LABELS_CHARS),
            "mindbigdata": (config.MINDBIGDATA_EEG,          config.MINDBIGDATA_LABELS),
            "imagenet"   : (config.MINDBIGDATA_IMAGENET_EEG, config.MINDBIGDATA_IMAGENET_LABELS),
        }
        eeg_path, label_path = eeg_map[args.dataset]
        train_loader, val_loader = get_eeg_loaders(eeg_path, label_path, batch_size=args.batch_size)

    n_channels = 5 if args.dataset == "imagenet" else 14
    if args.encoder == "transformer":
        encoder = TransformerEEGEncoder(
            n_channels = n_channels,
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

    optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=config.ENC_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch  = 0
    best_kmeans  = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_kmeans = ckpt.get("best_kmeans", 0.0)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    tag_str  = f"_{args.tag}" if args.tag else ""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"encoder_{args.encoder}_{args.dataset}{tag_str}.pth")

    patience_ctr = 0
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        encoder.train()
        epoch_loss = 0.0
        n_batches  = 0
        for eeg, labels in train_loader:
            eeg, labels = eeg.to(device), labels.to(device)
            optimizer.zero_grad()
            feat = encoder(eeg)
            loss = batch_semi_hard_triplet_loss(feat, labels, margin=args.margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            embs, lbls = extract_embeddings(encoder, val_loader, device)
            val_km = kmeans_accuracy(embs, lbls, n_clusters=len(np.unique(lbls)))
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
                break

    embs, lbls = extract_embeddings(encoder, val_loader, device)
    np.save(os.path.join(config.CHECKPOINT_DIR, f"embeddings_{args.encoder}_{args.dataset}{tag_str}.npy"), embs)
    np.save(os.path.join(config.CHECKPOINT_DIR, f"labels_{args.dataset}.npy"), lbls)
    return ckpt_path, best_kmeans

if __name__ == "__main__":
    args = parse_args()
    train(args)
