import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import config


# ── EEG augmentation ──────────────────────────────────────────────────────────
class EEGTransform:
    """Lightweight on-the-fly augmentation for EEG signals.

    All ops are applied randomly with moderate strength so evaluation
    (which does NOT use this transform) sees the clean signal.
    """
    def __init__(self, noise_std=0.0, shift_max=0, mask_len=0):
        self.noise_std = noise_std
        self.shift_max = shift_max
        self.mask_len  = mask_len

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Gaussian noise
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        # Circular time-shift
        if self.shift_max > 0:
            shift = random.randint(-self.shift_max, self.shift_max)
            x = torch.roll(x, shifts=shift, dims=-1)
        # Temporal masking (like SpecAugment for EEG)
        if self.mask_len > 0:
            t_len = x.shape[-1]
            start = random.randint(0, max(0, t_len - self.mask_len))
            x = x.clone()                          # avoid in-place on shared storage
            x[:, start : start + self.mask_len] = 0.0
        return x


# ── Shape correction helper ───────────────────────────────────────────────────
def _fix_eeg_shape(eeg: np.ndarray) -> np.ndarray:
    """Ensure EEG array is (N, Channels, Time)."""
    if eeg.size == 0:
        return eeg
    if eeg.ndim == 2:
        eeg = eeg[np.newaxis]                      # (C, T) → (1, C, T)
    if eeg.ndim == 3:
        C, T_cfg = eeg.shape[1], config.SEQ_LEN
        if C == T_cfg and eeg.shape[2] != T_cfg:   # (N, T, C) → (N, C, T)
            eeg = eeg.transpose(0, 2, 1)
    return eeg


# ── Base EEG dataset ──────────────────────────────────────────────────────────
class EEGDataset(Dataset):
    """Loads pre-processed EEG signals and integer class labels.

    NOTE: process_mindbigdata.py already Z-scores every channel before
    saving to disk (config.EEG_NORMALIZE=False by default).  Applying
    normalization again here collapses inter-sample variance and makes
    the encoder see near-identical inputs regardless of class.
    """
    def __init__(self, eeg_path, label_path, transform=None,
                 window_size=None, stride=None):
        self.eeg    = np.load(eeg_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.int64)
        self.eeg    = _fix_eeg_shape(self.eeg)
        self.transform   = transform
        self.window_size = window_size or self.eeg.shape[-1]
        self.stride      = stride or self.window_size
        n_time = self.eeg.shape[-1]
        self.num_windows = max(1, (n_time - self.window_size) // self.stride + 1)

    def __len__(self):
        return len(self.labels) * self.num_windows

    def __getitem__(self, idx):
        s_idx = idx // self.num_windows
        w_idx = idx  % self.num_windows
        eeg   = self.eeg[s_idx]
        start = w_idx * self.stride
        eeg   = eeg[:, start : start + self.window_size]

        # Optional in-loader normalisation (off by default — data is pre-normalised)
        if config.EEG_NORMALIZE:
            m = eeg.mean(axis=-1, keepdims=True)
            s = eeg.std( axis=-1, keepdims=True) + 1e-6
            eeg = (eeg - m) / s

        eeg = torch.from_numpy(eeg.copy())
        if self.transform:
            eeg = self.transform(eeg)
        return eeg, torch.tensor(int(self.labels[s_idx]), dtype=torch.long)


# ── Balanced batch sampler ────────────────────────────────────────────────────
class BalancedBatchSampler(torch.utils.data.Sampler):
    """Ensures each batch contains exactly n_per_class samples from each
    of n_classes_per_batch randomly chosen classes.

    Only useful for small class counts (≤ ~40).  For imagenet (500+
    classes) the default random shuffle is better (BALANCED_SAMPLING=False).
    """
    def __init__(self, labels, n_per_class, batch_size):
        self.labels             = np.array(labels)
        self.n_per_class        = n_per_class
        self.batch_size         = batch_size
        self.n_classes_per_batch = max(1, batch_size // n_per_class)
        self.label_to_indices   = {
            lbl: np.where(self.labels == lbl)[0]
            for lbl in np.unique(self.labels)
        }
        self.labels_list = list(self.label_to_indices.keys())

    def __iter__(self):
        n_batches = len(self.labels) // self.batch_size
        can_replace = len(self.labels_list) < self.n_classes_per_batch
        for _ in range(n_batches):
            classes = np.random.choice(self.labels_list,
                                       self.n_classes_per_batch,
                                       replace=can_replace)
            indices = []
            for lbl in classes:
                pool = self.label_to_indices[lbl]
                indices.extend(np.random.choice(pool, self.n_per_class, replace=True))
            np.random.shuffle(indices)
            yield indices[:self.batch_size]

    def __len__(self):
        return len(self.labels) // self.batch_size


# ── EEG dataloader factory ────────────────────────────────────────────────────
def get_eeg_loaders(eeg_path, label_path, batch_size,
                    val_split=0.2, seed=999, transform=None):
    full_ds = EEGDataset(eeg_path, label_path, transform=transform,
                         window_size=config.EEG_WINDOW_SIZE,
                         stride=config.EEG_WINDOW_STRIDE)
    n_val   = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    gen     = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    if config.BALANCED_SAMPLING:
        # Labels for training subset (index into full_ds.labels via subset indices)
        train_labels = [
            int(full_ds.labels[full_ds_idx // full_ds.num_windows])
            for full_ds_idx in train_ds.indices
        ]
        sampler      = BalancedBatchSampler(train_labels,
                                            config.SAMPLES_PER_CLASS,
                                            batch_size)
        train_loader = DataLoader(train_ds, batch_sampler=sampler,
                                  num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, drop_last=True,
                                  num_workers=2, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ── EEG + Image dataset (for GAN training with pre-saved images.npy) ─────────
class EEGImageDataset(Dataset):
    def __init__(self, eeg_path, label_path, image_path, transform=None):
        self.eeg    = np.load(eeg_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.int64)
        self.images = np.load(image_path)
        self.eeg    = _fix_eeg_shape(self.eeg)
        # Ensure images are (N, C, H, W)
        if self.images.ndim == 4 and self.images.shape[-1] == 3:
            self.images = self.images.transpose(0, 3, 1, 2)
        self.transform = transform or transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = torch.from_numpy(self.eeg[idx].copy())
        if config.EEG_NORMALIZE:
            eeg = (eeg - eeg.mean(-1, keepdim=True)) / (eeg.std(-1, keepdim=True) + 1e-6)
        lbl = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        img = torch.from_numpy(self.images[idx]).float()
        img = img / 127.5 - 1.0   # [0,255] → [-1,1]
        return eeg, img, lbl


# ── On-the-fly image loader (for GAN training without images.npy) ─────────────
class EEGImageOnTheFlyDataset(Dataset):
    """Loads ImageNet JPEG images directly from disk during training.

    Avoids creating a huge images.npy file that would exhaust Kaggle storage.
    Falls back to random noise if an image path is missing.
    """
    IMG_TRANSFORM = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def __init__(self, eeg_path, label_path, meta_path, imagenet_root):
        import json
        self.eeg    = np.load(eeg_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.int64)
        self.eeg    = _fix_eeg_shape(self.eeg)
        self.imagenet_root = imagenet_root

        # metadata.json maps index → original filename stem (e.g. "n01484850_xxx")
        self.filenames = []
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.filenames = meta.get("filenames", [])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = torch.from_numpy(self.eeg[idx].copy())
        if config.EEG_NORMALIZE:
            eeg = (eeg - eeg.mean(-1, keepdim=True)) / (eeg.std(-1, keepdim=True) + 1e-6)
        lbl = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        img = self._load_image(idx)
        return eeg, img, lbl

    def _load_image(self, idx) -> torch.Tensor:
        """Try to load real ImageNet image; fall back to noise on failure."""
        if idx < len(self.filenames) and self.imagenet_root:
            stem   = self.filenames[idx]           # e.g. "n01484850_xxx"
            synset = stem.split("_")[0]            # e.g. "n01484850"
            for ext in (".JPEG", ".jpg", ".png"):
                path = os.path.join(self.imagenet_root, synset, stem + ext)
                if os.path.exists(path):
                    try:
                        img = Image.open(path).convert("RGB")
                        return self.IMG_TRANSFORM(img)
                    except Exception:
                        break
        # fallback: random noise in [-1, 1]
        return torch.randn(config.NC, config.IMAGE_SIZE, config.IMAGE_SIZE)


def get_eeg_image_loaders(eeg_path, label_path, image_path,
                           batch_size, val_split=0.2, seed=999):
    full_ds = EEGImageDataset(eeg_path, label_path, image_path)
    n_val   = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    gen     = torch.Generator().manual_seed(seed)
    tr_ds, vl_ds = random_split(full_ds, [n_train, n_val], generator=gen)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       drop_last=True, num_workers=2, pin_memory=True)
    vl_ld = DataLoader(vl_ds, batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=True)
    return tr_ld, vl_ld


def get_eeg_image_loaders_otf(eeg_path, label_path, meta_path, imagenet_root,
                               batch_size, val_split=0.2, seed=999):
    """On-the-fly image loading variant — no images.npy required."""
    full_ds = EEGImageOnTheFlyDataset(eeg_path, label_path, meta_path, imagenet_root)
    n_val   = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    gen     = torch.Generator().manual_seed(seed)
    tr_ds, vl_ds = random_split(full_ds, [n_train, n_val], generator=gen)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       drop_last=True, num_workers=2, pin_memory=True)
    vl_ld = DataLoader(vl_ds, batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=True)
    return tr_ld, vl_ld


# ── Dummy datasets for smoke-testing ────────────────────────────────────────────
class DummyEEGDataset(Dataset):
    def __init__(self, n_samples=256, n_classes=10,
                 n_channels=None, seq_len=None):
        nc  = n_channels or config.N_CHANNELS
        sl  = seq_len    or config.SEQ_LEN
        self.labels      = np.random.randint(0, n_classes, n_samples)
        self.eeg         = np.random.randn(n_samples, nc, sl).astype(np.float32)
        self.num_windows = 1

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.eeg[idx]),
                torch.tensor(int(self.labels[idx]), dtype=torch.long))


class DummyEEGImageDataset(Dataset):
    def __init__(self, n_samples=256, n_classes=10,
                 n_channels=None, seq_len=None):
        nc  = n_channels or config.N_CHANNELS
        sl  = seq_len    or config.SEQ_LEN
        self.labels = np.random.randint(0, n_classes, n_samples)
        self.eeg    = np.random.randn(n_samples, nc, sl).astype(np.float32)
        self.images = np.random.randint(0, 256,
                          (n_samples, config.NC, config.IMAGE_SIZE, config.IMAGE_SIZE),
                          dtype=np.uint8)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).float() / 127.5 - 1.0
        return (torch.from_numpy(self.eeg[idx]),
                img,
                torch.tensor(int(self.labels[idx]), dtype=torch.long))
