import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import config
import random

class EEGTransform:
    def __init__(self, noise_std=0.0, shift_max=0, mask_len=0):
        self.noise_std = noise_std
        self.shift_max = shift_max
        self.mask_len  = mask_len

    def __call__(self, x):
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.shift_max > 0:
            shift = random.randint(-self.shift_max, self.shift_max)
            x = torch.roll(x, shifts=shift, dims=-1)
        if self.mask_len > 0:
            t_len = x.shape[-1]
            start = random.randint(0, max(0, t_len - self.mask_len))
            x[:, start : start + self.mask_len] = 0
        return x

def _fix_eeg_shape(eeg):
    if eeg.size == 0: return eeg
    
    # Handle shape (Samples, Channels, Time) or (Samples, Time, Channels)
    if eeg.ndim == 2:
        # If it's 2D, we assume (Channels, Time) and add a sample dimension
        eeg = eeg[np.newaxis, ...]
    
    if eeg.ndim == 3:
        # Check if it needs transposing (N, T, C) -> (N, C, T)
        # Assuming config.SEQ_LEN is the time dimension
        # If the middle dimension is the long one (SEQ_LEN), it's likely (N, T, C)
        # but only if the last dimension is NOT SEQ_LEN
        if eeg.shape[1] == config.SEQ_LEN and eeg.shape[2] != config.SEQ_LEN:
            eeg = eeg.transpose(0, 2, 1)
            
    return eeg

class EEGDataset(Dataset):
    def __init__(self, eeg_path, label_path, transform=None, window_size=None, stride=None):
        self.eeg = np.load(eeg_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.int64)
        self.eeg = _fix_eeg_shape(self.eeg)
        self.transform = transform
        self.window_size = window_size or self.eeg.shape[-1]
        self.stride      = stride or self.window_size
        self.num_windows = (self.eeg.shape[-1] - self.window_size) // self.stride + 1
        if self.num_windows < 1: self.num_windows = 1

    def __len__(self):
        return len(self.labels) * self.num_windows

    def __getitem__(self, idx):
        s_idx = idx // self.num_windows
        w_idx = idx % self.num_windows
        eeg   = self.eeg[s_idx]
        start = w_idx * self.stride
        eeg   = eeg[:, start : start + self.window_size]
        
        if config.EEG_NORMALIZE:
            m = eeg.mean(axis=-1, keepdims=True)
            s = eeg.std(axis=-1, keepdims=True) + 1e-6
            eeg = (eeg - m) / s
            
        eeg = torch.from_numpy(eeg)
        if self.transform: eeg = self.transform(eeg)
        return eeg, torch.tensor(self.labels[s_idx])

class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, n_per_class, batch_size):
        self.labels = np.array(labels)
        self.n_per_class = n_per_class
        self.batch_size = batch_size
        self.n_classes_per_batch = max(1, batch_size // n_per_class)
        self.label_to_indices = {i: np.where(self.labels == i)[0] for i in np.unique(self.labels)}
        self.labels_list = list(self.label_to_indices.keys())

    def __iter__(self):
        n_batches = len(self.labels) // self.batch_size
        for _ in range(n_batches):
            selected_labels = np.random.choice(self.labels_list, self.n_classes_per_batch, 
                                               replace=len(self.labels_list) < self.n_classes_per_batch)
            indices = []
            for lbl in selected_labels:
                t_indices = self.label_to_indices[lbl]
                indices.extend(np.random.choice(t_indices, self.n_per_class, replace=True))
            np.random.shuffle(indices)
            yield indices[:self.batch_size]

    def __len__(self):
        return (len(self.labels) // self.batch_size) * self.batch_size

def get_eeg_loaders(eeg_path, label_path, batch_size, val_split=0.2, seed=999, transform=None):
    full_ds = EEGDataset(eeg_path, label_path, transform=transform, 
                         window_size=config.EEG_WINDOW_SIZE, stride=config.EEG_WINDOW_STRIDE)
    n_val = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    
    if config.BALANCED_SAMPLING:
        train_labels = [full_ds.labels[i // full_ds.num_windows] for i in train_ds.indices]
        sampler = BalancedBatchSampler(train_labels, config.SAMPLES_PER_CLASS, batch_size)
        train_loader = DataLoader(train_ds, batch_sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

class DummyEEGDataset(Dataset):
    def __init__(self, n_samples=230, n_classes=10):
        self.labels = np.random.randint(0, n_classes, n_samples)
        self.eeg    = np.random.randn(n_samples, config.N_CHANNELS, config.SEQ_LEN).astype(np.float32)
        self.num_windows = 1
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return torch.from_numpy(self.eeg[idx]), torch.tensor(self.labels[idx])

class DummyEEGImageDataset(Dataset):
    def __init__(self, n_samples=230, n_classes=10):
        self.labels = np.random.randint(0, n_classes, n_samples)
        self.eeg    = np.random.randn(n_samples, config.N_CHANNELS, config.SEQ_LEN).astype(np.float32)
        self.images = np.random.randint(0, 256, (n_samples, config.NC, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).float() / 255.0
        img = (img - 0.5) / 0.5
        return torch.from_numpy(self.eeg[idx]), img, torch.tensor(self.labels[idx])

class EEGImageDataset(Dataset):
    def __init__(self, eeg_path, label_path, image_path, transform=None):
        self.eeg = np.load(eeg_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.int64)
        self.images = np.load(image_path)
        self.eeg = _fix_eeg_shape(self.eeg)
        if self.images.shape[-1] == 3: self.images = self.images.transpose(0, 3, 1, 2)
        self.transform = transform or transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        eeg = torch.from_numpy(self.eeg[idx])
        if config.EEG_NORMALIZE:
            eeg = (eeg - eeg.mean(dim=-1, keepdim=True)) / (eeg.std(dim=-1, keepdim=True) + 1e-6)
        lbl = torch.tensor(self.labels[idx])
        img = torch.from_numpy(self.images[idx]).float() / 255.0
        img = self.transform(img)
        return eeg, img, lbl

def get_eeg_image_loaders(eeg_path, label_path, image_path, batch_size, val_split=0.2, seed=999):
    full_ds = EEGImageDataset(eeg_path, label_path, image_path)
    n_val = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    tr_ds, vl_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    vl_ld = DataLoader(vl_ds, batch_size=batch_size, shuffle=False)
    return tr_ld, vl_ld
