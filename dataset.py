import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import config

def _fix_eeg_shape(eeg):
    if eeg.ndim == 2:
        eeg = eeg[np.newaxis, ...]
    if eeg.shape[1] == config.SEQ_LEN and eeg.shape[2] != config.SEQ_LEN:
        eeg = eeg.transpose(0, 2, 1)
    elif eeg.shape[2] == config.SEQ_LEN and eeg.shape[1] != config.SEQ_LEN:
        pass
    elif eeg.shape[1] == config.SEQ_LEN:
        eeg = eeg.transpose(0, 2, 1)
    return eeg

class EEGDataset(Dataset):
    def __init__(self, eeg_path, label_path):
        self.eeg = np.load(eeg_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.int64)
        self.eeg = _fix_eeg_shape(self.eeg)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = torch.from_numpy(self.eeg[idx])
        label = torch.tensor(self.labels[idx])
        return eeg, label

class DummyEEGDataset(Dataset):
    def __init__(self, n_samples=230, n_classes=10):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.eeg = np.random.randn(n_samples, config.N_CHANNELS, config.SEQ_LEN).astype(np.float32)
        self.labels = np.random.randint(0, n_classes, n_samples).astype(np.int64)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.eeg[idx]), torch.tensor(self.labels[idx])

class EEGImageDataset(Dataset):
    def __init__(self, eeg_path, label_path, image_path, transform=None):
        self.eeg = np.load(eeg_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.int64)
        self.images = np.load(image_path)
        self.eeg = _fix_eeg_shape(self.eeg)
        if self.images.shape[-1] == 3:
            self.images = self.images.transpose(0, 3, 1, 2)
        self.transform = transform or transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = torch.from_numpy(self.eeg[idx])
        label = torch.tensor(self.labels[idx])
        img = self.images[idx]
        img = torch.from_numpy(img).float() / 255.0
        if self.transform:
            img = self.transform(img)
        return eeg, img, label

class DummyEEGImageDataset(Dataset):
    def __init__(self, n_samples=230, n_classes=10):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.eeg = np.random.randn(n_samples, config.N_CHANNELS, config.SEQ_LEN).astype(np.float32)
        self.labels = np.random.randint(0, n_classes, n_samples).astype(np.int64)
        self.images = np.random.randint(0, 255, (n_samples, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)).astype(np.uint8)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        eeg = torch.from_numpy(self.eeg[idx])
        img = torch.from_numpy(self.images[idx]).float() / 255.0
        img = (img - 0.5) / 0.5
        label = torch.tensor(self.labels[idx])
        return eeg, img, label

def get_eeg_loaders(eeg_path, label_path, batch_size, val_split=0.2, seed=999):
    full_ds = EEGDataset(eeg_path, label_path)
    n_val = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def get_eeg_image_loaders(eeg_path, label_path, image_path, batch_size, val_split=0.2, seed=999):
    full_ds = EEGImageDataset(eeg_path, label_path, image_path)
    n_val = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
