from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from torchvision import models, transforms
from PIL import Image
import config

class InceptionScoreCalculator:
    def __init__(self, device=None, batch_size: int = 32, splits: int = config.IS_SPLITS):
        self.device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.splits     = splits
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def get_predictions(self, images: list[Image.Image]) -> np.ndarray:
        all_preds = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i: i + self.batch_size]
            tensors = torch.stack([self.transform(img) for img in batch]).to(self.device)
            logits = self.model(tensors)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(probs)
        return np.concatenate(all_preds, axis=0)

    def compute(self, images: list[Image.Image]) -> tuple[float, float]:
        preds = self.get_predictions(images)
        N     = preds.shape[0]
        split_scores = []
        for k in range(self.splits):
            part = preds[k * (N // self.splits): (k + 1) * (N // self.splits)]
            py   = part.mean(axis=0)
            kl   = [entropy(p, py) for p in part]
            split_scores.append(np.exp(np.mean(kl)))
        return float(np.mean(split_scores)), float(np.std(split_scores))

def kmeans_accuracy(
    embeddings : np.ndarray,
    labels     : np.ndarray,
    n_clusters : int | None = None,
    n_init     : int = config.KMEANS_N_INIT,
    seed       : int = config.SEED,
) -> float:
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
    cluster_ids = km.fit_predict(embeddings)
    cluster_to_label = {}
    for c in range(n_clusters):
        mask = cluster_ids == c
        if mask.sum() == 0:
            cluster_to_label[c] = 0
            continue
        cluster_to_label[c] = int(np.bincount(labels[mask]).argmax())
    predicted = np.array([cluster_to_label[c] for c in cluster_ids])
    return float(accuracy_score(labels, predicted))

class EISCCalculator:
    def __init__(self, device=None, batch_size: int = 32):
        from transformers import CLIPModel, CLIPProcessor
        self.device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model     = CLIPModel.from_pretrained(config.CLIP_MODEL).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        self.model.eval()

    @torch.no_grad()
    def _encode_images(self, images: list[Image.Image]) -> np.ndarray:
        all_embs = []
        for i in range(0, len(images), self.batch_size):
            batch  = images[i: i + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt",
                                    padding=True).to(self.device)
            embs   = self.model.get_image_features(**inputs)
            embs   = F.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    def compute(
        self,
        generated_images : list[Image.Image],
        real_images      : list[Image.Image],
    ) -> float:
        assert len(generated_images) == len(real_images)
        emb_gen  = self._encode_images(generated_images)
        emb_real = self._encode_images(real_images)
        cos_sims = (emb_gen * emb_real).sum(axis=1)
        return float(cos_sims.mean())

class FIDCalculator:
    def __init__(self, device=None, batch_size: int = 32):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.model.fc = nn.Identity()
        self.model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _get_features(self, images: list[Image.Image]) -> np.ndarray:
        feats = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i: i + self.batch_size]
            tensors = torch.stack([self.transform(img) for img in batch]).to(self.device)
            f = self.model(tensors)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    def compute(self, gen_images: list[Image.Image], real_images: list[Image.Image]) -> float:
        from scipy.linalg import sqrtm
        f_gen = self._get_features(gen_images)
        f_real = self._get_features(real_images)
        mu_gen, sigma_gen = f_gen.mean(axis=0), np.cov(f_gen, rowvar=False)
        mu_real, sigma_real = f_real.mean(axis=0), np.cov(f_real, rowvar=False)
        ssdiff = np.sum((mu_gen - mu_real)**2)
        covmean = sqrtm(sigma_gen.dot(sigma_real))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(ssdiff + np.trace(sigma_gen + sigma_real - 2.0 * covmean))

def tensor_to_pil_list(tensor: torch.Tensor) -> list[Image.Image]:
    imgs = []
    arr  = (tensor.detach().cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5)
    arr  = (arr * 255).clip(0, 255).astype(np.uint8)
    for i in range(arr.shape[0]):
        imgs.append(Image.fromarray(arr[i]))
    return imgs
