# EEG2GAN

**Transformer-based EEG Brain Signal to Image Generation**

> 🧠 Converting raw EEG brain signals into images using a Transformer encoder + conditional GAN.

---

## Overview

This repository upgrades standard LSTM-based EEG-to-Image models by implementing an Attention-based **Transformer Encoder**. It processes 128-sequence raw EEG signals across multiple channels (supports both 14-channel EPOC and 5-channel Insight hardware).

The encoded brain signals condition an advanced **DCGAN** (equipped with Hinge Loss, DiffAugment, and Mode-Seeking Loss) to generate 128×128 images.

---

## Results

### Main Comparison — MindBigData ImageNet (569 classes, 2439 samples)

| Method | IS ↑ | EISC ↑ | K-Means Acc ↑ | FID ↓ |
|--------|------|--------|---------------|-------|
| AC-GAN | 4.98 | 0.310 | — | — |
| ThoughtViz | 5.45 | 0.350 | — | — |
| LSTM Encoder (Baseline) | 6.15 ± 0.37 | 0.419 | 20.55% | 141.45 |
| **Transformer Encoder (Ours)** | **7.10** | **0.478** | **20.43%** | — |

> **EISC** = EEG-Image Semantic Consistency (CLIP-based cross-modal alignment score, higher = better)

### Fig 7 — IS vs EISC (Method Comparison)
![EISC vs IS](fig7_eisc_vs_is.png)

The Transformer encoder achieves **Pareto dominance** — higher image quality (IS) *and* higher EEG-image semantic alignment (EISC) simultaneously.

---

### Ablation Study — Loss Function & Augmentation

| Variant | IS ↑ | EISC ↑ | K-Means ↑ |
|---------|------|--------|-----------|
| L1 Loss (mean) | 6.10 | 0.42 | — |
| **L2 Loss (mean) ← best** | **6.90** | **0.43** | **↑** |
| L4 Loss (mean) | 6.50 | 0.41 | — |
| L2 + Cls head | 6.70 | 0.43 | — |
| L2, no DiffAugment | 6.20 | 0.40 | ↓ |

### Fig 4 — Ablation Bar Chart
![Ablation](fig4_ablation_bars.png)

---

### Fig 2 — t-SNE Embedding Space (LSTM vs Transformer)
![t-SNE](fig2_tsne.png)

### Fig 5 — GAN Training Curves (200 epochs, ImageNet)
![Training Curves](fig5_training_curves_imagenet.png)

Both encoders achieve stable GAN convergence (no mode collapse). Discriminator loss decays cleanly to ~0.34.

---

## Architecture

```
EEG signal (5 ch × 128 timesteps)
        │
        ▼
┌─────────────────────────┐
│  Transformer Encoder    │  ← Multi-head self-attention
│  (4 layers, 8 heads)    │    + positional encoding
└─────────┬───────────────┘
          │  128-dim embedding
          ▼
     [concat with noise]
          │
          ▼
┌─────────────────────────┐
│  DCGAN Generator        │  ← Hinge loss + Mode-Seeking
│  (128×128 RGB output)   │    + DiffAugment
└─────────────────────────┘
```

---

## Metrics

The pipeline features a comprehensive modern evaluation suite for cross-modal generation:

1. **Inception Score (IS)** — image quality and diversity
2. **Fréchet Inception Distance (FID)** — realism vs real image distribution
3. **K-Means Clustering Accuracy** — EEG feature separability (Hungarian matching)
4. **EISC (EEG-Image Semantic Consistency)** — custom CLIP-based metric verifying that the generated image semantically aligns with the original EEG brain signal

---

## Datasets Supported

| Dataset | Channels | Classes | Modality |
|---------|----------|---------|----------|
| ThoughtViz Objects | 14 (EPOC) | 10 | Objects |
| ThoughtViz Characters | 14 (EPOC) | 10 | Characters |
| MindBigData MNIST | 14 (EPOC) | 10 | Digits |
| **MindBigData ImageNet** | 5 (Insight) | **569** | ImageNet synsets |

---

## Usage

### 1. Pre-process MindBigData ImageNet
```bash
python process_mindbigdata.py \
  --mode imagenet \
  --input data/raw/csvs \
  --image_dir data/raw/images \
  --output data/imagenet
```

### 2. Train Everything & Extract Metrics
```bash
python run_all.py --dataset imagenet
```

### 3. Evaluate a Saved Checkpoint
```bash
python evaluate.py \
  --encoder_ckpt checkpoints/encoder_transformer_imagenet_main.pth \
  --gan_ckpt checkpoints/gan_transformer_imagenet_main.pth \
  --encoder_type transformer \
  --dataset imagenet \
  --output_csv results_main.csv
```

### 4. Extract K-Means Accuracy Locally (offline)
```bash
python extract_metrics.py
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `transformers`, `scikit-learn`, `scipy`, `open-clip-torch`, `tqdm`
