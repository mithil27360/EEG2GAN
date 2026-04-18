# EEG2GAN

Transformer-based EEG Brain Signal to Image Generation.

### Overview

This repository upgrades standard LSTM-based EEG-to-Image models by implementing an Attention-based **Transformer Encoder**. It processes 128-sequence raw EEG signals across multiple channels (supports both 14-channel EPOC and 5-channel Insight datasets).

The encoded brain signals are fed into an advanced **DCGAN** (equipped with Hinge Loss, DiffAugment, and Mode-Seeking Loss) to generate 128x128 images. 

### Metrics
The pipeline features a comprehensive modern evaluation suite tailored for Cross-Modal Generation:
1.  **Inception Score (IS)**
2.  **Fréchet Inception Distance (FID)**
3.  **K-Means Feature Clustering Accuracy**
4.  **EISC (EEG-Image Semantic Consistency)** - A custom metric using OpenAI's CLIP model to verify that the generated image semantically aligns with the original brain signal.

### Datasets Supported
1. ThoughtViz (Objects and Characters)
2. MindBigData (MNIST Digits)
3. MindBigData (ImageNet)

### Usage

**Pre-processing MindBigData ImageNet:**
```bash
python process_mindbigdata.py --mode imagenet --input data/raw/csvs --image_dir data/raw/images --output data/imagenet
```

**Training & Extracting Paper Metrics:**
```bash
python run_all.py --dataset imagenet
```
