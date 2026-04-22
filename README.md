# EEG2GAN: Transformer-based EEG-to-Image Generation

Implementation of a Transformer encoder coupled with a Conditional GAN to reconstruct visual stimuli from raw EEG brain signals.

**[ Research Report](https://drive.google.com/file/d/142n10mdMUsvHgl2oM4b1iqxxHEP4B0Ny/view?usp=sharing)**

## Results

Quantitative comparison on the MindBigData ImageNet dataset (569 classes). 

| Method | IS ↑ | EISC ↑ | K-Means Acc | FID ↓ |
|--------|------|--------|-------------|-------|
| ThoughtViz (2017) | 4.12 | 0.211 | 8.2% | 312.4 |
| LSTM Baseline | 6.15 | 0.419 | 20.4% | 141.4 |
| **EEG2GAN (Ours)** | **7.10** | **0.478** | **20.4%** | **128.9** |



## Visualizations

### Performance Metrics
Quantitative results including dataset statistics, confusion matrices, and ablation studies.

| Main Metrics | Dataset Stats |
|:---:|:---:|
| ![Results Table](results/figures/fig_results_table.png) | ![Dataset Stats](results/figures/fig_dataset_stats.png) |

| Confusion Matrix | Ablations |
|:---:|:---:|
| ![Confusion Matrix](results/figures/fig_confusion_matrix.png) | ![Ablations](results/figures/fig4_ablation_bars.png) |

### Signal Analysis
Power spectral density tracking across different cognitive states and manifold interpolation proving learned smoothness.

![EEG Spectra](results/figures/fig_eeg_spectra.png)



### Sample Generations
Representative grids across diverse ImageNet categories.

![Per-Class Grids](results/figures/fig_per_class_grids.png)



## Architecture

```mermaid
graph LR
    A[Raw EEG] --> B(Filter)
    B --> C(Transformer)
    C --> D{Concat}
    E[Noise z] --> D
    D --> F[EEG-Conditioned GAN]
    F --> G[Image]
```

- **Encoder:** Transformer with 2 layers and 4 attention heads (compared against an LSTM baseline).
- **GAN:** Standard Conditional DCGAN  with Hinge Loss and DiffAugment for stability.



## References

### Foundational GAN Research
- **Generative Adversarial Nets** | Ian J. Goodfellow et al. | [arXiv:1406.2661](https://arxiv.org/pdf/1406.2661)
- **Deep Convolutional Generative Adversarial Networks (DCGAN)** | Alec Radford et al. | [arXiv:1511.06434](https://arxiv.org/pdf/1511.06434)
- **Conditional Generative Adversarial Nets** | Mehdi Mirza & Simon Osindero | [arXiv:1411.1784](https://arxiv.org/pdf/1411.1784)

### EEG-to-Image & Signal Synthesis
- **EEG2IMAGE: Image Reconstruction from EEG Brain Signals** | Prajwal Singh et al. | [arXiv:2302.10121](https://arxiv.org/pdf/2302.10121)
- **EEG-GAN: Generative Adversarial Networks for EEG Signals** | Kay Gregor Hartmann et al. | [arXiv:1806.01875](https://arxiv.org/pdf/1806.01875)
- **Generating Visual Stimuli from EEG using Transformer-Encoder and GAN** | [arXiv:2402.10115](https://arxiv.org/pdf/2402.10115)
- **Guess What I Think: Streamlined EEG-to-Image with Diffusion** | Eleonora Lopez et al. | [arXiv:2410.02780](https://arxiv.org/pdf/2410.02780)

### Reviews & Survey
- **A Survey on Bridging EEG Signals and Generative AI** | Shreya Shukla et al. | [arXiv:2502.12048](https://arxiv.org/pdf/2502.12048)
- **Interpretable EEG-to-Image Generation with Semantic Prompts** | [arXiv:2507.07157](https://arxiv.org/pdf/2507.07157)
