# Mathematical Foundations of EEG2GAN

This document provides a rigorous mathematical breakdown of the components used in the EEG2GAN framework, covering signal processing, model architecture, optimization, and evaluation.

---

## 1. Signal Preprocessing

### 1.1 Temporal Filtering
To isolate biologically relevant neural oscillations, we apply a 4th-order Butterworth bandpass filter and an IIR notch filter. The transfer function in the $z$-domain is given by:
$$H(z) = \frac{\sum_{i=0}^n b_i z^{-i}}{\sum_{j=0}^n a_j z^{-j}}$$
1.  **Bandpass Filter**: Isolated between $f_{low} = 1.0$ Hz and $f_{high} = 50.0$ Hz.
2.  **Notch Filter**: Centered at $f_0 = 50.0$ Hz with a quality factor $Q = 30.0$.

### 1.2 Z-Score Normalization
Each EEG channel $c$ is standardized to have zero mean and unit variance:
$$\hat{x}_{c,t} = \frac{x_{c,t} - \mu_c}{\sigma_c + \epsilon}$$
where $\mu_c = \frac{1}{T}\sum_{t=1}^T x_{c,t}$ and $\sigma_c = \sqrt{\frac{1}{T}\sum_{t=1}^T (x_{c,t} - \mu_c)^2}$.

---

## 2. Encoder Architecture: Transformer

### 2.1 Scaled Dot-Product Attention
The core of the Transformer encoder is the attention mechanism, defined as:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Where $Q, K, V$ are Query, Key, and Value matrices derived from the input embeddings. The scaling factor $\frac{1}{\sqrt{d_k}}$ prevents vanishing gradients in the softmax during the training of large embedding dimensions.

### 2.2 Multi-Head Attention (MHA)
We project $Q, K, V$ into $h$ heads to capture parallel temporal features:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

---

## 3. Generative Model: Conditional DCGAN

### 3.1 Fractionally-Strided Convolution (Transposed Conv)
The Generator performs upsampling using transposed convolutions. The output spatial dimension $O$ for an input $I$ is calculated as:
$$O = (I - 1) \times s - 2p + k + p_{out}$$
where $s$ is stride, $p$ is padding, $k$ is kernel size, and $p_{out}$ is output padding. Every stage in our Generator uses $s=2, p=1, k=4$, effectively doubling the resolution.

### 3.2 Minimax Objective
The Conditional GAN game is defined by the objective function $V(D, G)$:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x|e)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z|e)|e))]$$
where $e$ is the EEG embedding from the Transformer encoder.

---

## 4. Optimization & Regularization

### 4.1 Hinge Loss
To improve stability, we implement the Hinge version of the GAN objective:
- **Discriminator**: $L_D = \mathbb{E}[\text{max}(0, 1 - D(x, e))] + \mathbb{E}[\text{max}(0, 1 + D(G(z, e), e))]$
- **Generator**: $L_G = -\mathbb{E}[D(G(z, e), e)]$

### 4.2 Mode-Seeking Loss ($L_{ms}$)
To prevent mode collapse by encouraging high diversity for different noise seeds $z_1, z_2$:
$$L_{ms} = -\frac{\|G(z_1, e) - G(z_2, e)\|_1}{\|z_1 - z_2\|_1 + \epsilon}$$

### 4.3 $R_1$ Gradient Penalty
A zero-centered gradient penalty applied to real samples to regularize the Discriminator's surface:
$$R_1 = \frac{\gamma}{2} \mathbb{E}_{x \sim p_{data}} [\|\nabla_x D(x, e)\|^2]$$

---

## 5. Evaluation Metrics

### 5.1 Inception Score (IS)
Measures quality and diversity using a pre-trained Inception-v3 model:
$$\text{IS}(G) = \exp(\mathbb{E}_{x \sim p_g} [D_{KL}(p(y|x) \| p(y))])$$

### 5.2 Fréchet Inception Distance (FID)
Quantifies the distance between feature distributions of real and generated images:
$$d^2 = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r\Sigma_g})$$

### 5.3 EEG-Image Semantic Consistency (EISC)
Defined as the average cosine similarity in the CLIP latent space:
$$\text{EISC} = \frac{1}{N}\sum_{i=1}^N \frac{\phi(G(z_i, e_i)) \cdot \psi(y_i)}{\|\phi\| \|\psi\|}$$
where $\phi$ is the CLIP image encoder and $\psi$ is the CLIP text/image category encoder.
