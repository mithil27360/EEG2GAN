import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ── Weight initialisation ─────────────────────────────────────────────────────
def weights_init(m: nn.Module):
    """Orthogonal init for conv layers; scaled normal for BN scale/bias.
    Guards against affine=False BatchNorm layers (weight/bias are None).
    """
    cls = m.__class__.__name__
    if "Conv" in cls and "SelfAttn" not in cls:
        if m.weight is not None:
            nn.init.orthogonal_(m.weight.data)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif "BatchNorm" in cls:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


# ── Conditional Batch Normalisation ──────────────────────────────────────────
class ConditionalBN(nn.Module):
    """Condition batch-norm scale (γ) and shift (β) on an EEG embedding.

    Uses affine=False BN (no learnable γ/β), then predicts γ and β from
    the conditioning vector via small linear layers.
    """
    def __init__(self, n_feat: int, cond_dim: int):
        super().__init__()
        self.bn       = nn.BatchNorm2d(n_feat, affine=False)
        self.fc_gamma = nn.Linear(cond_dim, n_feat)
        self.fc_beta  = nn.Linear(cond_dim, n_feat)
        # xavier_uniform_ gives each output dimension a different scaled
        # projection — nn.init.ones_ made every row identical (defeating
        # the purpose of per-channel conditioning).
        nn.init.xavier_uniform_(self.fc_gamma.weight)
        nn.init.ones_(self.fc_gamma.bias)    # start near identity (γ≈1)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out   = self.bn(x)                              # (B, C, H, W)
        gamma = self.fc_gamma(cond).view(-1, out.size(1), 1, 1)
        beta  = self.fc_beta (cond).view(-1, out.size(1), 1, 1)
        return out * gamma + beta


# ── Self-Attention (for Generator) ───────────────────────────────────────────
class SelfAttn(nn.Module):
    """Non-local self-attention block (Zhang et al., 2018 SAGAN)."""
    def __init__(self, in_dim: int):
        super().__init__()
        d = max(1, in_dim // 8)
        self.query = nn.Conv2d(in_dim, d,   1, bias=False)
        self.key   = nn.Conv2d(in_dim, d,   1, bias=False)
        self.value = nn.Conv2d(in_dim, in_dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, d)
        k = self.key  (x).view(B, -1, H * W)                   # (B, d, HW)
        v = self.value(x).view(B, -1, H * W)                   # (B, C, HW)
        attn = F.softmax(torch.bmm(q, k), dim=-1)              # (B, HW, HW)
        out  = torch.bmm(v, attn.permute(0, 2, 1))
        out  = out.view(B, C, H, W)
        return x + self.gamma * out


# ── Residual Up-sampling block (Generator) ────────────────────────────────────
class ResBlockUp(nn.Module):
    """Upsample + residual with ConditionalBN for class conditioning."""
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.cbn1  = ConditionalBN(in_ch,  cond_dim)
        self.cbn2  = ConditionalBN(out_ch, cond_dim)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.skip  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.cbn1(x,                       cond))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = F.relu(self.cbn2(h,                       cond))
        h = self.conv2(h)
        return h + self.skip(x)


# ── Residual Down-sampling block (Discriminator) ─────────────────────────────
class ResBlockDown(nn.Module):
    """Average-pool downsample + residual with Spectral Norm."""
    def __init__(self, in_ch: int, out_ch: int, first: bool = False):
        super().__init__()
        self.first  = first
        conv = lambda i, o, k=3, p=1: nn.utils.spectral_norm(
            nn.Conv2d(i, o, k, padding=p, bias=True))
        self.conv1  = conv(in_ch,  out_ch)
        self.conv2  = conv(out_ch, out_ch)
        self.skip   = nn.Sequential(
            nn.AvgPool2d(2),
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, bias=True)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x if self.first else F.leaky_relu(x, 0.1)
        h = F.leaky_relu(self.conv1(h), 0.1)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2)
        return h + self.skip(x)


# ── Generator ─────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    Conditional ResNet Generator: z ∈ ℝ^(NOISE_DIM+EEG_FEAT_DIM) → 128×128 RGB.

    Feature map sizes (with IMAGE_SIZE=128):
        z → FC → 4×4 → res1: 8×8 → res2: 16×16 → attn
        → res3: 32×32 → res4: 64×64 → res5: 128×128 → Conv2d(32,3)
    """
    def __init__(
        self,
        z_dim    : int = config.Z_DIM,
        nf       : int = config.NGF,
        cond_dim : int = config.EEG_FEAT_DIM,
        nc       : int = config.NC,
    ):
        super().__init__()
        self.z_dim    = z_dim
        self.cond_dim = cond_dim
        nf8 = nf * 8    # 512
        nf4 = nf * 4    # 256
        nf2 = nf * 2    # 128
        nf1 = nf        # 64
        nf_2 = nf // 2  # 32

        self.fc    = nn.Linear(z_dim, 4 * 4 * nf8,  bias=False)
        self.res1  = ResBlockUp(nf8, nf4,  cond_dim)   # 4 → 8
        self.res2  = ResBlockUp(nf4, nf2,  cond_dim)   # 8 → 16
        self.attn  = SelfAttn(nf2)                     # attention at 16×16
        self.res3  = ResBlockUp(nf2, nf1,  cond_dim)   # 16 → 32
        self.res4  = ResBlockUp(nf1, nf_2, cond_dim)   # 32 → 64
        self.res5  = ResBlockUp(nf_2, nf_2, cond_dim)  # 64 → 128
        self.out   = nn.Sequential(
            nn.BatchNorm2d(nf_2),
            nn.ReLU(True),
            nn.Conv2d(nf_2, nc, 3, padding=1),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def _split_z(self, z: torch.Tensor):
        """Split z into noise part and EEG conditioning part."""
        noise = z[:, :self.z_dim - self.cond_dim]
        cond  = z[:, -self.cond_dim:]
        return noise, cond

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _, cond = self._split_z(z)
        h = self.fc(z).view(-1, config.NGF * 8, 4, 4)   # (B, 512, 4, 4)
        h = self.res1(h, cond)
        h = self.res2(h, cond)
        h = self.attn(h)
        h = self.res3(h, cond)
        h = self.res4(h, cond)
        h = self.res5(h, cond)
        return self.out(h)


# ── Discriminator ─────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Projection Discriminator with Spectral Norm and Residual Blocks.
    Real/fake score + inner-product with EEG embedding for class conditioning.

    Input: 128×128 RGB → residual downsampling → GlobalAvgPool → linear
    """
    def __init__(
        self,
        nf       : int = config.NDF,
        cond_dim : int = config.EEG_FEAT_DIM,
        nc       : int = config.NC,
    ):
        super().__init__()
        nf2 = nf * 2    # 128
        nf4 = nf * 4    # 256
        nf8 = nf * 8    # 512
        sn  = nn.utils.spectral_norm

        self.res_in = ResBlockDown(nc,  nf,  first=True)   # 128 → 64
        self.res1   = ResBlockDown(nf,  nf2)               # 64  → 32
        self.attn   = SelfAttn(nf2)
        self.res2   = ResBlockDown(nf2, nf4)               # 32  → 16
        self.res3   = ResBlockDown(nf4, nf8)               # 16  → 8
        self.res4   = ResBlockDown(nf8, nf8)               # 8   → 4

        self.fc_out = sn(nn.Linear(nf8, 1,       bias=True))
        self.fc_emb = sn(nn.Linear(nf8, cond_dim, bias=False))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.res_in(x)
        h = self.res1(h)
        h = self.attn(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = F.relu(h)
        h = h.sum(dim=[2, 3])                              # global sum pooling
        score  = self.fc_out(h)                            # (B, 1)
        proj   = (self.fc_emb(h) * cond).sum(dim=1, keepdim=True)  # (B, 1)
        return score + proj
