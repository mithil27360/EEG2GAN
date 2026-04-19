import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import config

def weights_init(m):
    cls = m.__class__.__name__
    if "Conv" in cls and "SelfAttn" not in cls:
        nn.init.orthogonal_(m.weight.data)
    elif "BatchNorm" in cls:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# --------------------------------------------------------------------------- #
#  Shared building blocks
# --------------------------------------------------------------------------- #
class SelfAttention(nn.Module):
    """Non-local attention block (Zhang et al. 2019 SA-GAN)."""
    def __init__(self, in_ch):
        super().__init__()
        mid = max(1, in_ch // 8)
        self.q = spectral_norm(nn.Conv2d(in_ch, mid, 1, bias=False))
        self.k = spectral_norm(nn.Conv2d(in_ch, mid, 1, bias=False))
        self.v = spectral_norm(nn.Conv2d(in_ch, in_ch, 1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).view(B, -1, H * W).permute(0, 2, 1)   # (B, HW, C/8)
        k = self.k(x).view(B, -1, H * W)                     # (B, C/8, HW)
        attn = torch.softmax(torch.bmm(q, k) / (q.size(-1) ** 0.5), dim=-1)
        v = self.v(x).view(B, -1, H * W)                     # (B, C, HW)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x

class ConditionalBN(nn.Module):
    """Conditional BatchNorm2d — modulate scale/shift from EEG embedding."""
    def __init__(self, n_feat, cond_dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_feat, affine=False)
        self.fc_gamma = nn.Linear(cond_dim, n_feat)
        self.fc_beta  = nn.Linear(cond_dim, n_feat)
        nn.init.ones_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_beta.weight)

    def forward(self, x, cond):
        out   = self.bn(x)
        gamma = self.fc_gamma(cond).unsqueeze(-1).unsqueeze(-1)
        beta  = self.fc_beta(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * out + beta

# --------------------------------------------------------------------------- #
#  Generator — ResNet + Conditional BN + Self-Attention
# --------------------------------------------------------------------------- #
class ResBlockUp(nn.Module):
    """Upsampling residual block for Generator with conditional BN."""
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.cbn1   = ConditionalBN(in_ch, cond_dim)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.cbn2   = ConditionalBN(out_ch, cond_dim)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        for m in (self.conv1, self.conv2, self.skip):
            nn.init.orthogonal_(m.weight)

    def forward(self, x, cond):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        h  = F.relu(self.cbn1(x, cond))
        h  = F.interpolate(h, scale_factor=2, mode="nearest")
        h  = self.conv1(h)
        h  = F.relu(self.cbn2(h, cond))
        h  = self.conv2(h)
        return h + self.skip(up)

class Generator(nn.Module):
    """
    Conditional ResNet Generator.
    z = cat[noise (NOISE_DIM) | eeg_feat (EEG_FEAT_DIM)]
    EEG features condition every ResBlock via ConditionalBN.
    Architecture: 4x4 -> 8 -> 16 -> 32 -> 64 -> 128
    """
    def __init__(self, z_dim=config.Z_DIM, nc=config.NC, cond_dim=config.EEG_FEAT_DIM):
        super().__init__()
        self.noise_dim = config.NOISE_DIM
        self.cond_dim  = cond_dim
        nf = 512
        self.fc      = nn.Linear(config.NOISE_DIM, 4 * 4 * nf)
        self.res1    = ResBlockUp(nf,      nf // 2, cond_dim)   # 4  -> 8
        self.res2    = ResBlockUp(nf // 2, nf // 4, cond_dim)   # 8  -> 16
        self.res3    = ResBlockUp(nf // 4, nf // 8, cond_dim)   # 16 -> 32
        self.attn_g  = SelfAttention(nf // 8)                   # attention at 32x32
        self.res4    = ResBlockUp(nf // 8, nf // 16, cond_dim)  # 32 -> 64
        self.res5    = ResBlockUp(nf // 16, 16, cond_dim)       # 64 -> 128
        self.out = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, nc, 3, padding=1),
            nn.Tanh(),
        )
        nn.init.orthogonal_(self.fc.weight)

    def forward(self, z):
        # Split z into noise and EEG condition
        noise   = z[:, :self.noise_dim]
        eeg_cond = z[:, self.noise_dim:]
        B = noise.size(0)
        h = self.fc(noise).view(B, 512, 4, 4)
        h = self.res1(h, eeg_cond)
        h = self.res2(h, eeg_cond)
        h = self.res3(h, eeg_cond)
        h = self.attn_g(h)
        h = self.res4(h, eeg_cond)
        h = self.res5(h, eeg_cond)
        return self.out(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# --------------------------------------------------------------------------- #
#  Discriminator — Spectral Norm + Self Attention + Projection conditioning
# --------------------------------------------------------------------------- #
class ResBlockDown(nn.Module):
    """Downsampling residual block with spectral norm for Discriminator."""
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False))
        self.skip  = spectral_norm(nn.Conv2d(in_ch, out_ch, 1, bias=False))
        self.down  = downsample

    def forward(self, x):
        h = F.leaky_relu(x, 0.1)
        h = self.conv1(h)
        h = F.leaky_relu(h, 0.1)
        h = self.conv2(h)
        s = self.skip(x)
        if self.down:
            h = F.avg_pool2d(h, 2)
            s = F.avg_pool2d(s, 2)
        return h + s

class Discriminator(nn.Module):
    """
    Projection Discriminator (Miyato & Koyama 2018) with spectral norm.
    D(x, c) = φ(x)^T v(c) + ψ(φ(x))
    where v(c) projects the EEG condition into the feature space.
    """
    def __init__(self, nc=config.NC, eeg_feat_dim=config.EEG_FEAT_DIM):
        super().__init__()
        nf = 64
        # 128 -> 64 -> 32 (attn) -> 16 -> 8 -> 4 -> pool
        self.block_in = spectral_norm(nn.Conv2d(nc, nf, 3, padding=1, bias=False))
        self.b1 = ResBlockDown(nf,      nf * 2)   # 128 -> 64
        self.b2 = ResBlockDown(nf * 2,  nf * 4)   # 64  -> 32
        self.attn = SelfAttention(nf * 4)          # attention at 32x32
        self.b3 = ResBlockDown(nf * 4,  nf * 8)   # 32  -> 16
        self.b4 = ResBlockDown(nf * 8,  nf * 16)  # 16  -> 8
        self.b5 = ResBlockDown(nf * 16, nf * 16, downsample=False)  # 8 -> 8

        feat_size = nf * 16   # 1024

        # Unconditional head
        self.fc_uncond = spectral_norm(nn.Linear(feat_size, 1))
        # Projection head: EEG -> feat_size
        self.embed = spectral_norm(nn.Embedding(1, feat_size))   # placeholder
        self.proj  = spectral_norm(nn.Linear(eeg_feat_dim, feat_size, bias=False))

    def forward(self, img, eeg_feat):
        h = F.leaky_relu(self.block_in(img), 0.1)
        h = self.b1(h)
        h = self.b2(h)
        h = self.attn(h)
        h = self.b3(h)
        h = self.b4(h)
        h = self.b5(h)
        h = F.relu(h)
        h = h.sum(dim=[2, 3])          # global sum pooling → (B, feat_size)
        # Unconditional score
        out = self.fc_uncond(h)        # (B, 1)
        # Projection conditioning: inner product with EEG embed
        proj = self.proj(eeg_feat)     # (B, feat_size)
        out  = out + (h * proj).sum(dim=1, keepdim=True)
        return out.squeeze(1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# --------------------------------------------------------------------------- #
#  Losses
# --------------------------------------------------------------------------- #
def hinge_loss_d(real_scores, fake_scores):
    return (F.relu(1.0 - real_scores).mean() +
            F.relu(1.0 + fake_scores).mean())

def hinge_loss_g(fake_scores):
    return -fake_scores.mean()

def mode_seeking_loss(fake1, fake2, z1, z2, eps=1e-5):
    img_dist = (fake1 - fake2).abs().mean()
    z_dist   = (z1 - z2).abs().mean()
    return -(img_dist / (z_dist + eps))
