import torch
import torch.nn as nn
import torch.nn.functional as F
import config

def weights_init(m):
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

class ConditionalBN(nn.Module):
    def __init__(self, n_feat, cond_dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_feat, affine=False)
        self.fc_gamma = nn.Linear(cond_dim, n_feat)
        self.fc_beta = nn.Linear(cond_dim, n_feat)
        nn.init.xavier_uniform_(self.fc_gamma.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x, cond):
        out = self.bn(x)
        gamma = self.fc_gamma(cond).view(-1, out.size(1), 1, 1)
        beta = self.fc_beta(cond).view(-1, out.size(1), 1, 1)
        return out * gamma + beta

class SelfAttn(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        d = max(1, in_dim // 8)
        self.query = nn.Conv2d(in_dim, d, 1, bias=False)
        self.key = nn.Conv2d(in_dim, d, 1, bias=False)
        self.value = nn.Conv2d(in_dim, in_dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        attn = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return x + self.gamma * out

class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.cbn1 = ConditionalBN(in_ch, cond_dim)
        self.cbn2 = ConditionalBN(out_ch, cond_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
        )

    def forward(self, x, cond):
        h = F.relu(self.cbn1(x, cond))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = F.relu(self.cbn2(h, cond))
        h = self.conv2(h)
        return h + self.skip(x)

class ResBlockDown(nn.Module):
    def __init__(self, in_ch, out_ch, first=False):
        super().__init__()
        self.first = first
        conv = lambda i, o, k=3, p=1: nn.utils.spectral_norm(
            nn.Conv2d(i, o, k, padding=p, bias=True))
        self.conv1 = conv(in_ch, out_ch)
        self.conv2 = conv(out_ch, out_ch)
        self.skip = nn.Sequential(
            nn.AvgPool2d(2),
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, bias=True)),
        )

    def forward(self, x):
        h = x if self.first else F.leaky_relu(x, 0.1)
        h = F.leaky_relu(self.conv1(h), 0.1)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2)
        return h + self.skip(x)

class Generator(nn.Module):
    def __init__(
        self,
        z_dim=config.Z_DIM,
        nf=config.NGF,
        cond_dim=config.EEG_FEAT_DIM,
        nc=config.NC,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        nf8 = nf * 8
        nf4 = nf * 4
        nf2 = nf * 2
        nf1 = nf
        nf_2 = nf // 2
        self.fc = nn.Linear(z_dim, 4 * 4 * nf8, bias=False)
        self.res1 = ResBlockUp(nf8, nf4, cond_dim)
        self.res2 = ResBlockUp(nf4, nf2, cond_dim)
        self.attn = SelfAttn(nf2)
        self.res3 = ResBlockUp(nf2, nf1, cond_dim)
        self.res4 = ResBlockUp(nf1, nf_2, cond_dim)
        self.res5 = ResBlockUp(nf_2, nf_2, cond_dim)
        self.out = nn.Sequential(
            nn.BatchNorm2d(nf_2),
            nn.ReLU(True),
            nn.Conv2d(nf_2, nc, 3, padding=1),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def _split_z(self, z):
        noise = z[:, :self.z_dim - self.cond_dim]
        cond = z[:, -self.cond_dim:]
        return noise, cond

    def forward(self, z):
        _, cond = self._split_z(z)
        h = self.fc(z).view(-1, config.NGF * 8, 4, 4)
        h = self.res1(h, cond)
        h = self.res2(h, cond)
        h = self.attn(h)
        h = self.res3(h, cond)
        h = self.res4(h, cond)
        h = self.res5(h, cond)
        return self.out(h)

class Discriminator(nn.Module):
    def __init__(
        self,
        nf=config.NDF,
        cond_dim=config.EEG_FEAT_DIM,
        nc=config.NC,
    ):
        super().__init__()
        nf2 = nf * 2
        nf4 = nf * 4
        nf8 = nf * 8
        sn = nn.utils.spectral_norm
        self.res_in = ResBlockDown(nc, nf, first=True)
        self.res1 = ResBlockDown(nf, nf2)
        self.attn = SelfAttn(nf2)
        self.res2 = ResBlockDown(nf2, nf4)
        self.res3 = ResBlockDown(nf4, nf8)
        self.res4 = ResBlockDown(nf8, nf8)
        self.fc_out = sn(nn.Linear(nf8, 1, bias=True))
        self.fc_emb = sn(nn.Linear(nf8, cond_dim, bias=False))

    def forward(self, x, cond):
        h = self.res_in(x)
        h = self.res1(h)
        h = self.attn(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = F.relu(h)
        h = h.sum(dim=[2, 3])
        score = self.fc_out(h)
        proj = (self.fc_emb(h) * cond).sum(dim=1, keepdim=True)
        return score + proj
