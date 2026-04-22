import torch
import torch.nn as nn
import torch.nn.functional as F
import config

def weights_init(m):
    cls = m.__class__.__name__
    if cls.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif cls.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, z_dim=config.Z_DIM, nf=config.NGF, nc=config.NC):
        super().__init__()
        # Project noise + eeg features to 4x4x512
        self.fc = nn.Linear(z_dim, 4 * 4 * nf * 8)
        self.conv_layers = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(nf, nf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf // 2),
            nn.ReLU(True),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(nf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, config.NGF * 8, 4, 4)
        return self.conv_layers(h)

class Discriminator(nn.Module):
    def __init__(self, nc=config.NC, nf=config.NDF, eeg_feat_dim=config.EEG_FEAT_DIM):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(nc, nf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(config.LEAKY_SLOPE, inplace=True),
            # 64x64 -> 32x32
            nn.Conv2d(nf // 2, nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(config.LEAKY_SLOPE, inplace=True),
            # 32x32 -> 16x16
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(config.LEAKY_SLOPE, inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(config.LEAKY_SLOPE, inplace=True),
            # 8x8 -> 4x4
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(config.LEAKY_SLOPE, inplace=True),
        )
        # Joint evaluation of image features and EEG features
        self.fc = nn.Linear(4 * 4 * nf * 8 + eeg_feat_dim, 1)

    def forward(self, img, eeg_feat):
        h = self.conv_layers(img).view(img.size(0), -1)
        # Concatenate EEG embedding with flattened image features
        h = torch.cat([h, eeg_feat], dim=1)
        return self.fc(h)

def hinge_loss_d(real_scores, fake_scores):
    return (F.relu(1.0 - real_scores) + F.relu(1.0 + fake_scores)).mean()

def hinge_loss_g(fake_scores):
    return -fake_scores.mean()

def mode_seeking_loss(fake1, fake2, z1, z2, eps=1e-5):
    img_dist = (fake1 - fake2).abs().mean()
    z_dist = (z1 - z2).abs().mean()
    return -(img_dist / (z_dist + eps))
