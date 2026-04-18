import torch
import torch.nn as nn
import config

def weights_init(m):
    cls = m.__class__.__name__
    if cls.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif cls.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, z_dim=config.Z_DIM, nc=config.NC):
        super().__init__()
        self.fc = nn.Linear(z_dim, 4 * 4 * 512)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), -1)
        h = self.fc(z)
        h = h.view(h.size(0), 512, 4, 4)
        return self.conv_layers(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class Discriminator(nn.Module):
    def __init__(self, nc=config.NC, eeg_feat_dim=config.EEG_FEAT_DIM):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(config.LEAKY_SLOPE, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(config.LEAKY_SLOPE, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(config.LEAKY_SLOPE, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(config.LEAKY_SLOPE, inplace=True),
        )
        self.fc = nn.Linear(8 * 8 * 512 + eeg_feat_dim, 1)

    def forward(self, img, eeg_feat):
        h = self.conv_layers(img)
        h = h.view(img.size(0), -1)
        h = torch.cat([h, eeg_feat], dim=1)
        return self.fc(h).squeeze(1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def hinge_loss_d(real_scores, fake_scores):
    return (torch.relu(1.0 - real_scores).mean() +
            torch.relu(1.0 + fake_scores).mean())

def hinge_loss_g(fake_scores):
    return -fake_scores.mean()

def mode_seeking_loss(fake1, fake2, z1, z2, eps=1e-5):
    img_dist = (fake1 - fake2).abs().mean()
    z_dist = (z1 - z2).abs().mean()
    return -(img_dist / (z_dist + eps))
