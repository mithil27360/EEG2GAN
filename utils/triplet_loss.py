import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_distances(embeddings, squared=False):
    dot = embeddings @ embeddings.t()
    sq = dot.diag().unsqueeze(1) + dot.diag().unsqueeze(0)
    dist = (sq - 2 * dot).clamp(min=0)
    if not squared:
        dist = torch.sqrt(dist + 1e-12)
    return dist

def _get_triplet_mask(labels):
    B = labels.size(0)
    ids = torch.arange(B, device=labels.device)
    distinct = ids.unsqueeze(0) != ids.unsqueeze(1)
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    diff_label = ~same_label
    valid_ap = same_label & distinct
    mask = valid_ap.unsqueeze(2) & diff_label.unsqueeze(1)
    return mask

def batch_semi_hard_triplet_loss(embeddings, labels, margin=0.3, squared=False):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    dist_mat = pairwise_distances(embeddings, squared=squared)
    B = dist_mat.size(0)
    mask = _get_triplet_mask(labels)
    ap_dist = dist_mat.unsqueeze(2).expand(B, B, B)
    an_dist = dist_mat.unsqueeze(1).expand(B, B, B)
    loss_raw = ap_dist - an_dist + margin
    semi_hard_mask = (an_dist > ap_dist) & (loss_raw < margin)
    sh_mask = mask & semi_hard_mask
    if sh_mask.sum() == 0:
        sh_mask = mask
    triplet_loss = loss_raw * sh_mask.float()
    triplet_loss = triplet_loss.clamp(min=0)
    n_triplets = sh_mask.float().sum()
    loss = triplet_loss.sum() / (n_triplets + 1e-8)
    return loss

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        device = features.device
        B = features.size(0)
        sim = torch.mm(features, features.t()) / self.temperature
        eye = torch.eye(B, device=device).bool()
        sim.masked_fill_(eye, float('-inf'))
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()) & ~eye
        log_prob = F.log_softmax(sim, dim=1)
        n_pos = pos_mask.float().sum(dim=1)
        valid = n_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        loss = -(log_prob * pos_mask.float()).sum(dim=1)
        loss = loss[valid] / n_pos[valid]
        return loss.mean()
