"""Rotation and Shift Invariant Loss (RSIL), Eq. (2), and the baselines of Sec. IV-E.

    RSIL = 1/N sum_i [ L(T(I_a), T(I_p)) - L(T(I_a), T(I_n)) + m ]_+

with L the MRSD of Eq. (3). Using an aligner with no shifts and no rotation
gives the conventional triplet loss (TL, Eq. 1, with the MSE form of the
distance) and an aligner with shifts only gives the soft-shifted triplet loss
(SSTL).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RSIL(nn.Module):
    def __init__(self, aligner, margin=0.2):
        super().__init__()
        self.aligner = aligner
        self.margin = margin

    def forward(self, t_anchor, t_pos, t_neg):
        d_ap = self.aligner.mrsd(t_anchor, t_pos)
        d_an = self.aligner.mrsd(t_anchor, t_neg)
        loss = F.relu(d_ap - d_an + self.margin).mean()
        stats = {
            "d_ap": d_ap.detach().mean().item(),
            "d_an": d_an.detach().mean().item(),
            "active": (d_ap - d_an + self.margin > 0).float().mean().item(),
        }
        return loss, stats


@torch.no_grad()
def mine_batch_hard(templates, labels, aligner):
    """Hardest positive / hardest negative for every anchor of a PK batch (by MRSD)."""
    D = aligner.pairwise(templates, templates)                # D[i, j] = L(T_j, T_i)
    D = 0.5 * (D + D.t())
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pos_d = D.masked_fill(~same | eye, -float("inf"))
    neg_d = D.masked_fill(same, float("inf"))
    pos_idx = pos_d.argmax(dim=1)
    neg_idx = neg_d.argmin(dim=1)
    has_pos = torch.isfinite(pos_d.max(dim=1).values)
    has_neg = torch.isfinite(neg_d.min(dim=1).values)
    keep = has_pos & has_neg
    anchors = torch.nonzero(keep, as_tuple=False).squeeze(1)
    return anchors, pos_idx[anchors], neg_idx[anchors]
