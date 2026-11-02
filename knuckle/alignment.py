"""Rotation and shift alignment of templates and the MRSD distance.

Implements Eqs. (3)-(6) of the paper:

    D_{sw,sh,theta}(T1, T2) = 1/|C| * sum_{(x,y) in C} (T1^{(sw,sh,theta)}[x,y] - T2[x,y])^2      (4)
    L(T1, T2)               = min_{|sw|<=SW, |sh|<=SH, |theta|<=Theta} D_{sw,sh,theta}(T1, T2)      (3)

T1 is rotated by theta about the template centre and translated by (sw, sh)
pixels with an affine transform and bilinear interpolation (Eq. 6), realised
here with ``F.affine_grid`` / ``F.grid_sample``. C is the common region, i.e.
the output pixels whose bilinear support lies entirely inside T1.

Because every operation is differentiable and ``torch.min`` routes the
gradient only to the minimising transform, autograd reproduces exactly the
piecewise gradients of Eqs. (7)-(10): pixels outside the common region get a
zero gradient and the anchor gradient is pulled back through the bilinear
weights of Eq. (7).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotationShiftAligner(nn.Module):
    def __init__(self, max_shift_x=4, max_shift_y=4, angles=(0.0,), shift_step=1, min_overlap=0.5):
        super().__init__()
        params = [
            (float(sx), float(sy), float(a))
            for a in angles
            for sy in range(-max_shift_y, max_shift_y + 1, shift_step)
            for sx in range(-max_shift_x, max_shift_x + 1, shift_step)
        ]
        self.register_buffer("params", torch.tensor(params, dtype=torch.float32), persistent=False)
        self.min_overlap = min_overlap
        self.config = dict(max_shift_x=max_shift_x, max_shift_y=max_shift_y, angles=list(angles),
                           shift_step=shift_step, min_overlap=min_overlap)
        self._cache = {}

    @property
    def num_transforms(self):
        return self.params.shape[0]

    def _grid_mask(self, H, W, device, dtype):
        key = (H, W, str(device), dtype)
        if key not in self._cache:
            p = self.params.to(device=device, dtype=dtype)
            tx, ty, rad = p[:, 0], p[:, 1], p[:, 2] * math.pi / 180.0
            cos, sin = torch.cos(rad), torch.sin(rad)
            # Pixel coordinates centred on the map: in = R(-theta) (out - t).
            # Normalised coordinates (align_corners=False): x_n = 2 x / W, y_n = 2 y / H.
            A = torch.zeros(p.shape[0], 2, 3, device=device, dtype=dtype)
            A[:, 0, 0] = cos
            A[:, 0, 1] = sin * H / W
            A[:, 1, 0] = -sin * W / H
            A[:, 1, 1] = cos
            A[:, 0, 2] = -(2.0 / W) * (cos * tx + sin * ty)
            A[:, 1, 2] = -(2.0 / H) * (-sin * tx + cos * ty)
            grid = F.affine_grid(A, (p.shape[0], 1, H, W), align_corners=False)
            ones = torch.ones(p.shape[0], 1, H, W, device=device, dtype=dtype)
            cover = F.grid_sample(ones, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            mask = (cover[:, 0] > 1.0 - 1e-4).to(dtype)                       # (K, H, W) common region C
            area = mask.sum(dim=(1, 2))
            valid = area >= self.min_overlap * H * W
            self._cache[key] = (grid, mask, area.clamp(min=1.0), valid)
        return self._cache[key]

    def warp(self, t):
        """Warp templates t (B, C, H, W) with every transform -> (B, K, C, H, W)."""
        B, C, H, W = t.shape
        grid, _, _, _ = self._grid_mask(H, W, t.device, t.dtype)
        K = grid.shape[0]
        x = t.reshape(1, B * C, H, W).expand(K, -1, -1, -1)
        w = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return w.view(K, B, C, H, W).permute(1, 0, 2, 3, 4)

    def distances(self, t1, t2):
        """D_{sw,sh,theta}(t1_i, t2_i) for paired batches -> (B, K)."""
        B, C, H, W = t1.shape
        _, mask, area, valid = self._grid_mask(H, W, t1.device, t1.dtype)
        sq = (self.warp(t1) - t2.unsqueeze(1)) ** 2 * mask[None, :, None]
        d = sq.sum(dim=(2, 3, 4)) / (area * C)[None]
        return d.masked_fill(~valid[None], float("inf"))

    def mrsd(self, t1, t2, return_index=False):
        """Minimal rotational and shifted distance L(t1, t2) of Eq. (3) for paired batches."""
        d, idx = self.distances(t1, t2).min(dim=1)
        return (d, idx) if return_index else d

    @torch.no_grad()
    def pairwise(self, probes, gallery, budget=2 ** 26):
        """All-pairs MRSD: out[i, j] = L(gallery_j, probe_i), shape (M, N).

        The gallery template is the one being rotated/shifted (T1 in Eq. 3).
        Work is chunked so that at most ~``budget`` elements are materialised.
        """
        M, C, H, W = probes.shape
        N = gallery.shape[0]
        _, mask, area, valid = self._grid_mask(H, W, probes.device, probes.dtype)
        K = self.num_transforms
        per_pair = K * C * H * W
        g_chunk = max(1, min(N, budget // per_pair))
        out = probes.new_empty(M, N)
        for g0 in range(0, N, g_chunk):
            wg = self.warp(gallery[g0:g0 + g_chunk])                          # (n, K, C, H, W)
            n = wg.shape[0]
            p_chunk = max(1, budget // (n * per_pair))
            for p0 in range(0, M, p_chunk):
                p = probes[p0:p0 + p_chunk]
                sq = (wg.unsqueeze(0) - p[:, None, None]) ** 2 * mask[None, None, :, None]
                d = sq.sum(dim=(3, 4, 5)) / (area * C)[None, None]           # (m, n, K)
                d = d.masked_fill(~valid[None, None], float("inf"))
                out[p0:p0 + p.shape[0], g0:g0 + n] = d.min(dim=2).values
        return out


def build_aligner(loss_name, max_shift=4, angles=(-4.0, -2.0, 0.0, 2.0, 4.0), shift_step=1, min_overlap=0.5):
    """Aligner matching each loss: TL (none), SSTL (shift only), RSIL (shift + rotation)."""
    if loss_name == "tl":
        return RotationShiftAligner(0, 0, (0.0,), 1, min_overlap)
    if loss_name == "sstl":
        return RotationShiftAligner(max_shift, max_shift, (0.0,), shift_step, min_overlap)
    if loss_name == "rsil":
        return RotationShiftAligner(max_shift, max_shift, tuple(angles), shift_step, min_overlap)
    raise ValueError(f"Unknown loss '{loss_name}'")
