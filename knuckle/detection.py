"""Oriented finger knuckle detection (Sec. III): YOLOv5 + Circular Smooth Label.

Components implemented here
---------------------------
* Long-side five-parameter oriented box definition (cx, cy, w, h, theta) with
  w >= h and theta in [0, 180)  (Fig. 3a).
* Circular Smooth Label encoding of theta, Eq. (11) (Fig. 3b/c), with radius r = 6.
* ``OrientedDetect``: a YOLOv5 ``Detect`` head whose outputs are extended by
  180 angle-class logits per anchor.
* ``OrientedYoloLoss``: Loss = Loss_bbox + Loss_obj + Loss_class + Loss_angle,
  Eq. (12), where Loss_angle is the binary cross entropy between the predicted
  angle classes and the CSL targets of the cells responsible for an object, Eq. (13).
* Rotated NMS, decoding and cropping of the detected knuckle ROI so that it can be
  passed to the RFN matcher.

The head attaches to the multi-scale feature maps of any YOLOv5 backbone/neck
(e.g. the P3/P4/P5 outputs of the ultralytics/yolov5 models). The provided
``Knuckle_Images`` dataset already consists of segmented knuckle ROIs, so this
stage is not needed to train or evaluate the matcher on it; it is required
for full hand / complex-background images.
"""
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ANGLE_BINS = 180
DEFAULT_ANCHORS = ((10, 13, 16, 30, 33, 23), (30, 61, 62, 45, 59, 119), (116, 90, 156, 198, 373, 326))


# ----------------------------------------------------------------------------- box definitions
def poly_to_obb(poly):
    """4 corner points (4, 2) -> long-side (cx, cy, w, h, theta), w >= h, theta in [0, 180)."""
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(poly, dtype=np.float32).reshape(4, 2))
    if w < h:
        w, h, a = h, w, a + 90.0
    return cx, cy, w, h, a % 180.0


def obb_to_poly(cx, cy, w, h, theta):
    return cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(theta)))


# ----------------------------------------------------------------------------- CSL, Eq. (11)
def _window(d, radius, kind):
    if kind == "gaussian":
        sigma = radius / 2.0
        return torch.exp(-d ** 2 / (2 * sigma ** 2))
    if kind == "triangle":
        return 1.0 - d / radius
    if kind == "pulse":
        return (d == 0).to(d.dtype)
    if kind == "rectangular":
        return torch.ones_like(d)
    raise ValueError(kind)


def csl_encode(theta, num_bins=NUM_ANGLE_BINS, radius=6, window="gaussian"):
    """Circular smooth labels for angles theta (deg, any shape) -> (..., num_bins).

    The window g(x) is periodic with period t = 180 deg and is zero outside
    theta - r < x < theta + r.
    """
    theta = torch.as_tensor(theta, dtype=torch.float32)
    bins = torch.arange(num_bins, dtype=torch.float32, device=theta.device)
    period = float(num_bins)
    diff = torch.remainder(bins - theta.unsqueeze(-1), period)
    d = torch.minimum(diff, period - diff)                     # circular distance
    label = _window(d, radius, window)
    return torch.where(d < radius, label, torch.zeros_like(label))


def csl_decode(angle_logits):
    """Angle in degrees = index of the most probable angle class."""
    return angle_logits.argmax(dim=-1).float() * (180.0 / angle_logits.shape[-1])


def angular_loss(angle_logits, theta, radius=6, window="gaussian"):
    """Loss_angle of Eq. (13) for the predictions of object-responsible cells."""
    target = csl_encode(theta.detach(), angle_logits.shape[-1], radius, window).to(angle_logits)
    return F.binary_cross_entropy_with_logits(angle_logits, target)


# ----------------------------------------------------------------------------- head
class OrientedDetect(nn.Module):
    """YOLOv5 Detect head predicting (x, y, w, h, obj, classes..., 180 angle classes)."""

    def __init__(self, nc=1, anchors=DEFAULT_ANCHORS, ch=(256, 512, 1024), strides=(8, 16, 32),
                 num_angle_bins=NUM_ANGLE_BINS):
        super().__init__()
        self.nc, self.num_angle_bins = nc, num_angle_bins
        self.no = 5 + nc + num_angle_bins
        self.nl, self.na = len(anchors), len(anchors[0]) // 2
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32).view(self.nl, self.na, 2))
        self.register_buffer("stride", torch.tensor(strides, dtype=torch.float32))
        self.m = nn.ModuleList(nn.Conv2d(c, self.na * self.no, 1) for c in ch)
        self._init_biases()

    def _init_biases(self):
        for mi, s in zip(self.m, self.stride):
            b = mi.bias.view(self.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)               # objectness prior
            b.data[:, 5:5 + self.nc] += math.log(0.6 / (self.nc - 0.99)) if self.nc > 1 else 0.0
            mi.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, feats):
        """feats: list of (bs, ch_i, ny_i, nx_i) -> list of raw (bs, na, ny, nx, no)."""
        out = []
        for i, f in enumerate(feats):
            x = self.m[i](f)
            bs, _, ny, nx = x.shape
            out.append(x.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous())
        return out

    @torch.no_grad()
    def decode(self, preds):
        """Raw outputs -> (bs, N, no) with xywh in input pixels and sigmoid scores."""
        z = []
        for i, x in enumerate(preds):
            bs, na, ny, nx, no = x.shape
            yv, xv = torch.meshgrid(torch.arange(ny, device=x.device), torch.arange(nx, device=x.device),
                                    indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()
            y = x.sigmoid()
            xy = (y[..., :2] * 2 - 0.5 + grid) * self.stride[i]
            wh = (y[..., 2:4] * 2) ** 2 * self.anchors[i].view(1, na, 1, 1, 2)
            z.append(torch.cat((xy, wh, y[..., 4:]), -1).view(bs, -1, no))
        return torch.cat(z, 1)


# ----------------------------------------------------------------------------- loss, Eq. (12)
def bbox_ciou(box1, box2, eps=1e-7):
    """CIoU between (n, 4) xywh boxes."""
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = box1[:, 2], box1[:, 3] + eps
    w2, h2 = box2[:, 2], box2[:, 3] + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = (box2[:, 0] - box1[:, 0]) ** 2 + (box2[:, 1] - box1[:, 1]) ** 2
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


class OrientedYoloLoss:
    """YOLOv5 loss extended with the CSL angular classification loss.

    targets: (nt, 7) = [image_index, class, cx, cy, w, h, theta] with cx, cy,
    w, h normalised to [0, 1] by the input size (long-side definition) and
    theta in degrees [0, 180).
    """

    def __init__(self, head: OrientedDetect, box=0.05, obj=1.0, cls=0.5, angle=0.2, anchor_t=4.0,
                 csl_radius=6, csl_window="gaussian", balance=(4.0, 1.0, 0.4)):
        self.head = head
        self.hyp = dict(box=box, obj=obj, cls=cls, angle=angle)
        self.anchor_t = anchor_t
        self.csl_radius, self.csl_window = csl_radius, csl_window
        self.balance = balance
        self.bce = nn.BCEWithLogitsLoss()

    def build_targets(self, preds, targets):
        head, nt = self.head, targets.shape[0]
        device = targets.device
        tcls, tbox, tang, indices, anch = [], [], [], [], []
        gain = torch.ones(8, device=device)
        ai = torch.arange(head.na, device=device).float().view(head.na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(head.na, 1, 1), ai[..., None]), 2)   # (na, nt, 8)
        g = 0.5
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=device).float() * g
        for i in range(head.nl):
            anchors = head.anchors[i] / head.stride[i]
            ny, nx = preds[i].shape[2:4]
            gain[2:6] = torch.tensor([nx, ny, nx, ny], device=device, dtype=torch.float32)
            t = targets * gain
            if nt:
                r = t[..., 4:6] / anchors[:, None]
                t = t[torch.max(r, 1 / r).max(2)[0] < self.anchor_t]
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            b, c = t[:, 0].long(), t[:, 1].long()
            gxy, gwh = t[:, 2:4], t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij[:, 0].clamp(0, nx - 1), gij[:, 1].clamp(0, ny - 1)
            a = t[:, 7].long()
            indices.append((b, a, gj, gi))
            tbox.append(torch.cat((gxy - torch.stack((gi, gj), 1), gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)
            tang.append(t[:, 6])
        return tcls, tbox, tang, indices, anch

    def __call__(self, preds, targets):
        device = preds[0].device
        nc = self.head.nc
        lbox, lobj, lcls, lang = (torch.zeros(1, device=device) for _ in range(4))
        tcls, tbox, tang, indices, anch = self.build_targets(preds, targets)
        for i, pi in enumerate(preds):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros(pi.shape[:4], device=device, dtype=pi.dtype)
            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anch[i]
                iou = bbox_ciou(torch.cat((pxy, pwh), 1), tbox[i])
                lbox += (1.0 - iou).mean()
                tobj[b, a, gj, gi] = iou.detach().clamp(0).to(tobj.dtype)
                if nc > 1:
                    t = torch.zeros_like(ps[:, 5:5 + nc])
                    t[torch.arange(n, device=device), tcls[i]] = 1.0
                    lcls += self.bce(ps[:, 5:5 + nc], t)
                lang += angular_loss(ps[:, 5 + nc:], tang[i], self.csl_radius, self.csl_window)
            lobj += self.bce(pi[..., 4], tobj) * self.balance[i]
        bs = preds[0].shape[0]
        lbox, lobj, lcls, lang = (lbox * self.hyp["box"], lobj * self.hyp["obj"],
                                  lcls * self.hyp["cls"], lang * self.hyp["angle"])
        total = (lbox + lobj + lcls + lang) * bs
        return total, torch.cat((lbox, lobj, lcls, lang)).detach()


# ----------------------------------------------------------------------------- inference
def rotated_nms(decoded, nc=1, conf_thres=0.25, iou_thres=0.3, max_det=100):
    """decoded: (bs, N, no) from ``OrientedDetect.decode``.

    Returns per image an array (n, 7): cx, cy, w, h, theta, score, class.
    """
    results = []
    for x in decoded:
        obj = x[:, 4]
        cls_conf = x[:, 5:5 + nc] if nc > 1 else torch.ones_like(obj)[:, None]
        scores, cls = (obj[:, None] * cls_conf).max(1)
        keep = scores > conf_thres
        x, scores, cls = x[keep], scores[keep], cls[keep]
        if x.shape[0] == 0:
            results.append(np.zeros((0, 7), np.float32))
            continue
        theta = csl_decode(x[:, 5 + nc:])
        boxes = torch.cat((x[:, :4], theta[:, None]), 1).cpu().numpy()
        sc = scores.cpu().numpy()
        cl = cls.cpu().numpy()
        out = []
        for c in np.unique(cl):
            m = np.where(cl == c)[0]
            rects = [((float(b[0]), float(b[1])), (float(b[2]), float(b[3])), float(b[4])) for b in boxes[m]]
            k = cv2.dnn.NMSBoxesRotated(rects, sc[m].tolist(), conf_thres, iou_thres)
            k = np.array(k).reshape(-1)
            for idx in m[k]:
                out.append([*boxes[idx], sc[idx], c])
        out = np.array(sorted(out, key=lambda r: -r[5])[:max_det], np.float32).reshape(-1, 7)
        results.append(out)
    return results


def crop_knuckle_roi(image, obb, out_size=(100, 80), long_side="vertical"):
    """Rotate the image so that the oriented box is axis aligned and crop it.

    out_size is (H, W). ``long_side`` sets whether the box long side ends up
    vertical (as in Knuckle_Images, 100 x 80) or horizontal.
    """
    cx, cy, w, h, theta = [float(v) for v in obb[:5]]
    M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    roi = cv2.getRectSubPix(rotated, (int(round(w)), int(round(h))), (cx, cy))   # long side horizontal
    if long_side == "vertical":
        roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(roi, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)


def letterbox(image, new_size=640, color=114):
    h, w = image.shape[:2]
    r = new_size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new_size - nh) // 2, (new_size - nw) // 2
    canvas = np.full((new_size, new_size) + image.shape[2:], color, dtype=image.dtype)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, (left, top)


@torch.no_grad()
def detect_and_segment(feature_extractor, head, image_bgr, img_size=640, conf_thres=0.25, iou_thres=0.3,
                       roi_size=(100, 80), device="cpu"):
    """Full detection -> segmentation pipeline for one BGR image.

    feature_extractor: module mapping (1, 3, S, S) in [0, 1] to the list of
    feature maps expected by ``head`` (a YOLOv5 backbone + neck).
    Returns the list of oriented boxes (original image coordinates) and the
    grayscale knuckle ROIs ready for the RFN matcher.
    """
    img, r, (left, top) = letterbox(image_bgr, img_size)
    x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255.0)[None].to(device)
    dets = rotated_nms(head.decode(head(feature_extractor(x))), head.nc, conf_thres, iou_thres)[0]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    boxes, rois = [], []
    for d in dets:
        cx, cy, w, h = (d[0] - left) / r, (d[1] - top) / r, d[2] / r, d[3] / r
        obb = (cx, cy, w, h, d[4])
        boxes.append((*obb, d[5]))
        rois.append(crop_knuckle_roi(gray, obb, roi_size))
    return boxes, rois
