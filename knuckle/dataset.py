"""Knuckle ROI dataset handling.

The paper uses already detected/segmented knuckle ROIs as input to the
matching network. ``Knuckle_Images`` contains such ROIs named
``<subject>_<sample>.bmp`` (e.g. ``001_3.bmp``), which is the format parsed
here. Any folder with the same naming scheme can be used.

Protocols
---------
open   : subject-disjoint split, as for the PolyU v3.0 database in Sec. IV-A.1
         (the network is trained on subjects that never appear in the test set).
closed : every subject is seen during training using its first ``k`` samples;
         the remaining samples are probes and sample 1 is the gallery, similar
         to the HKPolyU hand-dorsal protocol (Sec. IV-A.2).
"""
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T

FNAME_RE = re.compile(r"^(\d+)_(\d+)\.(bmp|png|jpe?g|tiff?)$", re.IGNORECASE)


def scan_dataset(root):
    """Return a list of (path, subject_id, sample_idx) sorted by subject then sample.

    Sub-folders are searched too, so a nested layout such as
    ``Knuckle_Images/Knuckle_Images/001_1.bmp`` (e.g. after unzipping) also works.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset folder not found: {os.path.abspath(root)}")
    samples, seen = [], set()
    for dirpath, _, files in os.walk(root):
        for fname in files:
            m = FNAME_RE.match(fname)
            if m and fname not in seen:
                seen.add(fname)
                samples.append((os.path.join(dirpath, fname), int(m.group(1)), int(m.group(2))))
    if not samples:
        raise RuntimeError(f"No images named <subject>_<sample>.<ext> found in {root}")
    return sorted(samples, key=lambda s: (s[1], s[2]))


def make_split(samples, protocol="open", num_train_subjects=None, train_ratio=0.6,
               train_samples_per_subject=2, seed=0):
    """Split samples into train/test lists according to the protocol."""
    subjects = sorted({s for _, s, _ in samples})
    if protocol == "open":
        if num_train_subjects is None:
            num_train_subjects = int(round(len(subjects) * train_ratio))
        rng = random.Random(seed)
        shuffled = subjects[:]
        rng.shuffle(shuffled)
        train_ids = set(shuffled[:num_train_subjects])
        train = [s for s in samples if s[1] in train_ids]
        test = [s for s in samples if s[1] not in train_ids]
    elif protocol == "closed":
        train = [s for s in samples if s[2] <= train_samples_per_subject]
        test = [s for s in samples if s[2] > train_samples_per_subject]
    else:
        raise ValueError(f"Unknown protocol '{protocol}'")
    return train, test


def gallery_probe_split(samples, protocol="open", all_samples=None):
    """Gallery = first sample of each subject, probes = the rest.

    For the closed protocol the gallery is sample 1 of each subject (a training
    image, as the paper's enrolment/fine-tune sample) and the probes are the
    test samples.
    """
    if protocol == "closed":
        assert all_samples is not None
        test_subjects = {s for _, s, _ in samples}
        by_subj = defaultdict(list)
        for s in all_samples:
            if s[1] in test_subjects:
                by_subj[s[1]].append(s)
        gallery = [min(v, key=lambda s: s[2]) for _, v in sorted(by_subj.items())]
        probes = list(samples)
    else:
        by_subj = defaultdict(list)
        for s in samples:
            by_subj[s[1]].append(s)
        gallery, probes = [], []
        for _, v in sorted(by_subj.items()):
            v = sorted(v, key=lambda s: s[2])
            gallery.append(v[0])
            probes.extend(v[1:])
    return gallery, probes


def load_roi(path, img_size):
    """Load a grayscale ROI resized to img_size=(H, W), values in [0, 1], shape (1, H, W)."""
    img = Image.open(path).convert("L").resize((img_size[1], img_size[0]), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)


def standardize(x, eps=1e-6):
    """Per-image zero-mean / unit-variance normalisation (illumination robustness)."""
    return (x - x.mean(dim=(-2, -1), keepdim=True)) / (x.std(dim=(-2, -1), keepdim=True) + eps)


def train_augmentation(max_rotation=8.0, max_translate=0.06, scale=(0.95, 1.05)):
    """Mild geometric/photometric jitter emulating contactless pose variation."""
    return T.Compose([
        T.RandomAffine(degrees=max_rotation, translate=(max_translate, max_translate), scale=scale),
        T.ColorJitter(brightness=0.2, contrast=0.2),
    ])


class KnuckleDataset(Dataset):
    """Returns (image, subject_id). Images are cached in memory (ROIs are small)."""

    def __init__(self, samples, img_size=(128, 128), augment=False):
        self.samples = samples
        self.labels = [s for _, s, _ in samples]
        self.images = [load_roi(p, img_size) for p, _, _ in samples]
        self.transform = train_augmentation() if augment else None

    def __len__(self):
        return len(self.samples)

    def get_image(self, i):
        x = self.images[i]
        if self.transform is not None:
            x = self.transform(x)
        return standardize(x)

    def __getitem__(self, i):
        return self.get_image(i), self.labels[i]


class TripletDataset(Dataset):
    """Random (anchor, positive, negative) triplets, re-drawn on every access."""

    def __init__(self, base: KnuckleDataset, triplets_per_anchor=1):
        self.base = base
        self.by_label = defaultdict(list)
        for i, lab in enumerate(base.labels):
            self.by_label[lab].append(i)
        self.anchors = [i for i, lab in enumerate(base.labels) if len(self.by_label[lab]) > 1]
        self.label_set = list(self.by_label)
        if len(self.label_set) < 2:
            raise ValueError("Need at least two subjects with two or more samples to form triplets")
        self.triplets_per_anchor = triplets_per_anchor

    def __len__(self):
        return len(self.anchors) * self.triplets_per_anchor

    def __getitem__(self, idx):
        a = self.anchors[idx % len(self.anchors)]
        lab = self.base.labels[a]
        p = random.choice([j for j in self.by_label[lab] if j != a])
        neg_lab = random.choice([l for l in self.label_set if l != lab])
        n = random.choice(self.by_label[neg_lab])
        return self.base.get_image(a), self.base.get_image(p), self.base.get_image(n)


class PKBatchSampler(Sampler):
    """Batches of P subjects x K samples (for online batch-hard triplet mining)."""

    def __init__(self, labels, p=8, k=4, num_batches=None):
        self.by_label = defaultdict(list)
        for i, lab in enumerate(labels):
            self.by_label[lab].append(i)
        self.label_set = [l for l, v in self.by_label.items() if len(v) > 1]
        self.p = min(p, len(self.label_set))
        self.k = k
        self.num_batches = num_batches or max(1, len(labels) // (self.p * self.k))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for lab in random.sample(self.label_set, self.p):
                idx = self.by_label[lab]
                batch.extend(random.sample(idx, self.k) if len(idx) >= self.k
                             else random.choices(idx, k=self.k))
            yield batch
