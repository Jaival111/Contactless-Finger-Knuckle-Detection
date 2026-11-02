"""Verification (ROC / EER / GAR@FAR) and identification (CMC) metrics.

Scores are MRSD distances: smaller means more similar.
"""
import numpy as np
from sklearn.metrics import roc_curve


def roc_from_distances(genuine, impostor):
    """Return FAR, GAR and the distance thresholds."""
    genuine, impostor = np.asarray(genuine, float), np.asarray(impostor, float)
    y = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    scores = -np.concatenate([genuine, impostor])
    far, gar, thr = roc_curve(y, scores)
    return far, gar, -thr


def equal_error_rate(far, gar):
    frr = 1.0 - gar
    diff = far - frr
    i = np.argmax(diff >= 0)
    if i == 0:
        return float((far[0] + frr[0]) / 2)
    # linear interpolation between the two ROC points around the crossing
    x0, x1 = diff[i - 1], diff[i]
    w = x0 / (x0 - x1) if x0 != x1 else 0.0
    return float(far[i - 1] + w * (far[i] - far[i - 1]))


def gar_at_far(far, gar, target):
    return float(np.interp(target, far, gar))


def verification_report(genuine, impostor, far_targets=(1e-2, 1e-3, 1e-4)):
    far, gar, _ = roc_from_distances(genuine, impostor)
    report = {
        "num_genuine": int(len(genuine)),
        "num_impostor": int(len(impostor)),
        "EER": equal_error_rate(far, gar),
    }
    for t in far_targets:
        report[f"GAR@FAR={t:g}"] = gar_at_far(far, gar, t)
    return report, (far, gar)


def gallery_probe_scores(D, probe_labels, gallery_labels):
    """Split a (num_probes, num_gallery) distance matrix into genuine / impostor scores."""
    probe_labels = np.asarray(probe_labels)
    gallery_labels = np.asarray(gallery_labels)
    same = probe_labels[:, None] == gallery_labels[None, :]
    return D[same], D[~same]


def all_to_all_scores(D, labels):
    """Genuine / impostor scores from the upper triangle of a square distance matrix."""
    labels = np.asarray(labels)
    iu = np.triu_indices(len(labels), k=1)
    same = (labels[:, None] == labels[None, :])[iu]
    d = D[iu]
    return d[same], d[~same]


def cmc_curve(D, probe_labels, gallery_labels, max_rank=10):
    """Cumulative match characteristic: fraction of probes whose identity is within rank r."""
    probe_labels = np.asarray(probe_labels)
    gallery_labels = np.asarray(gallery_labels)
    order = np.argsort(D, axis=1)
    ranked = gallery_labels[order]
    hits = ranked == probe_labels[:, None]
    has_match = hits.any(axis=1)
    first = np.where(has_match, hits.argmax(axis=1), np.iinfo(np.int64).max)
    max_rank = min(max_rank, D.shape[1])
    return np.array([(first < r).mean() for r in range(1, max_rank + 1)])


def plot_roc(curves, path, title="Receiver Operating Characteristics"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (far, gar, eer) in curves.items():
        keep = far > 0
        ax.semilogx(far[keep], gar[keep], label=f"{name}; EER: {100 * eer:.2f}%")
    ax.set_xlabel("False Acceptance Rate")
    ax.set_ylabel("Genuine Acceptance Rate")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cmc(curves, path, title="Cumulative Match Characteristics"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, cmc in curves.items():
        ax.plot(np.arange(1, len(cmc) + 1), cmc, marker="o", label=f"{name}; Rank-1: {100 * cmc[0]:.2f}%")
    ax.set_xlabel("Cumulative Rank")
    ax.set_ylabel("Recognition Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
