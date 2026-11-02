"""Evaluate a trained model: MRSD matching, ROC / EER / GAR@FAR and CMC.

Example
-------
    python evaluate.py --checkpoint runs/rsil/last.pt

Templates are matched with the same minimal rotational and shifted distance
(MRSD, Eq. 3) that is used by RSIL during training. The rotation/shift search
range can be overridden at test time with --max_shift / --angles.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from knuckle.alignment import build_aligner
from knuckle.dataset import gallery_probe_split, load_roi, make_split, scan_dataset, standardize
from knuckle.metrics import (all_to_all_scores, cmc_curve, gallery_probe_scores, plot_cmc, plot_roc,
                             verification_report)
from knuckle.model import build_model


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", default=None, help="defaults to the training data_root")
    ap.add_argument("--out_dir", default=None, help="defaults to the checkpoint directory")
    ap.add_argument("--cross_db", action="store_true",
                    help="evaluate on ALL images of --data_root (cross-database protocol, Sec. IV-B)")
    ap.add_argument("--match_loss", choices=["rsil", "sstl", "tl"], default=None,
                    help="alignment used for matching (defaults to the training loss)")
    ap.add_argument("--max_shift", type=int, default=None)
    ap.add_argument("--angles", type=float, nargs="+", default=None)
    ap.add_argument("--max_rank", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


@torch.no_grad()
def extract_templates(model, samples, img_size, device, batch_size=64):
    out = []
    for i in range(0, len(samples), batch_size):
        x = torch.stack([standardize(load_roi(p, img_size)) for p, _, _ in samples[i:i + batch_size]])
        out.append(model(x.to(device)))
    return torch.cat(out)


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    targs = ckpt["args"]
    data_root = args.data_root or targs["data_root"]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)
    img_size = (targs["img_h"], targs["img_w"])

    model = build_model(targs["backbone"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    aligner = build_aligner(
        args.match_loss or targs["loss"],
        args.max_shift if args.max_shift is not None else targs["max_shift"],
        args.angles if args.angles is not None else targs["angles"],
        targs["shift_step"], targs["min_overlap"]).to(device)

    samples = scan_dataset(data_root)
    if args.cross_db:
        test_samples, protocol = samples, "open"
    else:
        protocol = targs["protocol"]
        # rebuild exactly the split stored with the checkpoint
        test_names = set(ckpt["split"]["test"])
        test_samples = [s for s in samples if os.path.basename(s[0]) in test_names]
        if not test_samples:
            _, test_samples = make_split(samples, protocol, targs["num_train_subjects"], targs["train_ratio"],
                                         targs["train_samples_per_subject"], targs["seed"])

    gallery, probes = gallery_probe_split(test_samples, protocol, all_samples=samples)
    print(f"Matching with {aligner.num_transforms} rotation/shift hypotheses | "
          f"gallery {len(gallery)} | probes {len(probes)}")

    t0 = time.time()
    g_t = extract_templates(model, gallery, img_size, device, args.batch_size)
    p_t = extract_templates(model, probes, img_size, device, args.batch_size)
    t_feat = (time.time() - t0) / (len(gallery) + len(probes))

    g_lab = np.array([s for _, s, _ in gallery])
    p_lab = np.array([s for _, s, _ in probes])

    t0 = time.time()
    D = aligner.pairwise(p_t, g_t).cpu().numpy()
    t_match = (time.time() - t0) / D.size

    genuine, impostor = gallery_probe_scores(D, p_lab, g_lab)
    report_gp, (far, gar) = verification_report(genuine, impostor)
    cmc = cmc_curve(D, p_lab, g_lab, args.max_rank)

    results = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "data_root": os.path.abspath(data_root),
        "protocol": "cross_db" if args.cross_db else protocol,
        "alignment": aligner.config,
        "gallery_probe": report_gp,
        "rank1": float(cmc[0]),
        "cmc": cmc.tolist(),
        "feature_extraction_s_per_image": t_feat,
        "matching_s_per_pair": t_match,
    }
    roc_curves = {"Gallery/probe": (far, gar, report_gp["EER"])}

    # all-to-all matching among every test image (verification only)
    all_t = torch.cat([g_t, p_t])
    all_lab = np.concatenate([g_lab, p_lab])
    D_all = aligner.pairwise(all_t, all_t).cpu().numpy()
    D_all = np.minimum(D_all, D_all.T)
    gen_all, imp_all = all_to_all_scores(D_all, all_lab)
    report_all, (far_a, gar_a) = verification_report(gen_all, imp_all)
    results["all_to_all"] = report_all
    roc_curves["All-to-all"] = (far_a, gar_a, report_all["EER"])

    plot_roc(roc_curves, os.path.join(out_dir, "roc.png"))
    plot_cmc({"Ours (RSIL)": cmc}, os.path.join(out_dir, "cmc.png"))
    np.savez(os.path.join(out_dir, "scores.npz"), D=D, probe_labels=p_lab, gallery_labels=g_lab,
             genuine=genuine, impostor=impostor, genuine_all=gen_all, impostor_all=imp_all)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({k: v for k, v in results.items() if k != "cmc"}, indent=2))
    print(f"Results, ROC/CMC plots and scores written to {out_dir}")


if __name__ == "__main__":
    main()
