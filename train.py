"""Train the RFN backbone with the Rotation and Shift Invariant Loss (RSIL).

Example
-------
    python train.py --data_root Knuckle_Images --loss rsil --epochs 120

The three shared-weight branches of the triplet architecture (Fig. 2) are
implemented by passing anchor, positive and negative images through the same
network in one batch.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from knuckle.alignment import build_aligner
from knuckle.dataset import (KnuckleDataset, PKBatchSampler, TripletDataset, make_split,
                             scan_dataset)
from knuckle.losses import RSIL, mine_batch_hard
from knuckle.model import build_model, count_parameters


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="Knuckle_Images")
    ap.add_argument("--out_dir", default="runs/rsil")
    # data / protocol
    ap.add_argument("--protocol", choices=["open", "closed"], default="open")
    ap.add_argument("--num_train_subjects", type=int, default=None,
                    help="open protocol: number of training subjects (default: train_ratio of all)")
    ap.add_argument("--train_ratio", type=float, default=0.6)
    ap.add_argument("--train_samples_per_subject", type=int, default=2,
                    help="closed protocol: samples 1..k of each subject are used for training")
    ap.add_argument("--img_h", type=int, default=128)
    ap.add_argument("--img_w", type=int, default=128)
    ap.add_argument("--no_augment", action="store_true")
    # model / loss
    ap.add_argument("--backbone", choices=["rfn", "defconv"], default="rfn")
    ap.add_argument("--loss", choices=["rsil", "sstl", "tl"], default="rsil")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--max_shift", type=int, default=4, help="SW = SH, in template pixels")
    ap.add_argument("--angles", type=float, nargs="+", default=[-4.0, -2.0, 0.0, 2.0, 4.0],
                    help="rotation angles (degrees) searched by RSIL, |theta| <= Theta")
    ap.add_argument("--shift_step", type=int, default=1)
    ap.add_argument("--min_overlap", type=float, default=0.5)
    # optimisation
    ap.add_argument("--sampler", choices=["random", "batch_hard"], default="random")
    ap.add_argument("--batch_size", type=int, default=16, help="triplets per batch (random sampler)")
    ap.add_argument("--triplets_per_anchor", type=int, default=4)
    ap.add_argument("--P", type=int, default=8, help="subjects per batch (batch_hard sampler)")
    ap.add_argument("--K", type=int, default=4, help="samples per subject (batch_hard sampler)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer, epoch, args, split):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
        "split": split,
    }, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    samples = scan_dataset(args.data_root)
    train_samples, test_samples = make_split(
        samples, args.protocol, args.num_train_subjects, args.train_ratio,
        args.train_samples_per_subject, args.seed)
    split = {
        "train": [os.path.basename(p) for p, _, _ in train_samples],
        "test": [os.path.basename(p) for p, _, _ in test_samples],
    }
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=1)
    print(f"{len(samples)} images | train: {len(train_samples)} images of "
          f"{len({s for _, s, _ in train_samples})} subjects | test: {len(test_samples)} images of "
          f"{len({s for _, s, _ in test_samples})} subjects")

    img_size = (args.img_h, args.img_w)
    base = KnuckleDataset(train_samples, img_size, augment=not args.no_augment)
    if args.sampler == "random":
        loader = DataLoader(TripletDataset(base, args.triplets_per_anchor), batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers, drop_last=True)
    else:
        loader = DataLoader(base, batch_sampler=PKBatchSampler(base.labels, args.P, args.K),
                            num_workers=args.num_workers)

    model = build_model(args.backbone).to(device)
    aligner = build_aligner(args.loss, args.max_shift, args.angles, args.shift_step, args.min_overlap).to(device)
    criterion = RSIL(aligner, args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"backbone={args.backbone} ({count_parameters(model) / 1e6:.2f}M params) | loss={args.loss} "
          f"| {aligner.num_transforms} rotation/shift hypotheses")

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot, n, agg = 0.0, 0, {"d_ap": 0.0, "d_an": 0.0, "active": 0.0}
        for batch in loader:
            if args.sampler == "random":
                xa, xp, xn = (b.to(device) for b in batch)
                B = xa.shape[0]
                t = model(torch.cat([xa, xp, xn]))
                ta, tp, tn = t[:B], t[B:2 * B], t[2 * B:]
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                t = model(x)
                a_idx, p_idx, n_idx = mine_batch_hard(t.detach(), y, aligner)
                if len(a_idx) == 0:
                    continue
                ta, tp, tn = t[a_idx], t[p_idx], t[n_idx]

            loss, stats = criterion(ta, tp, tn)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            tot += loss.item()
            n += 1
            for k in agg:
                agg[k] += stats[k]
        scheduler.step()

        rec = {"epoch": epoch, "loss": tot / max(n, 1), "lr": scheduler.get_last_lr()[0],
               "time_s": round(time.time() - t0, 2)}
        rec.update({k: v / max(n, 1) for k, v in agg.items()})
        print(json.dumps(rec))
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(os.path.join(args.out_dir, "last.pt"), model, optimizer, epoch, args, split)
    print(f"Saved checkpoint to {os.path.join(args.out_dir, 'last.pt')}")


if __name__ == "__main__":
    main()
