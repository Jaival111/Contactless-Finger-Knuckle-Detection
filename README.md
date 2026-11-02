# Completely Contactless Finger Knuckle Identification: implementation

This is a PyTorch implementation of *Zhou & Kumar, "Completely Contactless and Online Finger Knuckle
Identification for Real World Applications"* (JSTSP 2023, `JSTSP2023.pdf`). It uses `Knuckle_Images/`
in place of the paper's PolyU/Tsinghua databases. That folder holds 249 segmented knuckle ROIs
(80 x 100, grayscale) from 50 subjects, with about 5 samples per subject, named `<subject>_<sample>.bmp`.

## Mapping paper → code

| Paper | Code |
|---|---|
| Fig. 1 residual feature network (RFN), fully convolutional, 1-channel template at I/4 | `knuckle/model.py` → `RFN` |
| DefConvNet baseline (deformable convs) | `--backbone defconv` |
| Eq. (3)–(6) rotation + shift alignment with bilinear interpolation, common region C, MRSD | `knuckle/alignment.py` → `RotationShiftAligner` |
| Eq. (2) RSIL and its gradients Eq. (7)–(10) (obtained through autograd: `min` routes the gradient to the minimising transform, pixels outside C get zero gradient) | `knuckle/losses.py` → `RSIL` |
| TL / SSTL loss comparison (Sec. IV-E, Fig. 10) | `--loss tl` / `--loss sstl` |
| Triplet architecture with three shared branches (Fig. 2) | `train.py` |
| ROC, EER, GAR@FAR, CMC (Sec. IV) | `knuckle/metrics.py`, `evaluate.py` |
| Cross-database evaluation (Sec. IV-B) | `evaluate.py --cross_db --data_root <other_db>` |
| Sec. III oriented knuckle detection: long-side 5-parameter OBB, CSL Eq. (11) with r = 6, angular BCE loss Eq. (13), total loss Eq. (12), rotated NMS, ROI cropping | `knuckle/detection.py` |

Table II settings are the defaults: 128 x 128 input → 32 x 32 template, shift range ±4 pixels, and
rotations of ±2° and ±4° on top of 0°.

## Usage

```bash
pip install -r requirements.txt
```

Train with RSIL. The default is the subject-disjoint "open" protocol: 60% of subjects for training and
the unseen 40% for testing.

```bash
python train.py --data_root Knuckle_Images --loss rsil --epochs 120 --out_dir runs/rsil
```

Evaluate. This writes `results.json`, `roc.png`, `cmc.png` and `scores.npz` next to the checkpoint.

```bash
python evaluate.py --checkpoint runs/rsil/last.pt
```

Useful variations:

- `--protocol closed --train_samples_per_subject 2`: samples 1–2 of every subject are used for training.
  Sample 1 is the gallery and samples 3–5 are the probes, similar to the hand-dorsal protocol.
- `--sampler batch_hard --P 8 --K 4`: online hardest-positive/hardest-negative mining by MRSD.
- `--loss tl`, `--loss sstl`: loss ablation of Fig. 10. `--backbone defconv`: the DefConvNet baseline.
- `--max_shift`, `--angles`: the search range SW = SH and Θ (also overridable at test time).

### Evaluation protocols in `evaluate.py`

- **Gallery/probe:** the first sample of each test subject is enrolled and every other sample is a probe.
  This produces the genuine/impostor scores for ROC/EER and the CMC curve.
- **All-to-all:** every pair of test images, used for verification only.

## Notes and assumptions

- The paper leaves some details unspecified, so the code uses these choices:
  - Kernel sizes: 7/5/3 for conv1/conv2/others. As a result, the parameter count (≈1.4M) is larger than
    the 0.46M reported in Table II.
  - Margin `m = 0.2`.
  - Optimiser: Adam with lr 1e-3 and a cosine schedule.
  - Augmentation: mild random affine and brightness/contrast jitter.
  - Per-image standardisation of the ROI.
- The `Knuckle_Images` ROIs are already segmented, so the detector in `knuckle/detection.py` is not needed
  for this dataset. It provides the head, loss, decoding and ROI cropping for the full system on complex
  background images. To train it, attach `OrientedDetect` to a YOLOv5 backbone/neck (in place of
  YOLOv5's `Detect`) and train with `OrientedYoloLoss` on images annotated with oriented boxes.

## Results: this implementation vs. the paper

Current results come from `runs/rsil/results.json` (trained with `python train.py --data_root Knuckle_Images --loss rsil --epochs 120 --out_dir runs/rsil`, evaluated with `python evaluate.py --checkpoint runs/rsil/last.pt`).

### Setup of the current run

| Item | Value |
|---|---|
| Dataset | `Knuckle_Images`: 249 ROIs, 50 subjects, ≤5 samples each, 80 × 100 grayscale |
| Protocol | Open set, subject-disjoint (seed 0): **30 training subjects** (~149 images), **20 test subjects** (100 images) |
| Gallery / probes | 1st sample of each test subject enrolled (20) / remaining samples as probes (80) |
| Scores | Gallery/probe: 80 genuine, 1,520 impostor. All-to-all: 200 genuine, 4,750 impostor |
| Model / loss | RFN (1.38M params, 32 × 32 template), RSIL with shifts ±4 px and angles {0, ±2°, ±4°} (405 hypotheses), margin 0.2 |
| Training | 120 epochs, Adam lr 1e-3 with cosine decay, 16 random triplets per batch, augmentation on, CUDA |

### Headline comparison

| Database / protocol | Method | EER ↓ | GAR @ FAR=10⁻² | GAR @ FAR=10⁻³ | GAR @ FAR=10⁻⁴ | Rank-1 ↑ |
|---|---|---|---|---|---|---|
| **Knuckle_Images**, open set, gallery/probe | **This implementation (RSIL)** | **13.75%** | 72.50% | 63.75% | 60.00%* | **80.00%** |
| **Knuckle_Images**, open set, all-to-all | **This implementation (RSIL)** | **12.00%** | 77.50% | 70.00% | 64.50%* | – |
| PolyU Contactless FK v3.0 (C), one session | Paper (RSIL) | 2.21% | – | – | 91.48% | – |
| PolyU Contactless FK v3.0 (C), two session | Paper (RSIL) | 16.65% | – | – | – | – |
| PolyU Hand Dorsal (B), within-DB | Paper (RSIL) | 1.33% | – | – | 96.07% | 96.07% |
| PolyU 3D FK, 2D images (D(2)) | Paper (RSIL) | 1.60% | – | – | 96.05% | 96.05% |
| PolyU 3D FK, 3D images (D(1)) | Paper (RSIL) | 1.05% | – | – | 97.10% | – |
| Tsinghua THU-FVFDT3 (E), cross-DB | Paper (RSIL) | 0.16% | – | – | 99.80% | ≈100% |
| PolyU Hand Dorsal (B), cross-DB | Paper (RSIL) | 3.33% | – | – | – | – |

\* Only 1,520 (4,750) impostor scores exist, so the smallest measurable FAR is ≈6.6 × 10⁻⁴ (2.1 × 10⁻⁴). The GAR at FAR = 10⁻⁴ is therefore the value at the lowest measurable FAR, not a true estimate at 10⁻⁴.

Paper values are from Table I and Figs. 4–9 of `JSTSP2023.pdf`; "–" means the paper does not report that number.

### Paper baselines on its main database (C, one session) for context

| Method | EER (paper) |
|---|---|
| TCCV | 17.79% |
| FKNet | 6.46% |
| DefConvNet | 5.41% |
| EfficientNetV2-S | 4.06% |
| RFN + TL (triplet loss) | 2.86% |
| RFN + SSTL (shift-only) | 2.57% |
| **RFN + RSIL (paper)** | **2.21%** |
| **RFN + RSIL (this implementation, Knuckle_Images)** | **13.75%** (gallery/probe) / **12.00%** (all-to-all) |

### Complexity

| Item | Paper (Table II) | This implementation |
|---|---|---|
| Parameters | 0.46M | 1.38M |
| Template size | 32 × 32 | 32 × 32 |
| Feature extraction time | 0.01957 s/image (RTX 2080, single image) | 0.00206 s/image (batched, local GPU) |
| Matching time | not reported | 0.13 ms per pair (405 rotation/shift hypotheses) |

The timings were measured on different hardware with different batching, so they are not directly comparable.

### Interpretation

- The current EER (12–14%) is clearly worse than the paper's within-database EERs (1–2%). It is close to the paper's hardest setting (two-session v3.0, 16.65%), but that is a different and harder protocol.
- The gap is expected given the much smaller training set: 30 subjects and about 150 images here, against hundreds of subjects in the paper. The training log supports this. By epoch 120 only about 1.7% of random triplets still violate the margin (`active` in `train_log.jsonl`), so the network has largely fitted its training triplets but has seen too few identities to generalise well.
- With 20 test subjects and 80 genuine scores, each misclassified probe moves rank-1 by 1.25 points and GAR by 1.25 points. Treat these numbers as indicative, not precise.
- The numbers are not directly comparable to the paper because the datasets, protocols and score counts all differ. The comparison shows the size of the gap, not a like-for-like benchmark.

## Differences between the paper's methodology and this implementation

| Aspect | Paper | This implementation | Effect / reason |
|---|---|---|---|
| Training / test data | PolyU v3.0 (221 subjects), Hand Dorsal (712), 3D FK (190), Tsinghua (610) | `Knuckle_Images` (50 subjects, 249 images) | Far fewer identities to learn from and to test on; the main reason for the accuracy gap |
| Protocol | Database-specific: disjoint training subjects (v3.0, 117 train / 104 test), first-sample fine-tuning (Hand Dorsal), session 1 → session 2 (3D FK), cross-DB tests | Open set, random subject-disjoint 30 / 20 split; single session; no cross-DB test run | No second session or second database is available in `Knuckle_Images` |
| ROI acquisition | YOLOv5 + CSL oriented detector segments knuckles from complex-background hand images | Images are already segmented ROIs; the detector code (`knuckle/detection.py`) is implemented but not trained or used | No hand images with oriented-box labels are available |
| Input size | 128 × 128 ROI (32 × 32 template per Table II) | 80 × 100 ROIs resized to 128 × 128 (aspect ratio changed) | Keeps the paper's 32 × 32 template; slight geometric distortion |
| Backbone kernels | Not specified; 0.46M params | conv1 7×7, conv2 5×5, others 3×3, BatchNorm + ReLU; 1.38M params | Unknown details filled with standard RFN choices; network is ~3× larger |
| RSIL search range | Shift 4 px, "rotation with 4 angles" | Shift ±4 px (step 1), angles {−4, −2, 0, +2, +4}° | Exact angle values are not given in the paper |
| RSIL gradients | Derived by hand (Eqs. 7–10) | PyTorch autograd through `grid_sample` + `min` | Mathematically equivalent: gradient flows only through the minimising transform and the common region |
| Distance | Pixel-wise MSE over common region | Same; common region = pixels whose bilinear support is fully inside the warped template; transforms with <50% overlap are ignored | `min_overlap` is an added safeguard |
| Margin, optimiser, epochs, batch size | Not specified | m = 0.2, Adam 1e-3 + cosine, 120 epochs, batch 16 random triplets | Unspecified hyperparameters; not tuned |
| Triplet selection | Not specified | Random triplets (optional batch-hard mining with `--sampler batch_hard`) | Random sampling is the simplest choice; hard mining may help |
| Preprocessing / augmentation | Not specified for matching | Per-image standardisation; random affine (±8°, ±6% shift, 0.95–1.05 scale) and brightness/contrast jitter | Compensates for the small training set |
| Baselines | TCCV, FKNet, EfficientNetV2-S, DefConvNet, TL, SSTL | DefConvNet (`--backbone defconv`), TL (`--loss tl`) and SSTL (`--loss sstl`) implemented but not yet run; TCCV, FKNet, EfficientNetV2-S not implemented | Baseline numbers above are quoted from the paper |
| Online system | Cross-platform live identification app | Not implemented | Out of scope for offline evaluation |
