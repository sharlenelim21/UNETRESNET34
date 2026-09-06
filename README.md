# Cardiac Landmark Detection Model

This project trains and runs a deep learning model for detecting two cardiac
landmarks (RV insertion points: LM1 = superior/anterior, LM2 =
inferior/posterior) from 2-D slices extracted from NIfTI cardiac MRI volumes.
The model predicts two heatmaps, one per landmark, then converts the heatmap
peaks back into `(x1, y1, x2, y2)` pixel coordinates.

The main model is a ResNet-34 encoder with a UNet-style decoder, CBAM
attention, optional deep supervision, an optional auxiliary segmentation
head, and subpixel post-processing.

## Important Note

The current loaders follow the same general NIfTI 2-D slice loading concept
across all three datasets (`lv-landmark`, `acdc`, `rv_landmark`), but each is
customized for its own data layout. They do not use the MONAI preprocessing
pipeline.

## Cross-Domain Goal

The project trains on the **ACDC** dataset and aims to generalize
**zero-shot** to the **rv_landmark** dataset (and, ultimately, to arbitrary
user-uploaded scans) — i.e. train on one domain, work well on an unseen one.
This drives several design choices documented below: GroupNorm instead of
BatchNorm, robust percentile normalization instead of z-score, aggressive
segmentation-channel dropout, an auxiliary anatomy segmentation head, and
EMA teacher-student self-training on the target domain.

| Stage | Result (last measured) |
| --- | --- |
| Within-domain (ACDC val/test) | SDR@5 = 87.5%, MRE = 4.03px |
| Cross-domain (rv_landmark, before fixes) | SDR@5 = 21.2%, MRE = 15.49px |

## Project Structure

```text
UNETRESNET34/
|-- data/
|   |-- lv-landmark/         # original single-domain dataset
|   |   |-- Training/{images,masks}/
|   |   `-- Testing/{images,masks}/
|   |-- acdc/                # ACDC dataset (training domain)
|   |   |-- images/  masks/  points/
|   `-- rv_landmark/         # cross-domain target dataset
|       |-- train_images/  train_gt/  train_seg_multi/
|       |-- test_images/   test_gt/   test_seg_multi/
|       `-- pseudo_gt/       # written by pseudo_label_rv.py
|-- dataset/
|   |-- landmark_dataset.py       # lv-landmark loader (1-channel)
|   |-- acdc_landmark_dataset.py  # ACDC loader (1 or 2-channel, seg dropout, aux seg target)
|   `-- rv_landmark_dataset.py    # rv_landmark loader (1 or 2-channel)
|-- models/
|   |-- unet_resnet34.py     # ResNet34 + UNet, CBAM, deep supervision, aux seg head
|   `-- cyclegan.py          # CycleGAN for style-transfer experiments (see Notes)
|-- utils/
|   |-- heatmap.py           # Coordinates -> Gaussian heatmaps
|   |-- loss.py              # BCE + Dice + Wing (coord) + separation loss
|   |-- metrics.py           # MRE, SDR, per-landmark error, percentile metrics
|   |-- normalize.py         # Robust percentile normalization (cross-domain stable)
|   |-- postprocess.py       # Heatmap -> coordinates (soft/hard/subpixel argmax)
|   `-- visualize.py         # Prediction grids and training curves
|-- train.py                     # Original single-domain (lv-landmark) trainer
|-- finetune.py                  # Fine-tunes an lv-landmark checkpoint
|-- inference.py                 # Inference on lv-landmark volumes
|-- tta_eval.py                  # TTA evaluation for lv-landmark checkpoints
|-- train_acdc_1ch.py             # ACDC trainer, MRI-only input (fallback model)
|-- train_acdc_2ch.py             # ACDC trainer, MRI + seg-mask input (+ aux seg head)
|-- finetune_rv.py                 # Fine-tune an ACDC checkpoint on real rv_landmark labels
|-- pseudo_label_rv.py             # Generate pseudo-labels on rv_landmark train images
|-- finetune_rv_pseudo.py          # Fine-tune with real + pseudo + ACDC-replay data (weighted)
|-- finetune_rv_ema.py             # EMA teacher-student self-training (recommended)
|-- inference_rv.py                # Inference / evaluation on rv_landmark
|-- inference_acdc.py              # Inference / evaluation on ACDC
|-- ensemble_rv.py                 # Multi-checkpoint ensembling experiment
|-- generate_translated_dataset.py # CycleGAN-translated ACDC->RV style dataset
|-- train_cyclegan.py              # Trains the CycleGAN used above
|-- finetune_translated.py         # Fine-tune on CycleGAN-translated data
|-- finetune_mixed.py              # Fine-tune on a mixed multi-domain dataset
|-- check_points.py                 # Small checkpoint inspection utility
|-- requirements.txt
`-- notebooks/
    `-- run_landmark_model.ipynb
```

## What Each File Does

| File | Purpose |
| --- | --- |
| `train.py` | Original single-domain trainer for `lv-landmark`. Loads data, splits train/val, trains in phases, saves checkpoints to `checkpoints/`. |
| `finetune.py` | Fine-tunes a saved `lv-landmark` checkpoint with tighter loss settings. |
| `inference.py` | Predicts landmarks on an `lv-landmark` NIfTI volume (single slice, auto slice, or all slices). |
| `tta_eval.py` | Evaluates an `lv-landmark` checkpoint with test-time augmentation. |
| `train_acdc_1ch.py` | Trains on ACDC with **MRI-only** input. This is the fallback model used when a segmentation mask is missing or low quality at inference time. |
| `train_acdc_2ch.py` | Trains on ACDC with **MRI + segmentation mask** (2-channel) input. Supports the auxiliary anatomy segmentation head (`--seg-aux-weight`, `--seg-classes`) and seg-channel dropout (`--seg-dropout-prob`) for cross-domain robustness. |
| `finetune_rv.py` | Fine-tunes an ACDC checkpoint directly on the small labelled `rv_landmark` train split. |
| `pseudo_label_rv.py` | Runs TTA inference on `rv_landmark` train images with a trained checkpoint and writes confident predictions as pseudo-GT NIfTI files. |
| `finetune_rv_pseudo.py` | Fine-tunes using real `rv_landmark` GT + pseudo-labels + ACDC replay simultaneously, each with its own loss weight. |
| `finetune_rv_ema.py` | **Recommended cross-domain fine-tuning path.** Teacher-student EMA self-training: a teacher (EMA of the student) generates pseudo-heatmaps on the fly each step, with a confidence mask gating out low-certainty targets. Avoids the noise accumulation seen with discrete pseudo-label rounds. |
| `inference_rv.py` | Inference / full test-set evaluation on `rv_landmark`, with TTA and optional seg-mask input. |
| `inference_acdc.py` | Inference / evaluation on ACDC. |
| `ensemble_rv.py` | Ensembles multiple trained checkpoints on `rv_landmark`. |
| `train_cyclegan.py` / `generate_translated_dataset.py` / `finetune_translated.py` | CycleGAN-based style-transfer experiment: translate ACDC images toward RV appearance and fine-tune on the translated set. In practice this **corrupted RV geometry** and underperformed (see Notes). |
| `finetune_mixed.py` | Fine-tunes on a combined multi-domain dataset. |
| `dataset/landmark_dataset.py` | `lv-landmark` loader: reads `.nii.gz` volumes, extracts slices/landmarks, robust percentile normalization, augmentation, heatmap targets. |
| `dataset/acdc_landmark_dataset.py` | ACDC loader. 1 or 2-channel input, robust percentile normalization, seg-channel dropout, MRI-specific augmentations (bias field, blur, histogram perturb), and an optional `return_seg=True` mode that also returns the anatomy mask for the auxiliary seg head. |
| `dataset/rv_landmark_dataset.py` | `rv_landmark` loader: combined-heatmap GT with connected-component blob extraction, optional 2-channel seg input, robust percentile normalization, patient-level train/val split (`split_volumes`). |
| `models/unet_resnet34.py` | ResNet34-UNet architecture: CBAM attention, deep-supervision aux heads, optional InstanceNorm/GroupNorm swap, optional auxiliary segmentation head (`seg_classes`), cardiac-pretrained encoder loading, fallback UNet if `torchvision` is unavailable. |
| `models/cyclegan.py` | CycleGAN generator/discriminator used for the style-transfer experiment. |
| `utils/heatmap.py` | Converts landmark coordinates to Gaussian heatmaps. |
| `utils/loss.py` | Combined loss: per-landmark-weighted BCE + Dice + Wing (coordinate) + separation loss. Accepts an optional exact `gt_coords` target instead of deriving it from the heatmap via soft-argmax. |
| `utils/metrics.py` | MRE, per-landmark MRE, SDR at multiple pixel thresholds, per-sample MRE, percentiles. |
| `utils/normalize.py` | **Robust percentile normalization** — clips each volume to its [0.5, 99.5] percentile range and scales to [0,1]. Used everywhere instead of per-volume/per-slice z-score, because z-score statistics are dominated by background/FOV and do not transfer across scanners. |
| `utils/postprocess.py` | Converts predicted heatmaps to coordinates: soft argmax (differentiable, used in the loss), hard argmax, Gaussian subpixel argmax (used at inference/eval), quadratic subpixel argmax. |
| `utils/visualize.py` | Saves validation prediction grids and training-curve plots. |
| `requirements.txt` | Python dependencies. |

## Cross-Domain Design Changes

These changes exist specifically to make an ACDC-trained model generalize to
`rv_landmark` and to unseen scanners in general:

1. **Robust percentile normalization** (`utils/normalize.py`) instead of
   per-volume/per-slice z-score, used consistently in every dataset loader
   *and* in `inference_rv.py` / `pseudo_label_rv.py`. z-score statistics are
   dominated by background/FOV, which differs across scanners; percentile
   clipping is far more stable across acquisition protocols. This also fixed
   a real bug: `pseudo_label_rv.py` previously normalized per-slice while
   training normalized per-volume, silently corrupting every pseudo-label.

2. **GroupNorm instead of BatchNorm** (default `use_group_norm=True` in
   `train_acdc_1ch.py`, `train_acdc_2ch.py`, `inference_rv.py`, use
   `--no-group-norm` to opt out). BatchNorm's running statistics are computed
   on the training domain and do not transfer to an unseen scanner; GroupNorm
   has no running stats, so there's nothing domain-specific to carry over.

3. **Aggressive segmentation-channel dropout** (`--seg-dropout-prob`,
   default 0.5 in `train_acdc_2ch.py`). Zeroes the input seg-mask channel on
   half of training batches so the 2-channel model cannot become dependent on
   a segmentation mask that may be missing or unreliable on a real user
   upload.

4. **Exact-coordinate loss target** (`utils/loss.py`). The coordinate Wing
   loss now uses the true GT coordinates (passed in by the training loop)
   rather than re-deriving them from the target heatmap via soft-argmax.

5. **Auxiliary anatomy segmentation head** (`models/unet_resnet34.py`,
   `--seg-aux-weight`/`--seg-classes` in `train_acdc_2ch.py`). Predicts
   RV/myocardium/LV segmentation from the shared decoder feature, forcing
   the encoder to learn scanner-invariant anatomical shape rather than
   ACDC-specific texture. Disabled by default (`seg_classes=0`) so existing
   checkpoints without this head still load; checkpoints trained with it load
   into inference scripts via `strict=False` (the extra `seg_head.*` keys are
   simply ignored where the head isn't needed).

6. **EMA teacher-student self-training** (`finetune_rv_ema.py`). Instead of
   discrete pseudo-label rounds (which regressed after round 3 as label noise
   accumulated), a teacher model — an exponential moving average of the
   student — generates pseudo-heatmaps every step, so targets co-evolve with
   the model instead of going stale. A confidence mask (`--conf-threshold`)
   discards low-certainty teacher predictions before they can be learned.

## Data Format

`lv-landmark` (original dataset):

```text
data/lv-landmark/
|-- Training/
|   |-- images/DET0000101.nii.gz
|   `-- masks/DET0000101.nii.gz
`-- Testing/
    |-- images/DET0000301.nii.gz
    `-- masks/DET0000301.nii.gz
```

`acdc` (training domain for the cross-domain pipeline):

```text
data/acdc/
|-- images/patient001_frame01.nii.gz ...
|-- masks/patient001_frame01.nii.gz ...   # multi-class seg (0=bg,1=RV,2=myo,3=LV)
`-- points/patient001_frame01.nii.gz ...  # 4-D (H,W,S,2): ch0=LM1, ch1=LM2 binary masks
```

`rv_landmark` (cross-domain target dataset):

```text
data/rv_landmark/
|-- train_images/  train_gt/  train_seg_multi/
`-- test_images/   test_gt/   test_seg_multi/
```

For each image volume, the corresponding mask/GT file must share the same
filename and shape.

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch does not detect your GPU, reinstall PyTorch using the command
from the official PyTorch selector for your CUDA version.

### 3. Verify Python, PyTorch, and CUDA

```powershell
python -c "import sys, torch, torchvision; print(sys.executable); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected result for GPU training:

```text
cuda available: True
gpu: NVIDIA GeForce ...
```

## Workflow A — Original Single-Domain (`lv-landmark`)

```powershell
python test_dataset.py                # smoke test
python train.py                        # train
python finetune.py --checkpoint checkpoints/<run_name>/best_model.pth
python inference.py --checkpoint checkpoints/<run_name>/best_model.pth --image data/lv-landmark/Testing/images/DET0000301.nii.gz --auto --out inference_results
```

Training outputs:

```text
checkpoints/YYYY-MM-DD_HH-MM-SS/
|-- best_p2.pth
|-- best_model.pth
|-- last_model.pth
|-- training_curve.png
`-- grids/
```

## Workflow B — Cross-Domain (ACDC -> rv_landmark)

### 1. Train the ACDC base model

2-channel (MRI + seg mask), with GroupNorm and the auxiliary seg head on by
default:

```powershell
python train_acdc_2ch.py --seg-aux-weight 0.5
```

Or the 1-channel fallback model (used when a reliable seg mask isn't
available at inference time):

```powershell
python train_acdc_1ch.py
```

Checkpoints are saved to:

```text
acdc-checkpoints/<run_tag>_<timestamp>/
|-- best_p2.pth       # best checkpoint at end of phase 2 (curriculum)
|-- best_model.pth    # best checkpoint at end of phase 3 (precision squeeze) — use this one
|-- last_model.pth
|-- training_curve.png
|-- config.json
|-- results.json      # test-set metrics for this run
`-- grids/
```

`<run_tag>` reflects the settings used, e.g. `acdc_2ch_groupnorm`,
`acdc_2ch_instnorm`, `acdc_1ch`. Use `--no-group-norm` to train with
BatchNorm instead (not recommended for cross-domain).

### 2. Adapt to rv_landmark — recommended: EMA self-training

```powershell
python finetune_rv_ema.py --base-checkpoint acdc-checkpoints/acdc_2ch_groupnorm_<TIMESTAMP>/best_model.pth --in-channels 2 --group-norm --epochs 40 --conf-threshold 0.5 --ema-decay 0.99
```

Key flags:

| Flag | Meaning |
| --- | --- |
| `--ema-decay` | Teacher EMA decay (higher = smoother/slower-moving teacher). Default 0.99. |
| `--conf-threshold` | Minimum teacher heatmap peak for a prediction to count toward the consistency loss. Raise if training is unstable, lower if too few batches are contributing (the per-epoch log prints `con-batches=X/Y`). |
| `--consistency-weight` / `--consistency-rampup` | Weight and linear ramp-up (in epochs) for the consistency loss, so an early, still-noisy teacher isn't trusted too much. |
| `--acdc-weight` | Loss weight for ACDC replay samples mixed into training (keeps source-domain knowledge; 0 disables replay). |

Outputs:

```text
rv-checkpoints/finetune_ema_<in_channels>ch_<timestamp>/
|-- best_model.pth      # best of student/teacher by val P90 MRE — use this one
|-- last_student.pth
|-- last_teacher.pth
|-- history.json
|-- training_curve.png
`-- results.json
```

### 2 (alternative) — Discrete pseudo-labeling / direct fine-tune

These older paths are kept for comparison; EMA self-training generally beats
them and does not show the round-3 regression seen with repeated
pseudo-labeling:

```powershell
# Fine-tune directly on the small labelled rv_landmark split
python finetune_rv.py --checkpoint acdc-checkpoints/<run>/best_model.pth --in-channels 2

# Or: generate pseudo-labels, then fine-tune on real+pseudo+ACDC-replay
python pseudo_label_rv.py --checkpoint rv-checkpoints/<run>/best_model.pth --in-channels 2 --threshold 0.7
python finetune_rv_pseudo.py --base-checkpoint rv-checkpoints/<run>/best_model.pth --in-channels 2 --pseudo-weight 0.4 --acdc-weight 0.15
```

### 3. Evaluate on rv_landmark

```powershell
python inference_rv.py --checkpoint rv-checkpoints/finetune_ema_2ch_<TIMESTAMP>/best_model.pth --in-channels 2 --seg-dir data/rv_landmark/test_seg_multi --eval
```

Add `--no-group-norm` only if the checkpoint being loaded was trained with
BatchNorm. Drop `--seg-dir` to evaluate in MRI-only mode (the 2-channel
model automatically falls back to a blank seg channel).

## Option — Run in a Notebook

An `.ipynb` notebook is useful if you don't have a local NVIDIA GPU. Google
Colab or Kaggle with GPU enabled work well.

```text
notebooks/run_landmark_model.ipynb
```

1. Open the notebook in Colab/Kaggle/Jupyter/VS Code.
2. Enable GPU runtime (`Runtime -> Change runtime type -> GPU`).
3. Install requirements.
4. Upload/mount the dataset so the folder structure matches the layouts above.
5. Run `test_dataset.py`, then `train.py` (or `train_acdc_2ch.py`), then the
   relevant inference script.

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

## Common Issues

### VS Code says `Import "torchvision.models" could not be resolved`

VS Code/Pylance is likely using the wrong interpreter.

1. `Ctrl+Shift+P` -> `Python: Select Interpreter` -> choose `.\.venv\Scripts\python.exe`
2. `Ctrl+Shift+P` -> `Developer: Reload Window`

### CUDA is not available

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `False`: you likely installed a CPU-only PyTorch build, your
NVIDIA driver is missing/outdated, you're using the wrong environment, or
the machine has no NVIDIA GPU.

### `No valid samples found`

Check that image/mask (or GT/points) directories exist, filenames match
exactly, volumes share the same shape, and masks contain the expected
foreground labels on at least some slices.

### Checkpoint fails to load with unexpected/missing keys

- If loading an older checkpoint into a model built with `seg_classes>0`, or
  a seg-head checkpoint into a model with `seg_classes=0`: this is expected.
  `inference_rv.py` and `pseudo_label_rv.py` already load with
  `strict=False` and only report non-`seg_head` mismatches.
- If loading a GroupNorm checkpoint without `--group-norm` (or vice versa
  with `--no-group-norm`), the state dict will not match — pass the matching
  norm flag used at training time.

### `AttributeError: Can't pickle local object` when training on Windows

Windows `DataLoader` workers use `spawn`, which requires every dataset class
to be importable at module level — a class defined *inside* a function
cannot be pickled. If you add custom dataset wrappers, define them at module
scope (see `TaggedDataset` / `TaggedConcatDataset` in `finetune_rv_ema.py`
for the pattern), or run with `num_workers=0`.

### Windows Unicode print errors

```text
UnicodeEncodeError: 'charmap' codec can't encode character
```

Run Python with UTF-8 enabled:

```powershell
$env:PYTHONUTF8="1"
python train.py
```

## Notes

- Model input size is fixed at `256 x 256`.
- Training uses heatmap targets with a sigma curriculum (large sigma early
  for easier optimization, small sigma late for subpixel precision).
- Validation reports MRE, per-landmark MRE, SDR at 2/5/10 pixels, and MRE
  percentiles. Checkpoints are selected using **P90 MRE** so hard validation
  samples matter, not only the mean error.
- The CycleGAN-based style-transfer path (`train_cyclegan.py`,
  `generate_translated_dataset.py`, `finetune_translated.py`) was evaluated
  as a cross-domain approach but corrupted RV landmark geometry in practice
  and underperformed the normalization/GroupNorm/EMA approach above; it is
  kept in the repo for reference, not as the recommended path.
- Ensembling (`ensemble_rv.py`) of independently-trained checkpoints was also
  evaluated and underperformed due to correlated errors across models trained
  on the same source domain.
