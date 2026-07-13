"""
finetune_rv_ema.py — Teacher-student EMA self-training for RV insertion points
==============================================================================
Fixes the failure mode you saw with round-based pseudo-labelling (Round 3
regressed to 17.4% as noise accumulated). Instead of freezing pseudo-labels
from a past checkpoint, a *teacher* model — an exponential moving average (EMA)
of the student — produces pseudo-heatmaps ON THE FLY each step, so the labels
co-evolve with the model and never go stale. A confidence mask discards
low-certainty teacher predictions so garbage is never learned.

Losses on the student:
  1. Supervised loss on labelled data:
        - real rv_landmark train GT   (weight 1.0)
        - ACDC replay                 (weight --acdc-weight, keeps source knowledge)
  2. Consistency loss (student vs teacher) on the SAME rv_landmark images,
     student sees a strongly-augmented view, teacher sees a weakly-augmented
     view. Only applied where the teacher heatmap peak > --conf-threshold.

Why this beats discrete pseudo-label rounds:
  - Teacher is a temporal ensemble → smoother, lower-variance targets.
  - Confidence masking → noisy target dataset annotations can't dominate.
  - No round boundaries → no place for noise to accumulate and regress.

Normalisation is handled entirely inside the datasets (robust per-volume
percentiles), so train and this script are consistent by construction.

Usage:
    python finetune_rv_ema.py \\
        --base-checkpoint acdc-checkpoints/acdc_2ch_groupnorm_.../best_model.pth \\
        --in-channels 2 --group-norm \\
        --epochs 40 --conf-threshold 0.5 --ema-decay 0.99
"""

import argparse
import copy
import json
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from dataset.rv_landmark_dataset import RVLandmarkDataset, split_volumes
from dataset.acdc_landmark_dataset import ACDCLandmarkDataset
from models.unet_resnet34 import UNetResNet34
from utils.loss import HeatmapLoss
from utils.postprocess import gaussian_subpixel_argmax
from utils.metrics import (
    compute_mre, compute_mre_per_landmark,
    compute_sdr_multi, compute_per_sample_mre, compute_mre_percentiles,
)
from utils.visualize import save_epoch_grid, save_training_curve


# ── data paths ────────────────────────────────────────────────────────────────
RV_IMAGE_DIR = "data/rv_landmark/train_images"
RV_GT_DIR    = "data/rv_landmark/train_gt"
RV_SEG_DIR   = "data/rv_landmark/train_seg_multi"

TEST_IMAGE_DIR = "data/rv_landmark/test_images"
TEST_GT_DIR    = "data/rv_landmark/test_gt"
TEST_SEG_DIR   = "data/rv_landmark/test_seg_multi"

ACDC_IMAGE_DIR = "data/acdc/images"
ACDC_MASK_DIR  = "data/acdc/masks"
ACDC_RVIP_DIR  = "data/acdc/points"
ACDC_TRAIN_IDS = [f"patient{i:03d}" for i in range(1, 81)]

BATCH_SIZE   = 8
SEED         = 42
NUM_WORKERS  = 2
N_VIS        = 8
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 2.0
ENC_PREFIXES = ("enc0", "enc1", "enc2", "enc3", "enc4")

SIGMA_START = 4.0
SIGMA_END   = 1.5


# ── EMA teacher ────────────────────────────────────────────────────────────────

class EMATeacher:
    """
    Maintains an exponential moving average of the student's parameters and
    buffers. The teacher is used only in eval mode to generate targets; it is
    never updated by gradients.
    """
    def __init__(self, student: nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(student).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module):
        d = self.decay
        for te, se in zip(self.model.parameters(), student.parameters()):
            te.mul_(d).add_(se.detach(), alpha=1.0 - d)
        # Track buffers too (e.g. GroupNorm has none, but keep this general).
        for tb, sb in zip(self.model.buffers(), student.buffers()):
            tb.copy_(sb)

    @torch.no_grad()
    def predict(self, images):
        return torch.sigmoid(self.model(images))


# ── dataset wrappers (module-level so Windows spawn-based DataLoader workers
#    can pickle them — nested classes inside main() cannot be pickled) ─────────

class TaggedDataset(torch.utils.data.Dataset):
    """
    Wraps a dataset that returns (image, heatmap, coords) and appends an
    is_rv flag so the training loop can apply the ACDC replay weight and
    restrict the consistency loss to RV samples only.
    """
    def __init__(self, ds, is_rv):
        self.ds, self.is_rv = ds, is_rv

    def set_sigma(self, s):
        if hasattr(self.ds, "set_sigma"):
            self.ds.set_sigma(s)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, hm, co = self.ds[i]
        return img, hm, co, torch.tensor(1.0 if self.is_rv else 0.0)


class TaggedConcatDataset(torch.utils.data.Dataset):
    """ConcatDataset that propagates set_sigma to all child TaggedDatasets."""
    def __init__(self, parts):
        self._c = ConcatDataset(parts)
        self._parts = parts

    def set_sigma(self, s):
        for p in self._parts:
            p.set_sigma(s)

    def __len__(self):
        return len(self._c)

    def __getitem__(self, i):
        return self._c[i]


# ── helpers ────────────────────────────────────────────────────────────────────

def cosine_sigma(ep, total, s0, s1):
    if total <= 1:
        return s1
    t = ep / (total - 1)
    return s0 + 0.5 * (1 - math.cos(math.pi * t)) * (s1 - s0)


def enforce_superior_ordering_batch(coords):
    out  = coords.clone()
    swap = out[:, 1] > out[:, 3]
    if swap.any():
        tmp = out[swap].clone()
        out[swap, 0] = tmp[:, 2]
        out[swap, 1] = tmp[:, 3]
        out[swap, 2] = tmp[:, 0]
        out[swap, 3] = tmp[:, 1]
    return out


def strong_augment(imgs):
    """
    Cheap on-GPU strong augmentation for the student view: additive noise +
    random intensity gain/bias on the MRI channel (channel 0 only, so a seg
    channel — if present — is left intact). Geometry is kept identical to the
    teacher view so the two heatmaps remain pixel-aligned for consistency.
    """
    out = imgs.clone()
    b = out.size(0)
    gain = torch.empty(b, 1, 1, 1, device=out.device).uniform_(0.8, 1.2)
    bias = torch.empty(b, 1, 1, 1, device=out.device).uniform_(-0.1, 0.1)
    out[:, 0:1] = out[:, 0:1] * gain + bias
    out[:, 0:1] = out[:, 0:1] + torch.randn_like(out[:, 0:1]) * 0.05
    return out


def confidence_mask(teacher_hm, threshold):
    """
    Per-sample, per-channel confidence gate. Returns a [B,2,1,1] float mask that
    is 1 where the teacher's channel peak exceeds `threshold`, else 0. Applied to
    the consistency loss so uncertain teacher predictions contribute nothing.
    """
    peak = teacher_hm.amax(dim=(2, 3), keepdim=True)     # [B,2,1,1]
    return (peak > threshold).float()


# ── validation (real GT) ───────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, n_vis=N_VIS):
    model.eval()
    val_loss = tot_mre = tot_mre1 = tot_mre2 = 0.0
    tot_sdr = {2: 0.0, 5: 0.0, 10: 0.0}
    sample_mres = []
    vis_i, vis_p, vis_g = [], [], []

    for images, heatmaps, gt_coords in loader:
        images    = images.to(device)
        heatmaps  = heatmaps.to(device)
        gt_coords = gt_coords.to(device)

        logits    = model(images)
        loss, _   = criterion(logits.float(), heatmaps.float(),
                              gt_coords=gt_coords.float() / 256.0)
        val_loss += loss.item()

        pc = gaussian_subpixel_argmax(torch.sigmoid(logits), window=7)
        pc = enforce_superior_ordering_batch(pc)

        tot_mre  += compute_mre(pc, gt_coords).item()
        m1, m2    = compute_mre_per_landmark(pc, gt_coords)
        tot_mre1 += m1; tot_mre2 += m2
        s = compute_sdr_multi(pc, gt_coords, (2.0, 5.0, 10.0))
        for t in tot_sdr:
            tot_sdr[t] += s[t]
        sample_mres.extend(compute_per_sample_mre(pc, gt_coords).cpu().tolist())

        if len(vis_i) < n_vis:
            for j in range(min(n_vis - len(vis_i), images.size(0))):
                vis_i.append(images[j, 0].cpu().numpy())
                vis_p.append(pc[j].cpu().numpy())
                vis_g.append(gt_coords[j].cpu().numpy())

    n = len(loader)
    return {
        "val_loss": val_loss / n, "mre": tot_mre / n,
        "mre1": tot_mre1 / n, "mre2": tot_mre2 / n,
        "sdr": {t: tot_sdr[t] / n for t in tot_sdr},
        "pct": compute_mre_percentiles(sample_mres),
        "vis": (vis_i, vis_p, vis_g),
    }


# ── test evaluation with TTA + optimal LM matching ─────────────────────────────

def _tta(model, images, device):
    hms = []
    for fd in [[], [3], [2], [2, 3]]:
        v = torch.flip(images, fd) if fd else images
        with torch.no_grad():
            hm = torch.sigmoid(model(v))
        if fd:
            hm = torch.flip(hm, fd)
        hms.append(hm)
    return torch.stack(hms).mean(0)


def _enforce_and_match_np(pred_np, gt_np):
    p = pred_np.copy()
    if p[1] > p[3]:
        p = np.array([p[2], p[3], p[0], p[1]], dtype=np.float32)
    e_n = np.linalg.norm(p[:2] - gt_np[:2]) + np.linalg.norm(p[2:] - gt_np[2:])
    ps  = np.array([p[2], p[3], p[0], p[1]], dtype=np.float32)
    e_s = np.linalg.norm(ps[:2] - gt_np[:2]) + np.linalg.norm(ps[2:] - gt_np[2:])
    return ps if e_s < e_n else p


@torch.no_grad()
def evaluate_test(model, in_channels, device):
    seg_dir = TEST_SEG_DIR if in_channels == 2 else None
    try:
        test_ds = RVLandmarkDataset(
            image_dir=TEST_IMAGE_DIR, gt_dir=TEST_GT_DIR, seg_dir=seg_dir,
            in_channels=in_channels, augment=False,
            sigma=SIGMA_END, min_landmark_dist=20,
        )
    except Exception as exc:
        print(f"  [test eval skipped] {exc}")
        return None

    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    model.eval()
    sample_mres = []
    tot_mre = tot_mre1 = tot_mre2 = 0.0
    tot_sdr = {2: 0.0, 5: 0.0, 10: 0.0}

    for images, _, gt_coords in loader:
        images = images.to(device); gt_coords = gt_coords.to(device)
        avg = _tta(model, images, device)
        pc  = gaussian_subpixel_argmax(avg, window=7)
        pc_np, gt_np = pc.cpu().numpy(), gt_coords.cpu().numpy()
        pc_m = np.stack([_enforce_and_match_np(pc_np[b], gt_np[b])
                         for b in range(pc_np.shape[0])])
        pc_m = torch.tensor(pc_m, dtype=torch.float32, device=device)

        tot_mre  += compute_mre(pc_m, gt_coords).item()
        m1, m2    = compute_mre_per_landmark(pc_m, gt_coords)
        tot_mre1 += m1; tot_mre2 += m2
        s = compute_sdr_multi(pc_m, gt_coords, (2.0, 5.0, 10.0))
        for t in tot_sdr:
            tot_sdr[t] += s[t]
        sample_mres.extend(compute_per_sample_mre(pc_m, gt_coords).cpu().tolist())

    n = len(loader)
    return {
        "mre": tot_mre / n, "mre1": tot_mre1 / n, "mre2": tot_mre2 / n,
        "sdr": {t: tot_sdr[t] / n for t in tot_sdr},
        "pct": compute_mre_percentiles(sample_mres),
    }


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="EMA teacher-student self-training (RV)")
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--in-channels", type=int, default=2, choices=[1, 2])
    ap.add_argument("--group-norm",    dest="group_norm", action="store_true", default=True)
    ap.add_argument("--no-group-norm", dest="group_norm", action="store_false")
    ap.add_argument("--instance-norm", action="store_true")
    ap.add_argument("--epochs",        type=int,   default=40)
    ap.add_argument("--lr",            type=float, default=5e-5)
    ap.add_argument("--ema-decay",     type=float, default=0.99,
                    help="Teacher EMA decay (higher = smoother/slower teacher)")
    ap.add_argument("--conf-threshold", type=float, default=0.5,
                    help="Min teacher heatmap peak for the consistency loss to apply")
    ap.add_argument("--consistency-weight", type=float, default=1.0,
                    help="Weight of the student-teacher consistency loss")
    ap.add_argument("--consistency-rampup", type=int, default=5,
                    help="Epochs to linearly ramp the consistency weight from 0")
    ap.add_argument("--acdc-weight",   type=float, default=0.15,
                    help="Loss weight for ACDC replay (0 to disable replay)")
    ap.add_argument("--lm1-weight",    type=float, default=1.5)
    ap.add_argument("--val-frac",      type=float, default=0.2)
    ap.add_argument("--early-stop",    type=int,   default=10)
    ap.add_argument("--no-amp",        action="store_true")
    args = ap.parse_args()
    if args.instance_norm and args.group_norm:
        raise ValueError("Cannot use both --instance-norm and --group-norm "
                         "(pass --no-group-norm with --instance-norm)")

    torch.manual_seed(SEED); np.random.seed(SEED)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("rv-checkpoints", f"finetune_ema_{args.in_channels}ch_{ts}")
    os.makedirs(os.path.join(run_dir, "grids"), exist_ok=True)
    print(f"Run dir   : {run_dir}")
    print(f"Device    : {device}  AMP: {use_amp}")
    print(f"EMA decay : {args.ema_decay}   conf-threshold: {args.conf_threshold}")

    # ── datasets ────────────────────────────────────────────────────────────────
    train_files, val_files = split_volumes(RV_IMAGE_DIR, val_frac=args.val_frac, seed=SEED)
    print(f"RV train volumes: {len(train_files)}  val: {len(val_files)}")
    seg_dir = RV_SEG_DIR if args.in_channels == 2 else None

    # Labelled RV train (real GT) — supervised + also the source of unlabelled
    # images for the consistency loss (we simply ignore its GT for consistency).
    rv_train = RVLandmarkDataset(
        image_dir=RV_IMAGE_DIR, gt_dir=RV_GT_DIR, seg_dir=seg_dir,
        in_channels=args.in_channels, augment=True,
        sigma=SIGMA_START, min_landmark_dist=20, volume_whitelist=train_files,
    )
    parts = [rv_train]

    acdc_ds = None
    if args.acdc_weight > 0 and os.path.isdir(ACDC_IMAGE_DIR):
        try:
            acdc_ds = ACDCLandmarkDataset(
                image_dir=ACDC_IMAGE_DIR, mask_dir=ACDC_MASK_DIR, rvip_dir=ACDC_RVIP_DIR,
                patient_ids=ACDC_TRAIN_IDS, in_channels=args.in_channels,
                augment=True, sigma=SIGMA_START, min_landmark_dist=5,
            )
            parts.append(acdc_ds)
            print(f"ACDC replay: {len(acdc_ds)} slices (weight {args.acdc_weight})")
        except Exception as exc:
            print(f"  [ACDC skipped] {exc}")

    # A source flag per sample lets the loop apply the ACDC replay weight and
    # restrict the consistency loss to RV samples only.
    tagged = [TaggedDataset(rv_train, True)]
    if acdc_ds is not None:
        tagged.append(TaggedDataset(acdc_ds, False))

    train_ds = TaggedConcatDataset(tagged)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    val_ds = RVLandmarkDataset(
        image_dir=RV_IMAGE_DIR, gt_dir=RV_GT_DIR, seg_dir=seg_dir,
        in_channels=args.in_channels, augment=False,
        sigma=SIGMA_END, min_landmark_dist=20, volume_whitelist=val_files,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # ── model + EMA teacher ─────────────────────────────────────────────────────
    model = UNetResNet34(
        in_channels=args.in_channels, num_classes=2, dropout=0.0,
        pretrained=False, cardiac_pretrained=False,
        use_instance_norm=args.instance_norm, use_group_norm=args.group_norm,
    ).to(device)
    state = torch.load(args.base_checkpoint, map_location=device, weights_only=True)
    # strict=False: base checkpoint may carry aux seg_head.* keys not present here.
    missing, unexpected = model.load_state_dict(state, strict=False)
    non_seg = [k for k in unexpected if not k.startswith("seg_head")]
    print(f"Base ckpt loaded. missing={len(missing)} unexpected(non-seg)={len(non_seg)}")

    teacher = EMATeacher(model, decay=args.ema_decay)

    criterion = HeatmapLoss(
        coord_weight=20.0, sep_weight=0.5, sep_min_dist=0.08,
        wing_w=0.008, wing_eps=0.002, lm_weights=[2.0, 1.0], hard_k=BATCH_SIZE - 2,
    ).to(device)

    lm_w = torch.tensor([args.lm1_weight, 1.0], device=device).view(1, 2, 1, 1)

    optim  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history = []
    best_p90 = float("inf")
    no_improve = 0

    # baseline
    m0 = validate(model, val_loader, criterion, device)
    print(f"[baseline] MRE={m0['mre']:.2f}px  P90={m0['pct'][90]:.2f}px  "
          f"SDR@5={m0['sdr'][5]:.3f}")

    for ep in range(1, args.epochs + 1):
        sigma = cosine_sigma(ep - 1, args.epochs, SIGMA_START, SIGMA_END)
        train_ds.set_sigma(sigma)
        # Linear ramp-up avoids trusting a still-noisy teacher early on.
        cw = args.consistency_weight * min(1.0, ep / max(args.consistency_rampup, 1))

        model.train()
        run_sup = run_con = 0.0
        n_batches = n_con = 0

        for imgs, hms, coords, is_rv in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            hms    = (hms.to(device, non_blocking=True).float() * lm_w)
            coords = coords.to(device, non_blocking=True).float()
            is_rv  = is_rv.to(device, non_blocking=True)          # [B]

            # Per-sample supervised weight: RV=1.0, ACDC replay=acdc_weight.
            sup_w = torch.where(is_rv > 0.5,
                                torch.ones_like(is_rv),
                                torch.full_like(is_rv, args.acdc_weight))

            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                # Teacher target on the weak (given) view.
                teacher_hm = teacher.predict(imgs)               # [B,2,H,W] in [0,1]
                # Student sees a strong view of the same images.
                student_imgs = strong_augment(imgs)
                logits = model(student_imgs).float()

                # 1. Supervised loss (batch-mean scaled by mean supervised weight).
                sup_loss, parts = criterion(logits, hms, gt_coords=coords / 256.0)
                sup_loss = sup_loss * sup_w.mean()

                # 2. Confidence-masked consistency on RV samples only.
                mask = confidence_mask(teacher_hm, args.conf_threshold)   # [B,2,1,1]
                mask = mask * is_rv.view(-1, 1, 1, 1)                      # RV only
                pred_hm = torch.sigmoid(logits)
                denom = mask.sum().clamp_min(1.0)
                con_loss = ((pred_hm - teacher_hm) ** 2 * mask).sum() / denom

                loss = sup_loss + cw * con_loss

            if not torch.isfinite(loss):
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optim); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optim.step()

            # Update teacher AFTER the student step.
            teacher.update(model)

            run_sup += sup_loss.item()
            run_con += con_loss.item()
            n_batches += 1
            if mask.sum() > 0:
                n_con += 1

        sched.step()

        # Validate BOTH student and teacher; keep whichever is better (the teacher
        # is usually smoother and generalises better in self-training).
        vs = validate(model, val_loader, criterion, device)
        vt = validate(teacher.model, val_loader, criterion, device)
        which, v = ("teacher", vt) if vt["pct"][90] < vs["pct"][90] else ("student", vs)

        sup_avg = run_sup / max(n_batches, 1)
        con_avg = run_con / max(n_batches, 1)
        history.append({
            "epoch": ep, "sigma": sigma, "cw": cw,
            "sup": sup_avg, "con": con_avg,
            # train_loss = combined sup+consistency loss, kept under this key so
            # save_training_curve (shared with the other train/finetune scripts)
            # can plot it without a schema special-case.
            "train_loss": sup_avg + cw * con_avg,
            "val_loss": v["val_loss"], "mre": v["mre"], "sdr": v["sdr"][5],
            "which": which,
        })
        print(f"[EMA] Ep {ep:3d}  sigma={sigma:.2f}  cw={cw:.2f}  "
              f"sup={run_sup/max(n_batches,1):.3f} con={run_con/max(n_batches,1):.4f}  "
              f"[{which}] MRE={v['mre']:.2f}px (LM1={v['mre1']:.2f} LM2={v['mre2']:.2f})  "
              f"SDR@5={v['sdr'][5]:.3f}  P90={v['pct'][90]:.2f}px  "
              f"(con-batches={n_con}/{n_batches})")

        save_epoch_grid(*v["vis"], epoch=ep,
                        save_dir=os.path.join(run_dir, "grids"), n_samples=N_VIS)

        if v["pct"][90] < best_p90:
            best_p90 = v["pct"][90]
            no_improve = 0
            # Save the better of student/teacher as best_model.pth.
            best_state = (teacher.model if which == "teacher" else model).state_dict()
            torch.save(best_state, os.path.join(run_dir, "best_model.pth"))
            print(f"  -> new best P90={best_p90:.2f}px ({which} saved)")
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                print(f"  early-stop after {args.early_stop} epochs without improvement")
                break

        torch.save(model.state_dict(), os.path.join(run_dir, "last_student.pth"))
        torch.save(teacher.model.state_dict(), os.path.join(run_dir, "last_teacher.pth"))

    # ── persist + test ──────────────────────────────────────────────────────────
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    save_training_curve(history, run_dir)

    best_path = os.path.join(run_dir, "best_model.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    print("\nEvaluating best model on test set …")
    tm = evaluate_test(model, args.in_channels, device)
    if tm is not None:
        results = {
            "checkpoint": best_path, "in_channels": args.in_channels,
            "best_val_p90": best_p90,
            "test_mre": tm["mre"], "test_mre_lm1": tm["mre1"], "test_mre_lm2": tm["mre2"],
            "test_sdr2": tm["sdr"][2], "test_sdr5": tm["sdr"][5], "test_sdr10": tm["sdr"][10],
            "test_p50": tm["pct"][50], "test_p90": tm["pct"][90],
        }
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*44}")
        print("EMA SELF-TRAINING COMPLETE")
        print(f"Val  P90  : {best_p90:.2f}px")
        print(f"Test MRE  : {tm['mre']:.2f}px  (LM1={tm['mre1']:.2f} LM2={tm['mre2']:.2f})")
        print(f"Test SDR@5: {tm['sdr'][5]*100:.1f}%")
        print(f"Checkpoint: {best_path}")
        print(f"{'='*44}")
    else:
        print(f"\nDone. Best val P90 = {best_p90:.2f}px  ({best_path})")


if __name__ == "__main__":
    main()
