# -*- coding: utf-8 -*-
"""
Stage 2: Train CorrDiff UNet  — v13 (Proper Distribution-Preserving)
=====================================================================
DUAL-MODE TRAINER  ──  plug-and-play, 4 × H100

KEY FIXES vs v11:
─────────────────
1.  FLOW MATCHING mode:
    - Loss = PURE MSE(v_pred, v_target)  [NOT hybrid_sigma_loss]
    - Logit-Normal time sampling         [better than Beta(1.5,1.5)]
    - ODE integration step count = 20    [was 6 — too coarse]
    - No hard clamp in trajectory        [was clamp(-1, 7) — breaks tails]

2.  CORRDIFF_RESIDUAL mode:
    - Loss = PROPER weighted EDM MSE:  λ(σ)·‖D_pred − target‖²
    - σ_data derived from ACTUAL training data std (not 0.1925 magic number)
    - DDIM sampler uses consistent c_skip/c_out scalings throughout

3.  SIGMA_DATA:
    - Computed from real data at startup (first batch of training split)
    - Printed so you can hard-code it for future runs

4.  PhysicsGuide:
    - Soft dry attenuation (0.05×) instead of hard zeroing
    - Mass correction clamp = (0.05, 20.0)  [wide, allows monsoon extremes]

5.  VALIDATION:
    - DistributionMonitor reports P10/P50/P90/P99 bias every epoch
    - Best checkpoint selected on COMBINED score:
        score = val_loss + α·|mean_bias_pct|/100
      where α=0.5.  Prevents selecting checkpoints that are
      low-loss but heavily distribution-collapsed.
    - No QDM anywhere — the model must learn the true distribution.

6.  AUGMENTATION:
    - Consistent across all spatial inputs (fp, topo, d2m)
    - RandomGamma forces model to see extreme values during training

7.  CHECKPOINTING:
    - Isolated to checkpoints/v13/ to start fresh
    - Saves sigma_data in checkpoint so inference is consistent
"""

import os
import math
import time
import argparse
import traceback
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

try:
    import albumentations as A
    HAS_ALB = True
except ImportError:
    HAS_ALB = False
    warnings.warn("albumentations not found. pip install albumentations opencv-python-headless")

from Dataset import UpscaleDataset
from HybridNetwork import (CorrDiffRegressor, UNet, FlowMatching,
                            EDMPreconditioning, PhysicsGuide,
                            DistributionMonitor)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  MODE TOGGLE
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_MODE = os.environ.get("TRAIN_MODE", "flow_matching")
assert TRAIN_MODE in ("flow_matching", "corrdiff_residual"), \
    f"Bad TRAIN_MODE={TRAIN_MODE!r}"

# σ_data: std of log1p(precip) residuals after Stage-1 regressor.
# This is computed from real data at startup (see compute_sigma_data()).
# You can hard-code after first run to skip recomputation.
SIGMA_DATA_OVERRIDE = None   # e.g. set to 0.45 after first run

# ══════════════════════════════════════════════════════════════════════════════
# 1.  PATHS
# ══════════════════════════════════════════════════════════════════════════════
RF_PATH  = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/RF_1975to2023.nc"
ORO_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/oro.nc"
D2M_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/era5_aligned_to_rf.nc"
REG_CKPT = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/Variance/checkpoints/regressor/regressor_best.pth"
CKPT_DIR = "checkpoints/v13/"

# ══════════════════════════════════════════════════════════════════════════════
# 2.  HYPER-PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
BATCH        = 16
ACCUM_STEPS  = 2
LR           = 5e-5
MIN_LR       = LR * 0.01
EPOCHS       = 500
PATIENCE     = 200
T_COND       = 5
PRECIP_CH    = 0
BASE_CH      = 256
CHANNEL_MULT = (1, 2, 2, 4)
NRB          = 3
DROPOUT      = 0.15
FM_STEPS     = 20          # More steps → better ODE integration
CFG_SCALE    = 2.0         # Reduced from 2.5 — prevents over-sharpening
P_CFG_DROP   = 0.15
WEIGHT_DECAY = 1e-3
GRAD_CLIP    = 1.0
EMA_DECAY    = 0.9995
N_ENS        = 4           # Ensemble members for validation metrics
REG_IN_CH    = 2
REG_D2M_CH   = 1
D2M_CH       = 1
UNET_D2M_CH  = 1
UNET_VAR_MAP_CH = 1
TOPO_CH      = 3
GLOBAL_DIM   = 2
UNET_IN_CH   = 1 + 1 + T_COND   # noisy + mu + T temporal frames
DS_FACTOR    = 4

# Best-checkpoint score weights
SCORE_LOSS_WEIGHT      = 1.0
SCORE_MEANBIAS_WEIGHT  = 0.5   # penalise mean bias (normalised to %)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  EDM NOISE SCHEDULE  (Karras et al. 2022, Section B.6)
# ══════════════════════════════════════════════════════════════════════════════
def build_edm_schedule(n_steps, sigma_min=0.002, sigma_data=0.5, rho=7.0):
    """Deterministic noise schedule from σ_max=2·σ_data down to σ_min."""
    sigma_max = 2.0 * sigma_data
    steps     = torch.arange(n_steps, dtype=torch.float32) / max(n_steps - 1, 1)
    schedule  = (sigma_max ** (1 / rho) +
                 steps * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return schedule

# ══════════════════════════════════════════════════════════════════════════════
# 4.  COMPUTE σ_DATA FROM REAL TRAINING DATA
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_sigma_data(reg, trl_dataset, dev, max_batches=50,
                        batch_size=16, topo_fn=None, coarse_fn=None):
    """
    Compute std of (fine - regressor_mean) residuals in log1p space.
    This is the true σ_data for the EDM schedule.
    """
    if SIGMA_DATA_OVERRIDE is not None:
        print(f"[σ_data] Using override: {SIGMA_DATA_OVERRIDE}")
        return SIGMA_DATA_OVERRIDE

    print("[σ_data] Computing from real training data ...")
    loader = DataLoader(trl_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4, drop_last=True)
    residuals = []
    reg.eval()
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        fp      = b["fine"].to(dev)[:, PRECIP_CH:PRECIP_CH + 1]
        topo_1ch = b["topo"].to(dev)
        xi_raw   = F.avg_pool2d(fp, kernel_size=DS_FACTOR, stride=DS_FACTOR)
        var_map  = b["var_map"].to(dev)
        d2m      = b["d2m"].to(dev) if "d2m" in b else None
        gf       = torch.stack([b["doy"], b["hour"]], 1).float().to(dev)
        tp       = topo_fn(topo_1ch)
        xi       = coarse_fn(xi_raw, var_map)
        mu       = reg(xi, topo=tp, global_features=gf, d2m=d2m)
        residuals.append((fp - mu).cpu().float())

    all_res    = torch.cat(residuals)
    sigma_data = float(all_res.std().item())
    print(f"[σ_data] Computed: {sigma_data:.4f}  "
          f"(mean_residual={all_res.mean().item():.4f})")
    print(f"[σ_data] TIP: Set SIGMA_DATA_OVERRIDE={sigma_data:.4f} to skip recomputation")
    return sigma_data

# ══════════════════════════════════════════════════════════════════════════════
# 5.  EMA
# ══════════════════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay  = decay
        self.shadow = {k: v.clone().detach().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                  alpha=1 - self.decay)

    def apply_to(self, model):
        model.load_state_dict(
            {k: v.to(next(model.parameters()).device)
             for k, v in self.shadow.items()})

    def restore(self, model, state):
        model.load_state_dict(state)

# ══════════════════════════════════════════════════════════════════════════════
# 6.  AUGMENTATION  ──  CONSISTENT ACROSS ALL SPATIAL INPUTS
# ══════════════════════════════════════════════════════════════════════════════
def _make_aug_pipeline():
    if not HAS_ALB:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,
                           rotate_limit=10, border_mode=0, p=0.4),
        A.ElasticTransform(alpha=1.2, sigma=50.0, p=0.25),
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.1,
                                   contrast_limit=0.1, p=0.4),
        A.GaussNoise(var_limit=(1e-5, 5e-4), p=0.25),
        # Forces model to see extreme precipitation values in training
        A.RandomGamma(gamma_limit=(0.7, 1.5), p=0.3),
    ], additional_targets={"topo": "image", "d2m": "image"})


_AUG_PIPELINE = _make_aug_pipeline()


def augment_sample(fp_t, topo_t, d2m_t, aug_prob=0.5):
    def _rederive_coarse(fp):
        return F.avg_pool2d(fp.unsqueeze(0),
                            kernel_size=DS_FACTOR,
                            stride=DS_FACTOR).squeeze(0)

    if _AUG_PIPELINE is None or np.random.rand() > aug_prob:
        return fp_t, topo_t, _rederive_coarse(fp_t), d2m_t

    fp_np   = fp_t.squeeze(0).numpy().astype(np.float32)
    topo_np = topo_t.squeeze(0).numpy().astype(np.float32)
    d2m_np  = (d2m_t.squeeze(0).numpy().astype(np.float32)
               if d2m_t is not None else np.zeros_like(fp_np))

    result    = _AUG_PIPELINE(image=fp_np, topo=topo_np, d2m=d2m_np)
    fp_aug    = torch.from_numpy(result["image"]).unsqueeze(0)
    topo_aug  = torch.from_numpy(result["topo"]).unsqueeze(0)
    d2m_aug   = (torch.from_numpy(result["d2m"]).unsqueeze(0)
                 if d2m_t is not None else None)
    coarse_aug = _rederive_coarse(fp_aug)
    return fp_aug, topo_aug, coarse_aug, d2m_aug

# ══════════════════════════════════════════════════════════════════════════════
# 7.  TOPO HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def compute_slope_aspect(elev,
                          global_elev_max=8600.0,
                          global_slope_max=1.5):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=torch.float32, device=elev.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       dtype=torch.float32, device=elev.device).view(1, 1, 3, 3)
    e   = elev.float()
    dx  = F.conv2d(e, kx, padding=1)
    dy  = F.conv2d(e, ky, padding=1)
    slope  = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
    aspect = torch.atan2(dy, dx)

    def global_norm(t, g_min, g_max):
        return 2 * (t - g_min) / (g_max - g_min + 1e-8) - 1

    return torch.cat([
        global_norm(e,      0.0, global_elev_max),
        global_norm(slope,  0.0, global_slope_max),
        aspect / math.pi,
    ], dim=1)


def expand_topo(topo_1ch):
    return torch.cat([compute_slope_aspect(topo_1ch[i:i + 1])
                      for i in range(topo_1ch.shape[0])], dim=0)


def build_coarse_input(coarse, var_map):
    Hc, Wc = coarse.shape[-2], coarse.shape[-1]
    return torch.cat([coarse,
                      F.adaptive_avg_pool2d(var_map, (Hc, Wc))], dim=1)

# ══════════════════════════════════════════════════════════════════════════════
# 8.  TEMPORAL CONDITIONING
# ══════════════════════════════════════════════════════════════════════════════
def build_temporal_cond(batch, dev, n_frames=T_COND):
    if "tc_frames" in batch:
        tc = batch["tc_frames"].to(dev, non_blocking=True)
        if tc.shape[1] >= n_frames:
            return tc[:, :n_frames]
        pad = torch.zeros(tc.shape[0], n_frames - tc.shape[1],
                          *tc.shape[2:], device=dev)
        return torch.cat([tc, pad], dim=1)
    coarse    = batch["coarse"].to(dev, non_blocking=True)
    coarse_up = F.interpolate(coarse, scale_factor=4,
                               mode='bilinear', align_corners=False)
    return coarse_up.expand(-1, n_frames, -1, -1)

# ══════════════════════════════════════════════════════════════════════════════
# 9.  AUXILIARY METRICS  (for validation diagnostics)
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def weighted_pcc(pred, target, lat_w=None):
    B = pred.shape[0]
    p = pred.float().view(B, -1)
    t = target.float().view(B, -1)
    if lat_w is not None:
        w  = lat_w.view(1, -1).expand_as(p)
        w  = w / w.sum(dim=1, keepdim=True)
        pm = (p * w).sum(dim=1, keepdim=True)
        tm = (t * w).sum(dim=1, keepdim=True)
        pm_c = p - pm; tm_c = t - tm
        r = ((pm_c * tm_c * w).sum(dim=1) /
             (torch.sqrt((pm_c**2 * w).sum(dim=1) *
                         (tm_c**2 * w).sum(dim=1)) + 1e-8))
    else:
        pm = p.mean(dim=1, keepdim=True)
        tm = t.mean(dim=1, keepdim=True)
        pm_c = p - pm; tm_c = t - tm
        r = ((pm_c * tm_c).sum(dim=1) /
             (torch.sqrt((pm_c**2).sum(dim=1) *
                         (tm_c**2).sum(dim=1)) + 1e-8))
    return r.mean().item()


@torch.no_grad()
def crps_ensemble(samples, target):
    N   = samples.shape[0]
    mae = (samples - target.unsqueeze(0)).abs().mean(0)
    pair = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs()
    return (mae - 0.5 / N / (N - 1) * pair.sum([0, 1])).mean().item()


@torch.no_grad()
def psd_tail_ratio(pred, target, hff=0.3):
    P = torch.fft.rfft2(pred.float()).abs()
    T = torch.fft.rfft2(target.float()).abs()
    c = int((1 - hff) * P.shape[-1])
    return (P[..., c:].mean() / (T[..., c:].mean() + 1e-8)).item()


@torch.no_grad()
def fractions_skill_score(pred, target, threshold=0.5, window=5):
    p_bin  = (pred > threshold).float()
    t_bin  = (target > threshold).float()
    p_frac = F.avg_pool2d(p_bin, kernel_size=window, stride=1,
                          padding=window // 2)
    t_frac = F.avg_pool2d(t_bin, kernel_size=window, stride=1,
                          padding=window // 2)
    mse    = ((p_frac - t_frac) ** 2).mean(dim=[-1, -2])
    ref    = ((p_frac ** 2).mean(dim=[-1, -2]) +
              (t_frac ** 2).mean(dim=[-1, -2]))
    return (1.0 - mse / (ref + 1e-8)).mean().item()

# ══════════════════════════════════════════════════════════════════════════════
# 10.  DDP
# ══════════════════════════════════════════════════════════════════════════════
def setup():
    rank = int(os.environ.get("RANK", 0))
    ws   = int(os.environ.get("WORLD_SIZE", 1))
    lr_  = int(os.environ.get("LOCAL_RANK", 0))
    if ws > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(lr_)
    torch.backends.cudnn.benchmark      = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32     = True
    return rank, ws, lr_, torch.device(
        f"cuda:{lr_}" if torch.cuda.is_available() else "cpu")


def ar(t, ws):
    if ws > 1 and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= ws
    return t

# ══════════════════════════════════════════════════════════════════════════════
# 11.  REGRESSOR LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_regressor(ckpt_path, dev):
    ck  = torch.load(ckpt_path, map_location=dev)
    reg = CorrDiffRegressor(
        in_channels=ck.get("reg_in_channels", REG_IN_CH),
        out_channels=1,
        base_channels=64,
        channel_mult=(1, 2, 4),
        num_blocks=2,
        global_dim=GLOBAL_DIM,
        topo_channels=TOPO_CH,
        d2m_channels=ck.get("d2m_channels", REG_D2M_CH),
        use_d2m=ck.get("use_d2m", True),
    ).to(dev)
    state = {k.replace("module.", ""): v
             for k, v in ck["model_state_dict"].items()}
    reg.load_state_dict(state)
    reg.eval()
    for p in reg.parameters():
        p.requires_grad_(False)
    return reg

# ══════════════════════════════════════════════════════════════════════════════
# 12.  DDIM SAMPLER  (consistent EDM preconditioning)
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def ddim_sample(raw_model, mu, tc, tp, gf, d2m, var_map,
                edm_schedule, dev, edm_precond):
    """
    DDIM sampling with proper EDM c_skip / c_out / c_in / c_noise scalings.
    edm_precond: EDMPreconditioning instance (carries sigma_data).
    """
    B   = mu.shape[0]
    x_t = torch.randn_like(mu) * edm_precond.sigma_data
    sigmas = edm_schedule.to(dev)

    for i, sigma_cur in enumerate(sigmas):
        s_cur  = sigma_cur.view(1, 1, 1, 1)
        c_skip, c_out, c_in, c_noise = edm_precond.get_scalings(s_cur)
        c_n    = c_noise.expand(B).squeeze()

        x_in   = torch.cat([x_t, mu, tc], dim=1)
        D_pred  = raw_model(c_in * x_in, c_n, topo=tp,
                            global_features=gf, d2m=d2m,
                            var_map=var_map, T=T_COND)
        x0_hat  = c_skip * x_t[:, :1] + c_out * D_pred

        if i < len(sigmas) - 1:
            sigma_next = sigmas[i + 1].view(1, 1, 1, 1)
            x_t = x0_hat + sigma_next * (x_t - x0_hat) / s_cur.clamp(min=1e-8)
        else:
            x_t = x0_hat

    return x_t

# ══════════════════════════════════════════════════════════════════════════════
# 13.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train():
    rank, ws, lr_, dev = setup()
    os.makedirs(CKPT_DIR, exist_ok=True)

    SAVE    = os.path.join(CKPT_DIR,
                           f"unet_{TRAIN_MODE}_nrb{NRB}_best.pth")
    LATEST  = os.path.join(CKPT_DIR,
                           f"unet_{TRAIN_MODE}_nrb{NRB}_latest.pth")

    ds  = UpscaleDataset(RF_PATH, ORO_PATH, d2m_file=D2M_PATH,
                         split="train", normalize=True, device="cpu")
    n   = len(ds)
    trn = int(0.70 * n)
    van = int(0.10 * n)

    _loader_kwargs = dict(num_workers=8, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2,
                          drop_last=True)
    trl = DataLoader(Subset(ds, range(0, trn)),
                     BATCH, shuffle=True, **_loader_kwargs)
    val = DataLoader(Subset(ds, range(trn, trn + van)),
                     BATCH, shuffle=False,
                     num_workers=8, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

    reg = load_regressor(REG_CKPT, dev)
    if rank == 0:
        print(f"[Stage-1] Regressor loaded (frozen)")

    # ── Compute σ_data ──────────────────────────────────────────────────────
    trl_sub = Subset(ds, range(0, trn))
    sigma_data = compute_sigma_data(
        reg, trl_sub, dev, max_batches=50,
        topo_fn=expand_topo,
        coarse_fn=build_coarse_input)

    edm_precond  = EDMPreconditioning(sigma_data=sigma_data)
    edm_schedule = build_edm_schedule(FM_STEPS, sigma_data=sigma_data)

    if TRAIN_MODE == "flow_matching":
        fm          = FlowMatching(n_steps=FM_STEPS, cfg_scale=CFG_SCALE)
        loss_label  = "L_flow  "
    else:
        fm          = None
        loss_label  = "L_denoise"

    dist_monitor = DistributionMonitor()

    model = UNet(
        in_channels=UNET_IN_CH, out_channels=1,
        base_channels=BASE_CH, channel_mult=CHANNEL_MULT,
        num_res_blocks=NRB, dropout=DROPOUT,
        global_dim=GLOBAL_DIM, use_bottleneck_attention=True,
        topo_channels=TOPO_CH,
        use_d2m=True,         d2m_channels=UNET_D2M_CH,
        use_var_map=True,     var_map_channels=UNET_VAR_MAP_CH,
        temporal_frames=T_COND,
    ).to(dev)

    if ws > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[lr_], find_unused_parameters=False)
    raw = model.module if ws > 1 else model

    ema    = EMA(raw, decay=EMA_DECAY)
    opt    = AdamW(model.parameters(), lr=LR,
                   weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    scaler = GradScaler(device=dev.type)
    sched  = CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=1, eta_min=MIN_LR)

    start          = 0
    best_score     = float('inf')
    best_val_loss  = float('inf')
    no_improve     = 0

    if os.path.exists(LATEST):
        ck = torch.load(LATEST, map_location=dev)
        saved_sd = sigma_data
        ck_sd    = ck.get("sigma_data", None)
        if ck_sd is not None and abs(ck_sd - sigma_data) > 0.05:
            if rank == 0:
                print(f"[RESUME] σ_data mismatch "
                      f"(saved={ck_sd:.4f}, current={sigma_data:.4f}). "
                      f"Starting fresh.")
        else:
            try:
                raw.load_state_dict(
                    {k.replace("module.", ""): v
                     for k, v in ck["model_state_dict"].items()})
                opt.load_state_dict(ck["optimizer_state_dict"])
                start         = ck.get("epoch", 0) + 1
                best_score    = ck.get("best_score", float('inf'))
                best_val_loss = ck.get("val_loss", float('inf'))
                no_improve    = ck.get("no_improve", 0)
                if "ema_shadow" in ck:
                    ema.shadow = {k: v.to(dev)
                                  for k, v in ck["ema_shadow"].items()}
                if rank == 0:
                    print(f"[RESUME] ep={start}  "
                          f"best_score={best_score:.4f}")
            except RuntimeError as e:
                if rank == 0:
                    print(f"[RESUME ABORTED] shape mismatch, fresh start: {e}")

    if rank == 0:
        np_ = sum(p.numel() for p in model.parameters()
                  if p.requires_grad)
        print(f"[MODEL]  UNet {np_/1e6:.2f}M  mode={TRAIN_MODE}  "
              f"σ_data={sigma_data:.4f}")
        print(f"[COND]   d2m={UNET_D2M_CH}  var_map={UNET_VAR_MAP_CH}  "
              f"T_COND={T_COND}  FM_STEPS={FM_STEPS}")
        print(f"[OPTIM]  LR={LR}  ACCUM={ACCUM_STEPS}  "
              f"eff_batch={BATCH*ACCUM_STEPS}")
        hdr = (f"{'Ep':>5}|{loss_label:>10}|{'ValLoss':>8}|"
               f"{'Score':>8}|{'wPCC':>7}|{'CRPS':>8}|"
               f"{'PSD_r':>7}|{'FSS':>6}|{'MeanBias%':>10}|{'LR':>9}")
        print(hdr); print("-" * len(hdr))

    lat_w = None

    for ep in range(start, EPOCHS):
        model.train()
        t0 = time.time(); sum_ml = nb = 0.
        opt.zero_grad(set_to_none=True)

        for step, b in enumerate(trl, 1):
            try:
                fp       = b["fine"].to(dev, non_blocking=True)[:, PRECIP_CH:PRECIP_CH + 1]
                topo_1ch = b["topo"].to(dev, non_blocking=True)
                d2m      = b["d2m"].to(dev, non_blocking=True) if "d2m" in b else None
                var_map  = b["var_map"].to(dev, non_blocking=True)
                gf       = torch.stack([b["doy"], b["hour"]], 1).float().to(dev, non_blocking=True)
                tc       = build_temporal_cond(b, dev)

                # Inline augmentation (consistent flips)
                if torch.rand(1).item() < 0.5:
                    fp = fp.flip(-1); topo_1ch = topo_1ch.flip(-1)
                    if d2m is not None: d2m = d2m.flip(-1)
                    var_map = var_map.flip(-1)
                if torch.rand(1).item() < 0.5:
                    fp = fp.flip(-2); topo_1ch = topo_1ch.flip(-2)
                    if d2m is not None: d2m = d2m.flip(-2)
                    var_map = var_map.flip(-2)

                xi_raw = F.avg_pool2d(fp, kernel_size=DS_FACTOR, stride=DS_FACTOR)
                tp     = expand_topo(topo_1ch)
                xi     = build_coarse_input(xi_raw, var_map)

                with torch.no_grad():
                    mu = reg(xi, topo=tp, global_features=gf, d2m=d2m)

                residual = fp - mu
                cfg_drop = (torch.rand(fp.shape[0], device=dev) < P_CFG_DROP)

                # ── FLOW MATCHING ─────────────────────────────────────────
                if TRAIN_MODE == "flow_matching":
                    x_t, t_vec, v_star = fm.get_train_sample(residual)
                    x_in = torch.cat([x_t, mu, tc], dim=1)
                    with autocast(device_type=dev.type):
                        v_pred = model(x_in, t_vec, topo=tp,
                                       global_features=gf,
                                       cfg_drop=cfg_drop,
                                       d2m=d2m, var_map=var_map,
                                       T=T_COND)
                        # PURE MSE in velocity space — THE correct FM loss
                        loss = FlowMatching.loss(v_pred, v_star) / ACCUM_STEPS

                # ── CORRDIFF RESIDUAL  (EDM) ──────────────────────────────
                else:
                    idx     = torch.randint(0, len(edm_schedule),
                                            (fp.shape[0],))
                    sigma_t = edm_schedule[idx].to(dev).view(-1, 1, 1, 1)
                    eps     = torch.randn_like(residual)
                    x_t     = residual + sigma_t * eps

                    c_skip, c_out, c_in, c_noise = edm_precond.get_scalings(sigma_t)
                    c_n = c_noise.view(fp.shape[0])

                    x_in = torch.cat([x_t, mu, tc], dim=1)
                    with autocast(device_type=dev.type):
                        D_pred  = model(c_in * x_in, c_n, topo=tp,
                                        global_features=gf,
                                        cfg_drop=cfg_drop,
                                        d2m=d2m, var_map=var_map,
                                        T=T_COND)
                        x0_pred = c_skip * x_t[:, :1] + c_out * D_pred
                        # PROPER weighted EDM loss (not MAE, not sigma-scaled MAE)
                        loss = EDMPreconditioning.edm_loss(
                            x0_pred, residual, sigma_t,
                            edm_precond.sigma_data) / ACCUM_STEPS

                scaler.scale(loss).backward()

                if step % ACCUM_STEPS == 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    if not torch.isfinite(loss * ACCUM_STEPS):
                        if rank == 0:
                            print(f"[step={step}] NaN/Inf loss — skipping")
                        opt.zero_grad(set_to_none=True)
                        continue
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    ema.update(raw)

                with torch.no_grad():
                    sum_ml += loss.item() * ACCUM_STEPS
                    nb     += 1

            except Exception:
                if rank == 0:
                    traceback.print_exc()
                raise

        sched.step()
        te_ml  = ar(torch.tensor(sum_ml / max(nb, 1), device=dev), ws).item()
        lr_now = opt.param_groups[0]["lr"]

        # ── VALIDATION (EMA weights) ──────────────────────────────────────
        live_state = deepcopy(raw.state_dict())
        ema.apply_to(raw)
        model.eval()

        v_loss  = torch.tensor(0., device=dev)
        vn      = torch.tensor(0,  device=dev, dtype=torch.long)
        pcc_sum = crps_sum = psd_sum = fss_sum = 0.
        dist_metrics_all = []
        vb = 0

        with torch.no_grad():
            for b in val:
                fp      = b["fine"].to(dev, non_blocking=True)[:, PRECIP_CH:PRECIP_CH + 1]
                tp      = expand_topo(b["topo"].to(dev, non_blocking=True))
                gf      = torch.stack([b["doy"], b["hour"]], 1).float().to(dev, non_blocking=True)
                xi_raw  = b["coarse"].to(dev, non_blocking=True)
                var_map = b["var_map"].to(dev, non_blocking=True)
                d2m     = b["d2m"].to(dev, non_blocking=True) if "d2m" in b else None
                tc      = build_temporal_cond(b, dev)
                xi      = build_coarse_input(xi_raw, var_map)

                with autocast(device_type=dev.type):
                    mu = reg(xi, topo=tp, global_features=gf, d2m=d2m)

                residual = fp - mu

                # ── Validation loss (same as training) ────────────────────
                if TRAIN_MODE == "flow_matching":
                    x_t, t_vec, v_star = fm.get_train_sample(residual)
                    x_in = torch.cat([x_t, mu, tc], dim=1)
                    with autocast(device_type=dev.type):
                        v_pred = model(x_in, t_vec, topo=tp,
                                       global_features=gf,
                                       d2m=d2m, var_map=var_map,
                                       T=T_COND)
                        l_val = FlowMatching.loss(v_pred, v_star)
                else:
                    idx     = torch.randint(0, len(edm_schedule), (fp.shape[0],))
                    sigma_t = edm_schedule[idx].to(dev).view(-1, 1, 1, 1)
                    eps     = torch.randn_like(residual)
                    x_t     = residual + sigma_t * eps
                    c_skip, c_out, c_in, c_noise = edm_precond.get_scalings(sigma_t)
                    c_n     = c_noise.view(fp.shape[0])
                    x_in    = torch.cat([x_t, mu, tc], dim=1)
                    with autocast(device_type=dev.type):
                        D_pred  = model(c_in * x_in, c_n, topo=tp,
                                        global_features=gf,
                                        d2m=d2m, var_map=var_map,
                                        T=T_COND)
                        x0_pred = c_skip * x_t[:, :1] + c_out * D_pred
                        l_val   = EDMPreconditioning.edm_loss(
                            x0_pred, residual, sigma_t,
                            edm_precond.sigma_data)

                v_loss += l_val * fp.shape[0]
                vn     += fp.shape[0]

                # ── Generate samples for distribution evaluation ───────────
                samples = []
                for _ in range(N_ENS):
                    if TRAIN_MODE == "flow_matching":
                        x_cond = torch.cat([mu, tc], dim=1)
                        s = (fm.sample(raw, x_cond, topo=tp,
                                       global_features=gf,
                                       d2m=d2m, var_map=var_map,
                                       cfg_scale=CFG_SCALE, T=T_COND)
                             + mu)
                    else:
                        s = (ddim_sample(raw, mu, tc, tp, gf, d2m,
                                         var_map, edm_schedule, dev,
                                         edm_precond)
                             + mu)
                    s = PhysicsGuide.apply(s, xi_raw,
                                           enforce_mass=True,
                                           enforce_dry=True)
                    samples.append(s)

                samples_t = torch.stack(samples)
                best_s    = samples_t.mean(0)

                pcc_sum  += weighted_pcc(best_s, fp, lat_w)
                crps_sum += crps_ensemble(samples_t, fp)
                psd_sum  += psd_tail_ratio(best_s, fp)
                fss_sum  += fractions_skill_score(best_s, fp,
                                                   threshold=0.5, window=5)

                # Distribution monitoring
                dm = dist_monitor.evaluate(best_s, fp)
                dist_metrics_all.append(dm)
                vb += 1

        ema.restore(raw, live_state)
        model.train()

        if ws > 1 and dist.is_initialized():
            dist.barrier()
            for t__ in [v_loss, vn]:
                dist.all_reduce(t__, op=dist.ReduceOp.SUM)
            dist.barrier()

        vw       = (v_loss / vn.clamp(1)).item()
        wpcc     = pcc_sum  / max(vb, 1)
        crps_v   = crps_sum / max(vb, 1)
        psd_r    = psd_sum  / max(vb, 1)
        fss_v    = fss_sum  / max(vb, 1)

        # Aggregate distribution metrics
        mean_bias_pct = np.mean([dm["mean_bias_pct"]
                                  for dm in dist_metrics_all]) if dist_metrics_all else 0.
        mean_pred_mm  = np.mean([dm["mean_pred_mm"]
                                  for dm in dist_metrics_all]) if dist_metrics_all else 0.
        mean_obs_mm   = np.mean([dm["mean_obs_mm"]
                                  for dm in dist_metrics_all]) if dist_metrics_all else 0.
        p99_bias      = np.mean([dm.get("P99_bias_mm", 0.)
                                  for dm in dist_metrics_all]) if dist_metrics_all else 0.

        # Combined score: val_loss + α·|mean_bias_pct| / 100
        # Lower is better.  Prevents saving distribution-collapsed checkpoints.
        score = (SCORE_LOSS_WEIGHT * vw +
                 SCORE_MEANBIAS_WEIGHT * abs(mean_bias_pct) / 100.)

        el = time.time() - t0

        if rank == 0:
            star = " ★" if score < best_score else ""
            print(f"{ep:>5}|{te_ml:>10.5f}|{vw:>8.4f}|"
                  f"{score:>8.4f}|{wpcc:>7.4f}|{crps_v:>8.4f}|"
                  f"{psd_r:>7.3f}|{fss_v:>6.3f}|"
                  f"{mean_bias_pct:>+10.1f}|{lr_now:>9.2e}"
                  f"  [{el:.0f}s]{star}")

            # Detailed distribution report every 10 epochs
            if (ep % 10 == 0) and dist_metrics_all:
                dm_avg = {}
                for k in dist_metrics_all[0]:
                    dm_avg[k] = np.mean([d[k] for d in dist_metrics_all])
                print(dist_monitor.log_str(dm_avg))

            ck_base = {
                "epoch"               : ep,
                "model_state_dict"    : raw.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "ema_shadow"          : ema.shadow,
                "val_loss"            : vw,
                "best_score"          : score,
                "wpcc"                : wpcc,
                "crps"                : crps_v,
                "psd_tail_ratio"      : psd_r,
                "mean_bias_pct"       : mean_bias_pct,
                "mean_pred_mm"        : mean_pred_mm,
                "mean_obs_mm"         : mean_obs_mm,
                "p99_bias_mm"         : p99_bias,
                "no_improve"          : no_improve,
                "train_mode"          : TRAIN_MODE,
                "sigma_data"          : sigma_data,
                "unet_in_channels"    : UNET_IN_CH,
                "t_cond"              : T_COND,
                "d2m_channels"        : UNET_D2M_CH,
                "var_map_channels"    : UNET_VAR_MAP_CH,
                "reg_d2m_channels"    : REG_D2M_CH,
                "nrb"                 : NRB,
                "base_channels"       : BASE_CH,
                "channel_mult"        : list(CHANNEL_MULT),
            }
            torch.save(ck_base, LATEST)

            if score < best_score:
                best_score    = score
                best_val_loss = vw
                no_improve    = 0
                torch.save(ck_base, SAVE)
                print(f"  ★ BEST  score={score:.4f}  "
                      f"val_loss={vw:.4f}  wPCC={wpcc:.4f}  "
                      f"CRPS={crps_v:.4f}  mean_bias={mean_bias_pct:+.1f}%  "
                      f"pred_mm={mean_pred_mm:.2f}  obs_mm={mean_obs_mm:.2f}")
                if (ep + 1) % 50 == 0:
                    torch.save(ck_base, os.path.join(
                        CKPT_DIR,
                        f"unet_{TRAIN_MODE}_nrb{NRB}"
                        f"_ep{ep+1:04d}_score{score:.4f}.pth"))
            else:
                no_improve += 1

        if no_improve >= PATIENCE:
            if rank == 0:
                print(f"\n⚠  Early stop ep={ep+1}  "
                      f"best_score={best_score:.4f}")
            break

    if ws > 1 and dist.is_initialized():
        dist.destroy_process_group()

# ══════════════════════════════════════════════════════════════════════════════
# 14.  INFERENCE CLASS
# ══════════════════════════════════════════════════════════════════════════════
class CorrDiffInference:
    """
    Inference wrapper for trained CorrDiff model.
    Loads EMA weights.  No QDM post-processing.
    Returns mean, std, samples, and regressor output mu.
    """
    def __init__(self, reg_ckpt, unet_ckpt, device,
                 cfg_scale=CFG_SCALE, n_ens=8):
        self.dev       = device
        self.cfg_scale = cfg_scale
        self.n_ens     = n_ens
        self.reg       = load_regressor(reg_ckpt, device)

        ck             = torch.load(unet_ckpt, map_location=device)
        tc             = ck.get("t_cond", T_COND)
        self.tc        = tc
        self.train_mode = ck.get("train_mode", "flow_matching")
        _sd            = ck.get("sigma_data", 0.5)
        self.edm_precond = EDMPreconditioning(sigma_data=_sd)
        self.edm_sched = build_edm_schedule(FM_STEPS, sigma_data=_sd)
        self.fm        = FlowMatching(n_steps=FM_STEPS, cfg_scale=cfg_scale)

        self.unet = UNet(
            in_channels=ck.get("unet_in_channels", UNET_IN_CH),
            out_channels=1,
            base_channels=ck.get("base_channels", BASE_CH),
            channel_mult=tuple(ck.get("channel_mult", list(CHANNEL_MULT))),
            num_res_blocks=ck.get("nrb", NRB),
            dropout=0.,
            global_dim=GLOBAL_DIM,
            topo_channels=TOPO_CH,
            use_d2m=True,
            d2m_channels=ck.get("d2m_channels", UNET_D2M_CH),
            use_var_map=True,
            var_map_channels=ck.get("var_map_channels", UNET_VAR_MAP_CH),
            temporal_frames=tc,
        ).to(device)

        state = ({k: v.to(device) for k, v in ck["ema_shadow"].items()}
                 if "ema_shadow" in ck
                 else {k.replace("module.", ""): v
                       for k, v in ck["model_state_dict"].items()})
        self.unet.load_state_dict(state)
        self.unet.eval()

    @torch.no_grad()
    def predict(self, coarse, var_map, topo, d2m,
                doy, hour, tc_frames=None):
        dev     = self.dev
        coarse  = coarse.to(dev)
        var_map = var_map.to(dev)
        topo    = topo.to(dev)
        d2m     = d2m.to(dev)
        gf      = torch.stack([doy.to(dev), hour.to(dev)], dim=1).float()
        xi      = build_coarse_input(coarse, var_map)
        tp      = expand_topo(topo)

        if tc_frames is None:
            coarse_up = F.interpolate(coarse, scale_factor=4,
                                       mode='bilinear', align_corners=False)
            tc_frames = coarse_up.expand(-1, self.tc, -1, -1).to(dev)
        else:
            tc_frames = tc_frames.to(dev)

        mu = self.reg(xi, topo=tp, global_features=gf, d2m=d2m)

        samples = []
        for _ in range(self.n_ens):
            if self.train_mode == "flow_matching":
                x_cond = torch.cat([mu, tc_frames], dim=1)
                s = (self.fm.sample(self.unet, x_cond,
                                    topo=tp, global_features=gf,
                                    d2m=d2m, var_map=var_map,
                                    cfg_scale=self.cfg_scale,
                                    T=self.tc)
                     + mu)
            else:
                s = (ddim_sample(self.unet, mu, tc_frames, tp, gf,
                                  d2m, var_map, self.edm_sched,
                                  dev, self.edm_precond)
                     + mu)
            s = PhysicsGuide.apply(s, coarse,
                                   enforce_mass=True,
                                   enforce_dry=True)
            samples.append(s)

        samples = torch.stack(samples)
        return {
            "mean"   : samples.mean(0),
            "std"    : samples.std(0),
            "samples": samples,
            "mu"     : mu,
        }

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     default=None,
                        choices=["flow_matching", "corrdiff_residual"])
    parser.add_argument("--epochs",   type=int,   default=None)
    parser.add_argument("--batch",    type=int,   default=None)
    parser.add_argument("--lr",       type=float, default=None)
    parser.add_argument("--dropout",  type=float, default=None)
    parser.add_argument("--patience", type=int,   default=None)
    parser.add_argument("--sigma_data", type=float, default=None)
    args = parser.parse_args()

    if args.mode       is not None: TRAIN_MODE = args.mode
    if args.epochs     is not None: EPOCHS     = args.epochs
    if args.batch      is not None: BATCH      = args.batch
    if args.lr         is not None: LR         = args.lr
    if args.dropout    is not None: DROPOUT    = args.dropout
    if args.patience   is not None: PATIENCE   = args.patience
    if args.sigma_data is not None: SIGMA_DATA_OVERRIDE = args.sigma_data

    train()
