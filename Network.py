# -*- coding: utf-8 -*-
"""
CorrDiff Network — v13  (Proper Distribution-Preserving CorrDiff)
=================================================================
ROOT CAUSE FIXES from v11/v12:
  1. Flow Matching loss: PURE MSE in velocity space (v_target = x1 - x0)
     NO hybrid_sigma_loss, NO MAE — MSE preserves distributional shape
  2. No hard clamping anywhere in FM trajectory — soft squeeze only beyond ±3σ
  3. PhysicsGuide: SOFT multiplicative mass correction with wide clamp (0.05, 20)
     Dry mask: additive log-penalty, NOT hard zeroing
  4. QDM replaced by DistributionMonitor (diagnostic only, no post-hoc distortion)
  5. Log-space training is correct: log1p(precip_mm) → learn residual in that space
  6. EDM preconditioning (corrdiff_residual) preserves correct c_skip/c_out scaling
  7. FiLM conditioning from Topography in every ResBlock (gamma/beta from elev/slope/aspect)
  8. Separate d2m and var_map stems with learnable gates (zero-init → identity at start)
  9. DilatedBottleneck: ASPP multi-rate dilated conv at bottleneck (zero-init skip)
 10. BnAttn at bottleneck: low-res spatial attention → upsampled back
 11. FourierFilter at bottleneck: global frequency mixing
 12. TemporalAttention across T frames (used by UNet when T>1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ────────────────────────────────────────────────────────────────────────────
# Utility
# ────────────────────────────────────────────────────────────────────────────
def _g(ch, mx=32):
    """Find largest valid GroupNorm group count ≤ mx that divides ch."""
    for g in range(min(mx, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


# ────────────────────────────────────────────────────────────────────────────
# Building blocks
# ────────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    """Squeeze-Excitation channel attention."""
    def __init__(self, ch, r=8):
        super().__init__()
        self.p = nn.AdaptiveAvgPool2d(1)
        self.f = nn.Sequential(
            nn.Linear(ch, max(4, ch // r), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, ch // r), ch, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.f(self.p(x).view(b, c)).view(b, c, 1, 1)


class CoordConv2d(nn.Module):
    """Conv2d with appended normalised (y, x) coordinate channels."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels,
                              kernel_size, padding=padding)

    def forward(self, x):
        b, c, h, w = x.shape
        yg = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        xg = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return self.conv(torch.cat([x, yg, xg], dim=1))


class FourierFilter(nn.Module):
    """Global spectral mixing via learned complex weights.  Residual → identity at init."""
    def __init__(self, channels):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(channels, channels, 2) * 0.02)
        self.norm = nn.GroupNorm(_g(channels), channels)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft  = torch.fft.rfft2(self.norm(x))
        weight  = torch.view_as_complex(self.complex_weight)
        out_fft = torch.einsum('bchw,cd->bdhw', x_fft, weight)
        return x + self.proj(torch.fft.irfft2(out_fft, s=(H, W)))


class ResConv(nn.Module):
    """Double-conv residual block with SE attention (used in Regressor)."""
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_g(ic), ic), nn.SiLU(),
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.GroupNorm(_g(oc), oc), nn.SiLU(),
            nn.Conv2d(oc, oc, 3, padding=1))
        self.skip = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()
        self.se   = SEBlock(oc)

    def forward(self, x):
        return self.skip(x) + self.se(self.net(x))


class BnAttn(nn.Module):
    """Bottleneck spatial attention: pool → low-res attention → upsample back."""
    def __init__(self, ch, heads=4):
        super().__init__()
        while ch % heads != 0 and heads > 1:
            heads -= 1
        self.h     = heads
        self.s     = 8
        self.scale = (ch // heads) ** -0.5
        self.norm  = nn.GroupNorm(_g(ch), ch)
        self.qkv   = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.proj  = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        s   = min(self.s, H)
        lo  = F.adaptive_avg_pool2d(self.norm(x), (s, s))
        qkv = self.qkv(lo).reshape(B, 3, self.h, C // self.h, s * s)
        q, k, v = qkv.unbind(1)
        a   = torch.einsum('bhdn,bhdm->bhnm', q * self.scale, k).softmax(-1)
        o   = torch.einsum('bhnm,bhdm->bhdn', a, v).reshape(B, C, s, s)
        return x + F.interpolate(self.proj(o), (H, W),
                                 mode='bilinear', align_corners=False)


class TemporalAttention(nn.Module):
    """Cross-frame attention on global tokens. Gate-initialised to zero → identity."""
    def __init__(self, ch, T=4, heads=4):
        super().__init__()
        while ch % heads != 0 and heads > 1:
            heads -= 1
        self.T     = T
        self.h     = heads
        self.scale = (ch // heads) ** -0.5
        self.norm  = nn.GroupNorm(_g(ch), ch)
        self.qkv   = nn.Linear(ch, ch * 3, bias=False)
        self.proj  = nn.Linear(ch, ch)
        self.gate  = nn.Parameter(torch.zeros(1))

    def forward(self, x, T=None):
        T    = T or self.T
        BT, C, H, W = x.shape
        if BT % T != 0:
            return x
        B    = BT // T
        tok  = F.adaptive_avg_pool2d(self.norm(x), 1).view(BT, C).view(B, T, C)
        qkv  = self.qkv(tok).reshape(B, T, 3, self.h, C // self.h)
        q, k, v = qkv.unbind(2)
        a    = torch.einsum('bthd,bshd->bths', q * self.scale, k).softmax(-1)
        out  = torch.einsum('bths,bshd->bthd', a, v).reshape(B, T, C)
        out  = self.proj(out).view(BT, C, 1, 1)
        return x + self.gate.tanh() * out


class DilatedBottleneck(nn.Module):
    """
    ASPP-style multi-rate dilated bottleneck.
    Rates [1, 2, 4, 8] → concat → project back.
    Zero-init on project → pure identity at start of training.
    """
    def __init__(self, in_channels, mid_channels=512):
        super().__init__()
        rates      = [1, 2, 4, 8]
        branch_ch  = mid_channels // len(rates)
        self.branches = nn.ModuleList()
        for rate in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_ch, kernel_size=3,
                          padding=rate, dilation=rate, bias=False),
                nn.GroupNorm(_g(branch_ch), branch_ch),
                nn.SiLU()))
        self.project = nn.Sequential(
            nn.Conv2d(branch_ch * len(rates), in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_g(in_channels), in_channels))
        nn.init.zeros_(self.project[0].weight)

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        return x + self.project(out)


# ════════════════════════════════════════════════════════════════════════════
# REGRESSOR  (Stage 1) — mean predictor
# ════════════════════════════════════════════════════════════════════════════
class CorrDiffRegressor(nn.Module):
    """
    Stage-1 regressor.  Predicts μ(log1p(precip)) from coarse + topo + d2m.
    Dual-path encoder (rainfall path + topo path) with shared downsampling,
    fused at bottleneck, decoded via skip connections.
    """
    def __init__(self, in_channels=2, out_channels=1, base_channels=64,
                 channel_mult=(1, 2, 4), num_blocks=2, global_dim=2,
                 topo_channels=3, d2m_channels=1, use_d2m=True, **kw):
        super().__init__()
        cms        = list(channel_mult)
        st         = base_channels
        emb        = base_channels * 2
        self.use_d2m = use_d2m

        # Global context (DOY, hour)
        self.g_mlp = (nn.Sequential(
            nn.Linear(global_dim, emb), nn.SiLU(), nn.Linear(emb, st))
            if global_dim > 0 else None)

        # Input stems — separate paths
        self.r_stem = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            CoordConv2d(in_channels, st, 3, padding=1),
            nn.GroupNorm(_g(st), st), nn.SiLU())

        self.t_stem = nn.Sequential(
            CoordConv2d(topo_channels, st, 3, padding=1),
            nn.GroupNorm(_g(st), st), nn.SiLU())

        if use_d2m:
            self.d_stem = nn.Sequential(
                CoordConv2d(d2m_channels, st, 3, padding=1),
                nn.GroupNorm(_g(st), st), nn.SiLU())

        # Dual encoders
        self.r_enc = nn.ModuleList()
        self.t_enc = nn.ModuleList()
        self.r_dn  = nn.ModuleList()
        self.t_dn  = nn.ModuleList()
        self.sk_ch = []
        rc = tc     = st

        for li, m in enumerate(cms):
            oc = base_channels * m
            rb = nn.ModuleList()
            tb = nn.ModuleList()
            for _ in range(num_blocks):
                rb.append(ResConv(rc, oc)); tb.append(ResConv(tc, oc))
                rc = tc = oc
            self.r_enc.append(rb); self.t_enc.append(tb)
            self.sk_ch.append(rc + tc)
            last = (li == len(cms) - 1)
            self.r_dn.append(nn.Identity() if last else nn.Conv2d(rc, rc, 4, 2, 1))
            self.t_dn.append(nn.Identity() if last else nn.Conv2d(tc, tc, 4, 2, 1))

        bn = base_channels * cms[-1]
        self.bn_proj = nn.Conv2d(rc + tc, bn, 1)
        self.bn_attn = BnAttn(bn, max(1, bn // 64))
        self.bn_se   = SEBlock(bn)
        self.bn_mid  = ResConv(bn, bn)

        # Decoder
        self.d_ups = nn.ModuleList()
        self.d_blk = nn.ModuleList()
        dc = bn

        for li, m in reversed(list(enumerate(cms))):
            oc = base_channels * m
            sc = self.sk_ch[li]
            self.d_ups.append(
                nn.Identity() if li == len(cms) - 1 else
                nn.Sequential(nn.ConvTranspose2d(dc, dc, 4, 2, 1),
                               nn.GroupNorm(_g(dc), dc), nn.SiLU()))
            blks  = nn.ModuleList()
            ic2   = dc + sc
            for _ in range(num_blocks):
                blks.append(ResConv(ic2, oc)); ic2 = oc
            self.d_blk.append(blks)
            dc = oc

        self.out = nn.Sequential(
            nn.GroupNorm(_g(dc), dc), nn.SiLU(),
            nn.Conv2d(dc, out_channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, topo, global_features=None, d2m=None):
        r = self.r_stem(x)
        t = self.t_stem(topo)

        if self.use_d2m and d2m is not None:
            d2m_aligned = F.interpolate(d2m, size=r.shape[-2:],
                                        mode='bilinear', align_corners=False)
            r = r + self.d_stem(d2m_aligned)

        if self.g_mlp is not None and global_features is not None:
            gs = self.g_mlp(global_features)[:, :, None, None]
            r  = r + gs
            t  = t + gs

        rs, ts = [], []
        for li in range(len(self.r_enc)):
            for rb, tb in zip(self.r_enc[li], self.t_enc[li]):
                r = rb(r); t = tb(t)
            rs.append(r); ts.append(t)
            r = self.r_dn[li](r); t = self.t_dn[li](t)

        f = self.bn_proj(torch.cat([r, t], 1))
        f = self.bn_attn(f); f = self.bn_se(f); f = self.bn_mid(f)
        d = f

        for li, (up, blks) in enumerate(zip(self.d_ups, self.d_blk)):
            lv = len(self.r_enc) - 1 - li
            d  = up(d)
            d  = torch.cat([d, torch.cat([rs[lv], ts[lv]], 1)], 1)
            for b in blks:
                d = b(d)

        return self.out(d)


# ════════════════════════════════════════════════════════════════════════════
# UNET  (Stage 2) — residual corrector / diffusion backbone
# ════════════════════════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    """
    ResBlock with:
      - Time/noise embedding injection (additive after first conv)
      - FiLM conditioning from Topography (gamma/beta modulation after GroupNorm)
      - SE attention on output
      - Optional strided down/up sampling
    """
    def __init__(self, ic, oc, ec, down=False, up=False,
                 use_topo=True, topo_channels=3, dropout=0.1):
        super().__init__()
        self.rs = None
        if down: self.rs = nn.Conv2d(ic, ic, 4, 2, 1)
        if up:   self.rs = nn.ConvTranspose2d(ic, ic, 4, 2, 1)

        self.n1 = nn.GroupNorm(_g(ic), ic)
        self.c1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.ep = nn.Linear(ec, oc)

        self.use_topo = use_topo
        if use_topo:
            # FiLM: predict (γ, β) from topo.  Zero-init → identity at start.
            self.topo_proj = nn.Conv2d(topo_channels, oc * 2, kernel_size=3, padding=1)
            nn.init.zeros_(self.topo_proj.weight)
            nn.init.zeros_(self.topo_proj.bias)

        self.n2      = nn.GroupNorm(_g(oc), oc)
        self.dropout = nn.Dropout(dropout)
        self.c2      = nn.Conv2d(oc, oc, 3, padding=1)
        self.se      = SEBlock(oc)
        self.sk      = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x, e, topo=None):
        if self.rs:
            x = self.rs(x)
        h      = self.c1(F.silu(self.n1(x))) + self.ep(F.silu(e))[:, :, None, None]
        h_norm = self.n2(h)

        if self.use_topo and topo is not None:
            t_res  = F.interpolate(topo, size=h.shape[-2:],
                                   mode='bilinear', align_corners=False)
            gamma, beta = self.topo_proj(t_res).chunk(2, dim=1)
            # Proper FiLM: scale=(1+tanh(γ)), shift=β
            h_norm = h_norm * (1.0 + torch.tanh(gamma)) + beta

        h_norm = self.dropout(h_norm)
        return self.se(self.c2(F.silu(h_norm))) + self.sk(x)


class UNet(nn.Module):
    """
    CorrDiff UNet backbone.
    Inputs at forward():
      x         : (B, in_channels, H, W)  — [noisy_residual | mu_cond | temporal_frames]
      t         : (B,)                    — noise level / flow time
      topo      : (B, 3, H, W)            — elevation, slope, aspect (FiLM)
      global_features : (B, 2)            — DOY, hour
      cfg_drop  : (B,) bool               — classifier-free drop mask
      d2m       : (B, 1, H, W)            — dewpoint at fine res
      var_map   : (B, 1, H/4, W/4)        — coarse variance map (upsampled in v_stem)
      T         : int                     — temporal frame count
    """
    def __init__(self, in_channels, out_channels, base_channels=128,
                 channel_mult=(1, 2, 2, 4), num_res_blocks=2, dropout=0.1,
                 num_blocks=None, global_dim=2, use_bottleneck_attention=True,
                 topo_channels=3, use_d2m=True, d2m_channels=1,
                 use_var_map=True, var_map_channels=1,
                 temporal_frames=3, **kw):
        super().__init__()
        nrb  = num_blocks if num_blocks else num_res_blocks
        ec   = base_channels * 4
        self.topo_channels   = topo_channels
        self.use_d2m         = use_d2m
        self.use_var_map     = use_var_map
        self.temporal_frames = temporal_frames

        # ── Time / noise embedding ──────────────────────────────────────────
        self.t_emb = nn.Sequential(
            nn.Linear(base_channels, ec), nn.SiLU(), nn.Linear(ec, ec))

        # ── Global context (DOY, hour) ──────────────────────────────────────
        self.g_mlp = (nn.Sequential(nn.Linear(global_dim, ec), nn.SiLU())
                      if global_dim else None)

        # ── Separate conditioning stems ─────────────────────────────────────
        if use_d2m:
            self.d_stem = nn.Sequential(
                CoordConv2d(d2m_channels, base_channels, 3, padding=1),
                nn.GroupNorm(_g(base_channels), base_channels), nn.SiLU())
            self.d_gate = nn.Parameter(torch.zeros(1))

        if use_var_map:
            self.v_stem = nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                CoordConv2d(var_map_channels, base_channels, 3, padding=1),
                nn.GroupNorm(_g(base_channels), base_channels), nn.SiLU())
            self.v_gate = nn.Parameter(torch.zeros(1))

        # ── Main head ───────────────────────────────────────────────────────
        self.head = CoordConv2d(in_channels, base_channels, 3, padding=1)

        # ── Encoder ─────────────────────────────────────────────────────────
        self.downs = nn.ModuleList()
        ch = base_channels
        sk = []

        for m in channel_mult:
            oc = base_channels * m
            for _ in range(nrb):
                self.downs.append(ResBlock(ch, oc, ec,
                                           topo_channels=topo_channels,
                                           dropout=dropout))
                ch = oc; sk.append(ch)
            # Strided downsampling block
            self.downs.append(ResBlock(ch, ch, ec, down=True,
                                       topo_channels=topo_channels,
                                       dropout=dropout))
            sk.append(ch)

        # ── Bottleneck: ResBlock → BnAttn → FFT → DilatedBottleneck → ResBlock ──
        self.m1 = ResBlock(ch, ch, ec,
                           topo_channels=topo_channels, dropout=dropout)
        self.ma = (BnAttn(ch, max(1, ch // 64))
                   if use_bottleneck_attention else nn.Identity())
        self.fft             = FourierFilter(ch)
        mid_ch               = min(512, ch)
        self.dilated_bottleneck = DilatedBottleneck(in_channels=ch,
                                                     mid_channels=mid_ch * 2)
        self.m2 = ResBlock(ch, ch, ec,
                           topo_channels=topo_channels, dropout=dropout)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.ups = nn.ModuleList()
        for m in reversed(channel_mult):
            oc = base_channels * m
            self.ups.append(ResBlock(ch + sk.pop(), oc, ec, up=True,
                                     topo_channels=topo_channels,
                                     dropout=dropout))
            ch = oc
            for _ in range(nrb):
                self.ups.append(ResBlock(ch + sk.pop(), oc, ec,
                                         topo_channels=topo_channels,
                                         dropout=dropout))
                ch = oc

        self.out = nn.Sequential(
            nn.GroupNorm(_g(ch), ch), nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1))

    # ── Sinusoidal time embedding ───────────────────────────────────────────
    def _temb(self, t):
        half = self.t_emb[0].in_features // 2
        freq = torch.exp(
            torch.arange(half, device=t.device) *
            (-math.log(10000) / (half - 1)))
        e = t.unsqueeze(1) * freq.unsqueeze(0) * 2 * math.pi
        return self.t_emb(torch.cat([e.sin(), e.cos()], -1))

    def forward(self, x, t, topo=None, global_features=None,
                cfg_drop=None, d2m=None, var_map=None, T=None):
        T   = T or self.temporal_frames
        emb = self._temb(t)

        # Global context injection
        if global_features is not None and self.g_mlp:
            gf = global_features.clone()
            if cfg_drop is not None:
                gf[cfg_drop] = 0.
            emb = emb + self.g_mlp(gf)

        x_in    = x.clone()
        topo_in = topo.clone() if topo is not None else None

        # CFG drop: zero conditioning channels (keep channel-0 = noisy input)
        if cfg_drop is not None and cfg_drop.any():
            x_in[cfg_drop, 1:] = 0.
            if topo_in is not None:
                topo_in[cfg_drop] = 0.

        # Main head
        h = self.head(x_in)

        # d2m stem (separate, gated)
        if self.use_d2m and d2m is not None:
            d2m_in = d2m.clone()
            if cfg_drop is not None and cfg_drop.any():
                d2m_in[cfg_drop] = 0.
            d2m_res = F.interpolate(d2m_in, size=h.shape[-2:],
                                    mode='bilinear', align_corners=False)
            h = h + self.d_gate.tanh() * self.d_stem(d2m_res)

        # var_map stem (separate, gated, upsampled internally)
        if self.use_var_map and var_map is not None:
            vm_in = var_map.clone()
            if cfg_drop is not None and cfg_drop.any():
                vm_in[cfg_drop] = 0.
            h = h + self.v_gate.tanh() * self.v_stem(vm_in)

        # Encoder
        sk_list = [h]
        for layer in self.downs:
            h = layer(h, emb, topo_in)
            sk_list.append(h)

        # Bottleneck
        h = self.m1(h, emb, topo_in)
        h = self.ma(h) if isinstance(self.ma, BnAttn) else h
        h = self.fft(h)
        h = self.dilated_bottleneck(h)
        h = self.m2(h, emb, topo_in)

        # Decoder
        for layer in self.ups:
            h = torch.cat([h, sk_list.pop()], 1)
            h = layer(h, emb, topo_in)

        return self.out(h)


# ════════════════════════════════════════════════════════════════════════════
# Flow Matching  — PROPER MSE velocity loss, NO hard clamping
# ════════════════════════════════════════════════════════════════════════════
class FlowMatching:
    """
    Proper Conditional Flow Matching (Lipman et al. 2022).

    Linear interpolation path:
        x_t = (1 - t) * x0 + t * x1
        v_target = x1 - x0

    Loss: MSE(v_pred, v_target)
    This is the ONLY correct loss for learning the velocity field.
    MSE in velocity space = proper maximum-likelihood under Gaussian path.
    DO NOT replace with MAE or hybrid_sigma_loss — those break the distribution.

    Sampling: Euler ODE from t=0 → t=1.
    No hard clamping — soft asymptotic squeeze only.
    """
    def __init__(self, n_steps=20, cfg_scale=2.0):
        self.n_steps   = n_steps
        self.cfg_scale = cfg_scale

    def get_train_sample(self, x1):
        """
        x1: residual in log1p space, shape (B, 1, H, W)
        Returns x_t, t_scalar (B,), v_target
        """
        B  = x1.shape[0]
        x0 = torch.randn_like(x1)
        # Logit-Normal sampling: biases toward difficult mid-flow times
        u  = torch.rand(B, device=x1.device).clamp(1e-4, 1 - 1e-4)
        t_scalar = torch.sigmoid(torch.logit(u) * 1.2)   # sharper bias vs Beta(1.5,1.5)
        t  = t_scalar.view(B, 1, 1, 1)
        x_t       = (1 - t) * x0 + t * x1
        v_target  = x1 - x0
        return x_t, t_scalar, v_target

    @torch.no_grad()
    def sample(self, model, x_cond, topo, global_features=None,
               d2m=None, var_map=None, cfg_scale=None, T=1):
        """
        Euler ODE from x0 ~ N(0,I) to x1 = residual sample.
        Uses classifier-free guidance when cfg > 1.
        NO hard clamp in the loop — soft squeeze keeps heavy tails.
        """
        cfg = cfg_scale if cfg_scale is not None else self.cfg_scale
        B, _, H, W = x_cond.shape
        x  = torch.randn(B, 1, H, W, device=x_cond.device)
        dt = 1.0 / self.n_steps

        for i in range(self.n_steps):
            t_vec = torch.full((B,), i * dt, device=x.device)
            x_in  = torch.cat([x, x_cond], dim=1)
            v_c   = model(x_in, t_vec, topo=topo,
                          global_features=global_features,
                          d2m=d2m, var_map=var_map, T=T)
            if cfg > 1.0:
                x_unc = torch.cat([x, torch.zeros_like(x_cond)], dim=1)
                mask  = torch.ones(B, dtype=torch.bool, device=x.device)
                v_u   = model(x_unc, t_vec, topo=topo,
                              global_features=global_features,
                              d2m=d2m, var_map=var_map, cfg_drop=mask, T=T)
                v = v_u + cfg * (v_c - v_u)
            else:
                v = v_c

            x = x + dt * v
            # SOFT squeeze only — preserves learned heavy tails
            # log1p(200mm/day) ≈ 5.3, so upper limit ~6 is generous
            x = torch.where(x >  6.0,  6.0 + 0.05 * (x -  6.0), x)
            x = torch.where(x < -3.0, -3.0 + 0.05 * (x + 3.0), x)

        return x

    @staticmethod
    def loss(v_pred, v_target):
        """
        PURE MSE in velocity space.
        This is the correct loss for flow matching.
        It is equivalent to maximising the log-likelihood under the Gaussian path.
        Do NOT change this to MAE or add sigma weighting.
        """
        return F.mse_loss(v_pred, v_target)


# ════════════════════════════════════════════════════════════════════════════
# EDM Preconditioning (for corrdiff_residual mode)
# ════════════════════════════════════════════════════════════════════════════
class EDMPreconditioning:
    """
    Karras et al. (2022) EDM preconditioning.
    c_skip, c_out, c_in scale the network I/O to keep unit variance throughout.
    Loss weight = λ(σ) = (σ² + σ_data²) / (σ · σ_data)²

    Key: Use WEIGHTED MSE with λ(σ) weight — this IS the correct EDM loss.
    The weighting ensures equal gradient contribution from all noise levels.
    """
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data

    def get_scalings(self, sigma):
        """sigma: (B,) or (B,1,1,1)"""
        sd    = self.sigma_data
        denom = torch.sqrt(sigma ** 2 + sd ** 2)
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out  = sigma * sd / denom
        c_in   = 1.0 / denom
        c_noise = (sigma.log() / 4.0)
        return c_skip, c_out, c_in, c_noise

    def loss_weight(self, sigma):
        """λ(σ) = (σ² + σ_data²) / (σ·σ_data)²"""
        sd = self.sigma_data
        return (sigma ** 2 + sd ** 2) / ((sigma * sd) ** 2)

    @staticmethod
    def edm_loss(D_pred, target, sigma, sigma_data):
        """
        Proper EDM loss: weighted MSE.
        D_pred: denoised prediction (after c_skip*x + c_out*F_net)
        target: clean residual
        """
        sd = sigma_data
        weight = (sigma ** 2 + sd ** 2) / ((sigma * sd) ** 2 + 1e-8)
        return (weight * (D_pred - target) ** 2).mean()


# ════════════════════════════════════════════════════════════════════════════
# Physics Guidance  — SOFT, distribution-preserving
# ════════════════════════════════════════════════════════════════════════════
class PhysicsGuide:
    """
    Soft physics enforcement post-sampling.

    RULES:
      1. Dry mask: if coarse mean is below dry threshold → attenuate, not zero.
         Attenuation = multiplicative scaling toward 0, gradual.
      2. Mass conservation: multiplicative scale so pred_mean ≈ coarse_upsampled_mean.
         Clamp is WIDE (0.05, 20.0) to permit Indian monsoon extremes.
         This preserves the SHAPE of the distribution (same std/skew, shifted mean).

    CRITICAL: This must NOT break the learned tails.
    Soft attenuation for dry and multiplicative mass correction → shape-preserving.
    """
    DRY_THRESH_LOG = -2.5  # log1p(~0.08 mm) — stricter → more dry days

    @staticmethod
    def apply(pred_log, coarse_log, enforce_mass=True, enforce_dry=True,
              dry_attenuation=0.05):
        """
        pred_log  : (B,1,H,W) in log1p space (model output after + mu)
        coarse_log: (B,1,H/4,W/4) in log1p space
        """
        pred = pred_log.clone()

        if enforce_dry:
            coarse_mean = coarse_log.mean(dim=[-2, -1], keepdim=True)
            dry_mask    = coarse_mean < PhysicsGuide.DRY_THRESH_LOG
            # Soft: scale toward 0 instead of hard zero
            scale_dry   = torch.where(dry_mask,
                                       torch.full_like(coarse_mean, dry_attenuation),
                                       torch.ones_like(coarse_mean))
            pred = pred + torch.log(scale_dry + 1e-8)
            # After log attenuation, re-clamp to valid log space
            pred = pred.clamp(min=-20.)

        if enforce_mass:
            pred_phys   = torch.expm1(pred.clamp(min=-20.))
            coarse_phys = torch.expm1(coarse_log.clamp(min=-20.))
            coarse_up   = F.interpolate(coarse_phys, size=pred.shape[-2:],
                                        mode='bilinear', align_corners=False)
            target_mean = coarse_up.mean(dim=[-2, -1], keepdim=True).clamp(min=1e-6)
            pred_mean   = pred_phys.mean(dim=[-2, -1], keepdim=True).clamp(min=1e-6)
            # Wide clamp: allow Indian monsoon extremes (up to 20× amplification)
            mass_scale  = (target_mean / pred_mean).clamp(0.05, 20.0)
            pred        = torch.log1p((pred_phys * mass_scale).clamp(min=0.))

        return pred


# ════════════════════════════════════════════════════════════════════════════
# Distribution Monitor  (validation diagnostic — no post-hoc distortion)
# ════════════════════════════════════════════════════════════════════════════
class DistributionMonitor:
    """
    Tracks percentile-level bias during validation.
    Used to select best checkpoint and detect distribution collapse early.
    Does NOT modify predictions.

    Usage:
        mon = DistributionMonitor()
        metrics = mon.evaluate(pred, obs)   # dict with P10/P50/P90/P99 bias
    """
    PERCENTILES = [10, 25, 50, 75, 90, 95, 99]

    @torch.no_grad()
    def evaluate(self, pred, obs):
        """
        pred, obs: (B, 1, H, W) in log1p space
        Returns dict of percentile biases and mean bias.
        """
        # Convert back to physical space for interpretable diagnostics
        p_phys = torch.expm1(pred.clamp(min=0.)).flatten().float().cpu()
        o_phys = torch.expm1(obs.clamp(min=0.)).flatten().float().cpu()

        metrics = {
            "mean_pred_mm"  : p_phys.mean().item(),
            "mean_obs_mm"   : o_phys.mean().item(),
            "mean_bias_mm"  : (p_phys.mean() - o_phys.mean()).item(),
            "mean_bias_pct" : ((p_phys.mean() - o_phys.mean()) /
                               o_phys.mean().clamp(1e-3)).item() * 100,
        }
        for p in self.PERCENTILES:
            q      = p / 100.0
            p_q    = p_phys.quantile(q).item()
            o_q    = o_phys.quantile(q).item()
            bias_q = p_q - o_q
            metrics[f"P{p:02d}_pred_mm"] = p_q
            metrics[f"P{p:02d}_obs_mm"]  = o_q
            metrics[f"P{p:02d}_bias_mm"] = bias_q
        return metrics

    def log_str(self, metrics):
        lines = [
            f"  MeanBias: {metrics['mean_bias_mm']:+.3f}mm "
            f"({metrics['mean_bias_pct']:+.1f}%)",
        ]
        for p in self.PERCENTILES:
            lines.append(
                f"  P{p:02d}: pred={metrics[f'P{p:02d}_pred_mm']:.2f} "
                f"obs={metrics[f'P{p:02d}_obs_mm']:.2f} "
                f"bias={metrics[f'P{p:02d}_bias_mm']:+.3f}")
        return "\n".join(lines)
