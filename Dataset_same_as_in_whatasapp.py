import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

class UpscaleDataset(torch.utils.data.Dataset):
    PRECIPITATION_NAMES = [
        'rf', 'RF', 'rainfall', 'RAINFALL', 'precipitation', 'PRECIPITATION',
        'pr', 'PR', 'tp', 'TP', 'total_precipitation',
        'prec', 'PREC', 'rain', 'RAIN', 'precip', 'PRECIP'
    ]

    def __init__(
        self,
        nc_file,
        oro_file,
        d2m_file=None,
        downscale_factor=4,
        normalize=True,
        device="cuda",
        auto_detect_var=True,
        variable_name=None,
        split='train',
        train_frac=0.65,
        coarse_qdm=None,
    ):
        self.split      = split
        self.downscale  = downscale_factor
        self.has_d2m    = d2m_file is not None

        # ── LOAD PRECIP ──────────────────────────────────────────
        try:
            ds = xr.open_dataset(nc_file, engine="netcdf4")
        except Exception:
            ds = xr.open_dataset(nc_file, engine="h5netcdf")

        lat_name = next((c for c in ds.coords if c.lower() in ['lat', 'latitude']), None)
        lon_name = next((c for c in ds.coords if c.lower() in ['lon', 'longitude']), None)
        if lat_name and lon_name:
            ds = ds.sortby([lat_name, lon_name])

        # ── LOAD TOPO ────────────────────────────────────────────
        ds_oro = xr.open_dataset(oro_file, engine="netcdf4")
        lat_oro = next((c for c in ds_oro.coords if c.lower() in ['lat', 'latitude']), None)
        lon_oro = next((c for c in ds_oro.coords if c.lower() in ['lon', 'longitude']), None)
        if lat_oro and lon_oro:
            ds_oro = ds_oro.sortby([lat_oro, lon_oro])
        topo = ds_oro['topology'].values.astype(np.float32)
        topo = np.nan_to_num(topo, nan=0.0)

        # ── LOAD D2M ─────────────────────────────────────────────
        if self.has_d2m:
            ds_d2m = xr.open_dataset(d2m_file, engine="netcdf4")
            lat_d2m = next((c for c in ds_d2m.coords if c.lower() in ['lat', 'latitude']), None)
            lon_d2m = next((c for c in ds_d2m.coords if c.lower() in ['lon', 'longitude']), None)
            if lat_d2m and lon_d2m:
                ds_d2m = ds_d2m.sortby([lat_d2m, lon_d2m])
            d2m_var = None
            for v in ['d2m', 'D2M', 'dewpoint_temperature_2m', 'dew_point_temperature']:
                if v in ds_d2m.data_vars:
                    d2m_var = v; break
            if d2m_var is None:
                d2m_var = list(ds_d2m.data_vars)[0]
            d2m_raw = ds_d2m[d2m_var].values.astype(np.float32)
            if d2m_raw.ndim == 4:
                d2m_raw = d2m_raw[:, 0]
            d2m_raw = np.nan_to_num(d2m_raw, nan=0.0)
            if d2m_raw.mean() > 100:
                d2m_raw = d2m_raw - 273.15

        # ── VARIABLE SELECTION ───────────────────────────────────
        if variable_name is not None:
            data = ds[variable_name].values
        elif auto_detect_var:
            found = None
            for v in ds.data_vars:
                if any(k.lower() in v.lower() for k in self.PRECIPITATION_NAMES):
                    found = v; break
            if found is None:
                raise ValueError("No precipitation variable detected")
            data = ds[found].values
        else:
            data = ds[list(ds.data_vars)[0]].values

        data = data.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        data = np.clip(data, a_min=0.0, a_max=None)

        dt        = pd.to_datetime(ds["TIME"].values)
        self.doy  = torch.tensor(dt.dayofyear.values / 366.0, dtype=torch.float32)
        self.hour = torch.tensor(dt.hour.values / 24.0,       dtype=torch.float32)

        self.log_transformed = normalize

        # ── TOPO NORM ────────────────────────────────────────────
        topo_mean = topo.mean(); topo_std = topo.std() + 1e-8
        if normalize:
            topo = (topo - topo_mean) / topo_std
        self.topo_mean = topo_mean; self.topo_std = topo_std

        # ── D2M NORM ─────────────────────────────────────────────
        if self.has_d2m and normalize:
            self.d2m_mean = float(np.nanmean(d2m_raw))
            self.d2m_std  = float(np.nanstd(d2m_raw) + 1e-8)
            d2m_raw       = (d2m_raw - self.d2m_mean) / self.d2m_std
        elif self.has_d2m:
            self.d2m_mean = 0.0; self.d2m_std = 1.0

        # ── CROP TO MULTIPLE OF 16 ───────────────────────────────
        H, W = data.shape[1], data.shape[2]
        H    = (H // 16) * 16; W = (W // 16) * 16
        data = data[:, :H, :W]
        topo = topo[:H, :W]
        if self.has_d2m:
            d2m_raw = d2m_raw[:, :H, :W]

        self.H, self.W = H, W
        T = data.shape[0]
        self.doy  = self.doy[:T]
        self.hour = self.hour[:T]

        self.topo_tensor = torch.from_numpy(topo).unsqueeze(0).contiguous()
        if self.has_d2m:
            self.d2m_tensor = torch.from_numpy(d2m_raw).unsqueeze(1).contiguous()

        compute_device = (torch.device("cuda")
                          if device == "cuda" and torch.cuda.is_available()
                          else torch.device("cpu"))

        print(f"Dataset Split        : {self.split}")
        print(f"Fine resolution      : {self.H}x{self.W}")
        print(f"Coarse resolution    : {self.H//downscale_factor}x{self.W//downscale_factor}")
        print(f"D2M channel          : {'YES' if self.has_d2m else 'NO'}")

        # ── PRECOMPUTE FINE / COARSE ─────────────────────────────
        fine_all, coarse_all = [], []
        for i in range(0, T, 32):
            chunk  = torch.from_numpy(data[i:i+32]).unsqueeze(1)
            if compute_device.type == "cuda":
                chunk = chunk.to(compute_device, non_blocking=True)
            coarse = F.avg_pool2d(chunk, kernel_size=4, stride=4)
            if self.log_transformed:
                chunk  = torch.log1p(chunk)
                coarse = torch.log1p(coarse)
            fine_all.append(chunk.cpu())
            coarse_all.append(coarse.cpu())

        self.fine   = torch.cat(fine_all).contiguous()    # [T, 1, H,   W  ]
        self.coarse = torch.cat(coarse_all).contiguous()  # [T, 1, H/4, W/4]

        # ── VARIANCE MAP ─────────────────────────────────────────
        # Stays at COARSE resolution [1, H/4, W/4]
        # Network does the upsampling internally via v_stem
        coarse_raw = self.coarse[:, 0].numpy()            # [T, H/4, W/4]
        var_map = np.var(coarse_raw, axis=0)              # [H/4, W/4]
        var_map = np.nan_to_num(var_map, nan=0.0)

        vmin, vmax = var_map.min(), var_map.max()
        if vmax - vmin > 1e-7:
            var_map = (var_map - vmin) / (vmax - vmin + 1e-8)
        else:
            var_map = np.zeros_like(var_map)
            print("WARNING: Zero variance detected in dataset split!")

        # FIX: NO interpolation — keep at [1, H/4, W/4] (coarse res)
        # Regressor gets it via build_coarse_input() concat with coarse precip
        # UNet gets it via separate v_stem that upsamples internally
        var_t = torch.from_numpy(var_map).float().unsqueeze(0)  # [1, H/4, W/4]
        self.var_map_tensor = var_t.contiguous()

        print(f"Variance map         : shape={self.var_map_tensor.shape}  "
              f"min={self.var_map_tensor.min():.3f}  "
              f"max={self.var_map_tensor.max():.3f}")

        self.coarse_qdm = coarse_qdm if coarse_qdm is not None else self.coarse.clone()

    def __len__(self):
        return self.fine.shape[0]

    def __getitem__(self, idx):
        out = {
            "fine":       self.fine[idx],           # [1, H,   W  ]
            "coarse":     self.coarse[idx],         # [1, H/4, W/4]
            "coarse_qdm": self.coarse_qdm[idx],     # [1, H/4, W/4]
            "topo":       self.topo_tensor,         # [1, H,   W  ]
            "doy":        self.doy[idx],
            "hour":       self.hour[idx],
            "idx":        torch.tensor(idx, dtype=torch.long),
            "var_map":    self.var_map_tensor,      # [1, H/4, W/4]  ← coarse!
        }
        if self.has_d2m:
            out["d2m"] = self.d2m_tensor[idx]      # [1, H,   W  ]
        return out

    def denormalize(self, data):
        if self.log_transformed:
            return torch.expm1(data)
        return data

    def denormalize_d2m(self, data):
        return data * self.d2m_std + self.d2m_mean
