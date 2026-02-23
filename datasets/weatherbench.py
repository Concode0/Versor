# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""WeatherBench dataset loader.

Loading priority (tried in order):
    1. Cached .pt files  (fastest)
    2. NetCDF files via xarray  (ERA5 / WeatherBench1 on local disk)
    3. Zarr files via zarr + fsspec  (WeatherBench2 on GCS or local disk)
    4. Synthetic fallback  (always available, for development)

Install libraries:
    uv sync --extra weather        # xarray, netcdf4
    pip install zarr fsspec gcsfs  # WeatherBench2 cloud access
    pip install scipy              # for spatial resampling

WeatherBench2 data (zarr, publicly accessible):
    gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr
"""

from __future__ import annotations

import os
import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ---- Optional library detection ----
_HAS_XARRAY = False
try:
    import xarray as xr
    _HAS_XARRAY = True
except ImportError:
    pass

_HAS_ZARR = False
try:
    import zarr               # noqa: F401
    import fsspec             # noqa: F401
    _HAS_ZARR = True
except ImportError:
    pass

_HAS_SCIPY_NDIMAGE = False
try:
    from scipy.ndimage import zoom as _scipy_zoom
    _HAS_SCIPY_NDIMAGE = True
except ImportError:
    pass


def _resample_grid(arr: np.ndarray, target_H: int, target_W: int) -> np.ndarray:
    """Resample a (H, W) array to (target_H, target_W).

    Uses scipy.ndimage.zoom if available; otherwise nearest-neighbor via
    numpy slice indexing (less accurate but dependency-free).
    """
    if arr.shape == (target_H, target_W):
        return arr
    if _HAS_SCIPY_NDIMAGE:
        return _scipy_zoom(arr, (target_H / arr.shape[0], target_W / arr.shape[1]),
                           order=1)
    # Fallback: nearest-neighbor index selection
    h_idx = np.round(np.linspace(0, arr.shape[0] - 1, target_H)).astype(int)
    w_idx = np.round(np.linspace(0, arr.shape[1] - 1, target_W)).astype(int)
    return arr[np.ix_(h_idx, w_idx)]


# =====================================================================
# Spherical graph utilities
# =====================================================================

def _build_spherical_graph(H: int, W: int) -> torch.Tensor:
    """Build 8-neighbor connectivity on spherical lat/lon grid with pole nodes.

    Handles periodic boundary in longitude (wrap-around) and adds virtual
    singularity nodes at north and south poles to close the sphere topology.

    Without pole nodes, the top and bottom latitude rows are treated as flat
    edges — a topological breach. On a real sphere, all meridians converge
    at each pole. Virtual pole nodes fix this by connecting to every node
    in the adjacent row, enabling proper message flow across the poles.

    Node layout:
        0 .. H*W-1  : grid nodes (row-major, row 0 = north, row H-1 = south)
        H*W         : north pole virtual node
        H*W + 1     : south pole virtual node

    Args:
        H: Grid height (latitude points)
        W: Grid width (longitude points)

    Returns:
        edge_index: [2, E] connectivity tensor (includes pole edges)
    """
    north_pole = H * W
    south_pole = H * W + 1

    edges = []

    # Grid 8-neighbor edges (with periodic longitude)
    for i in range(H):
        for j in range(W):
            node = i * W + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni = i + di
                    nj = (j + dj) % W       # periodic in longitude
                    if 0 <= ni < H:
                        edges.append([node, ni * W + nj])

    # North pole <-> all nodes in top row
    for j in range(W):
        edges.append([north_pole, j])
        edges.append([j, north_pole])

    # South pole <-> all nodes in bottom row
    for j in range(W):
        node = (H - 1) * W + j
        edges.append([south_pole, node])
        edges.append([node, south_pole])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _lat_weights(H: int) -> torch.Tensor:
    """Latitude-dependent area weights (cos-lat scaling)."""
    lats = torch.linspace(90, -90, H)
    weights = torch.cos(lats * math.pi / 180)
    return weights / weights.sum()


def _num_nodes(H: int, W: int) -> int:
    """Total node count including virtual pole nodes."""
    return H * W + 2


# =====================================================================
# Dataset
# =====================================================================

class WeatherBenchDataset(Dataset):
    """WeatherBench dataset for global weather forecasting.

    Loads ERA5 reanalysis data on regular lat/lon grid.
    Creates (state_t, state_{t+lead_time}) pairs for forecasting.

    Supports variables: z500 (geopotential), t850 (temperature),
    u10/v10 (wind components), q (specific humidity).
    """

    def __init__(self, root: str, resolution: str = '5.625deg',
                 variables: Optional[List[str]] = None,
                 lead_time: int = 72, split: str = 'train',
                 years: Optional[List[int]] = None):
        self.root = root
        self.resolution = resolution
        self.variables = variables or ['z500', 't850']
        self.lead_time = lead_time
        self.split = split
        self.num_variables = len(self.variables)

        res_map = {'5.625deg': (32, 64), '2.8deg': (64, 128), '1.4deg': (128, 256)}
        self.H, self.W = res_map.get(resolution, (32, 64))

        self.edge_index = _build_spherical_graph(self.H, self.W)
        self.lat_weights = _lat_weights(self.H)
        self.num_nodes = _num_nodes(self.H, self.W)

        self.lat = torch.linspace(90, -90, self.H)
        self.lon = torch.linspace(0, 360 - 360 / self.W, self.W)

        self.data_list: List[dict] = []
        self._load_data()

    def _load_data(self):
        """Load data: cached .pt → NetCDF → Zarr → synthetic fallback."""
        cache_path = os.path.join(
            self.root, f'weatherbench_{self.resolution}_{self.split}.pt'
        )

        # 1. Cached .pt
        if os.path.exists(cache_path):
            self.data_list = torch.load(cache_path, weights_only=False)
            print(f">>> WeatherBench: loaded {len(self.data_list)} samples "
                  f"from cache ({self.split})")
            return

        # 2. NetCDF via xarray (ERA5 / WeatherBench1)
        if _HAS_XARRAY:
            if self._load_from_netcdf():
                os.makedirs(self.root, exist_ok=True)
                torch.save(self.data_list, cache_path)
                print(f">>> WeatherBench: cached {len(self.data_list)} "
                      f"NetCDF samples → {cache_path}")
                return
        else:
            print(">>> WeatherBench: xarray not installed, skipping NetCDF path "
                  "(install with: uv sync --extra weather)")

        # 3. Zarr via zarr+fsspec (WeatherBench2 local or GCS)
        if _HAS_ZARR:
            if self._load_from_zarr():
                os.makedirs(self.root, exist_ok=True)
                torch.save(self.data_list, cache_path)
                print(f">>> WeatherBench: cached {len(self.data_list)} "
                      f"zarr samples → {cache_path}")
                return
        else:
            print(">>> WeatherBench: zarr/fsspec not installed, skipping zarr path "
                  "(install with: pip install zarr fsspec gcsfs)")

        # 4. Synthetic fallback
        print(f">>> WeatherBench: no real data found, generating synthetic data "
              f"({self.split})")
        self._generate_synthetic_data()

    # ------------------------------------------------------------------
    # Path 2: NetCDF via xarray (ERA5)
    # ------------------------------------------------------------------

    def _load_from_netcdf(self) -> bool:
        """Load ERA5 reanalysis data from NetCDF files via xarray.

        Expected file patterns:
            {root}/{resolution}/{variable_name}.nc
            {root}/{resolution}/{variable_name}/  (directory with yearly .nc)

        Variable name aliases are tried in order (most-specific first).

        Standard year splits:
            train: 1979-2015 | val: 2016 | test: 2017-2018

        Returns:
            True if at least one sample was loaded.
        """
        var_to_file = {
            'z500': ['geopotential_500', 'geopotential', 'z500', 'z'],
            't850': ['temperature_850', 'temperature', 't850', 't'],
            'u10':  ['u_component_of_wind_10', 'u10', '10m_u_component_of_wind'],
            'v10':  ['v_component_of_wind_10', 'v10', '10m_v_component_of_wind'],
            'q':    ['specific_humidity', 'q'],
        }

        datasets: Dict[str, 'xr.Dataset'] = {}
        for var in self.variables:
            ds = self._try_open_netcdf(var, var_to_file.get(var, [var]))
            if ds is None:
                print(f">>> WeatherBench: NetCDF not found for variable '{var}' "
                      f"in {os.path.join(self.root, self.resolution)}")
                return False
            datasets[var] = ds

        split_years = {
            'train': list(range(1979, 2016)),
            'val':   [2016],
            'test':  [2017, 2018],
        }
        years = split_years.get(self.split, split_years['train'])

        first_ds = datasets[self.variables[0]]
        data_var = list(first_ds.data_vars)[0]
        time_mask = first_ds.time.dt.year.isin(years)
        times = first_ds.time[time_mask].values
        lead_td = np.timedelta64(self.lead_time, 'h')

        n_loaded = 0
        for t_idx in range(len(times) - 1):
            t = times[t_idx]
            t_plus = t + lead_td
            if t_plus > times[-1]:
                break

            state_t = torch.zeros(self.H, self.W, self.num_variables)
            state_tp = torch.zeros(self.H, self.W, self.num_variables)
            valid = True

            for v_idx, var in enumerate(self.variables):
                ds = datasets[var]
                dv = list(ds.data_vars)[0]
                try:
                    val_t  = ds[dv].sel(time=t,      method='nearest').values
                    val_tp = ds[dv].sel(time=t_plus, method='nearest').values
                    state_t[:, :, v_idx]  = torch.tensor(
                        _resample_grid(val_t,  self.H, self.W)[:self.H, :self.W],
                        dtype=torch.float32)
                    state_tp[:, :, v_idx] = torch.tensor(
                        _resample_grid(val_tp, self.H, self.W)[:self.H, :self.W],
                        dtype=torch.float32)
                except Exception as e:
                    warnings.warn(f"WeatherBench: skipping t={t} var={var}: {e}")
                    valid = False
                    break

            if valid:
                self.data_list.append({'state_t': state_t, 'state_t_plus': state_tp})
                n_loaded += 1

        for ds in datasets.values():
            ds.close()

        print(f">>> WeatherBench: loaded {n_loaded} NetCDF samples ({self.split})")
        return n_loaded > 0

    def _try_open_netcdf(self, var: str,
                         candidates: List[str]) -> Optional['xr.Dataset']:
        """Try to open an xarray Dataset from multiple filename candidates."""
        for name in candidates:
            res_dir = os.path.join(self.root, self.resolution)
            for ext in ['.nc', '.zarr']:
                path = os.path.join(res_dir, f'{name}{ext}')
                if os.path.exists(path):
                    try:
                        return xr.open_dataset(path)
                    except Exception:
                        pass
            var_dir = os.path.join(res_dir, name)
            if os.path.isdir(var_dir):
                try:
                    return xr.open_mfdataset(
                        os.path.join(var_dir, '*.nc'), combine='by_coords'
                    )
                except Exception:
                    pass
        return None

    # ------------------------------------------------------------------
    # Path 3: Zarr (WeatherBench2)
    # ------------------------------------------------------------------

    def _load_from_zarr(self) -> bool:
        """Load WeatherBench2 data from zarr stores (local or GCS).

        Checks these sources in order:
          a. Local zarr directory:  {root}/{resolution}/*.zarr
          b. GCS WeatherBench2:    gs://weatherbench2/... (requires gcsfs)

        WeatherBench2 variable names: 'geopotential', '2m_temperature',
        '10m_u_component_of_wind', '10m_v_component_of_wind', etc.

        Returns:
            True if at least one sample was loaded.
        """
        zarr_store = self._find_zarr_store()
        if zarr_store is None:
            return False

        try:
            import zarr as _zarr
        except ImportError:
            return False

        wb2_var_map = {
            'z500': ('geopotential', 500),
            't850': ('temperature', 850),
            't2m':  ('2m_temperature', None),
            'u10':  ('10m_u_component_of_wind', None),
            'v10':  ('10m_v_component_of_wind', None),
        }

        try:
            store = _zarr.open(zarr_store, mode='r')
        except Exception as e:
            warnings.warn(f"WeatherBench: failed to open zarr store {zarr_store}: {e}")
            return False

        split_years = {'train': list(range(1979, 2016)), 'val': [2016],
                       'test': [2017, 2018]}
        years = split_years.get(self.split, split_years['train'])

        # Determine time coordinate
        if 'time' not in store:
            warnings.warn("WeatherBench zarr: no 'time' coordinate found.")
            return False

        times_raw = store['time'][:]
        # Convert to datetime64 if needed
        if times_raw.dtype.kind in ('i', 'u', 'f'):
            # Assume hours since 1970-01-01
            times = (np.datetime64('1970-01-01') +
                     times_raw.astype('int64') * np.timedelta64(1, 'h'))
        else:
            times = times_raw.astype('datetime64[h]')

        year_arr = times.astype('datetime64[Y]').astype(int) + 1970
        year_mask = np.isin(year_arr, years)
        valid_times = times[year_mask]
        valid_time_idx = np.where(year_mask)[0]

        lead_td = np.timedelta64(self.lead_time, 'h')

        n_loaded = 0
        for i, (t, tidx) in enumerate(zip(valid_times, valid_time_idx)):
            t_plus = t + lead_td
            tp_candidates = np.searchsorted(times, t_plus)
            if tp_candidates >= len(times) or times[tp_candidates] != t_plus:
                continue

            state_t = torch.zeros(self.H, self.W, self.num_variables)
            state_tp = torch.zeros(self.H, self.W, self.num_variables)
            valid = True

            for v_idx, var in enumerate(self.variables):
                wb2_info = wb2_var_map.get(var)
                if wb2_info is None:
                    valid = False; break
                wb2_name, level = wb2_info
                if wb2_name not in store:
                    valid = False; break
                try:
                    arr = store[wb2_name]
                    if level is not None and arr.ndim == 4:
                        # Try to find level axis
                        if 'level' in store:
                            levels = store['level'][:]
                            lev_idx = np.argmin(np.abs(levels - level))
                            val_t  = arr[tidx,       lev_idx]
                            val_tp = arr[tp_candidates, lev_idx]
                        else:
                            val_t  = arr[tidx,       0]
                            val_tp = arr[tp_candidates, 0]
                    else:
                        val_t  = arr[tidx]
                        val_tp = arr[tp_candidates]

                    state_t[:, :, v_idx]  = torch.tensor(
                        _resample_grid(np.array(val_t),  self.H, self.W),
                        dtype=torch.float32)
                    state_tp[:, :, v_idx] = torch.tensor(
                        _resample_grid(np.array(val_tp), self.H, self.W),
                        dtype=torch.float32)
                except Exception as e:
                    warnings.warn(f"WeatherBench zarr: var={var} t={tidx}: {e}")
                    valid = False; break

            if valid:
                self.data_list.append({'state_t': state_t, 'state_t_plus': state_tp})
                n_loaded += 1

        print(f">>> WeatherBench: loaded {n_loaded} zarr samples ({self.split})")
        return n_loaded > 0

    def _find_zarr_store(self) -> Optional[str]:
        """Find a local zarr store or return a GCS URI.

        Checks local disk first, then tries the public WeatherBench2 GCS bucket
        if gcsfs is available (requires internet access).
        """
        # Local zarr
        local_candidates = [
            os.path.join(self.root, self.resolution, 'era5.zarr'),
            os.path.join(self.root, 'era5.zarr'),
            os.path.join(self.root, self.resolution),   # directory might be a zarr
        ]
        for p in local_candidates:
            if os.path.isdir(p) and os.path.exists(os.path.join(p, '.zattrs')):
                print(f">>> WeatherBench: found local zarr store at {p}")
                return p

        # GCS WeatherBench2 (requires gcsfs)
        try:
            import gcsfs     # noqa: F401
            # Canonical WeatherBench2 zarr on GCS
            res_to_gcs = {
                '5.625deg': 'gs://weatherbench2/datasets/era5/'
                            '1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr',
                '2.8deg':   'gs://weatherbench2/datasets/era5/'
                            '1959-2022-6h-128x64_equiangular_with_poles_conservative.zarr',
                '1.4deg':   'gs://weatherbench2/datasets/era5/'
                            '1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr',
            }
            gcs_uri = res_to_gcs.get(self.resolution)
            if gcs_uri:
                print(f">>> WeatherBench: trying GCS WeatherBench2 at {gcs_uri}")
                return gcs_uri
        except ImportError:
            pass

        return None

    # ------------------------------------------------------------------
    # Path 4: Synthetic fallback
    # ------------------------------------------------------------------

    def _generate_synthetic_data(self, num_samples: int = 200):
        """Generate synthetic weather data for development.

        Creates plausible mock data with latitude-dependent temperature
        and wave-like geopotential patterns.
        """
        rng = np.random.RandomState(42 if self.split == 'train' else 123)

        for _ in range(num_samples):
            state_t = torch.zeros(self.H, self.W, self.num_variables)

            for v_idx, var in enumerate(self.variables):
                if var == 'z500':
                    lat_prof = torch.cos(self.lat * math.pi / 180).unsqueeze(1).expand(-1, self.W)
                    wave = torch.sin(torch.arange(self.W).float() * 4 * math.pi / self.W).unsqueeze(0)
                    state_t[:, :, v_idx] = 50000 + 5000 * lat_prof + 1000 * wave
                    state_t[:, :, v_idx] += torch.tensor(
                        rng.randn(self.H, self.W) * 500, dtype=torch.float32)
                elif var == 't850':
                    lat_temp = 280 + 20 * torch.cos(
                        self.lat * math.pi / 180).unsqueeze(1).expand(-1, self.W)
                    state_t[:, :, v_idx] = lat_temp + torch.tensor(
                        rng.randn(self.H, self.W) * 5, dtype=torch.float32)
                else:
                    state_t[:, :, v_idx] = torch.tensor(
                        rng.randn(self.H, self.W), dtype=torch.float32)

            # Simulate advection: roll + small noise
            shift = rng.randint(1, 4)
            state_tp = torch.roll(state_t.clone(), shifts=shift, dims=1)
            state_tp += torch.tensor(
                rng.randn(self.H, self.W, self.num_variables) * 0.01,
                dtype=torch.float32) * state_t.std()

            self.data_list.append({'state_t': state_t, 'state_t_plus': state_tp})

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data_list[idx]
        return {
            'state_t':      sample['state_t'],
            'state_t_plus': sample['state_t_plus'],
            'edge_index':   self.edge_index,
            'lat':          self.lat,
            'lon':          self.lon,
            'lat_weights':  self.lat_weights,
            'num_nodes':    self.num_nodes,
        }


# =====================================================================
# Collate and loader factory
# =====================================================================

def collate_weatherbench(batch: list) -> dict:
    """Collate WeatherBench samples (all samples share the same grid)."""
    return {
        'state_t':      torch.stack([s['state_t']      for s in batch]),
        'state_t_plus': torch.stack([s['state_t_plus'] for s in batch]),
        'edge_index':   batch[0]['edge_index'],
        'lat':          batch[0]['lat'],
        'lon':          batch[0]['lon'],
        'lat_weights':  batch[0]['lat_weights'],
        'num_nodes':    batch[0]['num_nodes'],
    }


def get_weatherbench_loaders(root: str, resolution: str = '5.625deg',
                             variables: Optional[List[str]] = None,
                             lead_time: int = 72, batch_size: int = 4,
                             max_samples: Optional[int] = None,
                             years: Optional[List[int]] = None,
                             num_workers: int = 2,
                             pin_memory: bool = False):
    """Load WeatherBench dataset with train/val/test splits.

    Tries real data (NetCDF → zarr → GCS) and falls back to synthetic.

    Returns:
        train_loader, val_loader, test_loader, var_means, var_stds
    """
    if not _HAS_XARRAY and not _HAS_ZARR:
        print(">>> WeatherBench: no data loading libraries found. "
              "Install with:  uv sync --extra weather  or  pip install zarr fsspec")

    datasets: Dict[str, WeatherBenchDataset] = {}
    for split in ['train', 'val', 'test']:
        ds = WeatherBenchDataset(
            root=root, resolution=resolution, variables=variables,
            lead_time=lead_time, split=split, years=years
        )
        if max_samples and len(ds) > max_samples:
            ds.data_list = ds.data_list[:max_samples]
        datasets[split] = ds

    all_states = torch.stack([s['state_t'] for s in datasets['train'].data_list])
    var_means = all_states.mean(dim=[0, 1, 2])   # [C]
    var_stds  = all_states.std(dim=[0, 1, 2]).clamp(min=1e-6)

    H, W = datasets['train'].H, datasets['train'].W

    train_loader = DataLoader(datasets['train'], batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_weatherbench,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(datasets['val'],   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_weatherbench,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(datasets['test'],  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_weatherbench,
                              num_workers=num_workers, pin_memory=pin_memory)

    print(f">>> WeatherBench {resolution}: grid={H}×{W}, "
          f"vars={datasets['train'].num_variables}, lead={lead_time}h")
    print(f">>> {len(datasets['train'])}/{len(datasets['val'])}/"
          f"{len(datasets['test'])} train/val/test")
    for i, var in enumerate(datasets['train'].variables):
        print(f">>>   {var}: mean={var_means[i]:.2f}, std={var_stds[i]:.2f}")

    return train_loader, val_loader, test_loader, var_means, var_stds
