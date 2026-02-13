# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def _build_spherical_graph(H, W):
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

    # --- Grid 8-neighbor edges (with periodic longitude) ---
    for i in range(H):
        for j in range(W):
            node = i * W + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni = i + di
                    nj = (j + dj) % W  # Periodic in longitude
                    if 0 <= ni < H:
                        neighbor = ni * W + nj
                        edges.append([node, neighbor])

    # --- North pole <-> all nodes in top row (row 0) ---
    for j in range(W):
        node = j  # row 0
        edges.append([north_pole, node])
        edges.append([node, north_pole])

    # --- South pole <-> all nodes in bottom row (row H-1) ---
    for j in range(W):
        node = (H - 1) * W + j  # row H-1
        edges.append([south_pole, node])
        edges.append([node, south_pole])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def _lat_weights(H):
    """Compute latitude-dependent area weights for spherical grid.

    Accounts for cos(lat) scaling: cells near equator have more area.

    Args:
        H: Number of latitude points

    Returns:
        weights: [H] tensor of area weights (grid nodes only, no poles)
    """
    lats = torch.linspace(90, -90, H)
    weights = torch.cos(lats * math.pi / 180)
    return weights / weights.sum()


def _num_nodes(H, W):
    """Total node count including virtual pole nodes."""
    return H * W + 2


class WeatherBenchDataset(Dataset):
    """WeatherBench dataset for global weather forecasting.

    Loads ERA5 reanalysis data on regular lat/lon grid.
    Creates (state_t, state_{t+lead_time}) pairs for forecasting.

    Supports variables: z500 (geopotential), t850 (temperature),
    u10/v10 (wind components), q (specific humidity).
    """

    def __init__(self, root, resolution='5.625deg', variables=None,
                 lead_time=72, split='train', years=None):
        """Initialize WeatherBench dataset.

        Args:
            root: Root directory for data
            resolution: Grid resolution ('5.625deg', '2.8deg', '1.4deg')
            variables: List of variable names
            lead_time: Forecast lead time in hours
            split: 'train', 'val', or 'test'
            years: List of years to use (None for default split)
        """
        self.root = root
        self.resolution = resolution
        self.variables = variables or ['z500', 't850']
        self.lead_time = lead_time
        self.split = split
        self.num_variables = len(self.variables)

        # Grid dimensions based on resolution
        res_map = {
            '5.625deg': (32, 64),
            '2.8deg': (64, 128),
            '1.4deg': (128, 256),
        }
        self.H, self.W = res_map.get(resolution, (32, 64))

        # Build graph (includes virtual pole nodes)
        self.edge_index = _build_spherical_graph(self.H, self.W)
        self.lat_weights = _lat_weights(self.H)
        self.num_nodes = _num_nodes(self.H, self.W)  # H*W + 2 (grid + poles)

        # Lat/lon coordinates
        self.lat = torch.linspace(90, -90, self.H)
        self.lon = torch.linspace(0, 360 - 360 / self.W, self.W)

        self.data_list = []
        self._load_data()

    def _load_data(self):
        """Load data: cached .pt > NetCDF via xarray > synthetic fallback."""
        cache_path = os.path.join(self.root, f'weatherbench_{self.resolution}_{self.split}.pt')

        # 1. Try cached .pt
        if os.path.exists(cache_path):
            self.data_list = torch.load(cache_path, weights_only=False)
            print(f">>> WeatherBench: loaded {len(self.data_list)} samples from {cache_path}")
            return

        # 2. Try loading from NetCDF files
        loaded = self._load_from_netcdf()
        if loaded:
            # Cache the processed data
            os.makedirs(self.root, exist_ok=True)
            torch.save(self.data_list, cache_path)
            print(f">>> WeatherBench: cached {len(self.data_list)} samples to {cache_path}")
            return

        # 3. Fallback: synthetic data
        print(f">>> WeatherBench data not found, generating synthetic data")
        self._generate_synthetic_data()

    def _load_from_netcdf(self):
        """Load ERA5 reanalysis data from NetCDF files via xarray.

        Expected file patterns:
            {root}/{variable}_{resolution}.nc  (e.g. geopotential_5.625deg.nc)
            or {root}/{variable}/              (directory with yearly files)

        Standard year splits:
            train: 1979-2015
            val:   2016
            test:  2017-2018

        Returns:
            True if data was loaded successfully, False otherwise.
        """
        try:
            import xarray as xr
        except ImportError:
            return False

        # Variable name mapping
        var_to_file = {
            'z500': ['geopotential_500', 'geopotential', 'z500', 'z'],
            't850': ['temperature_850', 'temperature', 't850', 't'],
            'u10': ['u_component_of_wind_10', 'u10', '10m_u_component_of_wind'],
            'v10': ['v_component_of_wind_10', 'v10', '10m_v_component_of_wind'],
            'q': ['specific_humidity', 'q'],
        }

        # Try to find and load each variable
        datasets = {}
        for var in self.variables:
            ds = None
            candidates = var_to_file.get(var, [var])
            for name in candidates:
                # Try single file
                for ext in ['.nc', '.zarr']:
                    path = os.path.join(self.root, self.resolution, f'{name}{ext}')
                    if os.path.exists(path):
                        try:
                            ds = xr.open_dataset(path)
                            break
                        except Exception:
                            continue
                if ds is not None:
                    break
                # Try directory
                var_dir = os.path.join(self.root, self.resolution, name)
                if os.path.isdir(var_dir):
                    try:
                        ds = xr.open_mfdataset(os.path.join(var_dir, '*.nc'),
                                               combine='by_coords')
                        break
                    except Exception:
                        continue
            if ds is None:
                return False
            datasets[var] = ds

        # Define year-based splits
        split_years = {
            'train': list(range(1979, 2016)),
            'val': [2016],
            'test': [2017, 2018],
        }
        years = split_years.get(self.split, split_years['train'])

        # Extract data for each variable
        steps_ahead = self.lead_time  # in hours

        # Get the first variable to determine time axis
        first_var = self.variables[0]
        first_ds = datasets[first_var]
        data_var_name = list(first_ds.data_vars)[0]

        # Select years
        time_idx = first_ds.time.dt.year.isin(years)
        times = first_ds.time[time_idx].values

        # Create time pairs (t, t+lead_time)
        lead_td = np.timedelta64(self.lead_time, 'h')

        for t_idx in range(len(times) - 1):
            t = times[t_idx]
            t_plus = t + lead_td

            # Check if t_plus exists in the dataset
            if t_plus > times[-1]:
                break

            state_t = torch.zeros(self.H, self.W, self.num_variables)
            state_t_plus = torch.zeros(self.H, self.W, self.num_variables)

            valid = True
            for v_idx, var in enumerate(self.variables):
                ds = datasets[var]
                dv = list(ds.data_vars)[0]
                try:
                    val_t = ds[dv].sel(time=t, method='nearest').values
                    val_tp = ds[dv].sel(time=t_plus, method='nearest').values
                    # Coarsen to target resolution if needed
                    if val_t.shape != (self.H, self.W):
                        # Simple area averaging
                        from scipy.ndimage import zoom
                        val_t = zoom(val_t, (self.H / val_t.shape[0], self.W / val_t.shape[1]))
                        val_tp = zoom(val_tp, (self.H / val_tp.shape[0], self.W / val_tp.shape[1]))
                    state_t[:, :, v_idx] = torch.tensor(val_t[:self.H, :self.W], dtype=torch.float32)
                    state_t_plus[:, :, v_idx] = torch.tensor(val_tp[:self.H, :self.W], dtype=torch.float32)
                except Exception:
                    valid = False
                    break

            if valid:
                self.data_list.append({
                    'state_t': state_t,
                    'state_t_plus': state_t_plus,
                })

        # Clean up
        for ds in datasets.values():
            ds.close()

        return len(self.data_list) > 0

    def _generate_synthetic_data(self, num_samples=200):
        """Generate synthetic weather data for development.

        Creates plausible mock data with latitude-dependent temperature
        and wave-like geopotential patterns.
        """
        rng = np.random.RandomState(42 if self.split == 'train' else 123)

        # Time steps between samples (6h intervals)
        steps_ahead = self.lead_time // 6

        for _ in range(num_samples):
            # Generate state at time t
            state_t = torch.zeros(self.H, self.W, self.num_variables)

            for v_idx, var in enumerate(self.variables):
                if var == 'z500':
                    # Geopotential: latitude-dependent + wave pattern
                    lat_profile = torch.cos(self.lat * math.pi / 180).unsqueeze(1).expand(-1, self.W)
                    wave = torch.sin(torch.arange(self.W).float() * 4 * math.pi / self.W).unsqueeze(0)
                    state_t[:, :, v_idx] = 50000 + 5000 * lat_profile + 1000 * wave
                    state_t[:, :, v_idx] += torch.tensor(rng.randn(self.H, self.W) * 500, dtype=torch.float32)
                elif var == 't850':
                    # Temperature: warm at equator, cold at poles
                    lat_temp = 280 + 20 * torch.cos(self.lat * math.pi / 180).unsqueeze(1).expand(-1, self.W)
                    state_t[:, :, v_idx] = lat_temp
                    state_t[:, :, v_idx] += torch.tensor(rng.randn(self.H, self.W) * 5, dtype=torch.float32)
                else:
                    state_t[:, :, v_idx] = torch.tensor(rng.randn(self.H, self.W), dtype=torch.float32)

            # Generate state at t + lead_time (evolved version with noise)
            state_t_plus = state_t.clone()
            # Shift pattern slightly (simulate advection)
            shift = rng.randint(1, 4)
            state_t_plus = torch.roll(state_t_plus, shifts=shift, dims=1)
            state_t_plus += torch.tensor(rng.randn(self.H, self.W, self.num_variables) * 0.01, dtype=torch.float32) * state_t.std()

            self.data_list.append({
                'state_t': state_t,           # [H, W, C]
                'state_t_plus': state_t_plus,  # [H, W, C]
            })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        return {
            'state_t': sample['state_t'],
            'state_t_plus': sample['state_t_plus'],
            'edge_index': self.edge_index,
            'lat': self.lat,
            'lon': self.lon,
            'lat_weights': self.lat_weights,
            'num_nodes': self.num_nodes,
        }


def collate_weatherbench(batch):
    """Collate WeatherBench samples into a batch.

    All samples share the same grid, so we can simply stack.
    """
    return {
        'state_t': torch.stack([s['state_t'] for s in batch]),            # [B, H, W, C]
        'state_t_plus': torch.stack([s['state_t_plus'] for s in batch]),  # [B, H, W, C]
        'edge_index': batch[0]['edge_index'],                              # Shared grid
        'lat': batch[0]['lat'],
        'lon': batch[0]['lon'],
        'lat_weights': batch[0]['lat_weights'],
        'num_nodes': batch[0]['num_nodes'],
    }


def get_weatherbench_loaders(root, resolution='5.625deg', variables=None,
                             lead_time=72, batch_size=4, max_samples=None, years=None):
    """Load WeatherBench dataset with train/val/test splits.

    Returns:
        train_loader, val_loader, test_loader, var_means, var_stds
    """
    datasets = {}
    for split in ['train', 'val', 'test']:
        ds = WeatherBenchDataset(
            root=root, resolution=resolution, variables=variables,
            lead_time=lead_time, split=split, years=years
        )
        if max_samples and len(ds) > max_samples:
            ds.data_list = ds.data_list[:max_samples]
        datasets[split] = ds

    # Compute per-variable normalization stats from training set
    all_states = torch.stack([s['state_t'] for s in datasets['train'].data_list])
    var_means = all_states.mean(dim=[0, 1, 2])  # [C]
    var_stds = all_states.std(dim=[0, 1, 2])    # [C]

    H, W = datasets['train'].H, datasets['train'].W
    n_vars = datasets['train'].num_variables

    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_weatherbench)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, collate_fn=collate_weatherbench)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, collate_fn=collate_weatherbench)

    print(f">>> WeatherBench {resolution}: grid={H}x{W}, vars={n_vars}, lead={lead_time}h")
    print(f">>> {len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])} train/val/test")
    for i, var in enumerate(datasets['train'].variables):
        print(f">>>   {var}: mean={var_means[i]:.2f}, std={var_stds[i]:.2f}")

    return train_loader, val_loader, test_loader, var_means, var_stds
