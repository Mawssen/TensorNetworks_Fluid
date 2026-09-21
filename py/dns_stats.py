"""
dns_stats.py
------------
Statistics module for the PeleLM-FDF CO/H2 temporally evolving jet flame.

Reproduces the DNS statistics from Aitzhan et al., Combustion Theory and
Modelling (2022) [doi:10.1080/13647830.2022.2142673] for a single time
snapshot. Designed so the same functions can be called on either DNS or
MPS-truncated fields -- just pass different arrays in.

Conventions
-----------
Fields are stored on disk as big-endian float32, shape (nz, ny, nx) when
read flat from file. After loading we transpose to (nx, ny, nz) so that
axis 0 = streamwise (x), axis 1 = cross-stream (y), axis 2 = spanwise (z).
The center cube has extent 512^3 taken from the full DNS grid
(864 x 1008 x 576) following the user's existing workflow.

The temporally evolving jet is periodic in x and z, so Reynolds-averages
are taken over the x-z planes for each y.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class GridConfig:
    """Grid configuration for the Hawkes et al. CO/H2 DNS.

    All length scales are set from the DNS parameter file:
      - Physical domain: Lx, Ly, Lz (meters)
      - Jet velocity width H = 1.368e-3 m = 0.1368 cm
      - Full grid: 864 x 1008 x 576
      - Center cube extracted: 512^3

    Verified: Lx/H = 12.0, Ly/H = 14.0, Lz/H = 8.0 (matches paper exactly).
    """
    # Full DNS grid
    nx_full: int = 864
    ny_full: int = 1008
    nz_full: int = 576
    # Extracted center cube
    n: int = 512

    # Physical domain extents in meters (from DNS parameter file:
    # xmax = 1.6397 cm, ymax = 1.9133 cm, zmax = 1.0925 cm)
    Lx_m: float = 0.016397
    Ly_m: float = 0.019133
    Lz_m: float = 0.010925

    # Jet velocity width in meters (from DNS parameter file:
    # "jet velocity width 0.1368 cm")
    H: float = 1.368e-3

    # --- Physical (SI) spacings used for gradient operators ---
    @property
    def dx_m(self) -> float:
        return self.Lx_m / self.nx_full

    @property
    def dy_m(self) -> float:
        return self.Ly_m / self.ny_full

    @property
    def dz_m(self) -> float:
        return self.Lz_m / self.nz_full

    # --- Non-dimensional spacings (in units of H) used for plot axes ---
    @property
    def dx_over_H(self) -> float:
        return self.dx_m / self.H

    @property
    def dy_over_H(self) -> float:
        return self.dy_m / self.H

    @property
    def dz_over_H(self) -> float:
        return self.dz_m / self.H

    def y_over_H(self) -> np.ndarray:
        """Cross-stream coordinate (in units of H) for the center cube,
        centered at y = 0."""
        return self.y_over_H_for(self.n)

    def y_over_H_full(self) -> np.ndarray:
        """Cross-stream coordinate (in units of H) for the FULL grid,
        centered at y = 0."""
        return self.y_over_H_for(self.ny_full)

    def y_over_H_for(self, ny: int) -> np.ndarray:
        """Cross-stream coordinate (in units of H), centered at y = 0,
        for an arbitrary number of cells. Uses the physical dy.
        """
        dy = self.dy_over_H
        return (np.arange(ny) - (ny - 1) / 2.0) * dy


# Stoichiometric mixture fraction for the CO/H2 flame (Bilger's formulation,
# reported in the paper)
Z_ST_DEFAULT: float = 0.42
Y_OH_CUTOFF_DEFAULT: float = 7.0e-4  # extinction marker cutoff


# -----------------------------------------------------------------------------
# File I/O
# -----------------------------------------------------------------------------

def load_field(
    path: str,
    grid: GridConfig,
    dtype: str = ">f4",
) -> np.ndarray:
    """Load a single big-endian float32 raw binary field and extract the
    center cube of size grid.n^3, returned with axes (x, y, z).

    The cube is centered in ALL three directions (x, y, z). This differs
    from the earlier implementation that started at sx=0.
    """
    nx, ny, nz, n = grid.nx_full, grid.ny_full, grid.nz_full, grid.n
    data = np.fromfile(path, dtype=dtype)
    expected = nx * ny * nz
    if data.size != expected:
        raise ValueError(
            f"{path}: expected {expected} floats ({nz}x{ny}x{nx}), "
            f"got {data.size}"
        )
    data = data.reshape((nz, ny, nx))
    sz = nz // 2 - n // 2
    sy = ny // 2 - n // 2
    sx = nx // 2 - n // 2   # centered in x as well
    sub = data[sz:sz + n, sy:sy + n, sx:sx + n]
    # transpose to (x, y, z)
    return np.ascontiguousarray(np.transpose(sub, (2, 1, 0))).astype(np.float32)


def load_field_full(
    path: str,
    grid: GridConfig,
    dtype: str = ">f4",
) -> np.ndarray:
    """Load the FULL field (no center-cube extraction), returned with
    axes (x, y, z) shape = (nx_full, ny_full, nz_full).
    """
    nx, ny, nz = grid.nx_full, grid.ny_full, grid.nz_full
    data = np.fromfile(path, dtype=dtype)
    expected = nx * ny * nz
    if data.size != expected:
        raise ValueError(
            f"{path}: expected {expected} floats ({nz}x{ny}x{nx}), "
            f"got {data.size}"
        )
    data = data.reshape((nz, ny, nx))
    return np.ascontiguousarray(np.transpose(data, (2, 1, 0))).astype(np.float32)


def extract_center_cube(full: np.ndarray, grid: GridConfig) -> np.ndarray:
    """Extract the center-cube slice (centered in x, y, and z) used by
    the rest of the pipeline. Input axes: (x, y, z). Output: (n, n, n)."""
    nx, ny, nz = grid.nx_full, grid.ny_full, grid.nz_full
    n = grid.n
    sx = nx // 2 - n // 2
    sy = ny // 2 - n // 2
    sz = nz // 2 - n // 2
    return np.ascontiguousarray(
        full[sx:sx + n, sy:sy + n, sz:sz + n]
    )


def load_all_dns_fields(
    data_dir: str,
    timestep: str,
    grid: GridConfig,
    required_only: bool = False,
    region: str = "cube",
) -> Dict[str, np.ndarray]:
    """Load every field needed to reproduce the paper's statistics.

    Parameters
    ----------
    data_dir, timestep, grid
    required_only : bool
        If True, load only fields used by the statistics functions below.
    region : str
        'cube' -> extract center cube (grid.n^3)
        'full' -> return the full 864x1008x576 arrays
    """
    # Fields needed to reproduce every figure from the paper
    required = [
        "mixfrac",   # Z
        "T",         # temperature
        "Y_CO",
        "Y_CO2",
        "Y_OH",
        "Y_O2",
        "chi",       # scalar dissipation rate
    ]
    optional = [
        "alpha", "eps", "P", "rho",
        "rr_CO", "rr_OH",
        "u", "v", "w", "vis",
        "Y_H", "Y_H2", "Y_H2O", "Y_HCO", "Y_HO2", "Y_O",
    ]
    names = required if required_only else (required + optional)

    if region == "cube":
        loader = load_field
    elif region == "full":
        loader = load_field_full
    else:
        raise ValueError(f"Unknown region={region!r}; use 'cube' or 'full'.")

    fields: Dict[str, np.ndarray] = {}
    for name in names:
        path = os.path.join(data_dir, f"jet_{name}_{timestep}.dat")
        if not os.path.exists(path):
            if name in required:
                raise FileNotFoundError(f"Required field missing: {path}")
            continue
        fields[name] = loader(path, grid)
    return fields


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _xz_mean(field3d: np.ndarray) -> np.ndarray:
    """Reynolds average over periodic x-z planes (axes 0 and 2).
    Returns a 1D profile along y."""
    return field3d.mean(axis=(0, 2))


def _xz_var(field3d: np.ndarray) -> np.ndarray:
    """Variance over x-z planes at each y."""
    mean_y = _xz_mean(field3d)[None, :, None]
    return ((field3d - mean_y) ** 2).mean(axis=(0, 2))


def _xz_cov(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Covariance over x-z planes at each y."""
    ma = _xz_mean(a)[None, :, None]
    mb = _xz_mean(b)[None, :, None]
    return ((a - ma) * (b - mb)).mean(axis=(0, 2))


# -----------------------------------------------------------------------------
# Statistics  (one function per figure in the paper)
# -----------------------------------------------------------------------------

def mean_rms_profiles(
    fields: Dict[str, np.ndarray],
    variables: Tuple[str, ...] = ("mixfrac", "T", "Y_CO", "Y_CO2"),
) -> Dict[str, Dict[str, np.ndarray]]:
    """Figure 6: Reynolds-averaged mean and RMS vs y/H for each variable.

    For DNS (no filter defined), the 'total stress' r(a,a) reduces to the
    variance over x-z planes. RMS = sqrt(variance).
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for v in variables:
        if v not in fields:
            continue
        f = fields[v]
        mean_prof = _xz_mean(f)
        rms_prof = np.sqrt(np.maximum(_xz_var(f), 0.0))
        out[v] = {"mean": mean_prof, "rms": rms_prof}
    return out


def scalar_dissipation_profile(
    fields: Dict[str, np.ndarray],
) -> np.ndarray:
    """Figure 9: Reynolds-averaged scalar dissipation rate profile along y.

    Uses the chi field provided directly in the dataset. Definition in the
    paper: chi = 2 * gamma/rho * (grad Z . grad Z).
    """
    return _xz_mean(fields["chi"])


def mixture_fraction_thickness(
    fields: Dict[str, np.ndarray],
    grid: GridConfig,
    epsilon: float = 0.01,
    y_over_H: Optional[np.ndarray] = None,
) -> float:
    """Figure 8: delta_Z / (2H) at this timestep.

    Definition: delta_Z = 2 * argmin_{y>0} |Zbar(y) - epsilon|

    If y_over_H is None, uses the cube y coordinate. Pass the full-grid
    y coordinate when computing on the full domain.
    """
    Zmean = _xz_mean(fields["mixfrac"])
    if y_over_H is None:
        y_over_H = grid.y_over_H_for(Zmean.size)
    pos = y_over_H > 0
    yp = y_over_H[pos]
    Zp = Zmean[pos]
    idx = int(np.argmin(np.abs(Zp - epsilon)))
    return float(yp[idx])


def conditional_T_on_Z(
    fields: Dict[str, np.ndarray],
    n_bins: int = 50,
    z_range: Tuple[float, float] = (0.0, 1.0),
) -> Dict[str, np.ndarray]:
    """Figure 10: volume-averaged temperature conditioned on mixture fraction.

    Returns bin centers (psi_Z) and E[T | Z = psi_Z].
    """
    Z = fields["mixfrac"].ravel()
    T = fields["T"].ravel()
    edges = np.linspace(z_range[0], z_range[1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sum_T, _ = np.histogram(Z, bins=edges, weights=T)
    count, _ = np.histogram(Z, bins=edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_T = np.where(count > 0, sum_T / np.maximum(count, 1), np.nan)
    return {"psi_Z": centers, "E_T": mean_T, "count": count}


def extinction_marker(
    fields: Dict[str, np.ndarray],
    z_st: float = Z_ST_DEFAULT,
    z_tol: float = 0.02,
    y_oh_cutoff: float = Y_OH_CUTOFF_DEFAULT,
) -> Dict[str, float]:
    """Figure 11: (a) volume-averaged extinction marker, and
    (b) mean temperature on the stoichiometric surface, at this timestep.

    M_ext = H(Y_OH_cutoff - Y_OH) * indicator(|Z - Z_st| < z_tol)
    P_ext = <M_ext> / <indicator(|Z - Z_st| < z_tol)>   (conditional)

    Also reports the unconditional probability used in the paper's
    formulation:
        P(Z = Z_st, Y_OH <= Y_OH,c) ~ volume fraction where both hold.
    """
    Z = fields["mixfrac"]
    Y_OH = fields["Y_OH"]
    T = fields["T"]

    near_st = np.abs(Z - z_st) < z_tol
    extinguished = Y_OH <= y_oh_cutoff

    n_total = Z.size
    n_near_st = int(np.count_nonzero(near_st))
    n_both = int(np.count_nonzero(near_st & extinguished))

    # volume-averaged extinction marker (paper Fig 11a)
    vol_avg_marker = n_both / n_total

    # conditional extinction probability given near-stoichiometric
    if n_near_st > 0:
        cond_ext_prob = n_both / n_near_st
        # mean temperature on stoichiometric surface (paper Fig 11b)
        T_at_st = float(T[near_st].mean())
    else:
        cond_ext_prob = float("nan")
        T_at_st = float("nan")

    return {
        "volume_averaged_marker": vol_avg_marker,
        "conditional_extinction_prob": cond_ext_prob,
        "T_on_stoich_surface": T_at_st,
        "n_near_st_cells": n_near_st,
        "n_total_cells": n_total,
    }


def pdf_mixfrac_centerline(
    fields: Dict[str, np.ndarray],
    grid: GridConfig,
    n_planes: int = 8,
    n_bins: int = 60,
    z_range: Tuple[float, float] = (0.0, 1.0),
) -> Dict[str, np.ndarray]:
    """Figure 12: PDF of mixture fraction sampled from planes near y = 0.

    The paper's DNS uses 8 cross-stream planes near the centerline; we
    extract the n_planes planes closest to y = 0.
    """
    Z = fields["mixfrac"]
    ny = Z.shape[1]
    j0 = ny // 2
    half = n_planes // 2
    jlo = max(0, j0 - half)
    jhi = min(ny, j0 + (n_planes - half))
    sample = Z[:, jlo:jhi, :].ravel()
    edges = np.linspace(z_range[0], z_range[1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(sample, bins=edges, density=True)
    return {"psi_Z": centers, "pdf": hist, "n_samples": sample.size}


def joint_pdf_Z_YCO2(
    fields: Dict[str, np.ndarray],
    grid: GridConfig,
    n_planes: int = 8,
    n_bins: int = 60,
    z_range: Tuple[float, float] = (0.0, 1.0),
    yco2_range: Tuple[float, float] = (0.0, 0.17),
) -> Dict[str, np.ndarray]:
    """Figure 13: joint PDF of (Z, Y_CO2) on planes near y = 0."""
    Z = fields["mixfrac"]
    Y = fields["Y_CO2"]
    ny = Z.shape[1]
    j0 = ny // 2
    half = n_planes // 2
    jlo = max(0, j0 - half)
    jhi = min(ny, j0 + (n_planes - half))
    z_s = Z[:, jlo:jhi, :].ravel()
    y_s = Y[:, jlo:jhi, :].ravel()
    H, xe, ye = np.histogram2d(
        z_s, y_s,
        bins=[n_bins, n_bins],
        range=[z_range, yco2_range],
        density=True,
    )
    return {
        "psi_Z": 0.5 * (xe[:-1] + xe[1:]),
        "psi_YCO2": 0.5 * (ye[:-1] + ye[1:]),
        "joint_pdf": H,
        "n_samples": z_s.size,
    }


def manifold_scatter_sample(
    fields: Dict[str, np.ndarray],
    n_sample: int = 200000,
    rng_seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Figure 14: random subsample of (Z, Y_O2, Y_OH, T) for 3D scatter.

    Full-volume scatter with hundreds of millions of points is unreasonable;
    we draw a reproducible random subsample.
    """
    Z = fields["mixfrac"].ravel()
    Y_O2 = fields["Y_O2"].ravel()
    Y_OH = fields["Y_OH"].ravel()
    T = fields["T"].ravel()
    rng = np.random.default_rng(rng_seed)
    n = Z.size
    if n_sample >= n:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, size=n_sample, replace=False)
    return {
        "Z": Z[idx],
        "Y_O2": Y_O2[idx],
        "Y_OH": Y_OH[idx],
        "T": T[idx],
    }


# -----------------------------------------------------------------------------
# Top-level driver
# -----------------------------------------------------------------------------

def compute_all_stats(
    fields: Dict[str, np.ndarray],
    grid: GridConfig,
    z_st: float = Z_ST_DEFAULT,
) -> Dict:
    """Run every statistic and return a single dict suitable for np.savez.

    The size of the y-axis is inferred from the loaded fields, so this
    function works transparently on either the center cube or the full
    grid.
    """
    # infer ny from the first loaded field
    sample = next(iter(fields.values()))
    ny = sample.shape[1]
    y = grid.y_over_H_for(ny)

    profiles = mean_rms_profiles(fields)
    chi_prof = scalar_dissipation_profile(fields)
    dZ = mixture_fraction_thickness(fields, grid, y_over_H=y)
    condT = conditional_T_on_Z(fields)
    extinct = extinction_marker(fields, z_st=z_st)
    pdfZ = pdf_mixfrac_centerline(fields, grid)
    joint = joint_pdf_Z_YCO2(fields, grid)
    scatter = manifold_scatter_sample(fields)

    return {
        "y_over_H": y,
        "profiles": profiles,
        "chi_profile": chi_prof,
        "delta_Z_over_2H": dZ,
        "conditional_T_on_Z": condT,
        "extinction": extinct,
        "pdf_mixfrac": pdfZ,
        "joint_pdf_Z_YCO2": joint,
        "manifold_scatter": scatter,
        "z_st": z_st,
    }
