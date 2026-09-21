"""
mps_io.py
---------
Loader for MPS-truncated DNS fields stored as MATLAB v7.3 (.mat / HDF5)
files. Files are produced by the MPS truncation pipeline upstream and
follow the naming convention:

    truncated_jet_<var>_<timestep>.dat_512_il_cf<cutoff>.mat

inside a directory like:

    /ix/pgivi/moe32/Aidyn_DNS/stats/truncated_<timestep>/

Each .mat contains a 'u' dataset of shape (512, 512, 512) float32, which
is the truncated field. The HDF5 layout transposes axes relative to
MATLAB so we read it as (nz, ny, nx) and transpose to (nx, ny, nz) to
match the rest of the pipeline.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import h5py
import numpy as np


# Filename mapping: dns_stats's field key -> the variable string used in
# the truncated filename. For most variables the two are identical.
_MPS_FIELD_NAMES = {
    "mixfrac": "mixfrac",
    "T":       "T",
    "Y_CO":    "Y_CO",
    "Y_CO2":   "Y_CO2",
    "Y_OH":    "Y_OH",
    "Y_O2":    "Y_O2",
    "Y_H":     "Y_H",
    "Y_H2":    "Y_H2",
    "Y_H2O":   "Y_H2O",
    "Y_HCO":   "Y_HCO",
    "Y_HO2":   "Y_HO2",
    "Y_O":     "Y_O",
    "alpha":   "alpha",
    "rho":     "rho",
    "chi":     "chi",
    "P":       "P",
    "u":       "u",
    "v":       "v",
    "w":       "w",
    "vis":     "vis",
    "eps":     "eps",
    "rr_CO":   "rr_CO",
    "rr_OH":   "rr_OH",
}


# Mapping from sweep-parameter index to ordering string used in
# the upstream MPS truncation filename. sp=1 is the legacy default.
_SP_TO_ORDERING = {
    1: "il",      # interleaved (legacy)
    2: "seq",     # sequential
    3: "comb1",   # comb-style ordering, variant 1
    4: "combn",   # comb-style ordering, variant n
}


def ordering_for_sp(sp: int) -> str:
    """Return the ordering token used in the MPS .mat filename for a
    given sweep-parameter index sp. Raises if sp not recognized.
    """
    if sp not in _SP_TO_ORDERING:
        raise ValueError(f"Unknown sp={sp}; valid: {sorted(_SP_TO_ORDERING)}")
    return _SP_TO_ORDERING[sp]


def _mps_filename(var: str, timestep: str, cutoff_str: str,
                  ordering: str = "il") -> str:
    """Build the .mat filename used by the MPS pipeline.

    Examples:
        cutoff_str='0.001'  ordering='il'   -> ..._il_cf0.001.mat
        cutoff_str='1.0e-5' ordering='seq'  -> ..._seq_cf1.0e-5.mat
        cutoff_str='chi93'  ordering='il'   -> ..._il_chi93.mat    (chi-cap mode)

    Strings starting with 'chi' are treated as bond-dimension caps and
    produce a 'chi<N>' filename suffix instead of 'cf<value>'.
    """
    if cutoff_str.startswith("chi"):
        return f"truncated_jet_{var}_{timestep}.dat_512_{ordering}_{cutoff_str}.mat"
    return f"truncated_jet_{var}_{timestep}.dat_512_{ordering}_cf{cutoff_str}.mat"


def load_mps_field(
    path: str,
    dataset_key: str = "u",
    transpose_zyx_to_xyz: bool = True,
) -> np.ndarray:
    """Read a single MPS-truncated field from a v7.3 MAT file.

    The DNS pipeline keeps fields in (x, y, z) order. The upstream
    MPS truncation pipeline saves the array in (z, y, x) order
    (matching the original DNS .dat layout before transpose), so we
    transpose on load to match.

    Verified empirically: at a near-lossless cutoff (1e-5), the max
    pointwise difference between DNS T and MPS T drops from ~940 K
    (without transpose) to ~370 K (with transpose). The remaining
    ~370 K max / 8 K mean is genuine MPS truncation error.
    """
    with h5py.File(path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"Dataset '{dataset_key}' not found in {path}; "
                           f"available keys: {list(f.keys())}")
        arr = f[dataset_key][...]
    arr = np.ascontiguousarray(arr).astype(np.float32)
    if transpose_zyx_to_xyz:
        arr = np.ascontiguousarray(np.transpose(arr, (2, 1, 0)))
    return arr


def load_all_mps_fields(
    data_dir: str,
    timestep: str,
    cutoff_str: str,
    required_only: bool = False,
    ordering: str = "il",
    skip_chi: bool = False,
) -> Dict[str, np.ndarray]:
    """Load MPS-truncated fields for one (timestep, cutoff, ordering) tuple.

    Parameters
    ----------
    data_dir : str
        Directory containing truncated_jet_*_<timestep>.dat_*.mat files.
    timestep : str
        e.g. '0198'.
    cutoff_str : str
        e.g. '0.001', '0.0001', '1.0e-5' -- must match the filename
        suffix exactly.
    ordering : str
        'il', 'seq', 'comb1', 'combn'. The token between '512_' and '_cf'
        in the filename.
    required_only : bool
        Same semantics as DNS loader.
    skip_chi : bool
        If True, chi is dropped from the required list and not loaded
        even if present. Use this for mixed/recon modes where chi will
        be reconstructed downstream from mixfrac + alpha; loading the
        stored chi is wasted I/O.
    """
    required = ["mixfrac", "T", "Y_CO", "Y_CO2", "Y_OH", "Y_O2", "chi"]
    optional = ["alpha", "rho", "Y_H", "Y_H2", "Y_H2O", "Y_HCO",
                "Y_HO2", "Y_O", "P", "u", "v", "w", "vis", "eps",
                "rr_CO", "rr_OH"]
    if skip_chi:
        required = [n for n in required if n != "chi"]
        optional = [n for n in optional if n != "chi"]
    names = required if required_only else (required + optional)

    fields: Dict[str, np.ndarray] = {}
    for name in names:
        var = _MPS_FIELD_NAMES.get(name, name)
        path = os.path.join(data_dir,
                            _mps_filename(var, timestep, cutoff_str,
                                          ordering=ordering))
        if not os.path.exists(path):
            if name in required:
                raise FileNotFoundError(f"Required MPS field missing: {path}")
            continue
        fields[name] = load_mps_field(path)
    return fields


def cutoff_label(cutoff_str: str) -> str:
    """Convert a filename cutoff string to a human-readable label.

    Examples:
        '0.001'    -> '1e-3'
        '1.0e-5'   -> '1e-5'
        '0.0004'   -> '4e-4'
        '4.1e-4'   -> '4.1e-4'   (non-power-of-10 cutoff preserved)
        '0.000337' -> '3.37e-4'
        'chi93'    -> 'chi93'
    """
    if cutoff_str.startswith("chi"):
        return cutoff_str
    val = float(cutoff_str)
    if val <= 0:
        return cutoff_str
    log10v = np.log10(val)
    exp = int(np.floor(log10v))
    mantissa = val / (10.0 ** exp)
    # Within 1% of a clean power of 10 -> render as '1eN'
    if abs(mantissa - 1.0) < 0.01:
        return f"1e{exp}"
    # Within 1% of an integer mantissa -> render as 'NeM' (e.g. '4e-4')
    if abs(mantissa - round(mantissa)) < 0.01:
        return f"{int(round(mantissa))}e{exp}"
    # Otherwise keep up to two decimal digits, no trailing zeros
    s = f"{mantissa:.2f}".rstrip("0").rstrip(".")
    return f"{s}e{exp}"
