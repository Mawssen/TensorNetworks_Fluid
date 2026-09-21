"""
peps_io.py
----------
Loader for PEPS-truncated field files.

File layout (as of 2026-07):
    truncated_<TS>/PEPS/<var>_wf_D=9_periodic.mat
For var in {mixfrac, T, Y_CO, Y_CO2, Y_OH, Y_O2, alpha}.

Each .mat file is an HDF5 v7.3 with a single dataset 'Z' of shape
(512, 512, 512), float64. No bond-dimension array is stored; the
compression ratio is known externally (matched to LES CR=512).

Only one PEPS case is supported at a time (D=9, periodic). Extend
_peps_filename below if additional variants are added later.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import h5py
import numpy as np


# Fields to load into the stats pipeline (parallel to mps_io defaults).
_PEPS_REQUIRED = ["mixfrac", "T"]
_PEPS_OPTIONAL = ["alpha", "Y_CO", "Y_CO2", "Y_OH", "Y_O2"]


def _peps_filename(var: str, tag: str = "wf_D=9_periodic") -> str:
    """Build the PEPS filename for a given field variable."""
    return f"{var}_{tag}.mat"


def load_peps_field(path: str,
                     dataset_key: Optional[str] = None,
                     transpose_zyx_to_xyz: bool = True) -> np.ndarray:
    """Load a single PEPS-truncated field from a v7.3 .mat file.

    The dataset key inside PEPS files matches the field name:
      - mixfrac_wf_D=9_periodic.mat  -> key 'Z'
      - T_wf_D=9_periodic.mat        -> key 'T'
      - alpha_wf_D=9_periodic.mat    -> key 'alpha'
      - Y_CO_wf_D=9_periodic.mat     -> key 'Y_CO'
      etc.

    If dataset_key is None, infer it from the filename stem before '_wf_'.
    Falls back to whatever single dataset is inside the file.

    Data is stored as (512, 512, 512) float64. We return float32 to
    match the rest of the pipeline's memory footprint.

    PEPS files are written in MATLAB/Fortran order (z, y, x) but the DNS
    binary and MPS .mat files use C order (x, y, z). By default we
    transpose(2, 1, 0) so the returned axes match DNS's (x, y, z)
    convention. Verified via correlation with DNS: full-cube correlation
    is 0.996 with the transpose vs 0.79 without.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PEPS file not found: {path}")

    if dataset_key is None:
        base = os.path.basename(path)
        stem = base.split("_wf_", 1)[0] if "_wf_" in base else base.split(".")[0]
        dataset_key = "Z" if stem == "mixfrac" else stem

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if dataset_key not in f:
            if len(keys) == 1:
                dataset_key = keys[0]
            else:
                raise KeyError(f"'{dataset_key}' not in {path}; keys={keys}")
        arr = f[dataset_key][...]
    arr = np.asarray(arr, dtype=np.float32)
    if transpose_zyx_to_xyz and arr.ndim == 3:
        arr = np.transpose(arr, (2, 1, 0)).copy()
    return arr


def load_all_peps_fields(
    data_dir: str,
    tag: str = "wf_D=9_periodic",
    skip_chi: bool = True,
) -> Dict[str, np.ndarray]:
    """Load all available PEPS fields from a directory.

    Parameters
    ----------
    data_dir : str
        Directory containing files named '<var>_<tag>.mat'.
    tag : str
        Suffix identifying the PEPS variant. Default 'wf_D=9_periodic'.
    skip_chi : bool
        Ignored; kept for API parity with load_all_mps_fields. PEPS
        does not have a stored chi field so we always compute it from
        Z and alpha via derived_fields.compute_chi in the stats runner.

    Returns
    -------
    dict mapping field name -> (512, 512, 512) float32 array
    """
    fields: Dict[str, np.ndarray] = {}
    for var in _PEPS_REQUIRED:
        path = os.path.join(data_dir, _peps_filename(var, tag=tag))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required PEPS field missing: {path}")
        fields[var] = load_peps_field(path)

    for var in _PEPS_OPTIONAL:
        path = os.path.join(data_dir, _peps_filename(var, tag=tag))
        if not os.path.exists(path):
            print(f"[peps_io] Optional field missing (skipping): {path}")
            continue
        fields[var] = load_peps_field(path)

    return fields


def peps_case_label(tag: str = "wf_D=9_periodic") -> str:
    """Human-readable label for a PEPS case. Currently only one variant."""
    return f"PEPS D=9"
