"""
les_io.py
---------
Load fields from PeleLM AMReX plt files using yt, and extract a centered
sub-cube whose physical extent matches the DNS 512^3 center cube.

LES grid is 108 x 126 x 72 over the same physical domain as the 864 x 1008
x 576 DNS, so LES cells are 8x larger in each direction. The centered
64^3 LES sub-cube spans the same physical volume as the centered 512^3
DNS sub-cube (1/8 of each side).

LES (AMReX boxlib) field names differ from our internal names. Map:

    internal name       LES name (yt)
    -----------------------------
    mixfrac             "mixture_fraction"
    T                   "temp"
    Y_CO                "Y(CO)"
    Y_CO2               "Y(CO2)"
    Y_OH                "Y(OH)"
    Y_O2                "Y(O2)"
    Y_H                 "Y(H)"
    Y_H2                "Y(H2)"
    Y_H2O               "Y(H2O)"
    Y_HCO               "Y(HCO)"
    Y_HO2               "Y(HO2)"
    Y_O                 "Y(O)"
    rho                 "density"
    P                   "pressure"
    u                   "x_velocity"
    v                   "y_velocity"
    w                   "z_velocity"
    chi                 "scalar_dissipation_rate"   (if present in plt;
                                                     otherwise compute)

This loader is conservative: it asks for `internal_name` and looks up the
yt field via this map. If the variant doesn't exist for a particular
solver build, fallback variants are tried.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


# Mapping from internal name to ordered list of candidate yt field names.
# yt returns a (108, 126, 72)-shape array indexed as (x, y, z) for boxlib.
_LES_FIELD_CANDIDATES = {
    "mixfrac":  ["mixture_fraction", "mixfrac", "Z"],
    "T":        ["temp", "temperature", "T"],
    "Y_CO":     ["Y(CO)",  "Y_CO"],
    "Y_CO2":    ["Y(CO2)", "Y_CO2"],
    "Y_OH":     ["Y(OH)",  "Y_OH"],
    "Y_O2":     ["Y(O2)",  "Y_O2"],
    "Y_H":      ["Y(H)",   "Y_H"],
    "Y_H2":     ["Y(H2)",  "Y_H2"],
    "Y_H2O":    ["Y(H2O)", "Y_H2O"],
    "Y_HCO":    ["Y(HCO)", "Y_HCO"],
    "Y_HO2":    ["Y(HO2)", "Y_HO2"],
    "Y_O":      ["Y(O)",   "Y_O"],
    "rho":      ["density", "rho"],
    "P":        ["pressure", "P"],
    "u":        ["x_velocity", "u"],
    "v":        ["y_velocity", "v"],
    "w":        ["z_velocity", "w"],
    "chi":      ["Scalar_diss", "scalar_dissipation_rate", "chi"],
    "alpha":    ["alpha", "thermal_diffusivity"],
}


@dataclass(frozen=True)
class LESGrid:
    """Geometry of the LES grid (matches PeleLM plt20000 for jet_0198).
    Physical domain matches the DNS (Lx=1.6397cm, Ly=1.9133cm, Lz=1.0925cm)."""
    nx_full: int = 108
    ny_full: int = 126
    nz_full: int = 72
    # Centered cube extracted: 1/8 of full DNS cube = 64^3 cells, which
    # spans the same physical volume as the DNS 512^3 center cube.
    n: int = 64


def load_les_dataset(plt_path: str):
    """Open the AMReX plt file with yt; raise if missing or unreadable."""
    try:
        import yt
    except ImportError as e:
        raise RuntimeError(
            "yt is required to load LES plt files. Install with "
            "'pip install --user yt' or load the cluster module that "
            "provides it.") from e

    if not os.path.exists(plt_path):
        raise FileNotFoundError(f"LES plt path not found: {plt_path}")

    # Be polite: silence yt's noisy info logging if env says so
    if os.environ.get("LES_IO_SILENT_YT", "0").lower() in ("1", "true", "yes"):
        import yt as _yt
        _yt.set_log_level(40)   # ERROR

    return yt.load(plt_path)


def _find_yt_field(ds, internal_name: str) -> Tuple[str, str]:
    """Look up the (ftype, fname) yt field tuple for an internal name.
    Tries the boxlib type first, then any type with a matching name."""
    candidates = _LES_FIELD_CANDIDATES.get(internal_name, [internal_name])
    field_list = list(ds.field_list)
    field_by_name = {fname: (ftype, fname) for (ftype, fname) in field_list}
    for cand in candidates:
        if ("boxlib", cand) in field_list:
            return ("boxlib", cand)
        if cand in field_by_name:
            return field_by_name[cand]
    raise KeyError(
        f"Could not find LES field for internal name '{internal_name}'. "
        f"Tried {candidates}. Available fields: "
        f"{[f for _, f in field_list]}")


def load_les_full_field(ds, internal_name: str) -> np.ndarray:
    """Load a single LES field over the whole domain as a numpy array.

    Returns shape (nx, ny, nz) following the convention used elsewhere
    in this codebase (the same order DNS load_field returns).
    """
    field_tuple = _find_yt_field(ds, internal_name)
    cg = ds.covering_grid(
        level=0,
        left_edge=ds.domain_left_edge,
        dims=ds.domain_dimensions,
    )
    arr = cg[field_tuple].to_ndarray()
    # yt's covering_grid arr is (nx, ny, nz). Same order as our convention.
    return np.asarray(arr, dtype=np.float32)


def extract_center_cube(arr: np.ndarray, grid: LESGrid) -> np.ndarray:
    """Extract a centered grid.n^3 cube from a full-grid LES array.
    Spans the same physical volume as the DNS 512^3 center cube."""
    nx, ny, nz = arr.shape
    n = grid.n
    sx = nx // 2 - n // 2
    sy = ny // 2 - n // 2
    sz = nz // 2 - n // 2
    return arr[sx:sx + n, sy:sy + n, sz:sz + n].copy()


def load_les_field(ds, internal_name: str,
                   grid: Optional[LESGrid] = None) -> np.ndarray:
    """Load one LES field and return the centered sub-cube (64^3 by default)."""
    if grid is None:
        grid = LESGrid()
    full = load_les_full_field(ds, internal_name)
    return extract_center_cube(full, grid)


def load_all_les_fields(plt_path: str,
                        names_required: Optional[list] = None,
                        names_optional: Optional[list] = None,
                        ) -> Dict[str, np.ndarray]:
    """Load LES fields needed for stats. Skips missing optional fields,
    raises FileNotFoundError-style KeyError if a required field is absent.
    """
    if names_required is None:
        names_required = ["mixfrac", "T", "Y_CO", "Y_CO2", "Y_OH", "Y_O2"]
    if names_optional is None:
        names_optional = ["chi", "alpha", "rho", "Y_H", "Y_H2", "Y_H2O",
                          "Y_HCO", "Y_HO2", "Y_O", "P", "u", "v", "w"]

    ds = load_les_dataset(plt_path)
    grid = LESGrid()

    fields: Dict[str, np.ndarray] = {}
    for name in names_required:
        try:
            fields[name] = load_les_field(ds, name, grid)
        except KeyError as e:
            raise KeyError(
                f"Required LES field '{name}' could not be loaded: {e}") from e
    for name in names_optional:
        try:
            fields[name] = load_les_field(ds, name, grid)
        except KeyError:
            continue
    return fields


# --- coordinate helpers (matches DNS/MPS conventions) -------------------

def y_over_H_les(grid: Optional[LESGrid] = None,
                 H_m: float = 1.368e-3,
                 Ly_m: float = 0.019133) -> np.ndarray:
    """Return y/H for the LES center cube, centered at y=0. Uses the
    same H and Ly as the DNS so curves are directly comparable.
    """
    if grid is None:
        grid = LESGrid()
    # LES dy = Ly / ny_full
    dy = Ly_m / grid.ny_full
    return (np.arange(grid.n) - (grid.n - 1) / 2.0) * (dy / H_m)


# -----------------------------------------------------------------------------
# FDF micromixing chi from Tauplt + TauAllplt files
# -----------------------------------------------------------------------------
# Aitzhan's chi formula (from his post-processing code):
#   chi_LES_FDF = 2 * C_phi * Omega * <Z''^2>
# where
#   C_phi = 5    (IEM/Curl micromixing model constant)
#   Omega = micromixing frequency from Tauplt<TS>.temp (field 'Freq')
#   <Z''^2> = SGS variance of Z from TauAllplt<TS>.temp (field 'Z')
#
# The two files have the same domain/grid as the regular plt20000, so we
# read them via yt the same way.

_FDF_C_PHI: float = 5.0


def load_fdf_chi_from_tau_files(tau_plt_path: str,
                                 tauall_plt_path: str,
                                 grid: Optional[LESGrid] = None
                                 ) -> np.ndarray:
    """Load Omega and Z''^2 from the Tau/TauAll plt files and compute
    chi_LES_FDF on the centered LES cube.

    Returns a (n, n, n) array of chi values on the cube.
    """
    import yt
    if grid is None:
        grid = LESGrid()

    if not os.path.exists(tau_plt_path):
        raise FileNotFoundError(f"Tau plt not found: {tau_plt_path}")
    if not os.path.exists(tauall_plt_path):
        raise FileNotFoundError(f"TauAll plt not found: {tauall_plt_path}")

    ds_tau = yt.load(tau_plt_path)
    ds_all = yt.load(tauall_plt_path)

    # Aitzhan reads field 'Freq' from Tauplt and field 'Z' from TauAllplt.
    # Try both with and without explicit ftype tag.
    def _read(ds, candidates):
        fl = list(ds.field_list)
        names = {f for _, f in fl}
        for c in candidates:
            if ("boxlib", c) in fl:
                cg = ds.covering_grid(level=0,
                                       left_edge=ds.domain_left_edge,
                                       dims=ds.domain_dimensions)
                return np.asarray(cg[("boxlib", c)].to_ndarray(),
                                  dtype=np.float64)
            if c in names:
                cg = ds.covering_grid(level=0,
                                       left_edge=ds.domain_left_edge,
                                       dims=ds.domain_dimensions)
                # Find ftype for this name
                for ft, fn in fl:
                    if fn == c:
                        return np.asarray(cg[(ft, c)].to_ndarray(),
                                          dtype=np.float64)
        raise KeyError(
            f"Could not find any of {candidates} in {ds.parameter_filename}. "
            f"Available: {sorted(names)}")

    omega_full = _read(ds_tau, ["Freq", "freq", "Omega", "omega"])
    zvar_full = _read(ds_all, ["Z", "Z_variance", "ZZ", "Zvar"])

    # Extract centered cube
    omega = extract_center_cube(omega_full, grid)
    zvar = extract_center_cube(zvar_full, grid)

    # FDF chi
    chi = 2.0 * _FDF_C_PHI * omega * zvar
    return np.asarray(chi, dtype=np.float32)

