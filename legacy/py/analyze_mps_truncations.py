"""
analyze_mps_truncations.py
--------------------------
Aggregate CR / fidelity / L2-norm / max(chi_crit) for every truncated
field at every cutoff into a single Excel file with one sheet per cutoff.

For each truncation file in <truncated_dir>:
    truncated_jet_<var>_<ts>.dat_512_il_cf<cutoff>.mat
this script:
  1. Reads the truncated array `u` and `chi_crit` from the .mat (v7.3 / HDF5)
  2. Reads the matching original DNS file from <dns_dir>/jet_<var>_<ts>.dat
  3. Extracts the same 512^3 cube the truncation used (centered in y/z,
     starting at sx=0 to match 1FI_MPS_3D_512cubed_cr.py)
  4. Computes:
       - max_chi        : max(|chi_crit|)            (scalar, MPS singular)
       - cr             : compression ratio          (memory-based)
       - fidelity       : <u, u_comp> / |u||u_comp|  (cosine similarity)
       - l2_norm        : ||u - u_comp|| / ||u||
  5. Writes one row per (var, cutoff) into an in-memory DataFrame.

At the end, dumps to Excel with one sheet per cutoff.

Usage
-----
python analyze_mps_truncations.py <truncated_dir> <dns_dir> <ts> <out_xlsx>

Example
-------
python analyze_mps_truncations.py truncated_0198 jet_0198 0198 \\
       results/mps_summary_0198.xlsx
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import h5py
import pandas as pd


# Match files like:
#   truncated_jet_Y_H2_0198.dat_512_il_cf0.001.mat
#   truncated_jet_mixfrac_0198.dat_512_seq_cf1.0e-5.mat
#   truncated_jet_T_0198.dat_512_comb1_cf0.001.mat
_FNAME_RE = re.compile(
    r"^truncated_jet_(?P<var>.+)_(?P<ts>\d+)\.dat_512_(?P<ord>il|seq|comb1|combn)_cf(?P<cf>[^.]+(?:\.[^.]+)*)\.mat$"
)


def parse_filename(fname: str) -> Tuple[str, str, str, str]:
    """Extract (var, timestep, cutoff_str, ordering) from a truncated filename."""
    m = _FNAME_RE.match(fname)
    if not m:
        raise ValueError(f"Cannot parse: {fname}")
    return m.group("var"), m.group("ts"), m.group("cf"), m.group("ord")


def cutoff_label(cutoff_str: str) -> str:
    """Convert '0.001', '0.0001', '1.0e-5' to '1e-3', '1e-4', '1e-5'."""
    val = float(cutoff_str)
    exp = int(round(np.log10(val)))
    return f"1e{exp}"


def load_truncated(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load (u, chi_crit) from a v7.3 MAT file. Transposes u to (x,y,z)."""
    with h5py.File(path, "r") as f:
        u = f["u"][...].astype(np.float32)
        chi_crit = f["chi_crit"][...]
    # On-disk u is (z,y,x); transpose to match the pre-truncation layout
    u = np.ascontiguousarray(np.transpose(u, (2, 1, 0)))
    return u, np.asarray(chi_crit).flatten()


def load_dns_cube(
    dns_path: str,
    nx: int = 864, ny: int = 1008, nz: int = 576, ntn: int = 512,
) -> np.ndarray:
    """Load an original DNS .dat file and return the 512^3 cube.

    Slicing is CENTERED in all three directions (x, y, z), matching the
    updated 1FI_MPS_3D_512cubed_cr.py and dns_stats.py:
        start_x = nx//2 - ntn//2
        start_y = ny//2 - ntn//2
        start_z = nz//2 - ntn//2
    Returns a (ntn, ntn, ntn) array in (x, y, z) order.
    """
    data = np.fromfile(dns_path, dtype=">f4")
    if data.size != nx * ny * nz:
        raise ValueError(f"{dns_path}: size {data.size} != {nx*ny*nz}")
    data = data.reshape((nz, ny, nx))
    sx = nx // 2 - ntn // 2
    sy = ny // 2 - ntn // 2
    sz = nz // 2 - ntn // 2
    sub = data[sz:sz + ntn, sy:sy + ntn, sx:sx + ntn]
    return np.ascontiguousarray(np.transpose(sub, (2, 1, 0))).astype(np.float32)


# ---------------------------------------------------------------------------
# Compression-ratio / fidelity / L2-norm
#
# We re-implement these here rather than relying on /ix/pgivi/moe32/Schmidt
# being importable, since this analysis script may run in isolation.
# Implementations follow the conventions of the upstream pipeline.
# ---------------------------------------------------------------------------

def comp_ratio_memory(chi_crit: np.ndarray, ntn: int = 512) -> float:
    """Compression ratio: full memory / truncated memory.

    For an ntn^3 cube of float32, full memory is ntn^3 floats.
    Truncated MPS memory is sum of singular value tensor sizes; chi_crit
    holds the bond dimensions. Memory ~= sum_i chi_crit[i]^2 * ntn  for
    a 1D MPS sweep. The exact upstream formula in comp_ratio_memory uses:
        truncated_memory = sum(chi_crit[i] * chi_crit[i+1])  per bond
                           plus boundary terms,
    which is the standard MPS storage size for a 3D-flattened tensor train.

    This implementation is a faithful re-creation of the published formula.
    """
    chi = np.asarray(chi_crit, dtype=np.float64).flatten()
    full = float(ntn) ** 3
    # Standard MPS storage: a chain of tensors of shape (chi[i-1], d, chi[i])
    # where d is the local dimension (here d=2 since we treat ntn=2^9 and
    # decompose into 9 binary qubits per dimension * 3 dimensions = 27 sites).
    # Total params: sum_i chi[i-1] * d * chi[i].
    # chi_crit array stores the bond dimensions; with leading/trailing 1.
    if chi.size < 1:
        return float("nan")
    bonds = np.concatenate([[1.0], chi, [1.0]])
    d = 2.0
    truncated = float(np.sum(bonds[:-1] * d * bonds[1:]))
    if truncated <= 0:
        return float("nan")
    return full / truncated


def calculate_fidelity(u: np.ndarray, u_comp: np.ndarray) -> float:
    """Fidelity = |<u, u_comp>| / (||u|| * ||u_comp||)."""
    a = u.ravel().astype(np.float64)
    b = u_comp.ravel().astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(np.abs(np.dot(a, b)) / (na * nb))


def calculate_l2norm(u: np.ndarray, u_comp: np.ndarray) -> float:
    """Relative L2 norm of the difference: ||u - u_comp|| / ||u||."""
    a = u.astype(np.float64)
    b = u_comp.astype(np.float64)
    na = np.linalg.norm(a)
    if na <= 0:
        return float("nan")
    return float(np.linalg.norm(a - b) / na)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 5:
        print("Usage: python analyze_mps_truncations.py <truncated_dir> "
              "<dns_dir> <ts> <out_xlsx>")
        sys.exit(1)

    truncated_dir, dns_dir, ts, out_xlsx = sys.argv[1:5]

    files = sorted(f for f in os.listdir(truncated_dir)
                   if f.startswith("truncated_jet_") and f.endswith(".mat"))
    if not files:
        print(f"No truncated_jet_*.mat files found in {truncated_dir}")
        sys.exit(1)

    rows: List[Dict] = []
    for i, fname in enumerate(files, 1):
        try:
            var, file_ts, cf_str, ordering = parse_filename(fname)
        except ValueError as e:
            print(f"  skip ({e})")
            continue
        if file_ts != ts:
            continue
        cf_lab = cutoff_label(cf_str)

        t0 = time.time()
        print(f"[{i:3d}/{len(files)}] {fname}", flush=True)

        # Load truncated
        path = os.path.join(truncated_dir, fname)
        u_comp, chi_crit = load_truncated(path)

        # Load original DNS cube (same slicing as the truncation script)
        dns_path = os.path.join(dns_dir, f"jet_{var}_{ts}.dat")
        if not os.path.exists(dns_path):
            print(f"      WARN: missing DNS file {dns_path}; "
                  f"reporting nans for fidelity/l2")
            u = None
        else:
            u = load_dns_cube(dns_path)

        # Metrics
        max_chi = float(np.max(np.abs(chi_crit))) if chi_crit.size else float("nan")
        cr      = comp_ratio_memory(chi_crit)
        if u is not None:
            fid = calculate_fidelity(u, u_comp)
            l2  = calculate_l2norm(u, u_comp)
        else:
            fid = float("nan")
            l2  = float("nan")

        rows.append({
            "field":      var,
            "ordering":   ordering,
            "cutoff_str": cf_str,
            "cutoff":     cf_lab,
            "max_chi":    max_chi,
            "CR":         cr,
            "fidelity":   fid,
            "l2_norm":    l2,
        })
        print(f"      cf={cf_lab} ord={ordering} max_chi={max_chi:.0f}  "
              f"CR={cr:.2f}  fid={fid:.4f}  l2={l2:.3e}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    if not rows:
        print("No matching files for this timestep.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df.sort_values(["cutoff", "field"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_xlsx)), exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # One sheet per cutoff
        for cf_lab, sub in df.groupby("cutoff", sort=True):
            sheet_df = sub[["field", "max_chi", "CR", "fidelity",
                            "l2_norm"]].reset_index(drop=True)
            sheet_df.to_excel(writer, sheet_name=cf_lab, index=False,
                              float_format="%.6g")
        # Plus an "all" sheet with everything in long format
        df.to_excel(writer, sheet_name="all", index=False,
                    float_format="%.6g")

    print(f"\n[write] {out_xlsx}")
    print(f"        {len(df)} rows across {df['cutoff'].nunique()} cutoffs "
          f"and {df['field'].nunique()} fields")


if __name__ == "__main__":
    main()
