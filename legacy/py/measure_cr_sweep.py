"""
measure_cr_sweep.py
-------------------
Measure compression ratio (CR) for the mixfrac MPS sweep outputs.

Uses the memory-based CR formulation:
    mps_comp_memory = 2 * sum_{j=1..N} chi[j] * chi[j-1]
    CR = 2^(N+1) / mps_comp_memory
where chi is padded with 1 at both ends (open-boundary MPS).

Pandas-free: outputs are written via openpyxl directly so we don't hit the
numpy 1.x/2.x ABI mismatch in the cluster Python.

Usage
-----
python measure_cr_sweep.py [<truncated_dir>]

Default <truncated_dir> = first ./truncated_* in cwd
"""

from __future__ import annotations

import os
import sys
import glob
import re

import h5py
import numpy as np


def comp_ratio_memory(chi, cr_dof=False):
    """Memory compression ratio of DNS over truncated MPS.

    Parameters
    ----------
    chi : 1D array of bond dimensions (does NOT include the trivial 1 at
          either end; those are added).
    cr_dof : if True, also return the DOF-based CR.

    Returns
    -------
    CR_mem  (float)
    or (CR_mem, CR_dof) if cr_dof=True.
    """
    N3 = int(len(chi) + 1)
    chi = np.concatenate(([1], np.asarray(chi, dtype=np.int64), [1]))
    mps_comp_memory = 2 * sum(chi[j] * chi[j - 1] for j in range(1, len(chi)))
    cr_mem = (2.0 ** N3) / mps_comp_memory if mps_comp_memory != 0 else float("inf")
    if cr_dof:
        sum2_chi = sum(chi[j] ** 2 for j in range(1, len(chi) - 1))
        cr_dof_val = (2.0 ** N3) / (mps_comp_memory - sum2_chi)
        return cr_mem, cr_dof_val
    return cr_mem


def parse_label(fname):
    m = re.search(r"_(cf[\d.eE+-]+|chi\d+)\.mat$", fname)
    if not m:
        return ("unknown", "")
    label = m.group(1)
    param = label[2:] if label.startswith("cf") else label[3:]
    return (label, param)


def load_chi_crit(mat_path):
    with h5py.File(mat_path, "r") as f:
        if "chi_crit" not in f:
            raise KeyError(f"chi_crit not in {mat_path}; keys={list(f.keys())}")
        return f["chi_crit"][()].flatten().astype(np.int64)


def _write_xlsx(out_path, rows):
    try:
        import openpyxl
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("[warn] openpyxl not available; skipping xlsx output")
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "CR sweep"
    if not rows:
        wb.save(out_path)
        return
    headers = list(rows[0].keys())
    ws.append(headers)
    for row in rows:
        ws.append([row[h] for h in headers])
    for i in range(len(headers)):
        ws.column_dimensions[get_column_letter(i + 1)].width = 20
    wb.save(out_path)


def main():
    args = sys.argv[1:]
    if args:
        trunc_dir = args[0]
    else:
        cands = sorted(glob.glob("truncated_*"))
        if not cands:
            print("No truncated_* directory found in cwd; pass one explicitly.")
            sys.exit(1)
        trunc_dir = cands[0]

    timestep = trunc_dir.rstrip("/").split("_")[-1]
    pattern = os.path.join(trunc_dir, "truncated_jet_mixfrac_*.mat")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No mixfrac .mat files in {trunc_dir}")
        sys.exit(1)

    rows = []
    print(f"\n{'mode':>14s}  {'param':>10s}  {'max_chi':>8s}  "
          f"{'mean_chi':>10s}  {'mps_mem':>14s}  {'CR_mem':>10s}  "
          f"{'CR_dof':>10s}")
    print("-" * 86)
    for path in files:
        try:
            chi_crit = load_chi_crit(path)
        except Exception as e:
            print(f"  [skip] {os.path.basename(path)}: {e}")
            continue
        label, param = parse_label(os.path.basename(path))
        cr_mem, cr_dof = comp_ratio_memory(chi_crit, cr_dof=True)
        chi_pad = np.concatenate(([1], chi_crit, [1]))
        mps_mem = int(2 * sum(int(chi_pad[j]) * int(chi_pad[j - 1])
                              for j in range(1, len(chi_pad))))
        max_chi = int(chi_crit.max())
        mean_chi = float(chi_crit.mean())
        rows.append({
            "file": os.path.basename(path),
            "mode": label,
            "param": param,
            "max_chi": max_chi,
            "mean_chi": round(mean_chi, 2),
            "mps_mem": mps_mem,
            "CR_mem": round(cr_mem, 2),
            "CR_dof": round(cr_dof, 2),
        })
        print(f"  {label:>14s}  {param:>10s}  {max_chi:>8d}  "
              f"{mean_chi:>10.2f}  {mps_mem:>14,d}  {cr_mem:>10.2f}  "
              f"{cr_dof:>10.2f}")

    if not rows:
        sys.exit(1)

    rows_sorted = sorted(rows, key=lambda r: r["CR_mem"])
    out_xlsx = f"cr_sweep_{timestep}.xlsx"
    _write_xlsx(out_xlsx, rows_sorted)
    print(f"\n[write] {out_xlsx}")

    target = 512.0
    best = min(rows, key=lambda r: abs(r["CR_mem"] - target))
    delta = abs(best["CR_mem"] - target)
    print(f"\nClosest to LES CR=512: {best['mode']} "
          f"(CR_mem={best['CR_mem']:.2f}, |delta|={delta:.2f})")


if __name__ == "__main__":
    main()
