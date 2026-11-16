"""
make_filtered_cube.py
---------------------
Produce a box-filtered ("ideal LES") version of the DNS fields by averaging
each 8^3 block of the 512^3 DNS down to a 64^3 cube, and cache the results
to disk as .npy for reuse.

This is the 'Filtered' source: a perfect low-pass of the DNS with NO subgrid
model, so comparing the real LES-FDF against it isolates SGS-model error from
pure filtering error.

Fields filtered (only those the QC4PDE statistics need):
    mixfrac  (Z)        -> profiles, PDF, conditional T, joint PDF, contour
    temp     (T)        -> conditional T, manifold colour
    Y_O2               -> manifold
    Y_OH               -> manifold
    Y_CO2              -> joint PDF

Each cached file is written to:
    <cache-dir>/filtered_<field>_<timestep>_f<factor>.npy

Usage
-----
    python make_filtered_cube.py \
        --dns-data-dir /ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198 \
        --timestep 0198 \
        --factor 8 \
        --cache-dir /ix/pgivi/moe32/Aidyn_DNS/stats/filtered_0198 \
        [--fields mixfrac,temp,Y_O2,Y_OH,Y_CO2] \
        [--file-pattern "jet_{field}_{timestep}.dat"]

The default --file-pattern assumes DNS files are named like
jet_mixfrac_0198.dat. Adjust it if your T / species files follow a different
naming convention (use {field} and {timestep} placeholders).

Run on a login node with the venv + python module loaded.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_qc4pde import _parse   # noqa: E402


def block_average(cube: np.ndarray, factor: int = 8) -> np.ndarray:
    """Box-filter a cube by averaging each factor^3 block."""
    nx, ny, nz = cube.shape
    cx, cy, cz = (nx // factor) * factor, (ny // factor) * factor, \
                 (nz // factor) * factor
    if (cx, cy, cz) != (nx, ny, nz):
        print(f"    [warn] cropping {cube.shape} -> ({cx},{cy},{cz}) "
              f"to be divisible by {factor}")
    c = cube[:cx, :cy, :cz]
    c = c.reshape(cx // factor, factor,
                  cy // factor, factor,
                  cz // factor, factor)
    return c.mean(axis=(1, 3, 5))


def main():
    args = sys.argv[1:]
    dns_data_dir = _parse(args, "--dns-data-dir",
                          default="/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198")
    timestep = _parse(args, "--timestep", default="0198")
    factor = int(_parse(args, "--factor", default="8"))
    cache_dir = _parse(args, "--cache-dir",
                       default=f"/ix/pgivi/moe32/Aidyn_DNS/stats/"
                               f"filtered_{timestep}")
    fields_str = _parse(args, "--fields",
                        default="mixfrac,temp,Y_O2,Y_OH,Y_CO2")
    file_pattern = _parse(args, "--file-pattern",
                          default="jet_{field}_{timestep}.dat")
    fields = [f.strip() for f in fields_str.split(",") if f.strip()]

    os.makedirs(cache_dir, exist_ok=True)
    print(f"[filter] dns_data_dir = {dns_data_dir}")
    print(f"[filter] cache_dir    = {cache_dir}")
    print(f"[filter] factor       = {factor}  (512^3 -> {512//factor}^3)")
    print(f"[filter] fields       = {fields}")

    from dns_stats import GridConfig, load_field
    grid = GridConfig()

    t0 = time.time()
    for field in fields:
        fname = file_pattern.format(field=field, timestep=timestep)
        fpath = os.path.join(dns_data_dir, fname)
        out = os.path.join(cache_dir,
                           f"filtered_{field}_{timestep}_f{factor}.npy")
        if os.path.exists(out):
            print(f"    [skip] {field}: cache exists -> {out}")
            continue
        if not os.path.exists(fpath):
            print(f"    [MISS] {field}: DNS file not found -> {fpath}\n"
                  f"           (adjust --file-pattern if the naming differs)")
            continue
        print(f"    loading {field} from {fpath} ...")
        cube = np.asarray(load_field(fpath, grid), dtype=np.float64)
        filt = block_average(cube, factor)
        np.save(out, filt.astype(np.float32))
        print(f"    [ok] {field}: {cube.shape} -> {filt.shape}, "
              f"saved {out} ({time.time()-t0:.1f}s)")

    print(f"[filter] done in {time.time()-t0:.1f}s")
    print("[filter] NOTE: these cached cubes are the INPUT to the stats "
          "step.\n         The Filtered source still needs a stats pickle "
          "with the same\n         schema as the other sources (profiles, "
          "pdf_mixfrac, conditional_T_on_Z,\n         joint_pdf_Z_YCO2, "
          "manifold_scatter, y_over_H). Point your stats\n         "
          "computation at these .npy cubes to generate "
          f"filtered_stats_{timestep}.pkl.")


if __name__ == "__main__":
    main()
