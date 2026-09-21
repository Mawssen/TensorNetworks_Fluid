"""
run_filtered_stats.py
---------------------
Compute the QC4PDE statistics pickle for the box-filtered DNS ("Filtered"
source, an ideal-LES reference) using the SAME pipeline every other source
uses: assemble a `fields` dict and call dns_stats.compute_all_stats. This
guarantees the output pickle has an identical schema to dns/les/mps/peps, so
it plugs into plot_qc4pde.py with no special-casing.

Inputs are the cached 64^3 cubes written by make_filtered_cube.py:
    filtered_mixfrac_<ts>_f<f>.npy   (Z)
    filtered_T_<ts>_f<f>.npy         (temperature)
    filtered_Y_O2_<ts>_f<f>.npy
    filtered_Y_OH_<ts>_f<f>.npy
    filtered_Y_CO2_<ts>_f<f>.npy
    filtered_alpha_<ts>_f<f>.npy     (needed for chi; run make_filtered_cube
                                      with --fields alpha to produce it)

chi is reconstructed from the FILTERED Z and alpha via compute_chi, exactly
as the PEPS pipeline reconstructs it -- so the filtered chi is the chi of the
filtered field, not a filtered chi (the physically correct a-priori quantity).

IMPORTANT -- grid spacing:
    The filtered cube is 64^3 with 8x coarser spacing than DNS. chi involves
    a gradient, so it must use the FILTERED spacing (dx*8), not the DNS
    spacing, or the dissipation magnitude will be wrong by 8^2. This script
    builds a coarse GridConfig with dx,dy,dz multiplied by --factor.

Usage
-----
    python run_filtered_stats.py <cache_dir> <timestep> <out_dir> \
        [--factor 8] [--fd-order 4] [--periodic-xz false]

Produces:  <out_dir>/filtered_stats_<timestep>.pkl
"""

from __future__ import annotations

import os
import sys
import time
import pickle

import numpy as np

from dns_stats import GridConfig, compute_all_stats, Z_ST_DEFAULT, load_field
from derived_fields import compute_chi


def _parse(args, flag, default=None, cast=str):
    if flag in args:
        i = args.index(flag)
        v = args[i + 1]
        args.pop(i + 1); args.pop(i)
        return cast(v) if cast is not str else v
    return default


def _load_filtered(cache_dir, field, timestep, factor):
    p = os.path.join(cache_dir,
                     f"filtered_{field}_{timestep}_f{factor}.npy")
    if not os.path.exists(p):
        return None, p
    return np.load(p).astype(np.float32), p


def _coarse_spacing(grid, factor):
    """Return (dx, dy, dz) scaled for the coarse filtered cube.

    dx_m/dy_m/dz_m are read-only properties on GridConfig (derived from the
    domain lengths and full grid size), so we do NOT mutate the grid. The
    filtered cube has factor-times-larger cells, and only the gradient in
    compute_chi needs this coarse spacing, so we pass these floats directly.
    (This mirrors plot_dns_mps_comparison.py, which computes
    dx_les = coarsen * grid.dx_m as locals.)
    """
    return (grid.dx_m * factor, grid.dy_m * factor, grid.dz_m * factor)


def main():
    args = sys.argv[1:]
    factor = int(_parse(args, "--factor", default="8"))
    fd_order = int(_parse(args, "--fd-order", default="4"))
    periodic_str = _parse(args, "--periodic-xz", default="false")
    periodic_xz = periodic_str.lower() in ("true", "1", "yes", "on")
    dns_pkl = _parse(args, "--dns-pkl", default=None)

    if len(args) != 3:
        print("Usage: python run_filtered_stats.py <cache_dir> <timestep> "
              "<out_dir> [--factor 8] [--fd-order 4] [--periodic-xz false]")
        sys.exit(1)
    cache_dir, timestep, out_dir = args
    os.makedirs(out_dir, exist_ok=True)

    grid = GridConfig()
    dxc, dyc, dzc = _coarse_spacing(grid, factor)
    t0 = time.time()
    print(f"[filtered] cache_dir={cache_dir} timestep={timestep} "
          f"factor={factor}", flush=True)
    print(f"[filtered] coarse spacing for chi: dx={dxc:.3e} dy={dyc:.3e} "
          f"dz={dzc:.3e}  (DNS x{factor})", flush=True)

    # ----- assemble fields dict from cached filtered cubes ---------------
    fields = {}
    need = ["mixfrac", "T", "Y_O2", "Y_OH", "Y_CO2"]
    optional = ["Y_CO", "alpha"]
    for key in need + optional:
        arr, p = _load_filtered(cache_dir, key, timestep, factor)
        if arr is None:
            if key in need:
                print(f"[filtered] ERROR: required field '{key}' not cached "
                      f"at {p}\n           run make_filtered_cube.py "
                      f"--fields {key} first.", flush=True)
                sys.exit(1)
            else:
                print(f"[filtered] note: optional field '{key}' not cached "
                      f"({p}); continuing.", flush=True)
            continue
        fields[key] = arr
        print(f"    {key:>8s}: shape={arr.shape} "
              f"range=[{arr.min():+.3e},{arr.max():+.3e}]", flush=True)

    # ----- chi from filtered Z + alpha -----------------------------------
    if "rho" not in fields:
        fields["rho"] = np.ones_like(fields["mixfrac"], dtype=np.float32)
    if "alpha" in fields:
        print(f"[filtered] computing chi from FILTERED Z and alpha "
              f"(fd_order={fd_order}, periodic_xz={periodic_xz})...",
              flush=True)
        fields["chi"] = compute_chi(
            fields["mixfrac"], fields["alpha"], fields["rho"],
            dxc, dyc, dzc,
            alpha_is="thermal_diffusivity", order=fd_order,
            periodic_xz=periodic_xz,
        )
        print(f"       chi: peak={float(fields['chi'].max()):.3e}, "
              f"mean={float(fields['chi'].mean()):.3e}", flush=True)
    else:
        print("[filtered] WARNING: alpha not cached, so chi cannot be "
              "computed. The chi-overlay figure will lack a Filtered curve, "
              "but all other figures are unaffected. To include chi, run:\n"
              "    python make_filtered_cube.py --fields alpha ...",
              flush=True)

    # ----- run the identical stats pipeline ------------------------------
    t1 = time.time()
    print("[filtered] computing stats via compute_all_stats...", flush=True)
    results = compute_all_stats(fields, grid, z_st=Z_ST_DEFAULT)
    print(f"[filtered] stats done in {time.time()-t1:.1f}s", flush=True)

    # ----- FIX the cross-stream axis -------------------------------------
    # compute_all_stats built y_over_H from the fine `grid` spacing, but the
    # filtered field is 8x coarser, so its 64-point profiles were placed on
    # an axis spanning only ~1/factor of the true y/H range. Rebuild the
    # correct coarse y_over_H, and (if a DNS pickle is given) interpolate the
    # 1D profiles onto the DNS y_over_H so Filtered shares the exact same
    # axis as every other source (mirrors the LES handling in
    # plot_dns_mps_comparison.py).
    y_bad = np.asarray(results.get("y_over_H"))
    m = len(y_bad)
    y_coarse = (np.arange(m) - (m - 1) / 2.0) * (dyc / grid.H)

    if dns_pkl and os.path.exists(dns_pkl):
        with open(dns_pkl, "rb") as f:
            dns_res = pickle.load(f)
        y_target = np.asarray(dns_res["y_over_H"])
        print(f"[filtered] interpolating profiles from coarse y "
              f"[{y_coarse.min():.2f},{y_coarse.max():.2f}] ({m} pts) "
              f"onto DNS y ({len(y_target)} pts)", flush=True)

        def _reinterp(arr):
            a = np.asarray(arr, dtype=float)
            if a.ndim == 1 and a.shape[0] == m:
                return np.interp(y_target, y_coarse, a)
            return arr

        # profiles: dict of field -> {mean, rms, ...}
        if isinstance(results.get("profiles"), dict):
            for fld, d in results["profiles"].items():
                if isinstance(d, dict):
                    for stat in list(d.keys()):
                        d[stat] = _reinterp(d[stat])
        # any other top-level 1D arrays defined on the y grid
        for key in list(results.keys()):
            if key in ("profiles", "y_over_H"):
                continue
            v = results[key]
            if isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] == m:
                results[key] = _reinterp(v)
        results["y_over_H"] = y_target
        results["y_over_H_native"] = y_coarse
    else:
        # No DNS pickle: at least fix the axis to the correct coarse range so
        # Filtered is not crammed into 1/factor of the plot.
        print(f"[filtered] no --dns-pkl given; storing corrected coarse "
              f"y_over_H (range [{y_coarse.min():.2f},{y_coarse.max():.2f}]). "
              f"Profiles keep their 64-point coarse axis.", flush=True)
        results["y_over_H"] = y_coarse

    # metadata so the plotter/debugging can identify the source
    results["filtered_factor"] = factor
    results["fd_order"] = fd_order
    results["periodic_xz_cube"] = periodic_xz
    results["source_label"] = "Filtered"

    pkl_path = os.path.join(out_dir, f"filtered_stats_{timestep}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[filtered] saved {pkl_path}")
    print(f"[filtered] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
