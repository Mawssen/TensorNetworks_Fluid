"""
run_mps_stats.py
----------------
Compute statistics on MPS-truncated fields for a single (timestep, cutoff)
pair. Mirrors run_dns_stats.py but reads MAT files via mps_io.

Usage
-----
python run_mps_stats.py <data_dir> <timestep> <cutoff_str> <out_dir>
    [--z-chi-mode {stored,mixed,recon}] [--fd-order {2,4,6,8}]

Modes
-----
  stored : use MPS-truncated mixfrac and MPS-truncated chi as-is.
  mixed  : use MPS-truncated mixfrac for Z, but recompute chi from it
           via 2*alpha*|grad Z|^2 (isolates gradient-amplification of
           MPS noise without confounding with Bilger reconstruction).
  recon  : reconstruct Z via Bilger from MPS species, then recompute chi
           from that Z. Apples-to-apples with DNS reconstruction path.

Examples
--------
python run_mps_stats.py truncated_0198 0198 0.001  results/mps_0198/1e-3
python run_mps_stats.py truncated_0198 0198 0.0001 results/mps_0198/1e-4 --z-chi-mode mixed
python run_mps_stats.py truncated_0198 0198 1.0e-5 results/mps_0198/1e-5 --z-chi-mode recon
"""

from __future__ import annotations

import os
import sys
import time
import pickle

import numpy as np

from dns_stats import GridConfig, compute_all_stats, Z_ST_DEFAULT
from mps_io import load_all_mps_fields, cutoff_label, ordering_for_sp
from derived_fields import compute_Z_bilger, compute_chi, StreamConfig


def _parse_flag_with_value(args, flag, default):
    if flag in args:
        i = args.index(flag)
        if i + 1 >= len(args):
            print(f"Error: {flag} requires an argument")
            sys.exit(1)
        val = args[i + 1]
        args.pop(i + 1)
        args.pop(i)
        return val
    return default


def main():
    args = sys.argv[1:]
    z_chi_mode = _parse_flag_with_value(args, "--z-chi-mode", "stored")
    fd_order = int(_parse_flag_with_value(args, "--fd-order", "8"))
    periodic_xz_str = _parse_flag_with_value(args, "--periodic-xz", "false")
    periodic_xz_cube = periodic_xz_str.lower() in ("true", "1", "yes", "on")
    sp = int(_parse_flag_with_value(args, "--sp", "1"))
    if len(args) != 4:
        print("Usage: python run_mps_stats.py <data_dir> <timestep> "
              "<cutoff_str> <out_dir> [--z-chi-mode {stored,mixed,recon}] "
              "[--fd-order {2,4,6,8}] [--periodic-xz {true,false}] "
              "[--sp {1,2,3,4}]")
        sys.exit(1)
    if z_chi_mode not in ("stored", "mixed", "recon"):
        print(f"Invalid --z-chi-mode {z_chi_mode}; "
              "use 'stored', 'mixed', or 'recon'")
        sys.exit(1)
    if fd_order not in (2, 4, 6, 8):
        print(f"Invalid --fd-order {fd_order}; use 2, 4, 6, or 8")
        sys.exit(1)
    if sp not in (1, 2, 3, 4):
        print(f"Invalid --sp {sp}; use 1, 2, 3, or 4")
        sys.exit(1)
    ordering = ordering_for_sp(sp)

    data_dir, timestep, cutoff_str, out_dir = args
    os.makedirs(out_dir, exist_ok=True)

    grid = GridConfig()
    cf_label = cutoff_label(cutoff_str)

    t0 = time.time()
    print(f"[mps] data_dir={data_dir} timestep={timestep} "
          f"cutoff={cutoff_str} ({cf_label}) z_chi_mode={z_chi_mode} "
          f"fd_order={fd_order} periodic_xz_cube={periodic_xz_cube} "
          f"sp={sp} ({ordering})", flush=True)
    # In mixed/recon, chi will be recomputed from MPS-truncated mixfrac
    # and alpha. The stored MPS chi would be loaded and immediately
    # overwritten -- pure waste. Skip it.
    skip_chi = (z_chi_mode in ("mixed", "recon"))
    fields = load_all_mps_fields(data_dir, timestep, cutoff_str,
                                 required_only=False,
                                 ordering=ordering,
                                 skip_chi=skip_chi)
    print(f"[mps] loaded {len(fields)} fields in {time.time()-t0:.1f}s. "
          f"Keys: {sorted(fields.keys())}", flush=True)
    for k, v in fields.items():
        print(f"    {k:>10s}: shape={v.shape}, "
              f"min={float(v.min()):+.3e}, max={float(v.max()):+.3e}",
              flush=True)

    # z_chi_mode controls which fields we replace before stats:
    #
    #   stored : use MPS-truncated mixfrac and MPS-truncated chi as-is
    #            (no reconstruction)
    #   mixed  : use MPS-truncated mixfrac for Z, but recompute chi from it
    #            (isolates the gradient-amplification effect of MPS noise
    #             without confounding it with Bilger reconstruction)
    #   recon  : reconstruct Z via Bilger from MPS species, then recompute
    #            chi from that reconstructed Z (apples-to-apples with how
    #            we'd reconstruct from DNS species)
    if z_chi_mode == "recon":
        if all(k in fields for k in ("Y_CO", "Y_CO2", "Y_H2", "Y_H2O",
                                     "Y_O2", "Y_OH")):
            print("[mps] reconstructing Z (Bilger) from MPS species...",
                  flush=True)
            Z_rec, info = compute_Z_bilger(fields, streams=StreamConfig())
            fields["mixfrac"] = Z_rec
            print(f"      sum_Y mean={info['sumY_mean']:.4f}, "
                  f"clip frac={info['Z_clip_fraction']*100:.3f}%",
                  flush=True)
        else:
            print("[mps] WARN: missing species for Bilger; "
                  "keeping stored mixfrac", flush=True)

    if z_chi_mode in ("mixed", "recon"):
        if all(k in fields for k in ("mixfrac", "alpha", "rho")):
            src_label = "Bilger-reconstructed Z" if z_chi_mode == "recon" \
                        else "MPS-truncated Z"
            print(f"[mps] reconstructing chi from {src_label}, "
                  f"alpha, rho (FD order={fd_order})...", flush=True)
            chi_rec = compute_chi(
                fields["mixfrac"], fields["alpha"], fields["rho"],
                grid.dx_m, grid.dy_m, grid.dz_m,
                alpha_is="thermal_diffusivity", order=fd_order,
                periodic_xz=periodic_xz_cube,
            )
            fields["chi"] = chi_rec
        else:
            print("[mps] WARN: missing alpha/rho for chi reconstruction; "
                  "keeping stored chi", flush=True)

    t1 = time.time()
    print("[mps] computing stats...", flush=True)
    results = compute_all_stats(fields, grid, z_st=Z_ST_DEFAULT)
    print(f"[mps] stats done in {time.time()-t1:.1f}s", flush=True)

    ext = results["extinction"]
    print(f"\n=== [mps cf={cf_label} mode={z_chi_mode}] Summary ===")
    print(f"delta_Z / (2H)        = {results['delta_Z_over_2H']:.4f}")
    print(f"volume-averaged M_ext = {ext['volume_averaged_marker']:.3e}")
    print(f"cond. P(extinction)   = {ext['conditional_extinction_prob']:.3e}")
    print(f"T on stoich surface   = {ext['T_on_stoich_surface']:.2f} K")
    print(f"n (near-stoich cells) = {ext['n_near_st_cells']:,} / "
          f"{ext['n_total_cells']:,}")

    # Tag the metadata so the comparison plotter can label correctly
    results["mps_cutoff_str"] = cutoff_str
    results["mps_cutoff_label"] = cf_label
    results["z_chi_mode"] = z_chi_mode
    results["fd_order"] = fd_order
    results["periodic_xz_cube"] = periodic_xz_cube
    results["sp"] = sp
    results["ordering"] = ordering

    # Pickle filename always carries sp tag now (sp1 was the legacy default,
    # but encoding it in the name makes mixed-sp directories unambiguous).
    # FD_ORDER and PERIODIC_XZ tags only apply when chi is reconstructed.
    if z_chi_mode in ("recon", "mixed"):
        stem = (f"mps_stats_{timestep}_cf{cf_label}"
                f"_{z_chi_mode}_o{fd_order}_sp{sp}")
        if periodic_xz_cube:
            stem += "_p"
    else:
        stem = f"mps_stats_{timestep}_cf{cf_label}_{z_chi_mode}_sp{sp}"
    pkl_path = os.path.join(out_dir, stem + ".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[mps] save {pkl_path}")

    print(f"\nTotal wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
