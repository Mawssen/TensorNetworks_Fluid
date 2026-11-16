"""
run_les_stats.py
----------------
Compute a full statistics pickle for an LES plt file, structured identically
to the DNS stats pickle so the existing comparison plotter can consume it.

Usage
-----
python run_les_stats.py <plt_path> <out_dir> [--timestep TS]
    [--z-chi-mode {stored,mixed,recon}]    default 'stored'
    [--fd-order {2,4,6,8}]                  default 4
    [--periodic-xz {true,false}]            default false

Example
-------
python run_les_stats.py /ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198/LES_plt20000 \
       results/les_0198 --timestep 0198

Output
------
results/les_0198/les_stats_<TS>.pkl
"""

from __future__ import annotations

import os
import sys
import time
import pickle
from typing import Dict

import numpy as np

from dns_stats import GridConfig, compute_all_stats, Z_ST_DEFAULT
from derived_fields import compute_chi
from les_io import (load_all_les_fields, LESGrid, y_over_H_les)


def _parse_flag_with_value(args, name, default):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        args.pop(i + 1)
        args.pop(i)
        return v
    return default


def main():
    args = sys.argv[1:]
    timestep = _parse_flag_with_value(args, "--timestep", "0198")
    z_chi_mode = _parse_flag_with_value(args, "--z-chi-mode", "stored")
    fd_order = int(_parse_flag_with_value(args, "--fd-order", "4"))
    periodic_str = _parse_flag_with_value(args, "--periodic-xz", "false")
    periodic_xz_cube = periodic_str.lower() in ("true", "1", "yes", "on")
    tau_plt_path = _parse_flag_with_value(args, "--tau-plt", None)
    tauall_plt_path = _parse_flag_with_value(args, "--tauall-plt", None)

    if len(args) != 2:
        print("Usage: python run_les_stats.py <plt_path> <out_dir> "
              "[--timestep TS] [--z-chi-mode {stored,mixed,recon,fdf}] "
              "[--fd-order {2,4,6,8}] [--periodic-xz {true,false}] "
              "[--tau-plt PATH --tauall-plt PATH]")
        sys.exit(1)
    plt_path, out_dir = args
    if z_chi_mode not in ("stored", "mixed", "recon", "fdf"):
        print(f"Invalid --z-chi-mode {z_chi_mode}")
        sys.exit(1)
    if fd_order not in (2, 4, 6, 8):
        print(f"Invalid --fd-order {fd_order}")
        sys.exit(1)

    # Auto-detect FDF mode: if Tau files exist alongside the LES plt, use them.
    if z_chi_mode == "stored" and tau_plt_path is None and tauall_plt_path is None:
        # Look in same directory as plt for Tauplt<N>.temp + TauAllplt<N>.temp.
        # plt_path looks like '<dir>/LES_plt20000'; the N is the trailing digits.
        plt_dir = os.path.dirname(plt_path) or "."
        plt_base = os.path.basename(plt_path)
        digits = "".join(c for c in plt_base if c.isdigit())
        if digits:
            tau_cand = os.path.join(plt_dir, f"Tauplt{digits}.temp")
            tauall_cand = os.path.join(plt_dir, f"TauAllplt{digits}.temp")
            if os.path.exists(tau_cand) and os.path.exists(tauall_cand):
                tau_plt_path = tau_cand
                tauall_plt_path = tauall_cand
                z_chi_mode = "fdf"
                print(f"[les] Auto-detected Tau files; switching to FDF chi mode")

    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print(f"[les] plt_path={plt_path}", flush=True)
    print(f"[les] timestep={timestep}  z_chi_mode={z_chi_mode}  "
          f"fd_order={fd_order}  periodic_xz_cube={periodic_xz_cube}",
          flush=True)
    if z_chi_mode == "fdf":
        print(f"[les] tau_plt   = {tau_plt_path}", flush=True)
        print(f"[les] tauall_plt= {tauall_plt_path}", flush=True)

    # The fields LES will provide: Z, T, major species. chi is treated as
    # OPTIONAL even in stored mode -- not every PeleLM build writes it,
    # and the LES Scalar_diss field naming varies. If absent we'll fill
    # with zeros below.
    required = ["mixfrac", "T", "Y_CO", "Y_CO2", "Y_OH", "Y_O2"]
    optional = ["chi", "alpha", "rho", "Y_H", "Y_H2", "Y_H2O", "Y_HCO",
                "Y_HO2", "Y_O", "P", "u", "v", "w"]

    fields = load_all_les_fields(plt_path,
                                 names_required=required,
                                 names_optional=optional)
    print(f"[les] loaded fields in {time.time()-t0:.1f}s. "
          f"Keys: {sorted(fields.keys())}", flush=True)
    for k, v in fields.items():
        print(f"    {k:>10s}: shape={v.shape}, "
              f"min={float(v.min()):+.3e}, max={float(v.max()):+.3e}",
              flush=True)

    # --- chi handling --------------------------------------------------
    # LES grid spacings (in meters). LES is 1/8 DNS so dx = 8 * dx_DNS.
    # Use the DNS GridConfig for physical extents (same physical domain).
    dns_grid = GridConfig()
    les_grid = LESGrid()
    dx_les = dns_grid.Lx_m / les_grid.nx_full
    dy_les = dns_grid.Ly_m / les_grid.ny_full
    dz_les = dns_grid.Lz_m / les_grid.nz_full

    if z_chi_mode == "fdf":
        # FDF micromixing chi: chi = 2 * C_phi * Omega * <Z''^2>
        # where C_phi=5, Omega='Freq' from Tauplt, <Z''^2>='Z' from TauAllplt.
        # This is the LES-FDF closure model, matches Aitzhan's formulation.
        from les_io import load_fdf_chi_from_tau_files
        if tau_plt_path is None or tauall_plt_path is None:
            print("[les] ERROR: --z-chi-mode fdf requires --tau-plt and "
                  "--tauall-plt", flush=True)
            sys.exit(1)
        chi_fdf = load_fdf_chi_from_tau_files(tau_plt_path, tauall_plt_path,
                                              les_grid)
        fields["chi"] = chi_fdf
        print(f"[les] FDF chi loaded: shape={chi_fdf.shape}, "
              f"peak={chi_fdf.max():.3e}, mean={chi_fdf.mean():.3e}",
              flush=True)
    elif z_chi_mode in ("mixed", "recon"):
        if "alpha" not in fields:
            print(f"[les] WARN: alpha not in LES plt; cannot reconstruct chi. "
                  f"Falling back to stored chi if available.", flush=True)
        else:
            rho_arg = fields.get("rho")
            if rho_arg is None:
                rho_arg = np.ones_like(fields["alpha"])
            chi_rec = compute_chi(
                fields["mixfrac"], fields["alpha"], rho_arg,
                dx_les, dy_les, dz_les,
                alpha_is="thermal_diffusivity",
                order=fd_order,
                periodic_xz=periodic_xz_cube,
            )
            fields["chi"] = chi_rec
            print(f"[les] chi reconstructed: peak={chi_rec.max():.3e}",
                  flush=True)

    if "chi" not in fields:
        # Stats functions expect 'chi' to exist; provide zeros so other
        # statistics still compute and the field can be ignored on the
        # comparison plot side.
        fields["chi"] = np.zeros_like(fields["mixfrac"])
        print(f"[les] no chi available; using zeros placeholder", flush=True)

    # --- Run stats ------------------------------------------------------
    results = compute_all_stats(fields, dns_grid, z_st=Z_ST_DEFAULT)

    # Override y_over_H with the LES grid coordinate (compute_all_stats
    # uses DNS dy via GridConfig, which would be wrong by 8x for LES).
    results["y_over_H"] = y_over_H_les(les_grid,
                                       H_m=dns_grid.H,
                                       Ly_m=dns_grid.Ly_m)
    results["timestep"] = timestep
    results["source"] = "LES"
    results["plt_path"] = plt_path
    results["les_n"] = les_grid.n
    results["z_chi_mode"] = z_chi_mode
    results["fd_order"] = fd_order
    results["periodic_xz_cube"] = periodic_xz_cube

    # --- Save -----------------------------------------------------------
    pkl_path = os.path.join(out_dir, f"les_stats_{timestep}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[write] {pkl_path}")
    print(f"[les] total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
