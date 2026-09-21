"""
run_dns_stats.py
----------------
Driver: load DNS fields for one timestep, compute every statistic from the
paper, save results to an .npz.

Usage
-----
python run_dns_stats.py <data_dir> <timestep> <out_dir>
    [--validate] [--sweep] [--validate-full]
    [--region {cube,full,both}]
"""

from __future__ import annotations

import os
import sys
import time
import pickle

import numpy as np

from dns_stats import (
    GridConfig,
    load_all_dns_fields,
    compute_all_stats,
    Z_ST_DEFAULT,
)
from derived_fields import (
    validate_reconstructions,
    print_validation_report,
    StreamConfig,
    chi_fd_order_sweep,
    print_fd_order_sweep,
    validate_reconstructions_full,
    print_validation_full_report,
)


def _parse_flag_with_value(args, flag, default):
    """Pop '--flag VALUE' from args list and return VALUE (or default)."""
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


def run_region(
    data_dir: str, timestep: str, out_dir: str,
    grid: GridConfig, region: str,
    validate: bool, fd_order: int, periodic_xz_cube: bool,
) -> None:
    """Compute all stats for one region ('cube' or 'full') and save
    to <out_dir>/<region>/.
    """
    region_out = os.path.join(out_dir, region)
    os.makedirs(region_out, exist_ok=True)

    t0 = time.time()
    print(f"\n[{region}] loading fields...", flush=True)
    fields = load_all_dns_fields(data_dir, timestep, grid,
                                 required_only=False, region=region)
    print(f"[{region}] loaded in {time.time()-t0:.1f}s. Keys: "
          f"{sorted(fields.keys())}", flush=True)
    for k, v in fields.items():
        print(f"    {k:>10s}: shape={v.shape}, "
              f"min={float(v.min()):+.3e}, max={float(v.max()):+.3e}",
              flush=True)

    # Optional cube-only validation (only meaningful for the 'cube' region
    # because it compares stored cube chi to reconstruction from cube Z).
    validation = None
    if validate and region == "cube":
        t_val = time.time()
        print(f"[{region}] validating reconstructions "
              f"(fd_order={fd_order}, periodic_xz_cube={periodic_xz_cube})...",
              flush=True)
        validation = validate_reconstructions(
            fields, grid, streams=StreamConfig(),
            alpha_is="thermal_diffusivity",
            fd_order=fd_order,
            periodic_xz_cube=periodic_xz_cube,
        )
        print_validation_report(validation)
        print(f"[{region}] validate done in {time.time()-t_val:.1f}s",
              flush=True)

    t_s = time.time()
    print(f"[{region}] computing stats...", flush=True)
    results = compute_all_stats(fields, grid, z_st=Z_ST_DEFAULT)
    print(f"[{region}] stats done in {time.time()-t_s:.1f}s.", flush=True)

    ext = results["extinction"]
    print(f"\n=== [{region}] Summary ===")
    print(f"delta_Z / (2H)        = {results['delta_Z_over_2H']:.4f}")
    print(f"volume-averaged M_ext = {ext['volume_averaged_marker']:.3e}")
    print(f"cond. P(extinction)   = {ext['conditional_extinction_prob']:.3e}")
    print(f"T on stoich surface   = {ext['T_on_stoich_surface']:.2f} K")
    print(f"n (near-stoich cells) = {ext['n_near_st_cells']:,} / "
          f"{ext['n_total_cells']:,}")

    stem = f"dns_stats_{timestep}"
    pkl_path = os.path.join(region_out, stem + ".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[{region}] save {pkl_path}")

    if validation is not None:
        val_path = os.path.join(region_out, f"dns_validation_{timestep}.pkl")
        with open(val_path, "wb") as f:
            pickle.dump(validation, f)
        print(f"[{region}] save {val_path}")

    # Free memory before the next region
    del fields


def main():
    args = sys.argv[1:]
    validate = False
    sweep = False
    validate_full = False
    region_choice = _parse_flag_with_value(args, "--region", "both")
    fd_order = int(_parse_flag_with_value(args, "--fd-order", "8"))
    periodic_xz_str = _parse_flag_with_value(args, "--periodic-xz", "false")
    periodic_xz_cube = periodic_xz_str.lower() in ("true", "1", "yes", "on")
    if "--validate" in args:
        validate = True
        args.remove("--validate")
    if "--sweep" in args:
        sweep = True
        args.remove("--sweep")
    if "--validate-full" in args:
        validate_full = True
        args.remove("--validate-full")
    if len(args) != 3:
        print("Usage: python run_dns_stats.py <data_dir> <timestep> "
              "<out_dir> [--validate] [--sweep] [--validate-full] "
              "[--region {cube,full,both}] [--fd-order {2,4,6,8}] "
              "[--periodic-xz {true,false}]")
        sys.exit(1)

    data_dir, timestep, out_dir = args
    os.makedirs(out_dir, exist_ok=True)

    grid = GridConfig()

    if region_choice not in ("cube", "full", "both"):
        print(f"Invalid --region {region_choice}; use cube, full, or both")
        sys.exit(1)
    if fd_order not in (2, 4, 6, 8):
        print(f"Invalid --fd-order {fd_order}; use 2, 4, 6, or 8")
        sys.exit(1)

    print(f"[config] fd_order = {fd_order}, "
          f"periodic_xz_cube = {periodic_xz_cube}", flush=True)

    t_all = time.time()

    regions_to_run = []
    if region_choice in ("cube", "both"):
        regions_to_run.append("cube")
    if region_choice in ("full", "both"):
        regions_to_run.append("full")

    for region in regions_to_run:
        run_region(data_dir, timestep, out_dir, grid, region, validate,
                   fd_order=fd_order, periodic_xz_cube=periodic_xz_cube)

    # --- optional full-grid Z+chi validation (independent of the stats
    # regions; runs on full grid and crops to cube for comparison) ---
    if sweep or validate_full:
        # Re-load cube fields (cheap if already cached at OS level)
        print("\n[extra] loading cube fields for sweep / validate-full...",
              flush=True)
        cube_fields = load_all_dns_fields(data_dir, timestep, grid,
                                          required_only=False, region="cube")

        if sweep:
            ts = time.time()
            print(f"[sweep] running FD-order sweep for chi "
                  f"(periodic_xz_cube={periodic_xz_cube})...", flush=True)
            sw = chi_fd_order_sweep(cube_fields, grid,
                                    periodic_xz_cube=periodic_xz_cube)
            print_fd_order_sweep(sw)
            print(f"[sweep] done in {time.time()-ts:.1f}s", flush=True)

        if validate_full:
            tf = time.time()
            print(f"[validate-full] reconstructing Z and chi on full grid "
                  f"(fd_order={fd_order}, "
                  f"periodic_xz_cube={periodic_xz_cube})...", flush=True)
            validation_full = validate_reconstructions_full(
                data_dir, timestep, grid,
                streams=StreamConfig(),
                alpha_is="thermal_diffusivity",
                fd_order=fd_order,
                periodic_xz_cube=periodic_xz_cube,
            )
            print_validation_full_report(validation_full)
            vf_path = os.path.join(
                out_dir, f"dns_validation_full_{timestep}.pkl")
            with open(vf_path, "wb") as f:
                pickle.dump(validation_full, f)
            print(f"[validate-full] save {vf_path}")
            print(f"[validate-full] done in {time.time()-tf:.1f}s",
                  flush=True)

    print(f"\nTotal wall time: {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
