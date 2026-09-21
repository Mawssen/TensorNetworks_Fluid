"""
plot_field_scatter.py
---------------------
Generate ONE scatter plot of (DNS vs MPS) for a single (field, sp, cutoff)
combination. Designed to be run as a SLURM array task -- one plot per task.

For Z (mixfrac) and T: loads the stored fields directly.
For chi in --mode mixed: loads MPS-truncated mixfrac+alpha+rho, recomputes
chi from MPS-truncated Z, AND recomputes DNS chi the same way (apples-to-
apples, same FD operator both sides).

Renders 1M downsampled scatter points (uniform random subsample) with
the y=x reference line.

Usage
-----
python plot_field_scatter.py
    --field {mixfrac,T,chi}
    --sp {1,2,3,4}
    --cf {1e-2,1e-3,1e-4,1e-5}
    --timestep 0198
    --dns-data-dir /path/to/jet_0198
    --mps-data-dir /path/to/truncated_0198_sp{N}
    --out-dir /path/to/plots/scatter
    [--mode {stored,mixed}]                      # default mixed
    [--fd-order {2,4,6,8}]                       # default 8 (only for mixed chi)
    [--periodic-xz {true,false}]                 # default false
    [--n-sample N]                               # default 1000000
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dns_stats import GridConfig, load_field
from derived_fields import compute_chi
from mps_io import load_mps_field, ordering_for_sp, _mps_filename


_FIELD_LABELS = {
    "mixfrac": (r"$Z$",        (0.0, 1.0)),
    "T":       (r"$T$ [K]",    (300.0, 1700.0)),
    "chi":     (r"$\chi$ [1/s]", None),  # auto-range
}


def _parse(args, name, default=None, cast=str, required=False):
    if name in args:
        i = args.index(name)
        if i + 1 >= len(args):
            print(f"Error: {name} requires a value")
            sys.exit(1)
        v = args[i + 1]
        args.pop(i + 1); args.pop(i)
        return cast(v)
    if required:
        print(f"Error: {name} is required")
        sys.exit(1)
    return default


def _cf_to_str(cf_label: str) -> str:
    """Convert label '1e-3' -> filename suffix '0.001' / '1.0e-5'."""
    table = {
        "1e-2": "0.01",
        "1e-3": "0.001",
        "1e-4": "0.0001",
        "1e-5": "1.0e-5",
    }
    if cf_label not in table:
        raise ValueError(f"Unknown cutoff label: {cf_label}")
    return table[cf_label]


def _load_dns_field(dns_data_dir: str, timestep: str, var: str,
                    grid: GridConfig) -> np.ndarray:
    return load_field(os.path.join(dns_data_dir,
                                   f"jet_{var}_{timestep}.dat"), grid)


def _load_mps_field_one(mps_data_dir: str, timestep: str, var: str,
                        cf_str: str, ordering: str) -> np.ndarray:
    path = os.path.join(mps_data_dir,
                        _mps_filename(var, timestep, cf_str, ordering=ordering))
    return load_mps_field(path)


def _build_chi(mixfrac: np.ndarray, alpha: np.ndarray, rho: np.ndarray,
               grid: GridConfig, fd_order: int, periodic_xz_cube: bool
               ) -> np.ndarray:
    return compute_chi(
        mixfrac, alpha, rho,
        grid.dx_m, grid.dy_m, grid.dz_m,
        alpha_is="thermal_diffusivity",
        order=fd_order,
        periodic_xz=periodic_xz_cube,
    )


def _scatter_plot(dns_arr, mps_arr, field, cf_label, sp, out_path,
                  mode, fd_order, periodic_xz_cube, n_sample, rng):
    label, _ = _FIELD_LABELS[field]

    flat_dns = dns_arr.ravel()
    flat_mps = mps_arr.ravel()
    n_total = flat_dns.size
    if n_total > n_sample:
        idx = rng.choice(n_total, size=n_sample, replace=False)
        x = flat_dns[idx]
        y = flat_mps[idx]
    else:
        x = flat_dns
        y = flat_mps

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # Shared axis limits derived from the DNS field only. This is identical
    # for every (sp, cf) plot of a given field, so all plots use the same
    # axis range. MPS overshoots (esp. for chi at low cutoffs) extend beyond
    # DNS range and get visually clipped at the panel edge -- intentional,
    # tells you where MPS spikes.
    if field == "chi":
        # chi is heavy-tailed; use 99.5th percentile of DNS as upper edge
        # so the bulk of points are visible.
        lo = 0.0
        hi = float(np.percentile(flat_dns, 99.5))
    else:
        lo = float(flat_dns.min())
        hi = float(flat_dns.max())
    pad = 0.02 * (hi - lo)
    plot_lim = (lo - pad, hi + pad)

    ax.scatter(x, y, s=0.4, alpha=0.25, color="#2980b9",
               edgecolors="none", rasterized=True)
    ax.plot(plot_lim, plot_lim, "k-", lw=1.0, label="y = x")

    ax.set_xlim(*plot_lim); ax.set_ylim(*plot_lim)
    ax.set_xlabel(f"DNS {label}")
    ax.set_ylabel(f"MPS {label}")
    ordering = ordering_for_sp(sp)
    title = (f"{field} scatter  --  sp={sp} ({ordering}), cf={cf_label}, "
             f"mode={mode}")
    if field == "chi" and mode == "mixed":
        title += f", FD={fd_order}, per_xz={periodic_xz_cube}"
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.minorticks_off()
    ax.legend(loc="upper left", fontsize=9)

    # Diagnostics in corner -- computed on the FULL field, not the subsample
    rms_err = float(np.sqrt(np.mean((flat_dns - flat_mps) ** 2)))
    rel_l2 = float(np.linalg.norm(flat_dns - flat_mps) /
                   max(np.linalg.norm(flat_dns), 1e-30))
    corr = float(np.corrcoef(flat_dns, flat_mps)[0, 1])
    info = (f"n_sampled = {len(x):,} / {n_total:,}\n"
            f"RMS err   = {rms_err:.3e}\n"
            f"rel L2    = {rel_l2:.3e}\n"
            f"corr      = {corr:.5f}")
    ax.text(0.98, 0.02, info, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")
    return dict(rms=rms_err, rel_l2=rel_l2, corr=corr,
                n_total=n_total, n_sampled=len(x))


def main():
    args = sys.argv[1:]

    field = _parse(args, "--field", required=True)
    sp = _parse(args, "--sp", cast=int, required=True)
    cf_label = _parse(args, "--cf", required=True)
    timestep = _parse(args, "--timestep", default="0198")
    dns_data_dir = _parse(args, "--dns-data-dir", required=True)
    mps_data_dir = _parse(args, "--mps-data-dir", required=True)
    out_dir = _parse(args, "--out-dir", required=True)
    mode = _parse(args, "--mode", default="mixed")
    fd_order = _parse(args, "--fd-order", default=4, cast=int)
    periodic_str = _parse(args, "--periodic-xz", default="false")
    periodic_xz_cube = periodic_str.lower() in ("true", "1", "yes", "on")
    n_sample = _parse(args, "--n-sample", default=1_000_000, cast=int)
    seed = _parse(args, "--seed", default=42, cast=int)

    if field not in ("mixfrac", "T", "chi"):
        print(f"Invalid --field {field}; use mixfrac, T, or chi")
        sys.exit(1)
    if sp not in (1, 2, 3, 4):
        print(f"Invalid --sp {sp}")
        sys.exit(1)
    if mode not in ("stored", "mixed"):
        print(f"Invalid --mode {mode}; use stored or mixed")
        sys.exit(1)
    if cf_label not in ("1e-2", "1e-3", "1e-4", "1e-5"):
        print(f"Invalid --cf {cf_label}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    grid = GridConfig()
    ordering = ordering_for_sp(sp)
    cf_str = _cf_to_str(cf_label)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    print(f"[scatter] field={field}  sp={sp} ({ordering})  cf={cf_label}  "
          f"mode={mode}", flush=True)

    if field == "chi":
        # Need mixfrac + alpha + rho on both sides to reconstruct chi
        # with the same operator.
        Z_dns     = _load_dns_field(dns_data_dir, timestep, "mixfrac", grid)
        alpha_dns = _load_dns_field(dns_data_dir, timestep, "alpha",   grid)
        rho_dns   = _load_dns_field(dns_data_dir, timestep, "rho",     grid)
        chi_dns = _build_chi(Z_dns, alpha_dns, rho_dns, grid,
                             fd_order=fd_order,
                             periodic_xz_cube=periodic_xz_cube)
        del Z_dns, alpha_dns, rho_dns

        Z_mps     = _load_mps_field_one(mps_data_dir, timestep, "mixfrac",
                                        cf_str, ordering)
        alpha_mps = _load_mps_field_one(mps_data_dir, timestep, "alpha",
                                        cf_str, ordering)
        rho_mps   = _load_mps_field_one(mps_data_dir, timestep, "rho",
                                        cf_str, ordering)
        chi_mps = _build_chi(Z_mps, alpha_mps, rho_mps, grid,
                             fd_order=fd_order,
                             periodic_xz_cube=periodic_xz_cube)
        del Z_mps, alpha_mps, rho_mps

        dns_arr, mps_arr = chi_dns, chi_mps
    else:
        dns_arr = _load_dns_field(dns_data_dir, timestep, field, grid)
        mps_arr = _load_mps_field_one(mps_data_dir, timestep, field,
                                      cf_str, ordering)

    print(f"[scatter] loaded fields in {time.time()-t0:.1f}s; "
          f"shapes DNS={dns_arr.shape} MPS={mps_arr.shape}", flush=True)

    # Build output filename
    suffix = f"_o{fd_order}" if (field == "chi" and mode == "mixed") else ""
    if periodic_xz_cube:
        suffix += "_p"
    out_path = os.path.join(
        out_dir,
        f"scatter_{field}_sp{sp}_cf{cf_label}{suffix}.png")

    metrics = _scatter_plot(dns_arr, mps_arr, field, cf_label, sp, out_path,
                            mode=mode, fd_order=fd_order,
                            periodic_xz_cube=periodic_xz_cube,
                            n_sample=n_sample, rng=rng)

    # Persist metrics so an aggregator can build a per-field summary table.
    import json
    metrics_path = out_path.replace(".png", ".json")
    record = {
        "field": field,
        "sp": sp,
        "ordering": ordering,
        "cf_label": cf_label,
        "mode": mode,
        "fd_order": fd_order if (field == "chi" and mode == "mixed") else None,
        "periodic_xz_cube": periodic_xz_cube,
        "rms": metrics["rms"],
        "rel_l2": metrics["rel_l2"],
        "corr": metrics["corr"],
        "n_total": metrics["n_total"],
        "n_sampled": metrics["n_sampled"],
        "png": os.path.basename(out_path),
    }
    with open(metrics_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[metrics] {metrics_path}")

    print(f"[scatter] total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
