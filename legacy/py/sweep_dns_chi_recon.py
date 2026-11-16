"""
sweep_dns_chi_recon.py
----------------------
Compute chi reconstruction validation diagnostics for the DNS cube
across all combinations of FD order and periodic-xz setting.

For each (order, periodic) pair, computes:
  - pointwise rel L2(reconstructed_chi vs stored_chi)
  - pointwise correlation
  - profile rel L2 (after x-z averaging)
  - profile correlation
  - peak ratio (reconstructed peak / stored peak)
  - the x-z averaged chi profile itself (for plotting)

Loads the cube fields ONCE and reuses across all combinations.
Produces an overlay figure (fig09_chi_profile_sweep.png) that puts
DNS-stored and all 6 reconstructions on the same axes.

Usage
-----
python sweep_dns_chi_recon.py <data_dir> <timestep> <out_dir>

Example
-------
python sweep_dns_chi_recon.py jet_0198 0198 results/dns_0198/chi_sweep
"""

from __future__ import annotations

import os
import sys
import time
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dns_stats import GridConfig, load_all_dns_fields, _xz_mean
from derived_fields import (
    compute_chi, chi_diagnostics,
    relative_l2_error, pointwise_error_stats,
)


def _fold_profile(y, prof):
    """Fold a y-profile around y=0 into a single-sided y/H >= 0 view."""
    n = len(y)
    j0 = int(np.argmin(np.abs(y)))
    n_keep = min(n - j0, j0 + 1)
    p_pos = prof[j0:j0 + n_keep]
    p_neg = prof[j0 - np.arange(n_keep)]
    return y[j0:j0 + n_keep], 0.5 * (p_pos + p_neg)


def _plot_overlay(rows, dns_profile, y_over_H, out_path,
                  single_sided=True, ylim=(0.0, 1000.0)):
    """Overlay DNS chi profile plus all reconstructed cases."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    fig, ax = plt.subplots(figsize=(8.5, 6))

    # DNS reference
    if single_sided:
        yh, p = _fold_profile(y_over_H, dns_profile)
    else:
        yh, p = y_over_H, dns_profile
    ax.plot(yh, p, color="black", linestyle="-", linewidth=2.4,
            label="DNS (stored)", zorder=10)

    # Color/style per (order, periodic) combination
    color_by_order = {2: "#e67e22", 4: "#27ae60", 8: "#2980b9"}
    style_by_periodic = {False: "--", True: ":"}

    for row in rows:
        order = row["order"]
        periodic = row["periodic_xz"]
        prof = row["chi_profile"]
        if single_sided:
            _, prof = _fold_profile(y_over_H, prof)
        label = f"order={order}, periodic={periodic}"
        ax.plot(yh, prof,
                color=color_by_order[order],
                linestyle=style_by_periodic[periodic],
                linewidth=1.6,
                label=label)

    ax.set_xlabel(r"$y/H$")
    ax.set_ylabel(r"$\overline{\chi}$  [1/s]")
    ax.set_title("Scalar dissipation rate -- DNS stored vs FD reconstructions")
    if single_sided:
        ax.set_xlim(0, 5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")


def main():
    if len(sys.argv) != 4:
        print("Usage: python sweep_dns_chi_recon.py <data_dir> <timestep> "
              "<out_dir>")
        sys.exit(1)

    data_dir, timestep, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)

    grid = GridConfig()

    print(f"[sweep] Loading cube fields for {timestep}...", flush=True)
    t0 = time.time()
    fields = load_all_dns_fields(data_dir, timestep, grid,
                                 required_only=False, region="cube")
    print(f"        Loaded in {time.time() - t0:.1f}s. "
          f"Keys: {sorted(fields.keys())}", flush=True)

    required = ("mixfrac", "alpha", "rho", "chi")
    missing = [k for k in required if k not in fields]
    if missing:
        print(f"ERROR: missing required fields: {missing}")
        sys.exit(1)

    chi_ref = fields["chi"]
    Z = fields["mixfrac"]
    alpha = fields["alpha"]
    rho = fields["rho"]

    dns_profile = _xz_mean(chi_ref)
    y_over_H = grid.y_over_H()

    cases = [(o, p) for o in (2, 4, 8) for p in (False, True)]

    rows = []
    print(f"\n[sweep] Running {len(cases)} cases:")
    print(f"{'order':>5s}  {'periodic':>10s}  {'point_l2':>10s}  "
          f"{'point_corr':>11s}  {'prof_l2':>9s}  {'prof_corr':>10s}  "
          f"{'peak_ratio':>11s}  {'time(s)':>8s}")
    print("-" * 88)

    for order, periodic in cases:
        t1 = time.time()
        chi_rec = compute_chi(
            Z, alpha, rho,
            grid.dx_m, grid.dy_m, grid.dz_m,
            alpha_is="thermal_diffusivity",
            order=order,
            periodic_xz=periodic,
        )
        diag = chi_diagnostics(chi_rec, chi_ref)
        pw = pointwise_error_stats(chi_rec, chi_ref)
        chi_prof = _xz_mean(chi_rec)
        elapsed = time.time() - t1

        row = {
            "order": order,
            "periodic_xz": periodic,
            "pointwise_rel_l2": relative_l2_error(chi_rec, chi_ref),
            "pointwise_corr": pw["correlation"],
            "profile_rel_l2": diag["profile_rel_l2"],
            "profile_corr": diag["profile_correlation"],
            "peak_ratio": diag["peak_ratio"],
            "wall_time_s": elapsed,
            "chi_profile": chi_prof,
        }
        rows.append(row)
        print(f"{order:>5d}  {str(periodic):>10s}  "
              f"{row['pointwise_rel_l2']:>10.3e}  "
              f"{row['pointwise_corr']:>11.4f}  "
              f"{row['profile_rel_l2']:>9.3e}  "
              f"{row['profile_corr']:>10.4f}  "
              f"{row['peak_ratio']:>11.4f}  "
              f"{elapsed:>8.1f}", flush=True)

    # Save the table as a pickle and a plain-text summary
    out_data = {
        "rows": rows,
        "dns_chi_profile": dns_profile,
        "y_over_H": y_over_H,
        "timestep": timestep,
    }
    pkl_path = os.path.join(out_dir, f"chi_sweep_{timestep}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(out_data, f)
    print(f"\n[write] {pkl_path}")

    txt_path = os.path.join(out_dir, f"chi_sweep_{timestep}.txt")
    with open(txt_path, "w") as f:
        f.write(f"DNS chi reconstruction sweep, timestep {timestep}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'order':>5s}  {'periodic':>10s}  {'point_l2':>10s}  "
                f"{'point_corr':>11s}  {'prof_l2':>9s}  {'prof_corr':>10s}  "
                f"{'peak_ratio':>11s}\n")
        f.write("-" * 80 + "\n")
        for row in rows:
            f.write(f"{row['order']:>5d}  {str(row['periodic_xz']):>10s}  "
                    f"{row['pointwise_rel_l2']:>10.3e}  "
                    f"{row['pointwise_corr']:>11.4f}  "
                    f"{row['profile_rel_l2']:>9.3e}  "
                    f"{row['profile_corr']:>10.4f}  "
                    f"{row['peak_ratio']:>11.4f}\n")
    print(f"[write] {txt_path}")

    # Overlay plots: single-sided (y/H >= 0) and full
    _plot_overlay(rows, dns_profile, y_over_H,
                  os.path.join(out_dir,
                               f"fig09_chi_profile_sweep_{timestep}.png"),
                  single_sided=True, ylim=(0.0, 1000.0))
    _plot_overlay(rows, dns_profile, y_over_H,
                  os.path.join(out_dir,
                               f"fig09_chi_profile_sweep_{timestep}_full.png"),
                  single_sided=False, ylim=(0.0, 1000.0))
    # Also produce an autoscaled version so periodic cases (with their
    # huge boundary spikes) are fully visible.
    _plot_overlay(rows, dns_profile, y_over_H,
                  os.path.join(out_dir,
                               f"fig09_chi_profile_sweep_{timestep}_auto.png"),
                  single_sided=False, ylim=None)

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
