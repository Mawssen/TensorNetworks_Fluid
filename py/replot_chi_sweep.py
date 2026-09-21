"""
replot_chi_sweep.py
-------------------
Re-render the chi-profile overlay plot from a saved sweep pickle without
re-running the (expensive) field load.

Useful for tweaking ylim, colors, single-sided vs full, etc., after the
fact.

Usage
-----
python replot_chi_sweep.py <pkl_path> [--ymax YMAX] [--full]

Example
-------
python replot_chi_sweep.py results/dns_0198/chi_sweep/chi_sweep_0198.pkl \\
       --ymax 1500
"""

from __future__ import annotations

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fold_profile(y, prof):
    n = len(y)
    j0 = int(np.argmin(np.abs(y)))
    n_keep = min(n - j0, j0 + 1)
    p_pos = prof[j0:j0 + n_keep]
    p_neg = prof[j0 - np.arange(n_keep)]
    return y[j0:j0 + n_keep], 0.5 * (p_pos + p_neg)


def plot_overlay(rows, dns_profile, y_over_H, out_path,
                 single_sided=True, ylim=(0.0, 1000.0)):
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    fig, ax = plt.subplots(figsize=(8.5, 6))

    if single_sided:
        yh, p = _fold_profile(y_over_H, dns_profile)
    else:
        yh, p = y_over_H, dns_profile
    ax.plot(yh, p, color="black", linestyle="-", linewidth=2.4,
            label="DNS (stored)", zorder=10)

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
    args = sys.argv[1:]
    if not args:
        print("Usage: python replot_chi_sweep.py <pkl_path> "
              "[--ymax YMAX] [--full]")
        sys.exit(1)

    pkl_path = args[0]
    ymax = 1000.0
    full = False
    auto = False
    i = 1
    while i < len(args):
        if args[i] == "--ymax":
            ymax = float(args[i + 1])
            i += 2
        elif args[i] == "--full":
            full = True
            i += 1
        elif args[i] == "--auto":
            auto = True
            i += 1
        else:
            print(f"Unknown arg: {args[i]}")
            sys.exit(1)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Backwards compat: old pickle was just a list of rows; new one is a dict
    if isinstance(data, dict):
        rows = data["rows"]
        dns_profile = data["dns_chi_profile"]
        y_over_H = data["y_over_H"]
    else:
        print(f"ERROR: pickle is in old format (list of rows only); "
              f"need to re-run sweep_dns_chi_recon.py to capture profiles.")
        sys.exit(1)

    # If y_over_H was saved as an unevaluated bound method (bug in v1
    # of sweep_dns_chi_recon.py), reconstruct it from the profile length.
    if callable(y_over_H):
        try:
            y_over_H = y_over_H()
            assert hasattr(y_over_H, "__len__")
        except Exception:
            # Reconstruct using the physical grid spacing.
            # From s3d.in: Ly = 0.019133 m over 1008 cells, H = 1.368e-3 m
            # -> dy/H = (0.019133 / 1008) / 1.368e-3 = 0.01388
            n = len(dns_profile)
            dy_over_H = (0.019133 / 1008) / 1.368e-3
            y_over_H = (np.arange(n) - (n - 1) / 2.0) * dy_over_H
            print(f"[warn] y_over_H reconstructed from cube length n={n}, "
                  f"dy/H={dy_over_H:.6f} -> y/H range "
                  f"[{y_over_H[0]:.3f}, {y_over_H[-1]:.3f}]")

    out_dir = os.path.dirname(pkl_path) or "."
    ts = data.get("timestep", "")
    suffix = "_full" if full else ("_auto" if auto else "")
    out_path = os.path.join(
        out_dir, f"fig09_chi_profile_sweep_{ts}{suffix}.png")
    ylim = None if auto else (0.0, ymax)
    plot_overlay(rows, dns_profile, y_over_H, out_path,
                 single_sided=not (full or auto), ylim=ylim)


if __name__ == "__main__":
    main()
