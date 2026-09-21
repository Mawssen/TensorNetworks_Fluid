"""
plot_field_pdf.py
-----------------
For a single field, produce two flavors of PDF overlay plots:

  Flavor A: per-sp -> all 4 cutoffs on one plot, plus DNS reference
            (4 plots per field, one per sp)
  Flavor B: per-cf -> all 4 sps on one plot, plus DNS reference
            (4 plots per field, one per cf)

Total: 8 plots per field (24 across {mixfrac,T,chi} if scripted that way).

For chi in --mode mixed: chi is reconstructed via 2*alpha*|grad Z|^2 from
truncated mixfrac, applying the same operator on the DNS side too.

Usage
-----
python plot_field_pdf.py
    --field {mixfrac,T,chi}
    --timestep 0198
    --dns-data-dir /path/to/jet_0198
    --mps-root /path/to/dir/with/truncated_<TS>_sp<N>/  (parent dir)
    --out-dir /path/to/plots/pdf
    [--mode {stored,mixed}]            default mixed
    [--fd-order N]                     default 8 (chi only)
    [--periodic-xz {true,false}]       default false
    [--n-bins N]                       default 200
    [--no-log-y]                       if set, linear y axis (default: log)
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dns_stats import GridConfig, load_field
from derived_fields import compute_chi
from mps_io import load_mps_field, ordering_for_sp, _mps_filename


# Color schemes
_COLOR_BY_CF = {
    "1e-2": "#c0392b",
    "1e-3": "#e67e22",
    "1e-4": "#27ae60",
    "1e-5": "#2980b9",
}
_COLOR_BY_SP = {
    1: "#e67e22",
    2: "#27ae60",
    3: "#2980b9",
    4: "#8e44ad",
}
_ORD_NAME = {1: "il", 2: "seq", 3: "comb1", 4: "combn"}

_FIELD_LABELS = {
    "mixfrac": (r"$Z$",        (0.0, 1.0)),
    "T":       (r"$T$ [K]",    (300.0, 1700.0)),
    "chi":     (r"$\chi$ [1/s]", None),
}


def _parse(args, name, default=None, cast=str, required=False):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        args.pop(i + 1); args.pop(i)
        return cast(v)
    if required:
        print(f"Error: {name} is required")
        sys.exit(1)
    return default


def _has_flag(args, name):
    if name in args:
        args.remove(name)
        return True
    return False


def _cf_to_str(cf_label):
    table = {"1e-2": "0.01", "1e-3": "0.001",
             "1e-4": "0.0001", "1e-5": "1.0e-5"}
    return table[cf_label]


def _mps_dir(mps_root, timestep, sp):
    """Find the MPS data directory for a given sp.

    Layout: truncated_<TS>/sp<SP>/  (sp folder INSIDE timestep folder).
    Legacy fallbacks: truncated_<TS>_sp<SP>/ or truncated_<TS>/ (sp=1).
    """
    cand = os.path.join(mps_root, f"truncated_{timestep}", f"sp{sp}")
    if os.path.isdir(cand):
        return cand
    legacy_flat = os.path.join(mps_root, f"truncated_{timestep}_sp{sp}")
    if os.path.isdir(legacy_flat):
        return legacy_flat
    legacy_nosp = os.path.join(mps_root, f"truncated_{timestep}")
    if sp == 1 and os.path.isdir(legacy_nosp):
        return legacy_nosp
    return None


def _load_dns(dns_data_dir, timestep, var, grid):
    return load_field(os.path.join(dns_data_dir,
                                   f"jet_{var}_{timestep}.dat"), grid)


def _load_mps(mps_dir, timestep, var, cf_str, ordering):
    return load_mps_field(os.path.join(
        mps_dir, _mps_filename(var, timestep, cf_str, ordering=ordering)))


def _build_chi(Z, alpha, rho, grid, fd_order, periodic):
    return compute_chi(Z, alpha, rho,
                       grid.dx_m, grid.dy_m, grid.dz_m,
                       alpha_is="thermal_diffusivity",
                       order=fd_order, periodic_xz=periodic)


def _hist_pdf(arr, x_min, x_max, n_bins):
    """Compute normalized PDF (histogram) over arr."""
    flat = arr.ravel()
    hist, edges = np.histogram(flat, bins=n_bins,
                               range=(x_min, x_max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def main():
    args = sys.argv[1:]
    field = _parse(args, "--field", required=True)
    timestep = _parse(args, "--timestep", default="0198")
    dns_data_dir = _parse(args, "--dns-data-dir", required=True)
    mps_root = _parse(args, "--mps-root", required=True)
    out_dir = _parse(args, "--out-dir", required=True)
    mode = _parse(args, "--mode", default="mixed")
    fd_order = _parse(args, "--fd-order", default=4, cast=int)
    periodic_str = _parse(args, "--periodic-xz", default="false")
    periodic = periodic_str.lower() in ("true", "1", "yes", "on")
    n_bins = _parse(args, "--n-bins", default=200, cast=int)
    no_log_y = _has_flag(args, "--no-log-y")
    log_y = not no_log_y
    les_plt_path = _parse(args, "--les-plt", default=None)

    if field not in ("mixfrac", "T", "chi"):
        print(f"Invalid --field {field}")
        sys.exit(1)
    if mode not in ("stored", "mixed"):
        print(f"Invalid --mode {mode}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    grid = GridConfig()
    field_label, fixed_range = _FIELD_LABELS[field]

    SP_VALUES = [1, 2, 3, 4]
    CF_LABELS = ["1e-2", "1e-3", "1e-4", "1e-5"]

    t0 = time.time()
    print(f"[pdf] field={field}  mode={mode}  fd_order={fd_order}  "
          f"periodic={periodic}", flush=True)

    # ----- Load DNS field once -----
    if field == "chi":
        Z_dns = _load_dns(dns_data_dir, timestep, "mixfrac", grid)
        alpha_dns = _load_dns(dns_data_dir, timestep, "alpha", grid)
        rho_dns = _load_dns(dns_data_dir, timestep, "rho", grid)
        dns_arr = _build_chi(Z_dns, alpha_dns, rho_dns, grid,
                             fd_order, periodic)
        del Z_dns, alpha_dns, rho_dns
    else:
        dns_arr = _load_dns(dns_data_dir, timestep, field, grid)
    print(f"[pdf] DNS loaded in {time.time()-t0:.1f}s; "
          f"shape={dns_arr.shape}, range=[{dns_arr.min():.3g}, "
          f"{dns_arr.max():.3g}]", flush=True)

    # Determine x range
    if fixed_range is not None:
        x_min, x_max = fixed_range
    else:
        x_min = float(dns_arr.min())
        x_max = float(np.percentile(dns_arr, 99.95))   # clip extreme tail

    # PDF for DNS
    dns_x, dns_y = _hist_pdf(dns_arr, x_min, x_max, n_bins)
    del dns_arr

    # ----- Load LES field (if available) and compute PDF -----
    les_pdf = None
    if les_plt_path is None:
        candidate = os.path.join(dns_data_dir, "LES_plt20000")
        if os.path.isdir(candidate):
            les_plt_path = candidate
    if les_plt_path is not None and os.path.isdir(les_plt_path):
        try:
            from les_io import (load_les_dataset, load_les_field, LESGrid,
                                  load_fdf_chi_from_tau_files)
            les_grid = LESGrid()
            print(f"[pdf] LES plt: {les_plt_path}", flush=True)
            if field == "chi":
                digits = "".join(c for c in os.path.basename(les_plt_path)
                                 if c.isdigit())
                tau = os.path.join(os.path.dirname(les_plt_path),
                                    f"Tauplt{digits}.temp")
                tauall = os.path.join(os.path.dirname(les_plt_path),
                                       f"TauAllplt{digits}.temp")
                if os.path.exists(tau) and os.path.exists(tauall):
                    les_arr = load_fdf_chi_from_tau_files(tau, tauall, les_grid)
                    print(f"[pdf] LES chi via FDF: peak={les_arr.max():.3e}",
                          flush=True)
                else:
                    ds = load_les_dataset(les_plt_path)
                    les_arr = load_les_field(ds, "chi", les_grid)
            else:
                ds = load_les_dataset(les_plt_path)
                les_arr = load_les_field(ds, field, les_grid)
            les_pdf = _hist_pdf(les_arr, x_min, x_max, n_bins)
            del les_arr
        except Exception as e:
            print(f"[pdf] WARN: LES PDF failed: {e}", flush=True)
            les_pdf = None
    else:
        print(f"[pdf] No LES plt found; skipping LES overlay", flush=True)

    # ----- Load MPS arrays for all (sp, cf) -----
    # mps_pdfs[sp][cf] = (centers, pdf)
    mps_pdfs: Dict[int, Dict[str, tuple]] = {}
    for sp in SP_VALUES:
        mps_dir = _mps_dir(mps_root, timestep, sp)
        if mps_dir is None:
            print(f"[pdf] WARN: no MPS dir for sp={sp}; skipping", flush=True)
            continue
        ordering = ordering_for_sp(sp)
        mps_pdfs[sp] = {}
        for cf_label in CF_LABELS:
            cf_str = _cf_to_str(cf_label)
            tA = time.time()
            try:
                if field == "chi":
                    Z_mps = _load_mps(mps_dir, timestep, "mixfrac", cf_str, ordering)
                    alpha_mps = _load_mps(mps_dir, timestep, "alpha", cf_str, ordering)
                    rho_mps = _load_mps(mps_dir, timestep, "rho", cf_str, ordering)
                    mps_arr = _build_chi(Z_mps, alpha_mps, rho_mps, grid,
                                         fd_order, periodic)
                    del Z_mps, alpha_mps, rho_mps
                else:
                    mps_arr = _load_mps(mps_dir, timestep, field,
                                        cf_str, ordering)
            except FileNotFoundError as e:
                print(f"[pdf] WARN: missing files for sp={sp} cf={cf_label}: "
                      f"{e}", flush=True)
                continue
            x, y = _hist_pdf(mps_arr, x_min, x_max, n_bins)
            mps_pdfs[sp][cf_label] = (x, y)
            print(f"[pdf] sp={sp} cf={cf_label}: {time.time()-tA:.1f}s",
                  flush=True)
            del mps_arr

    # ----- Produce flavor A: per-sp plots (overlay all cf for each sp) -----
    for sp in SP_VALUES:
        if sp not in mps_pdfs or not mps_pdfs[sp]:
            continue
        ordering = ordering_for_sp(sp)
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        ax.plot(dns_x, dns_y, color="black", lw=2.0,
                label="DNS", zorder=10)
        if les_pdf is not None:
            ax.plot(les_pdf[0], les_pdf[1],
                    color="#c0392b", lw=2.0, linestyle="-",
                    marker="s", markevery=max(1, len(les_pdf[0]) // 25),
                    markersize=6, markerfacecolor="none",
                    markeredgewidth=1.5, label="LES-FDF", zorder=9)
        for cf_label in CF_LABELS:
            if cf_label not in mps_pdfs[sp]:
                continue
            x, y = mps_pdfs[sp][cf_label]
            ax.plot(x, y, color=_COLOR_BY_CF[cf_label],
                    linestyle="--", lw=1.5,
                    label=f"MPS cf={cf_label}")
        ax.set_xlabel(field_label)
        ax.set_ylabel(f"PDF({field_label})")
        ax.set_title(f"PDF of {field}  --  sp={sp} ({ordering})")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        suffix = ("" if field != "chi" else f"_o{fd_order}")
        suffix += ("_p" if periodic else "")
        out_path = os.path.join(out_dir,
                                f"pdf_{field}_sp{sp}{suffix}.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[plot] {out_path}")

    # ----- Produce flavor B: per-cf plots (overlay all sp for each cf) -----
    for cf_label in CF_LABELS:
        sps_with_data = [sp for sp in SP_VALUES
                         if sp in mps_pdfs and cf_label in mps_pdfs[sp]]
        if not sps_with_data:
            continue
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        ax.plot(dns_x, dns_y, color="black", lw=2.0,
                label="DNS", zorder=10)
        if les_pdf is not None:
            ax.plot(les_pdf[0], les_pdf[1],
                    color="#c0392b", lw=2.0, linestyle="-",
                    marker="s", markevery=max(1, len(les_pdf[0]) // 25),
                    markersize=6, markerfacecolor="none",
                    markeredgewidth=1.5, label="LES-FDF", zorder=9)
        for sp in sps_with_data:
            x, y = mps_pdfs[sp][cf_label]
            ord_name = _ORD_NAME[sp]
            ax.plot(x, y, color=_COLOR_BY_SP[sp],
                    linestyle="--", lw=1.5,
                    label=f"MPS sp={sp} ({ord_name})")
        ax.set_xlabel(field_label)
        ax.set_ylabel(f"PDF({field_label})")
        ax.set_title(f"PDF of {field}  --  cf={cf_label}")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        suffix = ("" if field != "chi" else f"_o{fd_order}")
        suffix += ("_p" if periodic else "")
        out_path = os.path.join(out_dir,
                                f"pdf_{field}_cf{cf_label}{suffix}.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[plot] {out_path}")

    print(f"[pdf] total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
