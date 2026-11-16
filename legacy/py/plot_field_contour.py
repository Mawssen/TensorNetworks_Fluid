"""
plot_field_contour.py
---------------------
For a single field, render mid-plane contour slices at z = nz/2 and y = ny/2
for the DNS reference and for every (sp, cutoff) MPS truncation case.

Produces 1 + 16 = 17 sources x 2 planes = 34 plots per field.

For chi in --mode mixed: chi is reconstructed via 2*alpha*|grad Z|^2 from
truncated mixfrac.

Color limits: derived from DNS field min/max so all MPS plots use the
same scale and are directly visually comparable.

Usage
-----
python plot_field_contour.py
    --field {mixfrac,T,chi}
    --timestep 0198
    --dns-data-dir /path/to/jet_0198
    --mps-root /path/to/dir/with/truncated_<TS>_sp<N>/
    --out-dir /path/to/plots/contour
    [--mode {stored,mixed}]            default mixed
    [--fd-order N]                     default 8
    [--periodic-xz {true,false}]       default false
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


_FIELD_INFO = {
    "mixfrac": dict(label=r"$Z$",        cmap="inferno"),
    "T":       dict(label=r"$T$ [K]",    cmap="inferno"),
    "chi":     dict(label=r"$\chi$ [1/s]", cmap="inferno"),
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


def _cf_to_str(cf_label):
    return {"1e-2": "0.01", "1e-3": "0.001",
            "1e-4": "0.0001", "1e-5": "1.0e-5"}[cf_label]


def _mps_dir(mps_root, timestep, sp):
    """Layout: truncated_<TS>/sp<SP>/ . Legacy fallbacks too."""
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


def _build_chi(Z, alpha, rho, grid, fd_order, periodic):
    return compute_chi(Z, alpha, rho,
                       grid.dx_m, grid.dy_m, grid.dz_m,
                       alpha_is="thermal_diffusivity",
                       order=fd_order, periodic_xz=periodic)


def _slice_mid(arr3d, axis):
    """Return a 2D slice through the volumetric mid of a (nx, ny, nz) array.
    axis 'z' -> slice at k = nz//2 -> result shape (nx, ny).
    axis 'y' -> slice at j = ny//2 -> result shape (nx, nz).
    """
    nx, ny, nz = arr3d.shape
    if axis == "z":
        return arr3d[:, :, nz // 2].copy()
    elif axis == "y":
        return arr3d[:, ny // 2, :].copy()
    raise ValueError(f"axis must be 'z' or 'y', got {axis}")


def _plot_slice(slab, axis, vmin, vmax, cmap, label, source, field, out_path):
    """Render a 2D imshow with consistent colorbar across plots."""
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
    # Use origin='lower' so the y-axis points up. arr orientation:
    # axis='z': slab is (nx, ny) -> imshow expects (rows, cols) =
    #          (ny rows, nx cols), so transpose
    if axis == "z":
        # plot as (ny rows, nx cols) -> y vertical, x horizontal
        im = ax.imshow(slab.T, origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
    else:
        # axis='y': slab is (nx, nz) -> rows=nz, cols=nx
        im = ax.imshow(slab.T, origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    ax.set_title(f"{field}  --  {source}  --  {axis}-mid plane")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")


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
    # Optional LES plt path. If provided (or auto-detected as <dns-data-dir>/LES_plt*),
    # a separate LES contour is produced per plane using the same color limits.
    les_plt_path = _parse(args, "--les-plt", default=None)

    if field not in ("mixfrac", "T", "chi"):
        print(f"Invalid --field {field}")
        sys.exit(1)
    if mode not in ("stored", "mixed"):
        print(f"Invalid --mode {mode}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    grid = GridConfig()
    info = _FIELD_INFO[field]

    SP_VALUES = [1, 2, 3, 4]
    CF_LABELS = ["1e-2", "1e-3", "1e-4", "1e-5"]
    PLANES = ("z", "y")

    t0 = time.time()
    print(f"[contour] field={field}  mode={mode}  fd_order={fd_order}  "
          f"periodic={periodic}", flush=True)

    # --- Load DNS reference ---
    if field == "chi":
        Z_dns = load_field(os.path.join(dns_data_dir,
                                        f"jet_mixfrac_{timestep}.dat"), grid)
        alpha_dns = load_field(os.path.join(dns_data_dir,
                                            f"jet_alpha_{timestep}.dat"), grid)
        rho_dns = load_field(os.path.join(dns_data_dir,
                                          f"jet_rho_{timestep}.dat"), grid)
        dns_arr = _build_chi(Z_dns, alpha_dns, rho_dns, grid,
                             fd_order, periodic)
        del Z_dns, alpha_dns, rho_dns
    else:
        dns_arr = load_field(os.path.join(dns_data_dir,
                                          f"jet_{field}_{timestep}.dat"), grid)
    print(f"[contour] DNS loaded in {time.time()-t0:.1f}s", flush=True)

    # Color limits from DNS
    if field == "chi":
        # chi is heavy-tailed; clip at 99.5th percentile so MPS spikes
        # don't dominate the colorbar
        vmin = 0.0
        vmax = float(np.percentile(dns_arr, 99.5))
    else:
        vmin = float(dns_arr.min())
        vmax = float(dns_arr.max())
    print(f"[contour] color range: [{vmin:.4g}, {vmax:.4g}]", flush=True)

    # --- DNS plots (1 per plane) ---
    suffix_chi = (f"_o{fd_order}" if field == "chi" else "")
    suffix_chi += ("_p" if periodic else "")

    for axis in PLANES:
        slab = _slice_mid(dns_arr, axis)
        out_path = os.path.join(
            out_dir, f"contour_{field}_{axis}_DNS{suffix_chi}.png")
        _plot_slice(slab, axis, vmin, vmax, info["cmap"],
                    info["label"], "DNS", field, out_path)

    del dns_arr

    # --- LES plots (auto-discover plt if not given) ---
    if les_plt_path is None:
        candidate = os.path.join(dns_data_dir, "LES_plt20000")
        if os.path.isdir(candidate):
            les_plt_path = candidate
    if les_plt_path is not None and os.path.isdir(les_plt_path):
        try:
            from les_io import (load_les_dataset, load_les_field, LESGrid,
                                  load_fdf_chi_from_tau_files)
            print(f"[contour] LES plt: {les_plt_path}", flush=True)
            les_grid = LESGrid()
            if field == "chi":
                # Prefer FDF chi from Tauplt/TauAllplt (Aitzhan's convention).
                digits = "".join(c for c in os.path.basename(les_plt_path)
                                 if c.isdigit())
                tau = os.path.join(os.path.dirname(les_plt_path),
                                    f"Tauplt{digits}.temp")
                tauall = os.path.join(os.path.dirname(les_plt_path),
                                       f"TauAllplt{digits}.temp")
                if os.path.exists(tau) and os.path.exists(tauall):
                    les_arr = load_fdf_chi_from_tau_files(tau, tauall, les_grid)
                    print(f"[contour] LES chi via FDF: peak={les_arr.max():.3e}",
                          flush=True)
                else:
                    ds = load_les_dataset(les_plt_path)
                    les_arr = load_les_field(ds, "chi", les_grid)
                    print(f"[contour] LES chi via Scalar_diss: "
                          f"peak={les_arr.max():.3e}", flush=True)
            else:
                ds = load_les_dataset(les_plt_path)
                les_arr = load_les_field(ds, field, les_grid)

            for axis in PLANES:
                slab = _slice_mid(les_arr, axis)
                out_path = os.path.join(
                    out_dir, f"contour_{field}_{axis}_LES{suffix_chi}.png")
                _plot_slice(slab, axis, vmin, vmax, info["cmap"],
                            info["label"], "LES-FDF", field, out_path)
            del les_arr
        except Exception as e:
            print(f"[contour] WARN: LES contour failed: {e}", flush=True)
    else:
        print(f"[contour] No LES plt found; skipping LES contour", flush=True)

    # --- MPS plots, looping over (sp, cf) ---
    for sp in SP_VALUES:
        mps_dir = _mps_dir(mps_root, timestep, sp)
        if mps_dir is None:
            print(f"[contour] WARN: no MPS dir for sp={sp}", flush=True)
            continue
        ordering = ordering_for_sp(sp)
        for cf_label in CF_LABELS:
            cf_str = _cf_to_str(cf_label)
            tA = time.time()
            try:
                if field == "chi":
                    Z = load_mps_field(os.path.join(
                        mps_dir, _mps_filename("mixfrac", timestep,
                                                cf_str, ordering=ordering)))
                    a = load_mps_field(os.path.join(
                        mps_dir, _mps_filename("alpha", timestep,
                                                cf_str, ordering=ordering)))
                    r = load_mps_field(os.path.join(
                        mps_dir, _mps_filename("rho", timestep,
                                                cf_str, ordering=ordering)))
                    mps_arr = _build_chi(Z, a, r, grid, fd_order, periodic)
                    del Z, a, r
                else:
                    mps_arr = load_mps_field(os.path.join(
                        mps_dir, _mps_filename(field, timestep,
                                                cf_str, ordering=ordering)))
            except FileNotFoundError as e:
                print(f"[contour] WARN: missing files for sp={sp} "
                      f"cf={cf_label}: {e}", flush=True)
                continue

            for axis in PLANES:
                slab = _slice_mid(mps_arr, axis)
                src_label = f"MPS sp={sp} ({ordering})  cf={cf_label}"
                out_path = os.path.join(
                    out_dir,
                    f"contour_{field}_{axis}_sp{sp}_cf{cf_label}{suffix_chi}.png")
                _plot_slice(slab, axis, vmin, vmax, info["cmap"],
                            info["label"], src_label, field, out_path)
            del mps_arr
            print(f"[contour] sp={sp} cf={cf_label}: {time.time()-tA:.1f}s",
                  flush=True)

    print(f"[contour] total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
