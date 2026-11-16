"""
plot_qc4pde.py
--------------
Generate paper-ready figures for the QC4PDE conference paper.

Produces the following figures in the output directory (default: QC4PDE/).
The three panel figures (contour, joint PDF, manifold) are saved as both
PNG (for review) and PDF (for LaTeX inclusion). The overlaid line figures
(mean/RMS, PDF, conditional T) are saved as PNG plus TEX.

Figures
-------
1. contour_mixfrac_{row,grid}.{png,pdf}
     x-y mid-Z slices of Z, one per source.
     Order: DNS -- LES -- MPS chi=93 -- MPS 4.1e-4 -- PEPS
2. mixfrac_mean_rms.{png,tex}
     Overlaid mean + RMS profile of Z vs y/H.
3. mixfrac_pdf.{png,tex}
     Overlaid PDF of Z.
4. conditional_T.{png,tex}
     Overlaid <T | Z>.
5. joint_pdf_{row,grid}.{png,pdf}
     (Z, Y_CO2) joint PDF contours, one per source.
6. manifold_{row,grid}.{png,pdf}
     3D compositional manifold scatters, one per source.

Usage
-----
    python plot_qc4pde.py [results_root=results] [out_dir=QC4PDE]
        [--timestep 0198]
        [--dns-data-dir /path/to/jet_0198]
        [--fd-order 4]
        [--periodic-xz false]
        [--sp 1]
        [--layout row|grid]      (default: row)
"""

from __future__ import annotations

import os
import sys
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from save_tikz import save_tikz_if_available


# ---------------------------------------------------------------------------
# File-format helper: save a figure as PNG plus any extra vector formats.
# ---------------------------------------------------------------------------

def _save_fig(fig, png_path: str, also: Tuple[str, ...] = ("pdf",),
              dpi: int = 200):
    """Save `fig` to `png_path` and to sibling files for each extra ext.

    e.g. also=('pdf',) writes both foo.png and foo.pdf. Returns the list of
    paths written (png first).
    """
    written = [png_path]
    fig.savefig(png_path, dpi=dpi)
    stem = os.path.splitext(png_path)[0]
    for ext in also:
        p = f"{stem}.{ext.lstrip('.')}"
        # PDF is vector; dpi only affects any rasterized elements.
        fig.savefig(p, dpi=dpi)
        written.append(p)
    return written


# ---------------------------------------------------------------------------
# LaTeX-style rendering with larger fonts (contour / joint-PDF / manifold).
# Falls back to mathtext if a system LaTeX install is unavailable.
# ---------------------------------------------------------------------------

def _latex_actually_works() -> bool:
    """Try a real usetex render; return True only if it succeeds.

    checkdep_usetex / a bare `which latex` can report success even when the
    install is missing style files (e.g. type1ec.sty), which then fails at
    draw time. This does a tiny real render to be sure.
    """
    import io
    saved = dict(plt.rcParams)
    try:
        plt.rcParams["text.usetex"] = True
        fig = plt.figure()
        fig.text(0.5, 0.5, r"$Y_{O_2}$")
        fig.savefig(io.BytesIO(), format="png", dpi=50)
        plt.close(fig)
        return True
    except Exception:
        return False
    finally:
        plt.rcParams.update(saved)


def _enable_latex_style():
    has_tex = _latex_actually_works()
    plt.rcParams.update({
        "text.usetex": bool(has_tex),
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "axes.formatter.use_mathtext": True,
    })
    if has_tex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        print("[qc4pde] LaTeX not usable; using mathtext (CM) fallback.")
    return bool(has_tex)


_HAS_TEX = _enable_latex_style()


def _set_n_ticks(ax, which: str = "both", nx: int = 5, ny: int = 4):
    """Force approximately a fixed number of *labeled* major ticks per axis.

    Uses MaxNLocator so the tick count is stable regardless of data range,
    unlike hiding every other auto-tick.
    """
    from matplotlib.ticker import MaxNLocator
    if which in ("x", "both"):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nx - 1, prune=None))
    if which in ("y", "both"):
        ax.yaxis.set_major_locator(MaxNLocator(nbins=ny - 1, prune=None))


def _skip_labels(ax, which: str = "both", step: int = 2):
    """Blank every `step`-th tick label (keeps the tick marks).

    Used for the 3D manifold panels where fonts still crowd otherwise.
    """
    if which in ("x", "both"):
        for i, lbl in enumerate(ax.get_xticklabels()):
            if i % step != 0:
                lbl.set_visible(False)
    if which in ("y", "both"):
        for i, lbl in enumerate(ax.get_yticklabels()):
            if i % step != 0:
                lbl.set_visible(False)


def _grid_shape(n: int, layout: str) -> Tuple[int, int]:
    """Return (nrows, ncols) for `n` panels under the given layout.

    layout='row'  -> single row (1 x n), the default behaviour.
    layout='grid' -> as square as possible, filling rows first. For the
                     usual n=4 (DNS, LES, MPS, PEPS) this is a 2x2 block;
                     n=5 gives 2x3 with one empty cell, n=3 gives 2x2 with
                     one empty, n<=2 stays a single row.
    """
    if layout == "row" or n <= 2:
        return 1, n
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _panel_pos(i: int, nrows: int, ncols: int) -> Tuple[int, int, bool, bool]:
    """Given a flat panel index, return (row, col, is_left_col, is_bottom_row).

    'is_bottom_row' accounts for partially filled grids: a panel is on the
    bottom if there is no panel directly beneath it.
    """
    r, c = divmod(i, ncols)
    is_left = (c == 0)
    is_bottom = (r == nrows - 1)
    return r, c, is_left, is_bottom


# ---------------------------------------------------------------------------
# Source ordering & style (final QC4PDE convention).
# Row order: DNS, LES, MPS chi=93, MPS 4.1e-4, PEPS.
# ---------------------------------------------------------------------------

_SOURCE_ORDER = ["DNS", "Filtered", "LES", "chi93", "4.1e-4", "PEPS"]

_STYLE = {
    "DNS":    dict(color="black",   linestyle="-",  lw=2.0, label="DNS"),
    "Filtered": dict(color="#7f8c8d", linestyle="--", lw=2.0,
                     label="Filtered DNS"),
    "LES":    dict(color="#c0392b", linestyle="-",  lw=2.0, label="LES"),
    "chi93":  dict(color="#16a085", linestyle="--", lw=2.0,
                    label="MPS $\\chi{=}93$"),
    "4.1e-4": dict(color="#9b59b6", linestyle="--", lw=2.0,
                    label="MPS $\\varepsilon{=}4.1{\\times}10^{-4}$"),
    "PEPS":   dict(color="#d35400", linestyle=":",  lw=2.4,
                    label="PEPS $D{=}9$"),
}

_TITLE = {
    "DNS":    "DNS",
    "Filtered": "Filtered DNS",
    "LES":    "LES",
    "chi93":  r"MPS $\chi=93$",
    "4.1e-4": r"MPS $\varepsilon=4.1{\times}10^{-4}$",
    "PEPS":   r"PEPS $D=9$",
}


# ---------------------------------------------------------------------------
# Argument parsing (unified with the rest of the codebase style)
# ---------------------------------------------------------------------------

def _parse(args, name, default=None, cast=str):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        args.pop(i + 1); args.pop(i)
        return cast(v) if cast is not str else v
    return default


# ---------------------------------------------------------------------------
# Pickle discovery
# ---------------------------------------------------------------------------

def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def _discover_pickles(results_root: str, timestep: str, fd_order: int,
                       sp: int, filtered_dir: str = None) -> Dict[str, str]:
    """Locate DNS/LES/PEPS/chi93/4.1e-4/Filtered pickles.

    filtered_dir, if given, is searched first for the Filtered pickle (its
    location often differs from results_root, e.g. under stats/).
    """
    paths: Dict[str, str] = {}

    # DNS
    dns_p = os.path.join(results_root, f"dns_{timestep}", "cube",
                         f"dns_stats_{timestep}.pkl")
    if os.path.exists(dns_p):
        paths["DNS"] = dns_p

    # LES
    les_p = os.path.join(results_root, f"les_{timestep}",
                         f"les_stats_{timestep}.pkl")
    if os.path.exists(les_p):
        paths["LES"] = les_p

    # Filtered (box-filtered DNS). Its pickle often lives outside
    # results_root (e.g. under stats/filtered_<ts>/). Search, in order:
    # an explicit --filtered-dir, then common spots relative to results_root
    # and its parent.
    _filt_name = f"filtered_stats_{timestep}.pkl"
    _filt_cands = []
    if filtered_dir:
        _filt_cands += [os.path.join(filtered_dir, _filt_name),
                        filtered_dir if filtered_dir.endswith(".pkl") else ""]
    _parent = os.path.dirname(os.path.abspath(results_root))
    _filt_cands += [
        os.path.join(results_root, f"filtered_{timestep}", _filt_name),
        os.path.join(results_root, _filt_name),
        os.path.join(_parent, f"filtered_{timestep}", _filt_name),
        os.path.join(".", f"filtered_{timestep}", _filt_name),
    ]
    for cand in _filt_cands:
        if cand and os.path.exists(cand):
            paths["Filtered"] = cand
            break

    # PEPS -- newest match
    import glob as _glob
    peps_cands = sorted(_glob.glob(
        os.path.join(results_root, f"peps_{timestep}",
                      f"peps_stats_{timestep}*.pkl")))
    if peps_cands:
        # Prefer one that mentions our fd_order
        pref = [c for c in peps_cands
                if f"_o{fd_order}" in os.path.basename(c)]
        paths["PEPS"] = pref[-1] if pref else peps_cands[-1]

    # MPS chi93 and 4.1e-4
    for label in ("chi93", "4.1e-4"):
        cand = os.path.join(
            results_root, f"mps_{timestep}_sp{sp}", label,
            f"mps_stats_{timestep}_cf{label}_mixed_o{fd_order}_sp{sp}.pkl")
        if os.path.exists(cand):
            paths[label] = cand
        else:
            # Try periodic variant
            cand_p = cand.replace(".pkl", "_p.pkl")
            if os.path.exists(cand_p):
                paths[label] = cand_p

    return paths


# ---------------------------------------------------------------------------
# Fold-symmetric helper (single-sided profiles at y/H >= 0)
# ---------------------------------------------------------------------------

def _fold_profile(y_over_H, prof, kind: str = "mean"):
    n = len(y_over_H)
    j0 = int(np.argmin(np.abs(y_over_H)))
    n_keep = min(n - j0, j0 + 1)
    p_pos = np.asarray(prof[j0:j0 + n_keep])
    p_neg = np.asarray(prof[j0 - np.arange(n_keep)])
    y_out = y_over_H[j0:j0 + n_keep]
    if kind == "rms":
        return y_out, np.sqrt(0.5 * (p_pos ** 2 + p_neg ** 2))
    return y_out, 0.5 * (p_pos + p_neg)


# ---------------------------------------------------------------------------
# Figure 1: contour row (x-y mid-Z slices of Z)
# ---------------------------------------------------------------------------

def _load_dns_slice_Z(dns_data_dir: str, timestep: str) -> np.ndarray:
    """Load a mid-z slice of DNS mixture fraction (centred 512^3 cube)."""
    from dns_stats import GridConfig, load_field
    grid = GridConfig()
    Z = load_field(os.path.join(dns_data_dir, f"jet_mixfrac_{timestep}.dat"),
                    grid)
    n = Z.shape[2]
    return Z[:, :, n // 2].astype(np.float32)   # (nx, ny) at mid-z


def _block_average(cube: np.ndarray, factor: int = 8) -> np.ndarray:
    """Box-filter a cube by averaging each factor^3 block (512^3 -> 64^3)."""
    nx, ny, nz = cube.shape
    cx, cy, cz = (nx // factor) * factor, (ny // factor) * factor, \
                 (nz // factor) * factor
    c = cube[:cx, :cy, :cz]
    c = c.reshape(cx // factor, factor,
                  cy // factor, factor,
                  cz // factor, factor)
    return c.mean(axis=(1, 3, 5))


def _load_filtered_slice_Z(dns_data_dir: str, timestep: str,
                           factor: int = 8) -> np.ndarray:
    """Mid-z slice of the box-filtered DNS (block-averaged 8^3 -> 64^3).

    This is the 'ideal LES' reference: a perfect low-pass of the DNS with
    no SGS model. Filtering the full cube then slicing keeps it consistent
    with the 64^3 filtered field used for the other Filtered statistics.
    """
    from dns_stats import GridConfig, load_field
    grid = GridConfig()
    Z = load_field(os.path.join(dns_data_dir, f"jet_mixfrac_{timestep}.dat"),
                    grid)
    Zf = _block_average(np.asarray(Z, dtype=np.float64), factor)
    n = Zf.shape[2]
    return Zf[:, :, n // 2].astype(np.float32)


def _load_les_slice_Z(les_plt_path: str) -> np.ndarray:
    """Load a mid-z slice of LES mixture fraction on the centred 64^3 cube."""
    from les_io import load_les_dataset, load_les_field, LESGrid
    ds = load_les_dataset(les_plt_path)
    grid = LESGrid()
    Z = load_les_field(ds, "mixfrac", grid)
    n = Z.shape[2]
    return Z[:, :, n // 2].astype(np.float32)


def _load_mps_slice_Z(mps_data_dir: str, timestep: str,
                       cutoff_str: str, ordering: str = "il") -> np.ndarray:
    """Load a mid-z slice of MPS mixture fraction."""
    from mps_io import load_mps_field, _mps_filename
    path = os.path.join(mps_data_dir,
                         _mps_filename("mixfrac", timestep, cutoff_str,
                                        ordering=ordering))
    Z = load_mps_field(path)
    n = Z.shape[2]
    return Z[:, :, n // 2].astype(np.float32)


def _load_peps_slice_Z(peps_dir: str,
                        tag: str = "wf_D=9_periodic") -> np.ndarray:
    """Load a mid-z slice of PEPS mixture fraction.

    load_peps_field applies transpose(2, 1, 0) internally so PEPS axes
    match DNS (x, y, z). After that, the raw mid-z slice already
    visually aligns with DNS -- no additional 2D transformation needed.
    """
    from peps_io import load_peps_field
    path = os.path.join(peps_dir, f"mixfrac_{tag}.mat")
    Z = load_peps_field(path)   # (512, 512, 512), (x, y, z) order
    n = Z.shape[2]
    return Z[:, :, n // 2].astype(np.float32)


def _peps_scale_and_sign(peps_slice: np.ndarray,
                          dns_slice: np.ndarray) -> np.ndarray:
    """Rescale a PEPS slice to match DNS L2 norm and fix sign via
    correlation. Small helper for the contour row where we work with 2D
    slices rather than the full 512^3 cube; approximate but close enough
    for visual comparison.
    """
    peps = peps_slice.astype(np.float64)
    dns = dns_slice.astype(np.float64)
    if np.sum(peps * dns) < 0:
        peps = -peps
    n_dns = np.sqrt(np.sum(dns ** 2))
    n_peps = np.sqrt(np.sum(peps ** 2))
    if n_peps > 0:
        peps = peps * (n_dns / n_peps)
    return np.clip(peps, 0.0, 1.0).astype(np.float32)


def plot_contour_row(sources_present: List[str], slices: Dict[str, np.ndarray],
                     out_dir: str, dx_dy: Tuple[float, float] = (1.0, 1.0),
                     H_m: float = 1.368e-3, layout: str = "row"):
    """x-y contour plots of Z, one per source, sharing a color scale.

    layout='row' (default) is a single row; layout='grid' arranges panels
    as a 2x2 block for n=4.
    """
    keys = [s for s in _SOURCE_ORDER if s in sources_present and s in slices]
    n = len(keys)
    nrows, ncols = _grid_shape(n, layout)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols, 3.6 * nrows),
                             constrained_layout=True, squeeze=False)
    axes_flat = axes.ravel()

    # Shared color scale from the DNS slice (physical intent: paint each
    # source with the same colormap so differences are visible).
    dns_slice = slices["DNS"]
    vmin = float(dns_slice.min())
    vmax = float(dns_slice.max())

    # Shared PHYSICAL extent for every panel. All sources represent the same
    # centred physical box; only their cell counts differ (LES is 8x coarser
    # than DNS/MPS/PEPS). The extent must therefore come from the physical
    # span of the DNS grid, NOT from each panel's own cell count times dx --
    # doing the latter shrinks the coarse LES panel by the resolution ratio
    # (this is why the LES x-axis previously read +/-0.4 instead of +/-3.5).
    dx, dy = dx_dy
    nxD, nyD = dns_slice.shape
    x_half = (nxD / 2.0) * dx / H_m
    y_half = (nyD / 2.0) * dy / H_m
    shared_extent = [-x_half, x_half, -y_half, y_half]

    im = None
    for i, key in enumerate(keys):
        ax = axes_flat[i]
        _, _, is_left, is_bottom = _panel_pos(i, nrows, ncols)
        arr = slices[key]
        # imshow uses (row=y, col=x). arr is (nx, ny), so transpose to
        # display with x horizontal, y vertical. Every panel uses the same
        # shared physical extent, so coarse and fine slices line up in
        # (x/H, y/H) regardless of their cell counts.
        im = ax.imshow(arr.T, origin="lower", cmap="inferno",
                        vmin=vmin, vmax=vmax,
                        extent=shared_extent,
                        aspect="equal", interpolation="nearest")
        ax.set_title(_TITLE[key], fontsize=20)
        ax.tick_params(labelsize=16)
        _set_n_ticks(ax, "x", nx=3)
        _set_n_ticks(ax, "y", ny=3)
        # x-label only on the bottom row; y-label only on the left column.
        if is_bottom:
            ax.set_xlabel(r"$x/H$", fontsize=20)
        else:
            ax.set_xticklabels([])
        if is_left:
            ax.set_ylabel(r"$y/H$", fontsize=20)
        else:
            ax.set_yticklabels([])

    # Hide any unused panels in a partially filled grid.
    for j in range(len(keys), len(axes_flat)):
        axes_flat[j].set_visible(False)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(r"$Z$", fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    suffix = "row" if (layout == "row" or n <= 2) else "grid"
    png = os.path.join(out_dir, f"contour_mixfrac_{suffix}.png")
    # PNG for review + PDF for LaTeX inclusion (no .tex per user request).
    _save_fig(fig, png, also=("pdf",))
    plt.close(fig)
    print(f"[qc4pde] {png} (+ .pdf)")


# ---------------------------------------------------------------------------
# Figure 2: Z mean + RMS
# ---------------------------------------------------------------------------

def plot_mixfrac_mean_rms(pkls: Dict[str, dict], out_dir: str,
                          single_sided: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key in _SOURCE_ORDER:
        if key not in pkls:
            continue
        r = pkls[key]
        if "profiles" not in r or "mixfrac" not in r["profiles"]:
            continue
        y = r["y_over_H"]
        m = r["profiles"]["mixfrac"]["mean"]
        s = r["profiles"]["mixfrac"]["rms"]
        if single_sided:
            yh_m, m = _fold_profile(y, m, "mean")
            yh_s, s = _fold_profile(y, s, "rms")
        else:
            yh_m = yh_s = y
        axes[0].plot(yh_m, m, **_STYLE[key])
        axes[1].plot(yh_s, s, **_STYLE[key])
    axes[0].set_xlabel(r"$y/H$"); axes[0].set_ylabel(r"$\overline{Z}$")
    axes[1].set_xlabel(r"$y/H$"); axes[1].set_ylabel(r"$Z_{\rm rms}$")
    for ax in axes:
        if single_sided:
            ax.set_xlim(0, 5)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8, framealpha=0.9)
    png = os.path.join(out_dir, "mixfrac_mean_rms.png")
    fig.savefig(png, dpi=200)
    save_tikz_if_available(fig, os.path.splitext(png)[0] + ".tex")
    plt.close(fig)
    print(f"[qc4pde] {png}")


# ---------------------------------------------------------------------------
# Figure 3: Z PDF
# ---------------------------------------------------------------------------

def plot_mixfrac_pdf(pkls: Dict[str, dict], out_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    for key in _SOURCE_ORDER:
        if key not in pkls:
            continue
        r = pkls[key]
        if "pdf_mixfrac" not in r:
            continue
        d = r["pdf_mixfrac"]
        ax.plot(d["psi_Z"], d["pdf"], **_STYLE[key])
    ax.set_xlabel(r"$\psi_Z$")
    ax.set_ylabel(r"$P(\psi_Z)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    png = os.path.join(out_dir, "mixfrac_pdf.png")
    fig.savefig(png, dpi=200)
    save_tikz_if_available(fig, os.path.splitext(png)[0] + ".tex")
    plt.close(fig)
    print(f"[qc4pde] {png}")


# ---------------------------------------------------------------------------
# Figure 4: conditional T vs Z
# ---------------------------------------------------------------------------

def plot_conditional_T(pkls: Dict[str, dict], out_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    for key in _SOURCE_ORDER:
        if key not in pkls:
            continue
        r = pkls[key]
        if "conditional_T_on_Z" not in r:
            continue
        d = r["conditional_T_on_Z"]
        ax.plot(d["psi_Z"], d["E_T"], **_STYLE[key])
    ax.set_xlabel(r"$\psi_Z$")
    ax.set_ylabel(r"$\langle T \mid Z=\psi_Z\rangle$  [K]")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    png = os.path.join(out_dir, "conditional_T.png")
    fig.savefig(png, dpi=200)
    save_tikz_if_available(fig, os.path.splitext(png)[0] + ".tex")
    plt.close(fig)
    print(f"[qc4pde] {png}")


# ---------------------------------------------------------------------------
# Figure 5: joint PDF row (Z, Y_CO2), no tex
# ---------------------------------------------------------------------------

def plot_joint_pdf_row(pkls: Dict[str, dict], out_dir: str,
                        levels=(5, 10, 20, 50, 100, 200),
                        layout: str = "row"):
    keys = [s for s in _SOURCE_ORDER if s in pkls]
    n = len(keys)
    nrows, ncols = _grid_shape(n, layout)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols, 3.4 * nrows),
                             constrained_layout=True, squeeze=False)
    axes_flat = axes.ravel()
    for i, key in enumerate(keys):
        ax = axes_flat[i]
        _, _, is_left, is_bottom = _panel_pos(i, nrows, ncols)
        r = pkls[key]
        if "joint_pdf_Z_YCO2" not in r:
            ax.set_visible(False)
            continue
        d = r["joint_pdf_Z_YCO2"]
        H = np.nan_to_num(np.asarray(d["joint_pdf"]).T,
                           nan=0.0, posinf=0.0, neginf=0.0)
        Z_c = d["psi_Z"]; Y_c = d["psi_YCO2"]
        vmax = float(H.max())
        plot_levels = [float(lv) for lv in levels if lv < vmax]
        if plot_levels:
            cs = ax.contour(Z_c, Y_c, H, levels=plot_levels,
                             cmap="winter", linewidths=1.5)
            ax.clabel(cs, inline=True, fontsize=11, fmt="%d")
        ax.set_title(_TITLE[key], fontsize=20)
        ax.tick_params(labelsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.17)
        ax.grid(True, alpha=0.2)
        _set_n_ticks(ax, "x", nx=5)
        _set_n_ticks(ax, "y", ny=4)
        if is_bottom:
            ax.set_xlabel(r"$\psi_Z$", fontsize=20)
        else:
            ax.set_xticklabels([])
        if is_left:
            ax.set_ylabel(r"$\psi_{Y_{CO_2}}$", fontsize=20)
        else:
            ax.set_yticklabels([])

    # Hide any unused panels in a partially filled grid.
    for j in range(len(keys), len(axes_flat)):
        axes_flat[j].set_visible(False)

    suffix = "row" if (layout == "row" or n <= 2) else "grid"
    png = os.path.join(out_dir, f"joint_pdf_{suffix}.png")
    # PNG for review + PDF for LaTeX inclusion (no .tex per user request).
    _save_fig(fig, png, also=("pdf",))
    plt.close(fig)
    print(f"[qc4pde] {png} (+ .pdf)")


# ---------------------------------------------------------------------------
# Figure 6: 3D manifold row, no tex
# ---------------------------------------------------------------------------

def plot_manifold_row(pkls: Dict[str, dict], out_dir: str,
                      layout: str = "row"):
    keys = [s for s in _SOURCE_ORDER if s in pkls
            and "manifold_scatter" in pkls[s]]
    n = len(keys)
    nrows, ncols = _grid_shape(n, layout)
    fig = plt.figure(figsize=(4.0 * ncols, 3.8 * nrows))
    sc = None
    for i, key in enumerate(keys):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        m = pkls[key]["manifold_scatter"]
        sc = ax.scatter(
            m["Z"], m["Y_O2"], m["Y_OH"] * 1000.0,
            c=m["T"], cmap="jet", s=1.0, alpha=0.9,
            vmin=500.0, vmax=1500.0, edgecolors="none",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.30)
        ax.set_zlim(0, 2.0)
        # Modest label size (close to original) so the LaTeX text stays
        # clean and does not intersect neighbouring axes. Every 3D panel
        # keeps all three axis labels regardless of grid position.
        ax.set_xlabel(r"$Z$", fontsize=13, labelpad=0)
        ax.set_ylabel(r"$Y_{O_2}$", fontsize=13, labelpad=0)
        ax.set_zlabel(r"$Y_{OH}\times 10^3$", fontsize=13, labelpad=2)
        # Pull tick labels in close to each axis (pad, not labelpad, moves
        # the tick *numbers* on 3D axes).
        ax.tick_params(axis="x", labelsize=11, pad=-2)
        ax.tick_params(axis="y", labelsize=11, pad=-2)
        ax.tick_params(axis="z", labelsize=11, pad=0)
        # The receding Y_O2 axis crowds badly, so cap it to a few evenly
        # spaced ticks (0.0, 0.1, 0.2, 0.3) instead of the auto ~5.
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
        try:
            ax.view_init(elev=15.0, azim=-70.0)
        except Exception:
            pass
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor("white")
            pane.set_edgecolor("0.7")
            pane.set_alpha(1.0)
        # Thin out x tick labels to reduce crowding (y is already capped
        # to 3 ticks above; keep all of them).
        _skip_labels(ax, "x", step=2)
        for j, lbl in enumerate(ax.get_zticklabels()):
            if j % 2 != 0:
                lbl.set_visible(False)
        ax.set_title(_TITLE[key], fontsize=13, pad=-4)

    # Leave room on the right for a shared colorbar and add gaps between
    # panels. constrained_layout is not used with 3D subplots, so set
    # margins explicitly for both row and grid.
    fig.subplots_adjust(left=0.02, right=0.90, wspace=0.10, hspace=0.10)
    cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.55, pad=0.06,
                        fraction=0.02)
    cbar.set_label(r"$T$ [K]", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    suffix = "row" if (layout == "row" or n <= 2) else "grid"
    png = os.path.join(out_dir, f"manifold_{suffix}.png")
    # PNG for review + PDF for LaTeX inclusion (no .tex per user request).
    _save_fig(fig, png, also=("pdf",))
    plt.close(fig)
    print(f"[qc4pde] {png} (+ .pdf)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    if len(args) >= 1 and not args[0].startswith("--"):
        results_root = args[0]
        args = args[1:]
    else:
        results_root = "results"
    if len(args) >= 1 and not args[0].startswith("--"):
        out_dir = args[0]
        args = args[1:]
    else:
        out_dir = "QC4PDE"

    timestep = _parse(args, "--timestep", default="0198")
    dns_data_dir = _parse(args, "--dns-data-dir",
                           default="/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198")
    les_plt_path = _parse(args, "--les-plt",
                           default=None)
    peps_dir = _parse(args, "--peps-dir",
                       default="/ix/pgivi/moe32/Aidyn_DNS/stats/truncated_0198/PEPS")
    mps_root = _parse(args, "--mps-root",
                       default="/ix/pgivi/moe32/Aidyn_DNS/stats")
    filtered_dir = _parse(args, "--filtered-dir",
                          default=f"/ix/pgivi/moe32/Aidyn_DNS/stats/"
                                  f"filtered_{timestep}")
    fd_order = int(_parse(args, "--fd-order", default="4"))
    periodic_str = _parse(args, "--periodic-xz", default="false")
    periodic_xz = periodic_str.lower() in ("true", "1", "yes", "on")
    sp = int(_parse(args, "--sp", default="1"))
    # --layout {row,grid}: row (single row) is the default; grid arranges the
    # 4 panels as a 2x2 block. Affects contour / joint-PDF / manifold.
    layout = _parse(args, "--layout", default="row").lower()
    if layout not in ("grid", "row"):
        print(f"[qc4pde] unknown --layout '{layout}', using 'row'")
        layout = "row"
    skip_contour = "--skip-contour" in args
    if skip_contour:
        args.remove("--skip-contour")

    # --exclude: comma-separated source labels to drop from all plots.
    # Valid labels: DNS, LES, chi93, 4.1e-4, PEPS.
    exclude_str = _parse(args, "--exclude", default="")
    exclude = [s.strip() for s in exclude_str.split(",") if s.strip()]
    if exclude:
        print(f"[qc4pde] Excluding sources: {exclude}")

    if les_plt_path is None:
        cand = os.path.join(dns_data_dir, "LES_plt20000")
        if os.path.isdir(cand):
            les_plt_path = cand

    os.makedirs(out_dir, exist_ok=True)
    print(f"[qc4pde] results_root = {results_root}")
    print(f"[qc4pde] out_dir      = {out_dir}")
    print(f"[qc4pde] timestep     = {timestep}, sp={sp}, fd_order={fd_order}")

    # ---- Load pickles ----------------------------------------------------
    paths = _discover_pickles(results_root, timestep, fd_order, sp,
                              filtered_dir=filtered_dir)
    print(f"[qc4pde] found pickles for: {sorted(paths.keys())}")
    for k, p in paths.items():
        print(f"  {k:>7s}: {p}")
    pkls = {k: _load(p) for k, p in paths.items()}
    for excl in exclude:
        if excl in pkls:
            pkls.pop(excl)
            print(f"[qc4pde] Dropped {excl} from plot inputs")

    # ---- Figures 2-6 -----------------------------------------------------
    print(f"[qc4pde] panel layout = {layout}")
    plot_mixfrac_mean_rms(pkls, out_dir, single_sided=True)
    plot_mixfrac_pdf(pkls, out_dir)
    plot_conditional_T(pkls, out_dir)
    plot_joint_pdf_row(pkls, out_dir, layout=layout)
    plot_manifold_row(pkls, out_dir, layout=layout)

    # ---- Figure 1 (contour row) needs field slices, not pickles ---------
    if skip_contour:
        print("[qc4pde] Skipping contour row (--skip-contour)")
        return

    print("[qc4pde] Building contour row (loads DNS/LES/MPS/PEPS Z slices)...")
    slices: Dict[str, np.ndarray] = {}
    t0 = time.time()

    # DNS
    if "DNS" in pkls:
        slices["DNS"] = _load_dns_slice_Z(dns_data_dir, timestep)
        print(f"    DNS slice loaded ({time.time()-t0:.1f}s)")

    # Filtered (box-filtered DNS, 8^3 -> 64^3): the ideal-LES reference.
    if "Filtered" in pkls:
        try:
            slices["Filtered"] = _load_filtered_slice_Z(dns_data_dir, timestep)
            print(f"    Filtered slice loaded ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    Filtered slice failed: {e}")

    # LES
    if "LES" in pkls and les_plt_path is not None and os.path.isdir(les_plt_path):
        try:
            slices["LES"] = _load_les_slice_Z(les_plt_path)
            print(f"    LES slice loaded ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    LES slice failed: {e}")

    # MPS cases
    from mps_io import ordering_for_sp
    ordering = ordering_for_sp(sp)
    mps_data_dir = os.path.join(mps_root, f"truncated_{timestep}", f"sp{sp}")
    if not os.path.isdir(mps_data_dir):
        mps_data_dir = os.path.join(mps_root, f"truncated_{timestep}_sp{sp}")
    if not os.path.isdir(mps_data_dir) and sp == 1:
        mps_data_dir = os.path.join(mps_root, f"truncated_{timestep}")
    for key, cutoff_str in [("chi93", "chi93"), ("4.1e-4", "0.00041")]:
        if key not in pkls:
            continue
        try:
            slices[key] = _load_mps_slice_Z(mps_data_dir, timestep,
                                              cutoff_str, ordering=ordering)
            print(f"    MPS {key} slice loaded ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    MPS {key} slice failed: {e}")

    # PEPS (with sign & L2 rescale on the slice against DNS slice)
    if "PEPS" in pkls and "DNS" in slices:
        try:
            peps_raw = _load_peps_slice_Z(peps_dir)
            slices["PEPS"] = _peps_scale_and_sign(peps_raw, slices["DNS"])
            print(f"    PEPS slice loaded + rescaled "
                  f"({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    PEPS slice failed: {e}")

    if slices:
        from dns_stats import GridConfig
        g = GridConfig()
        # All sources cover the same centred physical box; only cell counts
        # differ (LES is 8x coarser). plot_contour_row derives a single
        # shared physical extent from the DNS slice and applies it to every
        # panel, so coarse and fine slices share identical (x/H, y/H) axes.
        dx_dy = (g.dx_m, g.dy_m)
        plot_contour_row(list(slices.keys()), slices, out_dir,
                          dx_dy=dx_dy, H_m=g.H, layout=layout)

    print(f"[qc4pde] Done in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
