"""
plot_gradient_stats.py
----------------------
Gradient statistics of the mixture fraction Z, computed from the full
512^3 field cubes (NOT the mid-z slices). Mirrors the styling of
plot_qc4pde.plot_conditional_T.

Two figures
-----------
1. cond_gradZ.{png,pdf}        E(|grad Z| | Z = psi_Z)   vs psi_Z
2. cond_gradZ2.{png,pdf}       E(|grad Z|^2 | Z = psi_Z) vs psi_Z

Notes
-----
* All sources are kept at their native 512^3 resolution and share the same
  grid spacing, so the gradients are directly comparable. LES (64^3) is
  EXCLUDED: a gradient on an 8x coarser grid is not comparable and would sit
  artificially low for a resolution reason, not a physical one.
* |grad Z|^2 is the core of the scalar dissipation rate,
  chi = 2 (gamma / rho) |grad Z|^2, minus the 2 gamma/rho prefactor. So
  figure 2 connects directly to the scalar-dissipation analysis.
* Derivatives amplify high-frequency compression error. In particular, the
  fixed-chi MPS injects large gradient spikes at its 1D block seams, which
  dominate the conditional mean in the sparsely-sampled fuel-rich tail
  (psi_Z -> 1). By default the y-axis auto-scales to the DNS peak so these
  outliers clip at the top (still visible as lines leaving the frame) rather
  than flattening the physically meaningful comparison. Use --ymax / --ymax2
  to set explicit limits, or --exclude chi93 to drop that curve entirely.

Usage
-----
    python plot_gradient_stats.py [out_dir=QC4PDE]
        [--timestep 0198]
        [--dns-data-dir /path/to/jet_0198]
        [--peps-dir /path] [--mps-root /path]
        [--fd-order 4] [--sp 1]
        [--nbins-cond 64]
        [--ymax N] [--ymax2 N] [--ymax-scale 1.5]
        [--exclude chi93,...]

Run on a login node with the venv + python module loaded.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Reuse styling + helpers from the main plotting module so the figures match.
# ---------------------------------------------------------------------------

from plot_qc4pde import (
    _STYLE, _SOURCE_ORDER, _enable_latex_style, _save_fig, _parse,
)

# LES is intentionally excluded from gradient plots (see module docstring).
_GRAD_ORDER = [s for s in _SOURCE_ORDER if s != "LES"]

_HAS_TEX = _enable_latex_style()

Z_ST = 0.42   # stoichiometric mixture fraction (Hawkes CO/H2 flame)


# ---------------------------------------------------------------------------
# Full-cube loaders (NOT the mid-z slices used by the contour row).
# ---------------------------------------------------------------------------

def _load_dns_cube(dns_data_dir: str, timestep: str) -> np.ndarray:
    from dns_stats import GridConfig, load_field
    grid = GridConfig()
    Z = load_field(os.path.join(dns_data_dir, f"jet_mixfrac_{timestep}.dat"),
                   grid)
    return np.asarray(Z, dtype=np.float64)


def _load_mps_cube(mps_data_dir: str, timestep: str, cutoff_str: str,
                   ordering: str = "il") -> np.ndarray:
    from mps_io import load_mps_field, _mps_filename
    path = os.path.join(mps_data_dir,
                        _mps_filename("mixfrac", timestep, cutoff_str,
                                      ordering=ordering))
    return np.asarray(load_mps_field(path), dtype=np.float64)


def _load_peps_cube(peps_dir: str, tag: str = "wf_D=9_periodic") -> np.ndarray:
    from peps_io import load_peps_field
    path = os.path.join(peps_dir, f"mixfrac_{tag}.mat")
    return np.asarray(load_peps_field(path), dtype=np.float64)   # (x, y, z)


def _peps_scale_and_sign_cube(peps: np.ndarray,
                              dns: np.ndarray) -> np.ndarray:
    """Match PEPS to DNS L2 norm and fix global sign (full-cube version)."""
    if np.sum(peps * dns) < 0:
        peps = -peps
    n_dns = np.sqrt(np.sum(dns ** 2))
    n_peps = np.sqrt(np.sum(peps ** 2))
    if n_peps > 0:
        peps = peps * (n_dns / n_peps)
    return np.clip(peps, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Gradient magnitude, 4th-order central differences in physical units.
# ---------------------------------------------------------------------------

def grad_magnitude(Z: np.ndarray, dx: float, dy: float,
                   dz: float) -> np.ndarray:
    """|grad Z| via 4th-order central differences.

    np.gradient uses 2nd-order interior stencils; for a matched 4th-order
    scheme we roll the array. Periodic in x and z (streamwise/spanwise),
    one-sided fallback handled by np.gradient at the y edges.
    """
    # 4th-order central: f' = (-f[i+2] + 8 f[i+1] - 8 f[i-1] + f[i-2]) / (12 h)
    def d4(a, axis, h, periodic):
        if periodic:
            fp2 = np.roll(a, -2, axis=axis)
            fp1 = np.roll(a, -1, axis=axis)
            fm1 = np.roll(a,  1, axis=axis)
            fm2 = np.roll(a,  2, axis=axis)
            return (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h)
        # non-periodic (y): use np.gradient (2nd order) to avoid wrap-around
        return np.gradient(a, h, axis=axis, edge_order=2)

    gx = d4(Z, 0, dx, periodic=True)    # x streamwise (periodic)
    gy = d4(Z, 1, dy, periodic=False)   # y cross-stream (non-periodic)
    gz = d4(Z, 2, dz, periodic=True)    # z spanwise  (periodic)
    return np.sqrt(gx * gx + gy * gy + gz * gz)


# ---------------------------------------------------------------------------
# Conditional mean of a field G given Z, on a common psi_Z grid.
# ---------------------------------------------------------------------------

def conditional_mean(G: np.ndarray, Z: np.ndarray,
                     nbins: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """E(G | Z = psi) on `nbins` uniform bins of Z in [0, 1]."""
    zf = Z.ravel()
    gf = G.ravel()
    edges = np.linspace(0.0, 1.0, nbins + 1)
    which = np.clip(np.digitize(zf, edges) - 1, 0, nbins - 1)
    sums = np.bincount(which, weights=gf, minlength=nbins)
    cnts = np.bincount(which, minlength=nbins).astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnts > 0, sums / cnts, np.nan)
    return centers, mean


# ---------------------------------------------------------------------------
# Plotters (styled exactly like plot_conditional_T)
# ---------------------------------------------------------------------------

def _new_ax(ylabel: str, xlabel: str = r"$\psi_Z$"):
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_conditional_gradient(cond: Dict[str, Tuple[np.ndarray, np.ndarray]],
                              out_dir: str, squared: bool,
                              ymax: float = None, ymax_scale: float = 1.5):
    """Conditional mean of |grad Z| (or its square) vs Z.

    y-axis: if `ymax` is given, use it. Otherwise auto-scale to the DNS
    curve's peak times `ymax_scale`, so the physically meaningful data fills
    the axis and MPS seam-spike outliers clip at the top (they remain
    visible as lines exiting the frame rather than flattening everything).
    """
    ylabel = (r"$E(|\nabla Z|^2 \mid Z=\psi_Z)$  [m$^{-2}$]" if squared
              else r"$E(|\nabla Z| \mid Z=\psi_Z)$  [m$^{-1}$]")
    fig, ax = _new_ax(ylabel)
    for key in _GRAD_ORDER:
        if key not in cond:
            continue
        psi, val = cond[key]
        ax.plot(psi, val, **_STYLE[key])
    ax.axvline(Z_ST, color="0.6", linestyle=":", lw=1.5, label=r"$Z_{st}$")
    ax.set_xlim(0, 1)

    # Decide the y upper limit.
    if ymax is None:
        # Prefer DNS as the reference for "physically meaningful" range;
        # fall back to the median of available curves' maxima if DNS absent.
        ref = None
        if "DNS" in cond:
            ref = np.nanmax(cond["DNS"][1])
        else:
            maxes = [np.nanmax(v) for _, v in cond.values() if np.isfinite(v).any()]
            ref = np.nanmedian(maxes) if maxes else None
        if ref is not None and np.isfinite(ref) and ref > 0:
            ax.set_ylim(0, ymax_scale * ref)
        else:
            ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(0, ymax)

    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    name = "cond_gradZ2" if squared else "cond_gradZ"
    png = os.path.join(out_dir, f"{name}.png")
    _save_fig(fig, png, also=("pdf",))
    plt.close(fig)
    print(f"[grad] {png} (+ .pdf)")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    if len(args) >= 1 and not args[0].startswith("--"):
        out_dir = args[0]; args = args[1:]
    else:
        out_dir = "QC4PDE"

    timestep = _parse(args, "--timestep", default="0198")
    dns_data_dir = _parse(args, "--dns-data-dir",
                          default="/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198")
    peps_dir = _parse(args, "--peps-dir",
                      default="/ix/pgivi/moe32/Aidyn_DNS/stats/truncated_0198/PEPS")
    mps_root = _parse(args, "--mps-root",
                      default="/ix/pgivi/moe32/Aidyn_DNS/stats")
    fd_order = int(_parse(args, "--fd-order", default="4"))
    sp = int(_parse(args, "--sp", default="1"))
    nbins_cond = int(_parse(args, "--nbins-cond", default="64"))
    # y-axis control for the conditional-mean figures. --ymax N sets a hard
    # limit for the |grad Z| figure; --ymax2 N for the |grad Z|^2 figure.
    # If unset, the axis auto-scales to the DNS peak x --ymax-scale so MPS
    # seam-spike outliers clip visibly instead of flattening the plot.
    ymax = _parse(args, "--ymax", default=None)
    ymax2 = _parse(args, "--ymax2", default=None)
    ymax = float(ymax) if ymax is not None else None
    ymax2 = float(ymax2) if ymax2 is not None else None
    ymax_scale = float(_parse(args, "--ymax-scale", default="1.5"))
    exclude_str = _parse(args, "--exclude", default="")
    exclude = [s.strip() for s in exclude_str.split(",") if s.strip()]

    os.makedirs(out_dir, exist_ok=True)
    print(f"[grad] out_dir  = {out_dir}")
    print(f"[grad] timestep = {timestep}, sp={sp}, fd_order={fd_order}")
    print(f"[grad] sources  = {[s for s in _GRAD_ORDER if s not in exclude]} "
          f"(LES always excluded)")

    from dns_stats import GridConfig
    g = GridConfig()
    dx, dy, dz = g.dx_m, g.dy_m, g.dz_m

    # ---- load cubes ------------------------------------------------------
    cubes: Dict[str, np.ndarray] = {}
    t0 = time.time()

    if "DNS" not in exclude:
        cubes["DNS"] = _load_dns_cube(dns_data_dir, timestep)
        print(f"    DNS cube loaded ({time.time()-t0:.1f}s)")

    from mps_io import ordering_for_sp
    ordering = ordering_for_sp(sp)
    mps_data_dir = os.path.join(mps_root, f"truncated_{timestep}", f"sp{sp}")
    if not os.path.isdir(mps_data_dir):
        mps_data_dir = os.path.join(mps_root, f"truncated_{timestep}_sp{sp}")
    if not os.path.isdir(mps_data_dir) and sp == 1:
        mps_data_dir = os.path.join(mps_root, f"truncated_{timestep}")
    for key, cutoff_str in [("chi93", "chi93"), ("4.1e-4", "0.00041")]:
        if key in exclude:
            continue
        try:
            cubes[key] = _load_mps_cube(mps_data_dir, timestep, cutoff_str,
                                        ordering=ordering)
            print(f"    MPS {key} cube loaded ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    MPS {key} cube failed: {e}")

    if "PEPS" not in exclude:
        try:
            peps_raw = _load_peps_cube(peps_dir)
            if "DNS" in cubes:
                peps_raw = _peps_scale_and_sign_cube(peps_raw, cubes["DNS"])
            cubes["PEPS"] = peps_raw
            print(f"    PEPS cube loaded + rescaled ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    PEPS cube failed: {e}")

    if not cubes:
        print("[grad] no cubes loaded; nothing to do.")
        return

    # ---- gradients + conditional means ----------------------------------
    cond_g, cond_g2 = {}, {}
    for key in _GRAD_ORDER:
        if key not in cubes:
            continue
        Z = cubes[key]
        G = grad_magnitude(Z, dx, dy, dz)
        G2 = G * G
        psi, m = conditional_mean(G, Z, nbins=nbins_cond)
        cond_g[key] = (psi, m)
        psi2, m2 = conditional_mean(G2, Z, nbins=nbins_cond)
        cond_g2[key] = (psi2, m2)
        print(f"    stats done for {key} ({time.time()-t0:.1f}s)")

    # ---- plots -----------------------------------------------------------
    plot_conditional_gradient(cond_g, out_dir, squared=False,
                              ymax=ymax, ymax_scale=ymax_scale)
    plot_conditional_gradient(cond_g2, out_dir, squared=True,
                              ymax=ymax2, ymax_scale=ymax_scale)

    print(f"[grad] Done in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
