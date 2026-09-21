"""
diagnose_mps_spikes.py
----------------------
Test whether the spikes seen in E(|grad Z|^2 | Z) for the fixed-chi MPS are
(a) stable, seam-induced artifacts, or (b) sampling noise, and locate the
physical planes responsible.

Two diagnostics
---------------
1. BIN-COUNT ROBUSTNESS
   Recompute the conditional means E(|grad Z| | Z) and E(|grad Z|^2 | Z) for
   the chi93 MPS at several bin counts. Report the psi_Z location of the top
   spikes for each bin count. If the locations are stable across bin counts,
   the spikes are tied to real (seam) structure; if they wander, they are
   sampling noise.

2. EMPIRICAL SEAM DETECTION
   Rather than assume the "il" ordering, find the seams directly from the
   field: for each candidate plane (constant x, y, or z index), measure the
   mean squared jump across that plane, |Z[i+1] - Z[i]|^2, averaged over the
   plane. Real MPS block seams show up as regularly-spaced planes with
   anomalously large jumps. For the detected seam planes, report the mean Z
   on the plane -- that is the psi_Z where the seam dumps its gradient
   outliers, and should line up with the spike locations from diagnostic 1.

The comparison DNS field is used as a control: DNS should show NO such
regularly-spaced high-jump planes.

Usage
-----
    python diagnose_mps_spikes.py \
        --dns-data-dir /ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198 \
        --mps-root /ix/pgivi/moe32/Aidyn_DNS/stats \
        --timestep 0198 --sp 1 \
        [--cutoff chi93] [--out QC4PDE/mps_spike_diag.png]

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_qc4pde import _enable_latex_style, _parse   # noqa: E402
from plot_gradient_stats import (                       # noqa: E402
    grad_magnitude, conditional_mean, _load_dns_cube, _load_mps_cube,
)

_enable_latex_style()


# ---------------------------------------------------------------------------
# Diagnostic 1: bin-count robustness of the spikes
# ---------------------------------------------------------------------------

def top_spike_locations(psi: np.ndarray, val: np.ndarray, n_top: int = 3,
                        ref_level: float = None) -> List[Tuple[float, float]]:
    """Return the (psi_Z, value) of the largest local maxima of `val`.

    Excludes the sparse-sample end bins (first/last 3) so the extreme-Z
    tail rise is not mistaken for a spike.
    """
    v = np.array(val, dtype=float)
    v[:3] = np.nan
    v[-3:] = np.nan
    order = np.argsort(np.where(np.isfinite(v), v, -np.inf))[::-1]
    picks = []
    used = []
    for idx in order:
        if not np.isfinite(v[idx]):
            break
        # suppress neighbours of an already-picked peak (within 3 bins)
        if any(abs(idx - u) <= 3 for u in used):
            continue
        picks.append((float(psi[idx]), float(v[idx])))
        used.append(idx)
        if len(picks) >= n_top:
            break
    return picks


def bin_robustness(Z: np.ndarray, G: np.ndarray, G2: np.ndarray,
                   bin_counts=(48, 64, 96, 128)) -> Dict[int, dict]:
    out = {}
    for nb in bin_counts:
        psi, m = conditional_mean(G, Z, nbins=nb)
        psi2, m2 = conditional_mean(G2, Z, nbins=nb)
        out[nb] = {
            "psi": psi, "mean_g": m, "mean_g2": m2,
            "spikes_g": top_spike_locations(psi, m),
            "spikes_g2": top_spike_locations(psi2, m2),
        }
    return out


# ---------------------------------------------------------------------------
# Diagnostic 2: empirical seam detection
# ---------------------------------------------------------------------------

def plane_jump_profile(Z: np.ndarray, axis: int) -> np.ndarray:
    """Mean squared jump across each plane along `axis`.

    Returns an array of length (n_axis - 1); entry i is the mean over the
    plane of (Z[i+1] - Z[i])^2 along `axis`. Regular spikes in this profile
    mark seam planes.
    """
    d = np.diff(Z, axis=axis)
    other = tuple(a for a in range(3) if a != axis)
    return np.mean(d * d, axis=other)


def detect_seams(profile: np.ndarray, n_sigma: float = 4.0) -> np.ndarray:
    """Indices where the jump profile exceeds median + n_sigma * MAD."""
    med = np.median(profile)
    mad = np.median(np.abs(profile - med)) + 1e-30
    thresh = med + n_sigma * 1.4826 * mad
    return np.where(profile > thresh)[0]


def seam_Z_values(Z: np.ndarray, axis: int, seam_idx: np.ndarray
                  ) -> List[Tuple[int, float, float]]:
    """For each seam plane, return (index, mean_Z_on_plane, jump_value)."""
    other = tuple(a for a in range(3) if a != axis)
    out = []
    for i in seam_idx:
        # average the two planes bounding the seam
        plane = 0.5 * (np.take(Z, i, axis=axis) + np.take(Z, i + 1, axis=axis))
        out.append((int(i), float(plane.mean()), 0.0))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    dns_data_dir = _parse(args, "--dns-data-dir",
                          default="/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198")
    mps_root = _parse(args, "--mps-root",
                      default="/ix/pgivi/moe32/Aidyn_DNS/stats")
    timestep = _parse(args, "--timestep", default="0198")
    sp = int(_parse(args, "--sp", default="1"))
    cutoff = _parse(args, "--cutoff", default="chi93")
    n_sigma = float(_parse(args, "--n-sigma", default="4.0"))
    out_png = _parse(args, "--out", default="QC4PDE/mps_spike_diag.png")

    from dns_stats import GridConfig
    g = GridConfig()
    dx, dy, dz = g.dx_m, g.dy_m, g.dz_m
    H = g.H

    from mps_io import ordering_for_sp
    ordering = ordering_for_sp(sp)
    mps_dir = os.path.join(mps_root, f"truncated_{timestep}", f"sp{sp}")
    if not os.path.isdir(mps_dir):
        mps_dir = os.path.join(mps_root, f"truncated_{timestep}_sp{sp}")
    if not os.path.isdir(mps_dir) and sp == 1:
        mps_dir = os.path.join(mps_root, f"truncated_{timestep}")

    # Map the cutoff label to the string used in the MPS filename.
    # chi93 is a special (bond-dimension) label; anything else is treated as
    # a singular-value cutoff. Known aliases are normalised; otherwise the
    # value is passed through verbatim so arbitrary cutoffs work.
    _CUTOFF_ALIASES = {
        "chi93": "chi93",
        "4.1e-4": "0.00041",
        "0.00041": "0.00041",
    }
    cutoff_str = _CUTOFF_ALIASES.get(cutoff, cutoff)
    print(f"[spikes] loading MPS cube (cutoff label '{cutoff}' -> "
          f"file token '{cutoff_str}')...")
    t0 = time.time()

    # Echo the exact path we're about to open, and if it's missing, list the
    # cutoff tokens that DO exist in the directory (turns a raw h5py
    # traceback into an actionable message).
    from mps_io import _mps_filename
    _fname = _mps_filename("mixfrac", timestep, cutoff_str, ordering=ordering)
    _fpath = os.path.join(mps_dir, _fname)
    print(f"        path: {_fpath}")
    if not os.path.exists(_fpath):
        import glob as _glob
        import re as _re
        cands = sorted(_glob.glob(os.path.join(mps_dir, "*mixfrac*.mat")))
        toks = []
        for c in cands:
            mobj = _re.search(r"_cf([^.]*(?:\.[0-9e+-]+)?)\.mat$",
                              os.path.basename(c))
            toks.append(mobj.group(1) if mobj else os.path.basename(c))
        print(f"[spikes] ERROR: file not found.\n"
              f"         dir: {mps_dir}\n"
              f"         ordering token for sp={sp}: '{ordering}'\n"
              f"         available cutoff tokens: {toks}\n"
              f"         -> pass one of those to --cutoff (verbatim).")
        sys.exit(1)

    Zmps = _load_mps_cube(mps_dir, timestep, cutoff_str, ordering=ordering)
    print(f"        MPS shape={Zmps.shape} ({time.time()-t0:.1f}s)")

    print("[spikes] loading DNS cube (control)...")
    Zdns = _load_dns_cube(dns_data_dir, timestep)
    print(f"        DNS shape={Zdns.shape} ({time.time()-t0:.1f}s)")

    Gmps = grad_magnitude(Zmps, dx, dy, dz)
    G2mps = Gmps * Gmps

    # ---- Diagnostic 1 ----------------------------------------------------
    print("\n[spikes] === Diagnostic 1: bin-count robustness (MPS "
          f"{cutoff}) ===")
    rob = bin_robustness(Zmps, Gmps, G2mps)
    for nb, d in rob.items():
        s2 = ", ".join(f"psi={p:.3f}" for p, _ in d["spikes_g2"])
        s1 = ", ".join(f"psi={p:.3f}" for p, _ in d["spikes_g"])
        print(f"   nbins={nb:>3d}:  |grad Z|^2 spikes at [{s2}]")
        print(f"              |grad Z|   peaks  at [{s1}]")
    print("   -> If the |grad Z|^2 spike psi values are ~constant across "
          "nbins,\n      they are seam artifacts (stable), not sampling "
          "noise (which drifts).")

    # ---- Diagnostic 2 ----------------------------------------------------
    print("\n[spikes] === Diagnostic 2: empirical seam detection ===")
    axis_names = {0: "x", 1: "y", 2: "z"}
    seam_report = {}
    for axis in (0, 1, 2):
        prof_mps = plane_jump_profile(Zmps, axis)
        prof_dns = plane_jump_profile(Zdns, axis)
        seams = detect_seams(prof_mps, n_sigma=n_sigma)
        seams_dns = detect_seams(prof_dns, n_sigma=n_sigma)
        seam_report[axis] = (prof_mps, prof_dns, seams)
        zvals = seam_Z_values(Zmps, axis, seams)
        frac = len(seams) / max(len(prof_mps), 1)
        sat = "  [WARNING: >30% of planes flagged; raise --n-sigma]" \
            if frac > 0.30 else ""
        print(f"\n   axis {axis_names[axis]}: MPS has {len(seams)} seam "
              f"planes (>{n_sigma:g} MAD); DNS control has "
              f"{len(seams_dns)}.{sat}")
        if len(seams) > 0:
            # spacing regularity
            if len(seams) > 1:
                sp_gaps = np.diff(seams)
                print(f"      seam index spacing: {list(sp_gaps)} "
                      f"(regular spacing => block seams)")
            print("      seam planes and the Z they cut:")
            for idx, meanZ, _ in zvals[:12]:
                print(f"        idx={idx:>4d}  mean Z on plane = {meanZ:.3f}")
            print("      ^ compare these Z values to the |grad Z|^2 spike "
                  "psi_Z from Diagnostic 1.")

    # ---- Figure ----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    # top row: jump profiles per axis (MPS vs DNS)
    for axis in (0, 1, 2):
        prof_mps, prof_dns, seams = seam_report[axis]
        ax = axes[0, axis]
        ax.semilogy(prof_mps, color="#16a085", lw=1.0,
                    label=f"MPS {cutoff}")
        ax.semilogy(prof_dns, color="black", lw=1.0, alpha=0.7,
                    label="DNS")
        for s in seams:
            ax.axvline(s, color="red", lw=0.5, alpha=0.4)
        ax.set_title(f"plane-jump profile, axis {axis_names[axis]}")
        ax.set_xlabel("plane index")
        ax.set_ylabel(r"mean $(\Delta Z)^2$ across plane")
        ax.legend(fontsize=8)
    # bottom row: conditional means at different bin counts (g2)
    ax = axes[1, 0]
    for nb, d in rob.items():
        ax.plot(d["psi"], d["mean_g2"], lw=1.2, label=f"nbins={nb}")
    ax.set_title(r"$E(|\nabla Z|^2\mid Z)$ vs bin count (MPS "
                 f"{cutoff})")
    ax.set_xlabel(r"$\psi_Z$"); ax.set_ylabel(r"$E(|\nabla Z|^2\mid Z)$")
    ax.set_xlim(0, 1)
    # clip y so the plot is readable but spikes still visible
    dns_ref = np.nanmax(conditional_mean(grad_magnitude(Zdns, dx, dy, dz)**2,
                                         Zdns, nbins=64)[1])
    ax.set_ylim(0, 3.0 * dns_ref)
    ax.legend(fontsize=8)
    # bottom-middle: linear mean for contrast
    ax = axes[1, 1]
    for nb, d in rob.items():
        ax.plot(d["psi"], d["mean_g"], lw=1.2, label=f"nbins={nb}")
    ax.set_title(r"$E(|\nabla Z|\mid Z)$ vs bin count (MPS "
                 f"{cutoff})")
    ax.set_xlabel(r"$\psi_Z$"); ax.set_ylabel(r"$E(|\nabla Z|\mid Z)$")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    axes[1, 2].axis("off")

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"\n[spikes] wrote {out_png}")
    print(f"[spikes] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
