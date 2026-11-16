"""
diagnose_les_plane.py
---------------------
Verify that the LES mixture-fraction slice used in the QC4PDE contour row
is taken through the correct plane and with the correct axis orientation
relative to DNS.

Why this is needed
------------------
plot_qc4pde._load_les_slice_Z takes LES[:, :, n//2] (a "mid-z" slice),
mirroring the DNS loader. That is only correct if les_io returns the LES
cube in the SAME (x, y, z) axis order as DNS. If yt / les_io hands back a
cube in a different order (e.g. (z, y, x), which is common for AMReX /
Fortran-order data), then "mid-z" for LES is actually a slice through a
different physical plane, and the contour panel is misleading.

What this does
--------------
1. Loads full DNS (512^3) and LES (64^3) mixfrac cubes with the SAME
   loaders the pipeline uses.
2. Block-averages DNS 512^3 -> 64^3 so the two cubes are directly
   comparable cell-for-cell (LES CR = 8^3 = 512, so factor 8 per axis).
3. For every candidate LES axis permutation (6) x flip pattern, and for
   each slice axis (x, y, z) at the mid index, computes the Pearson
   correlation between the LES slice and the matching down-sampled-DNS
   slice.
4. Also sweeps the slice index along each axis (not just the midpoint),
   in case the jet centreline is not at the geometric centre.
5. Reports the best-correlating (permutation, slice-axis, index) combo and
   states whether the CURRENT pipeline choice -- LES[:, :, n//2], i.e.
   identity permutation, z-axis, mid index -- is in fact the best, or
   whether a different plane/orientation correlates better with DNS.

Usage
-----
    python diagnose_les_plane.py \
        --dns-data-dir /ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198 \
        --les-plt      /ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198/LES_plt20000 \
        --timestep     0198 \
        [--out QC4PDE/les_plane_diag.png]

Run it on the cluster via sbatch (venv + python module loaded), NOT on a
viz node -- same rule as the rest of the pipeline.
"""

from __future__ import annotations

import os
import sys
import itertools
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# tiny arg parser (matches the rest of the codebase style)
# ---------------------------------------------------------------------------

def _parse(args, name, default=None, cast=str):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        args.pop(i + 1); args.pop(i)
        return cast(v) if cast is not str else v
    return default


# ---------------------------------------------------------------------------
# loaders -- use the SAME code paths as the pipeline
# ---------------------------------------------------------------------------

def load_dns_cube(dns_data_dir: str, timestep: str) -> np.ndarray:
    from dns_stats import GridConfig, load_field
    grid = GridConfig()
    Z = load_field(os.path.join(dns_data_dir, f"jet_mixfrac_{timestep}.dat"),
                   grid)
    return np.asarray(Z, dtype=np.float64)


def load_les_cube(les_plt_path: str) -> np.ndarray:
    from les_io import load_les_dataset, load_les_field, LESGrid
    ds = load_les_dataset(les_plt_path)
    grid = LESGrid()
    Z = load_les_field(ds, "mixfrac", grid)
    return np.asarray(Z, dtype=np.float64)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def block_average(cube: np.ndarray, factor: int) -> np.ndarray:
    """Down-sample a cube by integer `factor` per axis via block mean.

    Crops to a multiple of `factor` first if needed.
    """
    nx, ny, nz = cube.shape
    cx, cy, cz = (nx // factor) * factor, (ny // factor) * factor, \
                 (nz // factor) * factor
    c = cube[:cx, :cy, :cz]
    c = c.reshape(cx // factor, factor,
                  cy // factor, factor,
                  cz // factor, factor)
    return c.mean(axis=(1, 3, 5))


def corr2d(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 2D arrays of the same shape."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na = np.sqrt(np.sum(a * a))
    nb = np.sqrt(np.sum(b * b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.sum(a * b) / (na * nb))


def resize_slice_to(a: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize a 2D slice to `shape` (no SciPy dep)."""
    ty, tx = shape
    sy, sx = a.shape
    iy = (np.linspace(0, sy - 1, ty)).round().astype(int)
    ix = (np.linspace(0, sx - 1, tx)).round().astype(int)
    return a[np.ix_(iy, ix)]


# All 6 axis permutations, each optionally with per-axis flips. We test the
# 6 permutations and, for each, the 8 flip combinations -- but flips of the
# in-plane axes only change orientation, not which plane; we still test them
# because a reversed axis would otherwise lower the correlation and mask a
# correct permutation.
_PERMS: List[Tuple[int, int, int]] = list(itertools.permutations((0, 1, 2)))
_FLIPS: List[Tuple[bool, bool, bool]] = list(
    itertools.product((False, True), repeat=3))

_AXIS_NAME = {0: "x", 1: "y", 2: "z"}


def apply_perm_flip(cube: np.ndarray, perm: Tuple[int, int, int],
                    flip: Tuple[bool, bool, bool]) -> np.ndarray:
    out = np.transpose(cube, perm)
    sl = tuple(slice(None, None, -1) if f else slice(None) for f in flip)
    return out[sl]


# ---------------------------------------------------------------------------
# main diagnostic
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    dns_data_dir = _parse(args, "--dns-data-dir",
                          default="/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198")
    les_plt = _parse(args, "--les-plt", default=None)
    timestep = _parse(args, "--timestep", default="0198")
    out_png = _parse(args, "--out", default="QC4PDE/les_plane_diag.png")

    if les_plt is None:
        les_plt = os.path.join(dns_data_dir, "LES_plt20000")

    print(f"[diag] DNS dir : {dns_data_dir}")
    print(f"[diag] LES plt : {les_plt}")
    print(f"[diag] timestep: {timestep}")

    print("[diag] loading DNS cube...")
    dns = load_dns_cube(dns_data_dir, timestep)
    print(f"       DNS shape = {dns.shape}, "
          f"min/max = {dns.min():.3g}/{dns.max():.3g}")

    print("[diag] loading LES cube...")
    les = load_les_cube(les_plt)
    print(f"       LES shape = {les.shape}, "
          f"min/max = {les.min():.3g}/{les.max():.3g}")

    # Down-sample DNS to the LES resolution so slices compare cell-for-cell.
    # Factor from the finest DNS axis vs finest LES axis.
    factor = int(round(max(dns.shape) / max(les.shape)))
    factor = max(factor, 1)
    print(f"[diag] block-averaging DNS by factor {factor} to match LES res")
    dns_c = block_average(dns, factor)
    print(f"       DNS_coarse shape = {dns_c.shape}")

    # The DNS coarse cube is our reference in (x, y, z). We hold DNS fixed and
    # search over LES orientations. For a given LES orientation and a given
    # slice axis `ax`, we compare the LES mid-plane slice to the DNS mid-plane
    # slice along the same axis. Correlation is orientation-robust because we
    # also sweep flips.
    #
    # To decide the *plane*, we sweep the slice index across the axis and take
    # the best correlation (the jet may not be centred), reported per axis.

    results = []  # (corr, perm, flip, slice_axis, dns_idx, les_idx)

    # For efficiency, precompute DNS slices per axis and index once.
    def slices_along(cube, ax):
        n = cube.shape[ax]
        return [(i, np.take(cube, i, axis=ax)) for i in range(n)]

    dns_slices = {ax: slices_along(dns_c, ax) for ax in (0, 1, 2)}

    for perm in _PERMS:
        for flip in _FLIPS:
            les_o = apply_perm_flip(les, perm, flip)
            if les_o.shape != dns_c.shape:
                # Shapes must match after orientation for a fair slice compare
                # (they will, since both are cubes of equal size, but guard).
                if sorted(les_o.shape) != sorted(dns_c.shape):
                    continue
            for ax in (0, 1, 2):
                n_les = les_o.shape[ax]
                n_dns = dns_c.shape[ax]
                # sweep LES slice index; compare to the SAME fractional DNS idx
                for li in range(n_les):
                    les_sl = np.take(les_o, li, axis=ax)
                    di = int(round(li * (n_dns - 1) / max(n_les - 1, 1)))
                    dns_sl = dns_slices[ax][di][1]
                    if les_sl.shape != dns_sl.shape:
                        les_sl = resize_slice_to(les_sl, dns_sl.shape)
                    c = corr2d(les_sl, dns_sl)
                    results.append((c, perm, flip, ax, di, li))

    results.sort(key=lambda r: -r[0])
    best = results[0]

    # Collapse to the best result per unique (perm, flip, slice-axis) so the
    # ranking shows distinct orientations rather than many index sweeps of
    # the same winner.
    seen = {}
    for c, perm, flip, ax, di, li in results:
        key = (perm, flip, ax)
        if key not in seen:
            seen[key] = (c, perm, flip, ax, di, li)
    unique = sorted(seen.values(), key=lambda r: -r[0])

    print("\n[diag] ===== TOP 10 DISTINCT orientations "
          "(perm, flip, slice-axis) by best correlation =====")
    for c, perm, flip, ax, di, li in unique[:10]:
        print(f"   corr={c:+.4f}  perm={perm} flip={flip} "
              f"axis={_AXIS_NAME[ax]} dns_idx={di} les_idx={li}")

    # What the CURRENT pipeline does: identity perm, no flip, z-axis (2),
    # mid index.
    cur_perm = (0, 1, 2)
    cur_flip = (False, False, False)
    cur_ax = 2
    n_les_z = les.shape[2]
    cur_li = n_les_z // 2
    cur_di = int(round(cur_li * (dns_c.shape[2] - 1) / max(n_les_z - 1, 1)))
    les_cur = np.take(les, cur_li, axis=cur_ax)
    dns_cur = dns_slices[cur_ax][cur_di][1]
    if les_cur.shape != dns_cur.shape:
        les_cur = resize_slice_to(les_cur, dns_cur.shape)
    cur_corr = corr2d(les_cur, dns_cur)

    print("\n[diag] ===== CURRENT pipeline choice =====")
    print(f"   LES[:, :, n//2]  ->  perm={cur_perm} flip={cur_flip} "
          f"axis={_AXIS_NAME[cur_ax]} les_idx={cur_li} dns_idx={cur_di}")
    print(f"   correlation with DNS = {cur_corr:+.4f}")

    bc, bperm, bflip, bax, bdi, bli = best
    print("\n[diag] ===== VERDICT =====")
    same_plane = (bax == cur_ax)
    same_all = (bperm == cur_perm and bflip == cur_flip and bax == cur_ax)
    if same_all:
        print(f"   OK: current choice IS the best-correlating orientation "
              f"(corr={cur_corr:+.4f}). LES is on the right plane.")
    elif same_plane and bperm == cur_perm:
        print(f"   Plane axis matches (z), but a flip {bflip} correlates "
              f"better ({bc:+.4f} vs {cur_corr:+.4f}). Likely a reversed "
              f"axis, not a wrong plane. Consider flipping.")
    else:
        print(f"   MISMATCH: best is perm={bperm} flip={bflip} "
              f"axis={_AXIS_NAME[bax]} (corr={bc:+.4f}), but the pipeline "
              f"uses z-axis identity (corr={cur_corr:+.4f}).")
        print(f"   -> LES may be slicing the wrong plane / wrong axis order.")

    # ---- Physics-based centreline check ---------------------------------
    # A jet is statistically symmetric about its cross-stream centreline.
    # For each axis, collapse the cube to a 1D mean profile along that axis
    # (averaging over the other two). The peak of that profile is the
    # centreline index. If DNS and LES agree on which axis is cross-stream
    # and where its centre sits, the profiles should peak at the same
    # fractional location. A near-flat profile means that axis is a
    # homogeneous (streamwise/spanwise) direction.
    print("\n[diag] ===== mean-Z profile per axis (centreline test) =====")
    print("   (fractional peak location; 'contrast' = (max-min)/max of the "
          "profile.\n    Cross-stream axis has HIGH contrast + a clear "
          "interior peak; homogeneous axes are flat.)")

    def axis_profile(cube, ax):
        other = tuple(a for a in (0, 1, 2) if a != ax)
        return cube.mean(axis=other)

    for ax in (0, 1, 2):
        pd = axis_profile(dns_c, ax)
        pl = axis_profile(les, ax)
        # normalise index to [0,1] for cross-resolution comparison
        fd = np.argmax(pd) / (len(pd) - 1)
        fl = np.argmax(pl) / (len(pl) - 1)
        cd = (pd.max() - pd.min()) / (abs(pd.max()) + 1e-30)
        cl = (pl.max() - pl.min()) / (abs(pl.max()) + 1e-30)
        flag = ""
        if cd > 0.5 and cl > 0.5 and abs(fd - fl) > 0.1:
            flag = "  <-- cross-stream peaks DISAGREE (centre offset!)"
        elif cd > 0.5 and cl > 0.5:
            flag = "  <-- cross-stream, peaks AGREE"
        print(f"   axis {_AXIS_NAME[ax]}: DNS peak@{fd:.2f} "
              f"(contrast {cd:.2f}) | LES peak@{fl:.2f} "
              f"(contrast {cl:.2f}){flag}")

    # Also print the DNS-coarse and LES centred-cube extents so cropping
    # windows can be compared by eye.
    print("\n[diag] NOTE: compare the LES domain edges from the yt header "
          "above\n    against the DNS crop window. A cross-stream peak that "
          "sits far\n    from 0.5 usually means the centred-cube crop is not "
          "centred on\n    the jet, which is a cropping-window bug, not an "
          "axis-order bug.")

    # Also print the best PER SLICE-AXIS under identity+no-flip, which most
    # directly answers "which plane should I be slicing".
    print("\n[diag] ===== best index per slice-axis (identity perm, no flip) "
          "=====")
    for ax in (0, 1, 2):
        sub = [r for r in results
               if r[1] == (0, 1, 2) and r[2] == (False, False, False)
               and r[3] == ax]
        if sub:
            sub.sort(key=lambda r: -r[0])
            c, _, _, _, di, li = sub[0]
            print(f"   axis {_AXIS_NAME[ax]}: best corr={c:+.4f} "
                  f"at les_idx={li} (dns_idx={di}); "
                  f"mid-plane corr="
                  f"{[r for r in sub if r[5] == les.shape[ax]//2][0][0]:+.4f}")

    # ---- Visual: DNS vs LES best slice vs LES current slice --------------
    les_best = apply_perm_flip(les, bperm, bflip)
    les_best_sl = np.take(les_best, bli, axis=bax)
    dns_best_sl = dns_slices[bax][bdi][1]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2),
                             constrained_layout=True)
    vmin = float(dns_c.min()); vmax = float(dns_c.max())
    axes[0].imshow(dns_best_sl.T, origin="lower", cmap="inferno",
                   vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title(f"DNS (coarse) axis={_AXIS_NAME[bax]} idx={bdi}")
    im = axes[1].imshow(
        resize_slice_to(les_best_sl, dns_best_sl.shape).T,
        origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title(f"LES BEST perm={bperm}\nflip={bflip} "
                      f"corr={bc:+.3f}")
    axes[2].imshow(
        resize_slice_to(les_cur, dns_cur.shape).T,
        origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, aspect="auto")
    axes[2].set_title(f"LES CURRENT [:,:,n//2]\ncorr={cur_corr:+.3f}")
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label="Z")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"\n[diag] wrote comparison figure -> {out_png}")


if __name__ == "__main__":
    main()
