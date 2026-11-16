"""
plot_chi_decomposition.py
-------------------------
One-off diagnostic: compare DNS vs two MPS cases (chi=93 and cf=4.1e-4)
for the intermediate quantities that go into scalar dissipation rate:

    Z              (mixture fraction)
    |grad Z|       (magnitude of gradient)
    |grad Z|^2     (dot product with itself)
    alpha          (thermal diffusivity)

Each MPS case is compared against DNS in three ways:
    - x-z averaged y-profile (mean and RMS)
    - PDF (histogram over the cube)
    - Scatter (1M downsampled points, DNS vs MPS per cell)

Gradient is computed with 4th-order non-periodic FD on the 512^3 cube
(matches the standard mixed-mode chi setup).

Usage
-----
python plot_chi_decomposition.py <results_root> <out_dir>
    [--timestep 0198]
    [--dns-data-dir /path/to/jet_0198]
    [--mps-root /path/to/parent/of/truncated_<TS>/sp<SP>]
    [--sp 1]
    [--fd-order 4]
    [--periodic-xz false]
    [--n-sample 1000000]
    [--cases chi93,4.1e-4]
    [--n-bins 200]

Example
-------
python plot_chi_decomposition.py results results/chi_decomp_0198 \
    --dns-data-dir /ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198 \
    --mps-root /ix/pgivi/moe32/Aidyn_DNS/stats \
    --sp 1
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dns_stats import GridConfig, load_field, _xz_mean, _xz_var
from derived_fields import _deriv_along_axis_nonperiodic
from mps_io import load_mps_field, ordering_for_sp, _mps_filename
from save_tikz import save_tikz_if_available


# Line style per source (matches conventions in plot_dns_mps_comparison.py).
_STYLE = {
    "DNS":    dict(color="black",   linestyle="-",  lw=2.0, label="DNS"),
    "PEPS":   dict(color="#d35400", linestyle=":",  lw=2.4,
                   label="PEPS D=9 (LES-matched)"),
    "chi93":  dict(color="#16a085", linestyle="--", lw=2.0,
                   label="MPS chi=93 (LES-matched)"),
    "4.1e-4": dict(color="#9b59b6", linestyle="--", lw=2.0,
                   label="MPS 4.1e-4 (LES-matched cutoff)"),
}

# The 4 diagnostic quantities.
_QUANTITY_LABELS = {
    "Z":         (r"$Z$",              None),
    "grad_Z":    (r"$|\nabla Z|$  [1/m]",   None),
    "grad_Z_sq": (r"$|\nabla Z|^2$  [1/m$^2$]",  None),
    "alpha":     (r"$\alpha$  [m$^2$/s]",   None),
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


def _grad_magnitude_and_sq(Z, dx, dy, dz, order, periodic_xz):
    """Compute |grad Z| and |grad Z|^2 with the same FD operator that
    compute_chi uses in non-periodic mode."""
    # non-periodic in all directions matches --periodic-xz false; if the
    # user set periodic_xz true, use compute_chi's periodic gradient
    # helper instead. Here we default to non-periodic (the standard case).
    if periodic_xz:
        from derived_fields import _gradient_periodic_x_z
        gx, gy, gz = _gradient_periodic_x_z(Z, dx, dy, dz, order=order)
    else:
        gx = _deriv_along_axis_nonperiodic(Z, dx, axis=0, order=order)
        gy = _deriv_along_axis_nonperiodic(Z, dy, axis=1, order=order)
        gz = _deriv_along_axis_nonperiodic(Z, dz, axis=2, order=order)
    grad_sq = gx * gx + gy * gy + gz * gz
    grad_mag = np.sqrt(np.maximum(grad_sq, 0.0))
    return grad_mag, grad_sq


def _rms_profile(field):
    """RMS profile computed as sqrt(x-z variance)."""
    return np.sqrt(np.maximum(_xz_var(field), 0.0))


def _fold_profile(y_over_H, prof, kind="mean"):
    """Fold a symmetric profile around y=0 to a single-sided view."""
    n = len(y_over_H)
    j0 = int(np.argmin(np.abs(y_over_H)))
    n_keep = min(n - j0, j0 + 1)
    p_pos = prof[j0:j0 + n_keep]
    p_neg = prof[j0 - np.arange(n_keep)]
    y_out = y_over_H[j0:j0 + n_keep]
    if kind == "rms":
        return y_out, np.sqrt(0.5 * (p_pos**2 + p_neg**2))
    return y_out, 0.5 * (p_pos + p_neg)


def _load_dns_cube(dns_data_dir, timestep, var, grid):
    return load_field(os.path.join(dns_data_dir,
                                    f"jet_{var}_{timestep}.dat"), grid)


def _load_mps_cube(mps_data_dir, timestep, var, cutoff_str, ordering):
    path = os.path.join(mps_data_dir,
                        _mps_filename(var, timestep, cutoff_str,
                                       ordering=ordering))
    return load_mps_field(path)


def _find_mps_data_dir(mps_root, timestep, sp):
    """Locate the sp subdirectory."""
    for cand in [
        os.path.join(mps_root, f"truncated_{timestep}", f"sp{sp}"),
        os.path.join(mps_root, f"truncated_{timestep}_sp{sp}"),
    ]:
        if os.path.isdir(cand):
            return cand
    if sp == 1:
        cand = os.path.join(mps_root, f"truncated_{timestep}")
        if os.path.isdir(cand):
            return cand
    return None


def _hist(arr, x_min, x_max, n_bins):
    h, e = np.histogram(arr.ravel(), bins=n_bins, range=(x_min, x_max),
                         density=True)
    c = 0.5 * (e[:-1] + e[1:])
    return c, h


def _plot_profiles(y_over_H, quantities: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                    out_dir: str, sources: List[str],
                    single_sided: bool = True):
    """2x4 grid: rows = (mean, rms), cols = 4 quantities.

    Y-axis limits use a robust scale (based on the DNS profile) so that
    one-sided FD boundary spikes at |y/H| near the domain edge don't
    blow up the plot.
    """
    keys = ["Z", "grad_Z", "grad_Z_sq", "alpha"]
    non_dns_sources = [s for s in sources if s != "DNS"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
    for ci, q in enumerate(keys):
        label, _ = _QUANTITY_LABELS[q]
        for ri, kind in enumerate(("mean", "rms")):
            ax = axes[ri, ci]
            lines_to_draw = []
            dns_prof = quantities[q]["DNS"][kind]
            for src in sources:
                if src not in quantities[q]:
                    continue
                prof = quantities[q][src][kind]
                if single_sided:
                    yh, p = _fold_profile(y_over_H, prof, kind=kind)
                else:
                    yh, p = y_over_H, prof
                lines_to_draw.append((src, yh, p))
            interior_slice = slice(4, -4) if len(dns_prof) > 12 else slice(None)
            dns_interior_max = float(np.nanmax(np.abs(dns_prof[interior_slice])))
            robust_max = dns_interior_max
            for src in non_dns_sources:
                if src not in quantities[q]:
                    continue
                p = quantities[q][src][kind]
                p_interior = p[interior_slice]
                p_max_int = float(np.nanmax(np.abs(p_interior)))
                if p_max_int < 3.0 * dns_interior_max:
                    robust_max = max(robust_max, p_max_int)
            robust_max *= 1.2
            if not np.isfinite(robust_max) or robust_max <= 0:
                robust_max = 1.0
            for src, yh, p in lines_to_draw:
                ax.plot(yh, p, **_STYLE[src])
            ax.set_ylim(0, robust_max)
            ax.set_title(f"{kind}({label})")
            if ri == 1:
                ax.set_xlabel(r"$y/H$")
            if single_sided:
                ax.set_xlim(0, 5)
            ax.grid(True, alpha=0.25)
            if ci == 0 and ri == 0:
                ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    suffix = "_single_sided" if single_sided else ""
    out_path = os.path.join(out_dir, f"chi_decomp_profiles{suffix}.png")
    fig.savefig(out_path, dpi=150)
    save_tikz_if_available(fig, os.path.splitext(out_path)[0] + ".tex")
    plt.close(fig)
    print(f"[plot] {out_path}")


def _plot_pdfs(quantities: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                out_dir: str, sources: List[str], log_y: bool = True):
    """1x4 grid of PDFs, DNS + MPS/PEPS cases overlaid per quantity."""
    keys = ["Z", "grad_Z", "grad_Z_sq", "alpha"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ci, q in enumerate(keys):
        label, _ = _QUANTITY_LABELS[q]
        ax = axes[ci]
        for src in sources:
            if src not in quantities[q]:
                continue
            centers = quantities[q][src]["pdf_centers"]
            pdf = quantities[q][src]["pdf_vals"]
            ax.plot(centers, pdf, **_STYLE[src])
        ax.set_title(f"PDF({label})")
        ax.set_xlabel(label)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        if ci == 0:
            ax.legend(loc="best", fontsize=9)
            ax.set_ylabel("PDF")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "chi_decomp_pdfs.png")
    fig.savefig(out_path, dpi=150)
    save_tikz_if_available(fig, os.path.splitext(out_path)[0] + ".tex")
    plt.close(fig)
    print(f"[plot] {out_path}")


def _plot_scatter_grid(scatter_data, out_dir, cases: List[str]):
    """4 rows (quantities) x N cols (cases). Each cell is a DNS vs case
    scatter with y=x reference and metrics.
    """
    keys = ["Z", "grad_Z", "grad_Z_sq", "alpha"]
    ncols = len(cases)
    if ncols == 0:
        return
    fig, axes = plt.subplots(4, ncols, figsize=(5.5 * ncols, 20),
                              squeeze=False)
    for ri, q in enumerate(keys):
        label, _ = _QUANTITY_LABELS[q]
        for ci, case in enumerate(cases):
            ax = axes[ri, ci]
            if case not in scatter_data[q]:
                ax.set_visible(False)
                continue
            x = scatter_data[q][case]["x"]
            y = scatter_data[q][case]["y"]
            axis_lim = scatter_data[q][case]["axis_lim"]
            ax.scatter(x, y, s=0.4, alpha=0.25, color="#2980b9",
                        edgecolors="none", rasterized=True)
            ax.plot(axis_lim, axis_lim, "k-", lw=1.0, label="y = x")
            ax.set_xlim(*axis_lim); ax.set_ylim(*axis_lim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"DNS {label}")
            other_prefix = "PEPS " if case == "PEPS" else "MPS "
            ax.set_ylabel(f"{other_prefix}{label}")
            src_label = _STYLE[case]["label"].replace("MPS ", "").replace("PEPS ", "")
            ax.set_title(f"{q}  --  {src_label}")
            ax.grid(False); ax.minorticks_off()
            metrics = scatter_data[q][case]["metrics"]
            info = (f"RMS err  = {metrics['rms']:.3e}\n"
                     f"rel L2   = {metrics['rel_l2']:.3e}\n"
                     f"corr     = {metrics['corr']:.5f}")
            ax.text(0.98, 0.02, info, transform=ax.transAxes,
                     ha="right", va="bottom", fontsize=8.5,
                     bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                ec="0.7", alpha=0.9))
    fig.tight_layout()
    out_path = os.path.join(out_dir, "chi_decomp_scatter.png")
    fig.savefig(out_path, dpi=150)
    # 3D-ish rasterized scatter -> no tikz (per user's instruction).
    plt.close(fig)
    print(f"[plot] {out_path}")


def _compute_metrics(dns_flat, mps_flat):
    err = dns_flat - mps_flat
    rms = float(np.sqrt(np.mean(err * err)))
    rel_l2 = float(np.linalg.norm(err) /
                    max(np.linalg.norm(dns_flat), 1e-30))
    corr = float(np.corrcoef(dns_flat, mps_flat)[0, 1])
    return dict(rms=rms, rel_l2=rel_l2, corr=corr)


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python plot_chi_decomposition.py <results_root> <out_dir> "
              "[--timestep 0198] [--dns-data-dir DIR] [--mps-root DIR] "
              "[--sp 1] [--fd-order 4] [--periodic-xz false] "
              "[--n-sample 1000000] [--cases chi93,4.1e-4] [--n-bins 200]")
        sys.exit(1)
    _results_root = args[0]
    out_dir = args[1]
    rest = args[2:]

    timestep = _parse(rest, "--timestep", default="0198")
    dns_data_dir = _parse(rest, "--dns-data-dir",
                           default="/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198")
    mps_root = _parse(rest, "--mps-root",
                      default="/ix/pgivi/moe32/Aidyn_DNS/stats")
    sp = int(_parse(rest, "--sp", default="1"))
    fd_order = int(_parse(rest, "--fd-order", default="4"))
    periodic_str = _parse(rest, "--periodic-xz", default="false")
    periodic_xz = periodic_str.lower() in ("true", "1", "yes", "on")
    n_sample = int(_parse(rest, "--n-sample", default="1000000"))
    cases_str = _parse(rest, "--cases", default="chi93,4.1e-4")
    cases = [c.strip() for c in cases_str.split(",") if c.strip()]
    n_bins = int(_parse(rest, "--n-bins", default="200"))
    seed = int(_parse(rest, "--seed", default="42"))

    # Scatter is expensive (1M-point scatter x 8 panels). Skip if flag set.
    skip_scatter = False
    if "--skip-scatter" in rest:
        rest.remove("--skip-scatter")
        skip_scatter = True

    if len(cases) < 1 or len(cases) > 4:
        print(f"Error: --cases must be 1 to 4 comma-separated labels; got {cases}")
        sys.exit(1)
    # Map cutoff label -> cutoff_str used in mps_io._mps_filename.
    # PEPS is handled separately (case=='PEPS' triggers peps_io path).
    def _to_cutoff_str(label):
        if label == "PEPS":
            return "PEPS"    # placeholder, not used
        if label == "chi93":
            return "chi93"
        if label == "4.1e-4":
            return "0.00041"
        try:
            v = float(label)
            return f"{v}"
        except ValueError:
            return label

    case_strs = {c: _to_cutoff_str(c) for c in cases}

    os.makedirs(out_dir, exist_ok=True)
    grid = GridConfig()
    ordering = ordering_for_sp(sp)
    mps_data_dir = _find_mps_data_dir(mps_root, timestep, sp)
    if mps_data_dir is None:
        print(f"Cannot find MPS data dir under {mps_root} for sp={sp}")
        sys.exit(1)

    print(f"[decomp] DNS dir = {dns_data_dir}")
    print(f"[decomp] MPS dir = {mps_data_dir}")
    print(f"[decomp] cases = {cases}")
    print(f"[decomp] FD order = {fd_order}, periodic_xz = {periodic_xz}")

    rng = np.random.default_rng(seed)

    # =========================================================================
    # Load DNS Z and alpha; compute grad_Z and grad_Z^2
    # =========================================================================
    t0 = time.time()
    print(f"[decomp] Loading DNS Z + alpha...", flush=True)
    Z_dns = _load_dns_cube(dns_data_dir, timestep, "mixfrac", grid).astype(np.float32)
    alpha_dns = _load_dns_cube(dns_data_dir, timestep, "alpha", grid).astype(np.float32)
    print(f"         loaded in {time.time()-t0:.1f}s  shapes Z={Z_dns.shape} "
          f"alpha={alpha_dns.shape}", flush=True)

    t1 = time.time()
    print(f"[decomp] Computing DNS |grad Z| and |grad Z|^2...", flush=True)
    grad_mag_dns, grad_sq_dns = _grad_magnitude_and_sq(
        Z_dns, grid.dx_m, grid.dy_m, grid.dz_m,
        order=fd_order, periodic_xz=periodic_xz)
    grad_mag_dns = grad_mag_dns.astype(np.float32)
    grad_sq_dns = grad_sq_dns.astype(np.float32)
    print(f"         computed in {time.time()-t1:.1f}s", flush=True)

    dns_fields = {"Z": Z_dns, "grad_Z": grad_mag_dns,
                    "grad_Z_sq": grad_sq_dns, "alpha": alpha_dns}

    # =========================================================================
    # Load each MPS case; compute grad_Z and grad_Z^2
    # =========================================================================
    mps_fields: Dict[str, Dict[str, np.ndarray]] = {}
    for case in cases:
        t2 = time.time()
        if case == "PEPS":
            # PEPS case: load from truncated_<TS>/PEPS/<var>_<tag>.mat via peps_io
            from peps_io import load_peps_field, _peps_filename
            peps_dir = os.path.join(mps_root, f"truncated_{timestep}", "PEPS")
            if not os.path.isdir(peps_dir):
                print(f"[decomp] WARN: PEPS dir not found: {peps_dir}; skipping",
                      flush=True)
                continue
            print(f"[decomp] Loading PEPS case (tag=wf_D=9_periodic)",
                  flush=True)
            Z_mps = load_peps_field(
                os.path.join(peps_dir, _peps_filename("mixfrac"))
            ).astype(np.float32)
            alpha_mps = load_peps_field(
                os.path.join(peps_dir, _peps_filename("alpha"))
            ).astype(np.float32)
        else:
            cutoff_str = case_strs[case]
            print(f"[decomp] Loading MPS case '{case}' (cutoff_str={cutoff_str})",
                  flush=True)
            Z_mps = _load_mps_cube(mps_data_dir, timestep, "mixfrac",
                                    cutoff_str, ordering).astype(np.float32)
            alpha_mps = _load_mps_cube(mps_data_dir, timestep, "alpha",
                                        cutoff_str, ordering).astype(np.float32)
        grad_mag_mps, grad_sq_mps = _grad_magnitude_and_sq(
            Z_mps, grid.dx_m, grid.dy_m, grid.dz_m,
            order=fd_order, periodic_xz=periodic_xz)
        mps_fields[case] = {
            "Z": Z_mps,
            "grad_Z": grad_mag_mps.astype(np.float32),
            "grad_Z_sq": grad_sq_mps.astype(np.float32),
            "alpha": alpha_mps,
        }
        print(f"         loaded + computed in {time.time()-t2:.1f}s",
              flush=True)

    # =========================================================================
    # Build profile / PDF / scatter data structures
    # =========================================================================
    y_over_H = grid.y_over_H()

    quantities: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    scatter_data: Dict[str, Dict[str, Dict]] = {}

    for q in ("Z", "grad_Z", "grad_Z_sq", "alpha"):
        # Profiles
        quantities[q] = {}
        for src, arr in [("DNS", dns_fields[q])] + [
                (case, mps_fields[case][q]) for case in cases]:
            quantities[q][src] = {
                "mean": _xz_mean(arr),
                "rms":  _rms_profile(arr),
            }

        # PDF range from DNS: use [min, 99.5th percentile] to avoid tails
        dns_arr = dns_fields[q]
        dns_flat = dns_arr.ravel()
        if q == "Z":
            x_min, x_max = 0.0, 1.0
        elif q == "alpha":
            x_min = float(dns_arr.min())
            x_max = float(dns_arr.max())
        else:
            x_min = 0.0
            x_max = float(np.percentile(dns_arr, 99.5))

        # DNS PDF
        c, h = _hist(dns_flat, x_min, x_max, n_bins)
        quantities[q]["DNS"]["pdf_centers"] = c
        quantities[q]["DNS"]["pdf_vals"] = h
        # MPS PDFs
        for case in cases:
            c2, h2 = _hist(mps_fields[case][q].ravel(), x_min, x_max, n_bins)
            quantities[q][case]["pdf_centers"] = c2
            quantities[q][case]["pdf_vals"] = h2

        # Scatter: subsample same indices for DNS and each MPS
        n_total = dns_flat.size
        if n_total > n_sample:
            idx = rng.choice(n_total, size=n_sample, replace=False)
        else:
            idx = np.arange(n_total)

        scatter_data[q] = {}
        # Axis limits shared per quantity: [DNS min - pad, DNS 99.5th percentile + pad]
        lo = float(dns_flat.min())
        hi = (float(np.percentile(dns_flat, 99.5))
               if q in ("grad_Z", "grad_Z_sq") else float(dns_flat.max()))
        pad = 0.02 * (hi - lo)
        axis_lim = (lo - pad, hi + pad)

        for case in cases:
            mps_flat = mps_fields[case][q].ravel()
            x = dns_flat[idx]
            y = mps_flat[idx]
            metrics = _compute_metrics(dns_flat, mps_flat)
            scatter_data[q][case] = dict(x=x, y=y, axis_lim=axis_lim,
                                          metrics=metrics)

    # =========================================================================
    # Render
    # =========================================================================
    all_sources = ["DNS"] + cases
    _plot_profiles(y_over_H, quantities, out_dir, sources=all_sources,
                    single_sided=True)
    _plot_profiles(y_over_H, quantities, out_dir, sources=all_sources,
                    single_sided=False)
    _plot_pdfs(quantities, out_dir, sources=all_sources, log_y=True)
    if not skip_scatter:
        _plot_scatter_grid(scatter_data, out_dir, cases=cases)
    else:
        print("[decomp] Skipping scatter plots (--skip-scatter)")

    # Write a text summary of scatter metrics
    txt_path = os.path.join(out_dir, "chi_decomp_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("Per-cell error metrics between DNS and MPS cases\n")
        f.write(f"Timestep: {timestep}, sp={sp} ({ordering}), FD={fd_order}, "
                 f"periodic_xz={periodic_xz}\n\n")
        f.write(f"{'quantity':>10s}  {'case':>10s}  {'RMS':>12s}  "
                 f"{'rel_L2':>12s}  {'corr':>12s}\n")
        f.write("-" * 62 + "\n")
        for q in ("Z", "grad_Z", "grad_Z_sq", "alpha"):
            for case in cases:
                m = scatter_data[q][case]["metrics"]
                f.write(f"{q:>10s}  {case:>10s}  {m['rms']:>12.4e}  "
                         f"{m['rel_l2']:>12.4e}  {m['corr']:>12.5f}\n")
    print(f"[write] {txt_path}")

    print(f"\n[decomp] total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
