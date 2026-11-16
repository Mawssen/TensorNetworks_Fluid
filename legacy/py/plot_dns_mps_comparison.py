"""
plot_dns_mps_comparison.py
--------------------------
Overlay DNS and MPS-truncated statistics on a single figure for direct
visual comparison.

Usage
-----
python plot_dns_mps_comparison.py <dns_pkl> <mps_pkl_1e-3> <mps_pkl_1e-4>
    <mps_pkl_1e-5> <out_dir>

Or pass a directory containing all of them and let the script auto-discover:

python plot_dns_mps_comparison.py <results_root> <out_dir>

Where <results_root> contains:
    dns_<ts>/cube/dns_stats_<ts>.pkl
    mps_<ts>/1e-3/mps_stats_<ts>_cf1e-3_*.pkl
    mps_<ts>/1e-4/mps_stats_<ts>_cf1e-4_*.pkl
    mps_<ts>/1e-5/mps_stats_<ts>_cf1e-5_*.pkl

Layout
------
- Combined overlay plots (DNS + 3 MPS lines): mean+RMS, chi profile,
  conditional T, PDF Z
- Separate plots per source (4 figures): joint PDF, manifold

Color scheme:
  DNS:  black,  solid
  1e-3: orange, dashed
  1e-4: green,  dashed
  1e-5: blue,   dashed
"""

from __future__ import annotations

import os
import sys
import glob
import time
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from save_tikz import save_tikz_if_available


def _save_2d(fig, png_path: str, save_tex: bool = True) -> None:
    """Save a 2D figure to PNG and optionally to a matching .tex (tikz).
    Skip .tex for 3D scatter (manifold) figures.
    """
    fig.savefig(png_path)
    if save_tex:
        tex_path = os.path.splitext(png_path)[0] + ".tex"
        save_tikz_if_available(fig, tex_path)


# Visual style for each source
_STYLE = {
    "DNS":    dict(color="black",   linestyle="-",  marker=None, label="DNS",   lw=2.0),
    "LES":    dict(color="#c0392b", linestyle="-.", marker=None, label="LES-FDF", lw=2.0),
    "PEPS":   dict(color="#d35400", linestyle=":",  marker=None, label="PEPS D=9 (LES-matched)", lw=2.4),
    "chi93":  dict(color="#16a085", linestyle="--", marker=None, label="MPS chi=93 (LES-matched)", lw=2.0),
    "4.1e-4": dict(color="#9b59b6", linestyle="--", marker=None, label="MPS 4.1e-4 (LES-matched cutoff)", lw=2.0),
    "1e-2":   dict(color="#34495e", linestyle="--", marker=None, label="MPS 1e-2", lw=1.6),
    "1e-3":   dict(color="#e67e22", linestyle="--", marker=None, label="MPS 1e-3", lw=1.6),
    "1e-4":   dict(color="#27ae60", linestyle="--", marker=None, label="MPS 1e-4", lw=1.6),
    "1e-5":   dict(color="#2980b9", linestyle="--", marker=None, label="MPS 1e-5", lw=1.6),
}


def _setup_mpl():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.8,
    })


# -----------------------------------------------------------------------------
# Folding (single-sided y/H >= 0)
# -----------------------------------------------------------------------------

def _fold_profile(y, prof, kind="mean"):
    n = len(y)
    j0 = int(np.argmin(np.abs(y)))
    n_keep = min(n - j0, j0 + 1)
    p_pos = prof[j0:j0 + n_keep]
    p_neg = prof[j0 - np.arange(n_keep)]
    if kind == "mean":
        return y[j0:j0 + n_keep], 0.5 * (p_pos + p_neg)
    elif kind == "rms":
        return y[j0:j0 + n_keep], np.sqrt(0.5 * (p_pos**2 + p_neg**2))
    raise ValueError(f"kind={kind!r}")


# -----------------------------------------------------------------------------
# Auto-discovery of pickles
# -----------------------------------------------------------------------------

def _extract_meta_from_mps(path: str) -> Tuple[Optional[str], Optional[str],
                                               Optional[str], Optional[int],
                                               Optional[bool], Optional[int]]:
    """Extract (timestep, cutoff_label, mode, fd_order, periodic, sp) from MPS pickle name.
    Recognizes:
      'mps_stats_0198_cf1e-3_stored.pkl'              -> ('0198','1e-3','stored',None,False,None)
      'mps_stats_0198_cf1e-3_stored_sp1.pkl'          -> ('0198','1e-3','stored',None,False,1)
      'mps_stats_0198_cf1e-3_recon_o8.pkl'            -> ('0198','1e-3','recon',8,False,None)
      'mps_stats_0198_cf1e-3_recon_o8_sp2.pkl'        -> ('0198','1e-3','recon',8,False,2)
      'mps_stats_0198_cf1e-3_recon_o8_sp2_p.pkl'      -> ('0198','1e-3','recon',8,True,2)
    """
    base = os.path.basename(path)
    if not base.startswith("mps_stats_") or not base.endswith(".pkl"):
        return None, None, None, None, None, None
    stem = base[len("mps_stats_"):-len(".pkl")]
    parts = stem.split("_")
    if len(parts) < 3:
        return None, None, None, None, None, None
    ts = parts[0]
    cf_part = parts[1]
    mode = parts[2]
    if not cf_part.startswith("cf"):
        return None, None, None, None, None, None
    cf_label = cf_part[2:]

    fd_order = None
    sp = None
    periodic = False
    # Walk the trailing tokens looking for o<N>, sp<N>, p
    for tok in parts[3:]:
        if tok.startswith("o") and tok[1:].isdigit():
            fd_order = int(tok[1:])
        elif tok.startswith("sp") and tok[2:].isdigit():
            sp = int(tok[2:])
        elif tok == "p":
            periodic = True
    return ts, cf_label, mode, fd_order, periodic, sp


def discover_results(root: str, timestep: Optional[str] = None,
                     mode: Optional[str] = None,
                     fd_order: Optional[int] = None,
                     periodic: Optional[bool] = None,
                     sp: Optional[int] = None,
                     ) -> Tuple[Optional[str], Dict[str, str], Optional[str],
                                Optional[int], Optional[bool], Optional[int]]:
    """Walk root looking for DNS and MPS pickles.

    Filters MPS pickles by timestep, mode, fd_order, periodicity, and sp.
    If timestep is None, auto-detected from MPS filenames.
    If mode is None, alphabetically first match wins.
    If mode is recon/mixed and fd_order is None, highest order found wins.
    If periodic is None, defaults to non-periodic (False).
    If sp is None, the lowest sp found wins (sp=1 is the legacy default).

    Returns (dns_pkl_path, {cutoff_label: mps_pkl_path}, mode_used,
             fd_order_used, periodic_used, sp_used).
    """
    candidate_mps = sorted(glob.glob(
        os.path.join(root, "**", "mps_stats_*.pkl"), recursive=True))

    target_periodic = periodic if periodic is not None else False

    inferred_ts: Optional[str] = None
    inferred_mode: Optional[str] = None
    inferred_sp: Optional[int] = None
    available_orders: list = []
    available_sps: list = []
    for cand in candidate_mps:
        ts, cf_label, m, o, p, s = _extract_meta_from_mps(cand)
        if ts is None:
            continue
        # Legacy compat: pickles without an sp tag are interleaved -> sp=1
        s_effective = s if s is not None else 1
        if timestep is not None and ts != timestep:
            continue
        if mode is not None and m != mode:
            continue
        if fd_order is not None and o is not None and o != fd_order:
            continue
        if p != target_periodic:
            continue
        if sp is not None and s_effective != sp:
            continue
        if inferred_ts is None:
            inferred_ts = ts
            inferred_mode = m
            inferred_sp = s_effective
        if m == (mode or inferred_mode):
            if o is not None:
                available_orders.append(o)
            available_sps.append(s_effective)

    chosen_ts = timestep or inferred_ts
    chosen_mode = mode or inferred_mode

    if chosen_mode in ("recon", "mixed"):
        if fd_order is not None:
            chosen_order = fd_order
        elif available_orders:
            chosen_order = max(set(available_orders))
        else:
            chosen_order = None
    else:
        chosen_order = None

    if sp is not None:
        chosen_sp = sp
    elif available_sps:
        chosen_sp = min(set(available_sps))
    else:
        chosen_sp = inferred_sp   # could be None for legacy un-tagged pickles

    # Second pass: collect MPS pickles for the chosen settings
    mps: Dict[str, str] = {}
    if chosen_ts is not None and chosen_mode is not None:
        for cand in candidate_mps:
            ts, cf_label, m, o, p, s = _extract_meta_from_mps(cand)
            if ts != chosen_ts or m != chosen_mode:
                continue
            if chosen_mode in ("recon", "mixed") and chosen_order is not None and o != chosen_order:
                continue
            if p != target_periodic:
                continue
            # Legacy compat: pickles without an sp tag predate the sp
            # feature and were always interleaved (sp=1). Treat None as 1.
            s_effective = s if s is not None else 1
            if chosen_sp is not None and s_effective != chosen_sp:
                continue
            mps[cf_label] = cand

    # Third pass: find DNS pickle whose timestep matches
    dns_pkl: Optional[str] = None
    if chosen_ts is not None:
        patterns = [
            os.path.join(root, f"dns_{chosen_ts}", "cube",
                         f"dns_stats_{chosen_ts}.pkl"),
            os.path.join(root, "**", f"dns_stats_{chosen_ts}.pkl"),
        ]
        for pat in patterns:
            hits = sorted(glob.glob(pat, recursive=True))
            for h in hits:
                if "/cube/" in h or os.path.basename(h) == f"dns_stats_{chosen_ts}.pkl":
                    dns_pkl = h
                    break
            if dns_pkl is not None:
                break

    if chosen_ts is not None:
        msg = f"[discover] timestep={chosen_ts}, mode={chosen_mode}"
        if chosen_order is not None:
            msg += f", fd_order={chosen_order}"
        msg += f", periodic={target_periodic}"
        if chosen_sp is not None:
            msg += f", sp={chosen_sp}"
        print(msg)
    return dns_pkl, mps, chosen_mode, chosen_order, target_periodic, chosen_sp


# -----------------------------------------------------------------------------
# Combined overlay plots
# -----------------------------------------------------------------------------

def plot_mean_rms_overlay(
    dns, mps_dict, out_dir,
    single_sided: bool = True,
    ylim_overrides: Optional[dict] = None,
    les=None,
    peps=None,
):
    """Mean + RMS for Z, T, Y_CO, Y_CO2 with DNS and all MPS overlaid."""
    DEFAULT_YLIM = {
        "mixfrac": (0.0, 1.0),
        "T":       (0.0, 1300.0),
        "Y_CO":    (0.0, 0.6),
        "Y_CO2":   (0.0, 0.12),
    }
    if ylim_overrides:
        DEFAULT_YLIM = {**DEFAULT_YLIM, **ylim_overrides}
    variables = [
        ("mixfrac", r"$Z$"),
        ("T",       r"$T$ [K]"),
        ("Y_CO",    r"$Y_{CO}$"),
        ("Y_CO2",   r"$Y_{CO_2}$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    sources = [("DNS", dns)]
    if les is not None:
        sources.append(("LES", les))
    if peps is not None:
        sources.append(("PEPS", peps))
    sources += [(k, mps_dict[k]) for k in sorted(mps_dict)]

    for ax, (key, label) in zip(axes.ravel(), variables):
        # First pass: mean profiles
        for tag, results in sources:
            if key not in results["profiles"]:
                continue
            y = results["y_over_H"]
            d = results["profiles"][key]
            if single_sided:
                yh, mean_p = _fold_profile(y, d["mean"], "mean")
                _,  rms_p  = _fold_profile(y, d["rms"],  "rms")
            else:
                yh, mean_p, rms_p = y, d["mean"], d["rms"]
            style = dict(_STYLE[tag])
            ax.plot(yh, mean_p, label=f"{style.pop('label')}", **style)
            # RMS as the same color, lighter style
            rms_style = dict(_STYLE[tag])
            rms_style.pop("label")
            rms_style["linestyle"] = ":"
            rms_style["lw"] = max(rms_style.get("lw", 1.6) - 0.4, 0.8)
            ax.plot(yh, rms_p, **rms_style)
        ax.set_xlabel(r"$y/H$")
        ax.set_ylabel(label)
        if single_sided:
            ax.set_xlim(0, 5)
        if key in DEFAULT_YLIM:
            ax.set_ylim(*DEFAULT_YLIM[key])
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Mean (solid/dashed) and RMS (dotted) profiles  --  DNS vs MPS")
    fig.tight_layout()
    fname = ("fig06_mean_rms_overlay_single_sided.png" if single_sided
             else "fig06_mean_rms_overlay.png")
    path = os.path.join(out_dir, fname)
    fig.savefig(path)
    save_tikz_if_available(fig, os.path.splitext(path)[0] + '.tex')
    plt.close(fig)
    print(f"[plot] {path}")


def plot_chi_overlay(dns, mps_dict, out_dir, single_sided=True,
                     ylim=(0.0, 1000.0), les=None, peps=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    sources = [("DNS", dns)]
    if les is not None:
        sources.append(("LES", les))
    if peps is not None:
        sources.append(("PEPS", peps))
    sources += [(k, mps_dict[k]) for k in sorted(mps_dict)]
    for tag, results in sources:
        y = results["y_over_H"]
        chi = results["chi_profile"]
        if single_sided:
            yh, chi_p = _fold_profile(y, chi, "mean")
        else:
            yh, chi_p = y, chi
        ax.plot(yh, chi_p, **_STYLE[tag])
    ax.set_xlabel(r"$y/H$")
    ax.set_ylabel(r"$\overline{\chi}$  [1/s]")
    ax.set_title("Scalar dissipation rate  --  DNS vs MPS")
    if single_sided:
        ax.set_xlim(0, 5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    fname = ("fig09_chi_overlay_single_sided.png" if single_sided
             else "fig09_chi_overlay.png")
    path = os.path.join(out_dir, fname)
    fig.savefig(path)
    save_tikz_if_available(fig, os.path.splitext(path)[0] + '.tex')
    plt.close(fig)
    print(f"[plot] {path}")


def plot_conditional_T_overlay(dns, mps_dict, out_dir,
                               xlim=(0.0, 1.0), ylim=(500.0, 1400.0),
                               les=None, peps=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    sources = [("DNS", dns)]
    if les is not None:
        sources.append(("LES", les))
    if peps is not None:
        sources.append(("PEPS", peps))
    sources += [(k, mps_dict[k]) for k in sorted(mps_dict)]
    for tag, results in sources:
        d = results["conditional_T_on_Z"]
        ax.plot(d["psi_Z"], d["E_T"], **_STYLE[tag])
    ax.axvline(dns["z_st"], color="gray", ls=":", alpha=0.6,
               label=r"$Z_{st}$")
    ax.set_xlabel(r"$\psi_Z$")
    ax.set_ylabel(r"$E(T \mid Z = \psi_Z)$  [K]")
    ax.set_title("Conditional mean temperature  --  DNS vs MPS")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "fig10_conditional_T_overlay.png")
    fig.savefig(path)
    save_tikz_if_available(fig, os.path.splitext(path)[0] + '.tex')
    plt.close(fig)
    print(f"[plot] {path}")


def plot_pdf_Z_overlay(dns, mps_dict, out_dir,
                       xlim=(0.0, 1.0), ylim=(0.0, 6.0),
                       les=None, peps=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    sources = [("DNS", dns)]
    if les is not None:
        sources.append(("LES", les))
    if peps is not None:
        sources.append(("PEPS", peps))
    sources += [(k, mps_dict[k]) for k in sorted(mps_dict)]
    for tag, results in sources:
        d = results["pdf_mixfrac"]
        ax.plot(d["psi_Z"], d["pdf"], **_STYLE[tag])
    ax.set_xlabel(r"$\psi_Z$")
    ax.set_ylabel(r"PDF$(Z)$")
    ax.set_title("Mixture-fraction PDF near centerline  --  DNS vs MPS")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "fig12_pdf_Z_overlay.png")
    fig.savefig(path)
    save_tikz_if_available(fig, os.path.splitext(path)[0] + '.tex')
    plt.close(fig)
    print(f"[plot] {path}")


# -----------------------------------------------------------------------------
# Per-source plots (joint PDF, manifold)
# -----------------------------------------------------------------------------

def _plot_joint_pdf_one(results, out_path, source_label,
                        xlim=(0.0, 1.0), ylim=(0.0, 0.17),
                        levels=(5, 10, 20, 50, 100, 200)):
    d = results["joint_pdf_Z_YCO2"]
    Z_c = d["psi_Z"]; Y_c = d["psi_YCO2"]
    H = d["joint_pdf"].T
    # Guard against NaN/inf that can appear when the source field's
    # histogram range had zero in-range samples (division by 0 in
    # np.histogram2d density normalization).
    if not np.isfinite(H).any():
        print(f"[plot] joint PDF ({source_label}): all values NaN/inf; skipping. "
              f"(Likely no samples in fixed histogram range.)")
        return
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(H.max())
    plot_levels = [float(lv) for lv in levels if lv < vmax]
    if not plot_levels:
        print(f"[plot] joint PDF ({source_label}): all levels above "
              f"vmax={vmax:.3f}; skipping")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contour(Z_c, Y_c, H, levels=plot_levels,
                    cmap="winter", linewidths=2.0)
    ax.clabel(cs, inline=True, fontsize=9, fmt="%d")
    j_peak, i_peak = np.unravel_index(int(np.argmax(H)), H.shape)
    ax.plot(Z_c[i_peak], Y_c[j_peak], "ko", ms=8)
    ax.set_xlabel(r"$\psi_Z$")
    ax.set_ylabel(r"$\psi_{Y_{CO_2}}$")
    ax.set_title(f"Joint PDF of $(Z, Y_{{CO_2}})$  --  {source_label}")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(out_path)
    save_tikz_if_available(fig, os.path.splitext(out_path)[0] + '.tex')
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_joint_pdf_per_source(dns, mps_dict, out_dir, les=None, peps=None):
    _plot_joint_pdf_one(dns, os.path.join(out_dir, "fig13_joint_pdf_DNS.png"),
                        "DNS")
    if les is not None:
        _plot_joint_pdf_one(
            les, os.path.join(out_dir, "fig13_joint_pdf_LES.png"),
            "LES-FDF")
    if peps is not None:
        _plot_joint_pdf_one(
            peps, os.path.join(out_dir, "fig13_joint_pdf_PEPS.png"),
            "PEPS D=9")
    for cf_label, results in sorted(mps_dict.items()):
        _plot_joint_pdf_one(
            results,
            os.path.join(out_dir, f"fig13_joint_pdf_MPS_{cf_label}.png"),
            f"MPS {cf_label}",
        )


def _plot_manifold_one(results, out_path, source_label,
                       xlim=(0.0, 1.0), ylim=(0.0, 0.3), zlim=(0.0, 2.0),
                       T_range=(500.0, 1500.0),
                       elev=10.0, azim=-75.0, roll=0.0):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    d = results["manifold_scatter"]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        d["Z"], d["Y_O2"], d["Y_OH"] * 1000.0,
        c=d["T"], cmap="jet", s=1.0, alpha=1.0,
        vmin=T_range[0], vmax=T_range[1],
        edgecolors="none",
    )
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.10, label=r"$T$ [K]")
    ax.set_xlabel(r"$Z$")
    ax.set_ylabel(r"$Y_{O_2}$")
    ax.set_zlabel(r"$Y_{OH}\times 10^3$")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    if zlim is not None: ax.set_zlim(*zlim)
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("white")
        pane.set_edgecolor("0.6")
        pane.set_alpha(1.0)
    ax.set_title(f"Compositional manifold  --  {source_label}")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_manifold_per_source(dns, mps_dict, out_dir, les=None, peps=None):
    _plot_manifold_one(dns, os.path.join(out_dir, "fig14_manifold_DNS.png"),
                       "DNS")
    if les is not None:
        _plot_manifold_one(
            les, os.path.join(out_dir, "fig14_manifold_LES.png"),
            "LES-FDF")
    if peps is not None:
        _plot_manifold_one(
            peps, os.path.join(out_dir, "fig14_manifold_PEPS.png"),
            "PEPS D=9")
    for cf_label, results in sorted(mps_dict.items()):
        _plot_manifold_one(
            results,
            os.path.join(out_dir, f"fig14_manifold_MPS_{cf_label}.png"),
            f"MPS {cf_label}",
        )


# -----------------------------------------------------------------------------
# Error metrics summary
# -----------------------------------------------------------------------------

def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float("nan")


def write_error_summary(dns, mps_dict, out_dir):
    lines = ["DNS vs MPS error summary",
             "========================",
             "",
             "Relative L2 error of MPS profile vs DNS profile.",
             ""]
    header = f"{'statistic':<28s}" + "".join(
        f"{cf:>12s}" for cf in sorted(mps_dict))
    lines.append(header)
    lines.append("-" * len(header))

    profile_keys = []
    for v in ("mixfrac", "T", "Y_CO", "Y_CO2"):
        if v in dns["profiles"]:
            profile_keys.append((f"{v} mean", "profiles", v, "mean"))
            profile_keys.append((f"{v} rms",  "profiles", v, "rms"))
    for label, *_ in profile_keys:
        pass

    for label, top, sub, leaf in profile_keys:
        dns_prof = dns[top][sub][leaf]
        row = f"{label:<28s}"
        for cf in sorted(mps_dict):
            mps_prof = mps_dict[cf][top][sub][leaf]
            row += f"{_rel_l2(mps_prof, dns_prof):>12.3e}"
        lines.append(row)

    # chi profile
    row = f"{'chi profile':<28s}"
    for cf in sorted(mps_dict):
        row += f"{_rel_l2(mps_dict[cf]['chi_profile'], dns['chi_profile']):>12.3e}"
    lines.append(row)

    # Conditional T (handle nan-padded entries)
    row = f"{'E(T|Z) (where defined)':<28s}"
    dns_ET = dns["conditional_T_on_Z"]["E_T"]
    for cf in sorted(mps_dict):
        mps_ET = mps_dict[cf]["conditional_T_on_Z"]["E_T"]
        mask = np.isfinite(dns_ET) & np.isfinite(mps_ET)
        if not mask.any():
            row += f"{'n/a':>12s}"
        else:
            row += f"{_rel_l2(mps_ET[mask], dns_ET[mask]):>12.3e}"
    lines.append(row)

    # PDF Z
    row = f"{'PDF(Z) marginal':<28s}"
    for cf in sorted(mps_dict):
        row += f"{_rel_l2(mps_dict[cf]['pdf_mixfrac']['pdf'], dns['pdf_mixfrac']['pdf']):>12.3e}"
    lines.append(row)

    # Scalar diagnostics
    lines.append("")
    lines.append("Scalar statistics (DNS value, then MPS at each cutoff):")
    lines.append("")
    lines.append(f"{'delta_Z/(2H)':<28s}{dns['delta_Z_over_2H']:>12.4f}" +
                 "".join(f"{mps_dict[cf]['delta_Z_over_2H']:>12.4f}"
                         for cf in sorted(mps_dict)))
    lines.append(f"{'M_ext (vol-avg)':<28s}"
                 f"{dns['extinction']['volume_averaged_marker']:>12.3e}" +
                 "".join(f"{mps_dict[cf]['extinction']['volume_averaged_marker']:>12.3e}"
                         for cf in sorted(mps_dict)))
    lines.append(f"{'T on stoich [K]':<28s}"
                 f"{dns['extinction']['T_on_stoich_surface']:>12.2f}" +
                 "".join(f"{mps_dict[cf]['extinction']['T_on_stoich_surface']:>12.2f}"
                         for cf in sorted(mps_dict)))

    path = os.path.join(out_dir, "comparison_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[write] {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def _recompute_dns_chi_profile(
    dns: Dict, dns_data_dir: str, timestep: str,
    fd_order: int, periodic_xz_cube: bool,
) -> None:
    """Mutate dns in-place: replace dns['chi_profile'] with one computed
    from the DNS cube fields using the SAME FD operator that the MPS
    pipeline used in mixed/recon mode. Apples-to-apples comparison.

    Loads only the three fields needed (mixfrac, alpha, rho), computes
    chi via 2*alpha*|grad Z|^2, x-z averages.
    """
    from dns_stats import GridConfig, load_field, _xz_mean
    from derived_fields import compute_chi
    import os

    grid = GridConfig()
    print(f"[plot] Recomputing DNS chi with order={fd_order}, "
          f"periodic_xz={periodic_xz_cube}, from {dns_data_dir}",
          flush=True)
    t0 = time.time()
    Z = load_field(os.path.join(dns_data_dir, f"jet_mixfrac_{timestep}.dat"),
                   grid)
    alpha = load_field(os.path.join(dns_data_dir, f"jet_alpha_{timestep}.dat"),
                       grid)
    rho = load_field(os.path.join(dns_data_dir, f"jet_rho_{timestep}.dat"),
                     grid)
    print(f"       loaded 3 cube fields in {time.time()-t0:.1f}s", flush=True)

    chi_rec = compute_chi(
        Z, alpha, rho,
        grid.dx_m, grid.dy_m, grid.dz_m,
        alpha_is="thermal_diffusivity",
        order=fd_order,
        periodic_xz=periodic_xz_cube,
    )
    chi_prof_rec = _xz_mean(chi_rec)
    chi_prof_stored = dns["chi_profile"]
    dns["chi_profile_stored"] = chi_prof_stored   # keep original for ref
    dns["chi_profile"] = chi_prof_rec
    dns["chi_recomputed"] = True
    dns["chi_recomp_fd_order"] = fd_order
    dns["chi_recomp_periodic_xz_cube"] = periodic_xz_cube
    print(f"       reconstructed peak={chi_prof_rec.max():.1f},  "
          f"stored peak={chi_prof_stored.max():.1f}", flush=True)


def _filter_dns_chi_to_les_grid(
    dns: Dict, dns_data_dir: str, timestep: str,
    fd_order: int, periodic_xz_cube: bool,
    coarsen: int = 8,
) -> None:
    """Replace dns['chi_profile'] with chi computed on DNS box-filtered
    to the LES grid (coarsen-fold spatial average in each direction).

    This is the apples-to-apples comparison for LES-FDF validation:
       chi_filtered_DNS = 2 * alpha_filtered * |grad(Z_filtered)|^2
    on the coarsened DNS field, then x-z averaged for the profile.

    The result is a y-profile at the LES grid resolution (n_les_y points).
    To keep the comparison plotter's existing y-axis usable, we then
    *interpolate* the LES-grid profile back onto the DNS-cube y_over_H
    coordinate so the plot still has 512 y-samples.

    Stashes the unfiltered/un-filtered profiles under '_stored' / '_recomp'
    keys for reference.
    """
    from dns_stats import GridConfig, load_field, _xz_mean
    from derived_fields import compute_chi
    import os

    grid = GridConfig()
    print(f"[plot] Filtering DNS to LES grid (coarsen={coarsen}x) then "
          f"computing chi (order={fd_order}, periodic_xz={periodic_xz_cube})",
          flush=True)
    t0 = time.time()
    Z = load_field(os.path.join(dns_data_dir, f"jet_mixfrac_{timestep}.dat"),
                   grid).astype(np.float64)
    alpha = load_field(os.path.join(dns_data_dir,
                                     f"jet_alpha_{timestep}.dat"),
                       grid).astype(np.float64)
    rho = load_field(os.path.join(dns_data_dir, f"jet_rho_{timestep}.dat"),
                     grid).astype(np.float64)
    print(f"       loaded 3 cube fields in {time.time()-t0:.1f}s", flush=True)

    # Box-average to LES grid: shape (n, n, n) -> (n//c, n//c, n//c)
    n = Z.shape[0]
    if n % coarsen != 0:
        # Trim from each side so the cube is divisible by coarsen
        trim = n % coarsen
        a = trim // 2
        b = trim - a
        Z = Z[a:n - b, a:n - b, a:n - b]
        alpha = alpha[a:n - b, a:n - b, a:n - b]
        rho = rho[a:n - b, a:n - b, a:n - b]
        n = Z.shape[0]
        print(f"       trimmed cube to {n}^3 to be divisible by {coarsen}",
              flush=True)
    m = n // coarsen
    Z_les = Z.reshape(m, coarsen, m, coarsen, m, coarsen).mean(axis=(1, 3, 5))
    a_les = alpha.reshape(m, coarsen, m, coarsen, m, coarsen).mean(axis=(1, 3, 5))
    r_les = rho.reshape(m, coarsen, m, coarsen, m, coarsen).mean(axis=(1, 3, 5))
    del Z, alpha, rho

    # LES grid spacings (m): coarsen * DNS dx
    dx_les = coarsen * grid.dx_m
    dy_les = coarsen * grid.dy_m
    dz_les = coarsen * grid.dz_m

    chi_les = compute_chi(
        Z_les.astype(np.float32), a_les.astype(np.float32),
        r_les.astype(np.float32),
        dx_les, dy_les, dz_les,
        alpha_is="thermal_diffusivity",
        order=fd_order,
        periodic_xz=periodic_xz_cube,
    )
    # x-z average -> shape (m,)
    chi_prof_les = chi_les.mean(axis=(0, 2))

    # y-coord on LES grid (centered)
    y_les = (np.arange(m) - (m - 1) / 2.0) * (dy_les / grid.H)

    # Interpolate back onto DNS-cube y_over_H so the plot can use existing axis
    y_dns = dns["y_over_H"]
    chi_prof_on_dns = np.interp(y_dns, y_les, chi_prof_les)

    chi_prof_prev = dns.get("chi_profile")
    if chi_prof_prev is not None:
        dns["chi_profile_unfiltered"] = chi_prof_prev
    dns["chi_profile"] = chi_prof_on_dns
    dns["chi_profile_les_y"] = y_les
    dns["chi_profile_les_y_native"] = chi_prof_les
    dns["chi_filtered_to_les"] = True
    dns["chi_filter_coarsen"] = coarsen
    print(f"       filtered peak={chi_prof_les.max():.1f}, "
          f"previous (unfiltered) peak="
          f"{(chi_prof_prev.max() if chi_prof_prev is not None else float('nan')):.1f}",
          flush=True)


def main():
    args = sys.argv[1:]

    # Optional --timestep flag
    timestep = None
    if "--timestep" in args:
        i = args.index("--timestep")
        if i + 1 >= len(args):
            print("Error: --timestep requires an argument")
            sys.exit(1)
        timestep = args[i + 1]
        args.pop(i + 1)
        args.pop(i)

    # Optional --mode flag (stored | mixed | recon)
    mode = None
    if "--mode" in args:
        i = args.index("--mode")
        if i + 1 >= len(args):
            print("Error: --mode requires an argument")
            sys.exit(1)
        mode = args[i + 1]
        args.pop(i + 1)
        args.pop(i)
        if mode not in ("stored", "mixed", "recon"):
            print(f"Invalid --mode {mode}; "
                  "use 'stored', 'mixed', or 'recon'")
            sys.exit(1)

    # Optional --fd-order flag (relevant only in recon mode)
    fd_order = None
    if "--fd-order" in args:
        i = args.index("--fd-order")
        if i + 1 >= len(args):
            print("Error: --fd-order requires an argument")
            sys.exit(1)
        fd_order = int(args[i + 1])
        args.pop(i + 1)
        args.pop(i)
        if fd_order not in (2, 4, 6, 8):
            print(f"Invalid --fd-order {fd_order}; use 2, 4, 6, or 8")
            sys.exit(1)

    # Optional --periodic-xz flag (true | false; default false)
    periodic = None
    if "--periodic-xz" in args:
        i = args.index("--periodic-xz")
        if i + 1 >= len(args):
            print("Error: --periodic-xz requires an argument")
            sys.exit(1)
        periodic = args[i + 1].lower() in ("true", "1", "yes", "on")
        args.pop(i + 1)
        args.pop(i)

    # Optional --sp flag (1|2|3|4)
    sp = None
    if "--sp" in args:
        i = args.index("--sp")
        if i + 1 >= len(args):
            print("Error: --sp requires an argument")
            sys.exit(1)
        sp = int(args[i + 1])
        args.pop(i + 1)
        args.pop(i)
        if sp not in (1, 2, 3, 4):
            print(f"Invalid --sp {sp}; use 1, 2, 3, or 4")
            sys.exit(1)

    # Optional --dns-data-dir flag for on-the-fly DNS chi reconstruction.
    # If provided AND mode is mixed/recon, the comparison plotter loads
    # DNS mixfrac/alpha/rho from this directory, recomputes chi with the
    # SAME FD order and periodicity used by the MPS pipeline, and uses
    # that as the DNS chi profile. Apples-to-apples gradient operator.
    dns_data_dir = None
    if "--dns-data-dir" in args:
        i = args.index("--dns-data-dir")
        if i + 1 >= len(args):
            print("Error: --dns-data-dir requires an argument")
            sys.exit(1)
        dns_data_dir = args[i + 1]
        args.pop(i + 1)
        args.pop(i)

    # Optional --les-pkl flag for overlaying an LES stats pickle as a
    # third source. If 'auto', search the results root for les_stats_<TS>.pkl
    # under les_<TS>/. If a path is provided, use it directly.
    # If 'none' or omitted, no LES overlay.
    les_pkl_arg = None
    if "--les-pkl" in args:
        i = args.index("--les-pkl")
        if i + 1 >= len(args):
            print("Error: --les-pkl requires an argument")
            sys.exit(1)
        les_pkl_arg = args[i + 1]
        args.pop(i + 1)
        args.pop(i)

    # Optional --peps-pkl: mirrors --les-pkl but for the PEPS source.
    # Same semantics: 'auto' or omitted -> auto-discovery of
    # results_root/peps_<TS>/peps_stats_<TS>_*.pkl; 'none' -> skip;
    # a path -> use it directly.
    peps_pkl_arg = None
    if "--peps-pkl" in args:
        i = args.index("--peps-pkl")
        if i + 1 >= len(args):
            print("Error: --peps-pkl requires an argument")
            sys.exit(1)
        peps_pkl_arg = args[i + 1]
        args.pop(i + 1)
        args.pop(i)

    # Optional --exclude-cf: comma-separated cutoff labels to drop from
    # plots (e.g. "1e-2" to skip the noisiest MPS curve). Useful when one
    # cutoff dominates the y range and obscures the others.
    exclude_cfs: list = []
    if "--exclude-cf" in args:
        i = args.index("--exclude-cf")
        if i + 1 >= len(args):
            print("Error: --exclude-cf requires an argument")
            sys.exit(1)
        exclude_cfs = [c.strip() for c in args[i + 1].split(",")
                       if c.strip()]
        args.pop(i + 1)
        args.pop(i)

    # Per-source plots can be slow (3D manifold scatter especially).
    # Flags to skip them individually or together.
    skip_manifold = False
    if "--skip-manifold" in args:
        args.remove("--skip-manifold")
        skip_manifold = True
    skip_joint_pdf = False
    if "--skip-joint-pdf" in args:
        args.remove("--skip-joint-pdf")
        skip_joint_pdf = True
    if "--skip-per-source" in args:
        args.remove("--skip-per-source")
        skip_manifold = True
        skip_joint_pdf = True

    # Filter DNS to LES grid before computing chi profile. Used to match
    # LES-FDF validation conventions (Aitzhan 2022 figures). Coarsening is
    # DNS_n / LES_n in each direction; default 8 matches our 512->64.
    filter_dns_to_les = False
    coarsen = 8
    if "--filter-dns-to-les" in args:
        args.remove("--filter-dns-to-les")
        filter_dns_to_les = True
    if "--coarsen" in args:
        i = args.index("--coarsen")
        coarsen = int(args[i + 1])
        args.pop(i + 1); args.pop(i)

    if len(args) == 2:
        root, out_dir = args
        dns_pkl, mps_pkls, mode_used, fd_order_used, periodic_used, sp_used = (
            discover_results(root, timestep=timestep, mode=mode,
                             fd_order=fd_order, periodic=periodic, sp=sp))
        if dns_pkl is None:
            print(f"Could not find DNS pickle under {root}"
                  + (f" matching timestep {timestep}" if timestep else ""))
            sys.exit(1)
        if not mps_pkls:
            print(f"Could not find any MPS pickles under {root}"
                  + (f" matching timestep {timestep}" if timestep else "")
                  + (f" mode {mode}" if mode else "")
                  + (f" fd_order {fd_order}" if fd_order else "")
                  + (f" periodic={periodic}" if periodic is not None else "")
                  + (f" sp={sp}" if sp is not None else ""))
            sys.exit(1)
    elif len(args) == 5:
        dns_pkl, p1, p2, p3, out_dir = args
        mps_pkls = {"1e-3": p1, "1e-4": p2, "1e-5": p3}
        mode_used = mode
        fd_order_used = fd_order
        periodic_used = periodic
        sp_used = sp
    else:
        print("Usage:")
        print("  python plot_dns_mps_comparison.py <results_root> <out_dir> "
              "[--timestep TS] [--mode {stored,mixed,recon}] "
              "[--fd-order {2,4,6,8}] [--periodic-xz {true,false}] "
              "[--sp {1,2,3,4}] [--dns-data-dir DIR] [--les-pkl PATH|auto|none] "
              "[--peps-pkl PATH|auto|none] "
              "[--exclude-cf CF1,CF2,...]")
        print("  python plot_dns_mps_comparison.py <dns.pkl> "
              "<mps_1e-3.pkl> <mps_1e-4.pkl> <mps_1e-5.pkl> <out_dir>")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    print(f"DNS pickle : {dns_pkl}")
    for k, p in sorted(mps_pkls.items()):
        print(f"MPS {k:>4s}  : {p}")

    dns = _load(dns_pkl)
    mps_dict = {k: _load(v) for k, v in mps_pkls.items()}

    # Drop excluded cutoffs (e.g. --exclude-cf 1e-2 to hide noisy curves).
    if exclude_cfs:
        for cf in exclude_cfs:
            if cf in mps_dict:
                print(f"[plot] Excluding cutoff cf={cf} from plots")
                mps_dict.pop(cf)
            else:
                print(f"[plot] Note: --exclude-cf {cf} requested but cutoff "
                      f"not in loaded set; skipping")

    # ----- LES discovery / load -----
    # Default behavior when in the discover branch (len(args) == 2):
    # if no --les-pkl was provided, look for results_root/les_<TS>/les_stats_<TS>.pkl
    # and overlay if present. Pass --les-pkl=none to skip; --les-pkl=<path>
    # to point at a specific file.
    les = None
    les_pkl_path: Optional[str] = None
    if les_pkl_arg is not None and les_pkl_arg.lower() != "none":
        if les_pkl_arg.lower() == "auto":
            les_pkl_arg = None  # fall through to auto-discovery below
        else:
            les_pkl_path = les_pkl_arg

    if les_pkl_path is None and les_pkl_arg is None and len(sys.argv) >= 3 \
       and os.path.isdir(sys.argv[1]):
        ts_for_les = timestep or (dns.get("timestep") if isinstance(dns, dict) else None) or "0198"
        candidate = os.path.join(sys.argv[1], f"les_{ts_for_les}",
                                  f"les_stats_{ts_for_les}.pkl")
        if os.path.exists(candidate):
            les_pkl_path = candidate

    if les_pkl_path is not None:
        if os.path.exists(les_pkl_path):
            les = _load(les_pkl_path)
            print(f"LES pickle: {les_pkl_path}")
        else:
            print(f"[plot] WARN: --les-pkl path not found: {les_pkl_path}")

    # ----- PEPS discovery / load (mirrors LES) -----
    peps = None
    peps_pkl_path: Optional[str] = None
    if peps_pkl_arg is not None and peps_pkl_arg.lower() != "none":
        if peps_pkl_arg.lower() == "auto":
            peps_pkl_arg = None
        else:
            peps_pkl_path = peps_pkl_arg

    if peps_pkl_path is None and peps_pkl_arg is None and len(sys.argv) >= 3 \
       and os.path.isdir(sys.argv[1]):
        ts_for_peps = timestep or (dns.get("timestep") if isinstance(dns, dict) else None) or "0198"
        peps_dir = os.path.join(sys.argv[1], f"peps_{ts_for_peps}")
        if os.path.isdir(peps_dir):
            # Find the newest peps_stats_<TS>*.pkl in the directory.
            import glob as _glob
            cands = sorted(_glob.glob(
                os.path.join(peps_dir, f"peps_stats_{ts_for_peps}*.pkl")))
            if cands:
                # Prefer the one matching our fd_order if we can tell.
                preferred = None
                if fd_order_used is not None:
                    for c in cands:
                        if f"_o{fd_order_used}" in os.path.basename(c):
                            preferred = c
                            break
                peps_pkl_path = preferred or cands[-1]

    if peps_pkl_path is not None:
        if os.path.exists(peps_pkl_path):
            peps = _load(peps_pkl_path)
            print(f"PEPS pickle: {peps_pkl_path}")
        else:
            print(f"[plot] WARN: --peps-pkl path not found: {peps_pkl_path}")

    # Apples-to-apples chi: when MPS reconstructed chi with a non-trivial
    # operator (mixed or recon mode), recompute DNS chi the same way.
    if mode_used in ("mixed", "recon") and dns_data_dir is not None:
        ts_for_recomp = timestep or dns.get("timestep") or "0198"
        order_for_recomp = fd_order_used if fd_order_used is not None else 8
        per_for_recomp = periodic_used if periodic_used is not None else False
        _recompute_dns_chi_profile(
            dns, dns_data_dir, ts_for_recomp,
            fd_order=order_for_recomp,
            periodic_xz_cube=per_for_recomp,
        )
    elif mode_used in ("mixed", "recon") and dns_data_dir is None:
        print("[plot] WARN: mode is mixed/recon but --dns-data-dir not "
              "provided; DNS chi profile will be the stored S3D chi "
              "(asymmetric comparison vs MPS reconstructed chi).",
              flush=True)

    # LES-style chi: filter DNS to LES grid, then compute chi there.
    # Used together with the FDF-modeled LES chi for the standard
    # LES-FDF validation comparison.
    if filter_dns_to_les:
        if dns_data_dir is None:
            print("[plot] WARN: --filter-dns-to-les requires --dns-data-dir; "
                  "skipping filtered DNS chi.", flush=True)
        else:
            ts_for_filt = timestep or dns.get("timestep") or "0198"
            order_for_filt = fd_order_used if fd_order_used is not None else 8
            per_for_filt = periodic_used if periodic_used is not None else False
            _filter_dns_chi_to_les_grid(
                dns, dns_data_dir, ts_for_filt,
                fd_order=order_for_filt,
                periodic_xz_cube=per_for_filt,
                coarsen=coarsen,
            )

    _setup_mpl()
    plot_mean_rms_overlay(dns, mps_dict, out_dir, single_sided=True, les=les, peps=peps)
    plot_mean_rms_overlay(dns, mps_dict, out_dir, single_sided=False, les=les, peps=peps)
    plot_chi_overlay(dns, mps_dict, out_dir, single_sided=True, les=les, peps=peps)
    plot_chi_overlay(dns, mps_dict, out_dir, single_sided=False, les=les, peps=peps)
    plot_conditional_T_overlay(dns, mps_dict, out_dir, les=les, peps=peps)
    plot_pdf_Z_overlay(dns, mps_dict, out_dir, les=les, peps=peps)
    if not skip_joint_pdf:
        plot_joint_pdf_per_source(dns, mps_dict, out_dir, les=les, peps=peps)
    else:
        print("[plot] Skipping joint PDF per-source plots")
    if not skip_manifold:
        plot_manifold_per_source(dns, mps_dict, out_dir, les=les, peps=peps)
    else:
        print("[plot] Skipping manifold per-source plots")
    write_error_summary(dns, mps_dict, out_dir)


if __name__ == "__main__":
    main()
