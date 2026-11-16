"""
plot_sp_overlay.py
------------------
Cross-sp overlay: for a fixed (mode, fd_order, periodic, cutoff), put
DNS plus all available sp values on the same plot.

Goal: visualize how the MPS ordering scheme (interleaved / sequential /
comb1 / combn) affects truncation quality at a given cutoff and chi
mode.

Layouts
-------
For each cutoff that has multiple sp pickles, generates four overlay plots
(in the chosen output dir):
    chi_profile_sp_<cf>.png    -- x-z averaged chi(y/H), DNS + each sp
    pdf_Z_sp_<cf>.png          -- PDF(Z) marginal
    cond_T_sp_<cf>.png         -- E(T|Z)
    mean_rms_Z_sp_<cf>.png     -- Z mean+RMS profile

Usage
-----
python plot_sp_overlay.py <results_root> <out_dir>
    [--timestep TS] [--mode {stored,mixed,recon}]
    [--fd-order {2,4,6,8}] [--periodic-xz {true,false}]
    [--cutoff LABEL]    # if given, only that cf; else all that have >=2 sps

Example
-------
python plot_sp_overlay.py results results/sp_overlay_0198_stored \
       --mode stored --timestep 0198
"""

from __future__ import annotations

import os
import sys
import glob
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Colors for sp values
_SP_STYLE = {
    1: dict(color="#e67e22", linestyle="--", lw=1.6, label="sp=1 (il)"),
    2: dict(color="#27ae60", linestyle="--", lw=1.6, label="sp=2 (seq)"),
    3: dict(color="#2980b9", linestyle="--", lw=1.6, label="sp=3 (comb1)"),
    4: dict(color="#8e44ad", linestyle="--", lw=1.6, label="sp=4 (combn)"),
}
_DNS_STYLE = dict(color="black", linestyle="-", lw=2.0, label="DNS", zorder=10)


def _setup_mpl():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def _fold_profile(y, prof, kind="mean"):
    n = len(y)
    j0 = int(np.argmin(np.abs(y)))
    n_keep = min(n - j0, j0 + 1)
    p_pos = prof[j0:j0 + n_keep]
    p_neg = prof[j0 - np.arange(n_keep)]
    if kind == "rms":
        return y[j0:j0 + n_keep], np.sqrt(0.5 * (p_pos**2 + p_neg**2))
    return y[j0:j0 + n_keep], 0.5 * (p_pos + p_neg)


def _extract_meta_from_mps(path):
    """Same parser as in plot_dns_mps_comparison.py"""
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
    fd_order = None; sp = None; periodic = False
    for tok in parts[3:]:
        if tok.startswith("o") and tok[1:].isdigit():
            fd_order = int(tok[1:])
        elif tok.startswith("sp") and tok[2:].isdigit():
            sp = int(tok[2:])
        elif tok == "p":
            periodic = True
    return ts, cf_label, mode, fd_order, periodic, sp


def discover_sp_pickles(root, timestep, mode, fd_order, periodic):
    """Find all sp pickles matching the (timestep, mode, fd_order, periodic)
    tuple. Returns nested dict: {cf_label: {sp: pkl_path}} and the DNS pkl.
    """
    candidate = sorted(glob.glob(
        os.path.join(root, "**", "mps_stats_*.pkl"), recursive=True))
    by_cf_sp: Dict[str, Dict[int, str]] = {}
    for cand in candidate:
        ts, cf_label, m, o, p, s = _extract_meta_from_mps(cand)
        if ts is None or s is None:
            continue
        if timestep is not None and ts != timestep:
            continue
        if mode is not None and m != mode:
            continue
        if mode in ("recon", "mixed") and fd_order is not None and o != fd_order:
            continue
        if periodic is not None and p != periodic:
            continue
        by_cf_sp.setdefault(cf_label, {})[s] = cand

    # Find matching DNS pkl
    dns_pkl = None
    if timestep:
        patterns = [
            os.path.join(root, f"dns_{timestep}", "cube",
                         f"dns_stats_{timestep}.pkl"),
            os.path.join(root, "**", f"dns_stats_{timestep}.pkl"),
        ]
        for pat in patterns:
            hits = sorted(glob.glob(pat, recursive=True))
            for h in hits:
                if "/cube/" in h or os.path.basename(h) == f"dns_stats_{timestep}.pkl":
                    dns_pkl = h
                    break
            if dns_pkl is not None:
                break
    return dns_pkl, by_cf_sp


def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def _plot_chi_profile(dns, sps, out_path, cf_label,
                      single_sided=True, ylim=(0.0, 1000.0)):
    fig, ax = plt.subplots(figsize=(7, 5))
    y = dns["y_over_H"]
    chi = dns["chi_profile"]
    if single_sided:
        yh, p = _fold_profile(y, chi)
    else:
        yh, p = y, chi
    ax.plot(yh, p, **_DNS_STYLE)
    for sp_val in sorted(sps):
        results = sps[sp_val]
        chi = results["chi_profile"]
        if single_sided:
            _, p = _fold_profile(y, chi)
        else:
            p = chi
        ax.plot(yh, p, **_SP_STYLE[sp_val])
    ax.set_xlabel(r"$y/H$")
    ax.set_ylabel(r"$\overline{\chi}$  [1/s]")
    ax.set_title(f"Scalar dissipation rate -- DNS vs MPS sp variants  "
                 f"(cf={cf_label})")
    if single_sided:
        ax.set_xlim(0, 5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")


def _plot_pdf_Z(dns, sps, out_path, cf_label,
                xlim=(0.0, 1.0), ylim=(0.0, 6.0)):
    fig, ax = plt.subplots(figsize=(7, 5))
    d = dns["pdf_mixfrac"]
    ax.plot(d["psi_Z"], d["pdf"], **_DNS_STYLE)
    for sp_val in sorted(sps):
        d = sps[sp_val]["pdf_mixfrac"]
        ax.plot(d["psi_Z"], d["pdf"], **_SP_STYLE[sp_val])
    ax.set_xlabel(r"$\psi_Z$")
    ax.set_ylabel(r"PDF$(Z)$")
    ax.set_title(f"PDF(Z) -- DNS vs MPS sp variants  (cf={cf_label})")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")


def _plot_cond_T(dns, sps, out_path, cf_label,
                 xlim=(0.0, 1.0), ylim=(500.0, 1400.0)):
    fig, ax = plt.subplots(figsize=(7, 5))
    d = dns["conditional_T_on_Z"]
    ax.plot(d["psi_Z"], d["E_T"], **_DNS_STYLE)
    for sp_val in sorted(sps):
        d = sps[sp_val]["conditional_T_on_Z"]
        ax.plot(d["psi_Z"], d["E_T"], **_SP_STYLE[sp_val])
    ax.axvline(dns["z_st"], color="gray", ls=":", alpha=0.6,
               label=r"$Z_{st}$")
    ax.set_xlabel(r"$\psi_Z$")
    ax.set_ylabel(r"$E(T \mid Z = \psi_Z)$  [K]")
    ax.set_title(f"E(T|Z) -- DNS vs MPS sp variants  (cf={cf_label})")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")


def _plot_Z_meanrms(dns, sps, out_path, cf_label, single_sided=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    y = dns["y_over_H"]
    if "mixfrac" not in dns["profiles"]:
        plt.close(fig)
        return
    d = dns["profiles"]["mixfrac"]
    if single_sided:
        yh, mean_p = _fold_profile(y, d["mean"], "mean")
        _,  rms_p  = _fold_profile(y, d["rms"],  "rms")
    else:
        yh, mean_p, rms_p = y, d["mean"], d["rms"]
    style = dict(_DNS_STYLE); style.pop("label")
    ax.plot(yh, mean_p, label="DNS mean", **style)
    rms_style = dict(style); rms_style["linestyle"] = ":"
    ax.plot(yh, rms_p, label="DNS rms", **rms_style)

    for sp_val in sorted(sps):
        if "mixfrac" not in sps[sp_val]["profiles"]:
            continue
        d = sps[sp_val]["profiles"]["mixfrac"]
        if single_sided:
            _, mean_p = _fold_profile(y, d["mean"], "mean")
            _, rms_p  = _fold_profile(y, d["rms"],  "rms")
        else:
            mean_p, rms_p = d["mean"], d["rms"]
        style = dict(_SP_STYLE[sp_val])
        lbl = style.pop("label")
        ax.plot(yh, mean_p, label=f"{lbl} mean", **style)
        rs = dict(style)
        rs["linestyle"] = ":"; rs["lw"] = max(rs.get("lw", 1.6) - 0.4, 0.8)
        ax.plot(yh, rms_p, label=f"{lbl} rms", **rs)

    ax.set_xlabel(r"$y/H$")
    ax.set_ylabel(r"$Z$")
    ax.set_title(f"Z mean (solid/dashed) and RMS (dotted)  "
                 f"-- DNS vs MPS sp variants  (cf={cf_label})")
    if single_sided:
        ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python plot_sp_overlay.py <results_root> <out_dir>")
        print("    [--timestep TS] [--mode {stored,mixed,recon}]")
        print("    [--fd-order {2,4,6,8}] [--periodic-xz {true,false}]")
        print("    [--cutoff LABEL]")
        sys.exit(1)

    root = args[0]; out_dir = args[1]
    rest = args[2:]

    def _pop_flag(name, default=None, cast=str):
        if name in rest:
            i = rest.index(name)
            v = rest[i + 1]
            rest.pop(i + 1); rest.pop(i)
            return cast(v)
        return default

    timestep = _pop_flag("--timestep")
    mode = _pop_flag("--mode")
    fd_order = _pop_flag("--fd-order", cast=int)
    periodic_str = _pop_flag("--periodic-xz", default="false")
    periodic = periodic_str.lower() in ("true", "1", "yes", "on")
    only_cutoff = _pop_flag("--cutoff")

    if mode is not None and mode not in ("stored", "mixed", "recon"):
        print(f"Invalid --mode {mode}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    _setup_mpl()

    print(f"[discover] root={root}  ts={timestep}  mode={mode}  "
          f"fd_order={fd_order}  periodic={periodic}")
    dns_pkl, by_cf_sp = discover_sp_pickles(root, timestep, mode,
                                             fd_order, periodic)
    if dns_pkl is None:
        print(f"DNS pickle not found under {root}")
        sys.exit(1)
    if not by_cf_sp:
        print("No MPS sp pickles found.")
        sys.exit(1)

    dns = _load(dns_pkl)
    print(f"DNS: {dns_pkl}")

    for cf_label in sorted(by_cf_sp):
        if only_cutoff is not None and cf_label != only_cutoff:
            continue
        sp_pkls = by_cf_sp[cf_label]
        if len(sp_pkls) < 2:
            print(f"  cf={cf_label}: only {len(sp_pkls)} sp variant; skipping")
            continue
        print(f"\n  cf={cf_label}:")
        sps = {}
        for sp_val, p in sorted(sp_pkls.items()):
            print(f"    sp={sp_val}: {p}")
            sps[sp_val] = _load(p)

        _plot_chi_profile(dns, sps,
                          os.path.join(out_dir, f"chi_profile_sp_{cf_label}.png"),
                          cf_label)
        _plot_pdf_Z(dns, sps,
                    os.path.join(out_dir, f"pdf_Z_sp_{cf_label}.png"),
                    cf_label)
        _plot_cond_T(dns, sps,
                     os.path.join(out_dir, f"cond_T_sp_{cf_label}.png"),
                     cf_label)
        _plot_Z_meanrms(dns, sps,
                        os.path.join(out_dir, f"mean_rms_Z_sp_{cf_label}.png"),
                        cf_label)


if __name__ == "__main__":
    main()
