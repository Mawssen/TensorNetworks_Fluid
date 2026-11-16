"""
compute_qc4pde_metrics.py
-------------------------
Compute two families of metrics for QC4PDE paper reporting:

1. 3D field infidelity of Z between DNS and each of: MPS chi=93, MPS 4.1e-4, PEPS.
   Skipped for LES (LES grid is 64^3 vs DNS 512^3; not meaningful).

   Fidelity (real fields, standard inner-product form):
       F = |<psi|phi>|^2 / (||psi||^2 * ||phi||^2)
     Infidelity: I = 1 - F.

2. Mean and RMS profile errors between DNS and each source (LES included).
   Reported metrics:
     - rel_L2:   ||src - dns||_2 / ||dns||_2
     - RMS_diff: sqrt(mean((src - dns)^2))
     - peak_err: max|src - dns| / max|dns|

Only the mixture fraction Z is analysed. Writes a plain-text summary to
QC4PDE/metrics_Z.txt.

Usage
-----
    python compute_qc4pde_metrics.py [results_root=results] [out_dir=QC4PDE]
        [--timestep 0198]
        [--dns-data-dir /path/to/jet_0198]
        [--peps-dir /path/to/PEPS]
        [--mps-root /path/to/parent/of/truncated_<TS>/sp<SP>]
        [--sp 1]
        [--fd-order 4]
"""

from __future__ import annotations

import os
import sys
import time
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np


def _parse(args, name, default=None, cast=str):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        args.pop(i + 1); args.pop(i)
        return cast(v) if cast is not str else v
    return default


def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Field loading helpers (mirroring plot_qc4pde.py conventions).
# -----------------------------------------------------------------------------

def _load_dns_Z(dns_data_dir, timestep):
    from dns_stats import GridConfig, load_field
    grid = GridConfig()
    return load_field(os.path.join(dns_data_dir,
                                     f"jet_mixfrac_{timestep}.dat"),
                       grid).astype(np.float32)


def _load_mps_Z(mps_data_dir, timestep, cutoff_str, ordering):
    from mps_io import load_mps_field, _mps_filename
    path = os.path.join(mps_data_dir,
                         _mps_filename("mixfrac", timestep, cutoff_str,
                                        ordering=ordering))
    return load_mps_field(path).astype(np.float32)


def _load_peps_Z(peps_dir, dns_Z):
    """Load PEPS mixfrac and apply the same L2-norm rescale + sign fix
    that run_peps_stats.py applied, so the field is on the DNS scale."""
    from peps_io import load_peps_field
    path = os.path.join(peps_dir, "mixfrac_wf_D=9_periodic.mat")
    Z = load_peps_field(path)   # (512, 512, 512), float32
    Zf64 = Z.astype(np.float64)
    dnsf64 = dns_Z.astype(np.float64)
    if float(np.sum(Zf64 * dnsf64)) < 0:
        Zf64 = -Zf64
    n_dns = float(np.sqrt(np.sum(dnsf64 * dnsf64)))
    n_peps = float(np.sqrt(np.sum(Zf64 * Zf64)))
    if n_peps > 0:
        Zf64 = Zf64 * (n_dns / n_peps)
    return Zf64.astype(np.float32)   # unclipped (fidelity uses raw values)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def field_fidelity(dns_arr: np.ndarray, other_arr: np.ndarray) -> Dict:
    """Standard quantum-state fidelity for real vectors:
        F = |<dns, other>|^2 / (||dns||^2 * ||other||^2)
    Returns dict with fidelity, infidelity, dot product, both norms, cos(theta).
    """
    a = dns_arr.astype(np.float64).ravel()
    b = other_arr.astype(np.float64).ravel()
    dot = float(np.sum(a * b))
    na = float(np.sqrt(np.sum(a * a)))
    nb = float(np.sqrt(np.sum(b * b)))
    denom = na * nb
    if denom <= 0.0:
        return dict(fidelity=float("nan"), infidelity=float("nan"),
                     dot=dot, norm_dns=na, norm_other=nb, cos_theta=float("nan"))
    cos_theta = dot / denom
    fidelity = cos_theta ** 2
    return dict(fidelity=fidelity, infidelity=1.0 - fidelity,
                 dot=dot, norm_dns=na, norm_other=nb, cos_theta=cos_theta)


def profile_errors(dns_prof: np.ndarray, other_prof: np.ndarray) -> Dict:
    """Errors between two 1D profiles."""
    a = np.asarray(dns_prof, dtype=np.float64)
    b = np.asarray(other_prof, dtype=np.float64)
    if a.shape != b.shape:
        # If shapes differ (e.g. LES on 64-point y_over_H vs DNS on 512),
        # interpolate other_prof to DNS y grid. But we can't do that here
        # because we don't have y_over_H arrays; expect caller to align.
        return dict(rel_L2=float("nan"), rms_diff=float("nan"),
                     peak_err=float("nan"), note=f"shape mismatch {a.shape} vs {b.shape}")
    diff = a - b
    dns_norm = float(np.sqrt(np.sum(a * a)))
    err_norm = float(np.sqrt(np.sum(diff * diff)))
    rel_l2 = err_norm / dns_norm if dns_norm > 0 else float("nan")
    rms_diff = float(np.sqrt(np.mean(diff * diff)))
    dns_peak = float(np.max(np.abs(a)))
    peak_err = (float(np.max(np.abs(diff))) / dns_peak
                 if dns_peak > 0 else float("nan"))
    return dict(rel_L2=rel_l2, rms_diff=rms_diff, peak_err=peak_err)


def profile_errors_interp(y_dns, dns_prof, y_other, other_prof) -> Dict:
    """Errors for two profiles with (possibly) different y grids.

    Interpolate other_prof onto y_dns, then compute standard metrics.
    """
    a = np.asarray(dns_prof, dtype=np.float64)
    b_on_y_dns = np.interp(np.asarray(y_dns, dtype=np.float64),
                             np.asarray(y_other, dtype=np.float64),
                             np.asarray(other_prof, dtype=np.float64))
    return profile_errors(a, b_on_y_dns)


# -----------------------------------------------------------------------------
# Pickle discovery (mirrors plot_qc4pde.py)
# -----------------------------------------------------------------------------

def _discover_pickles(results_root, timestep, fd_order, sp):
    import glob
    paths = {}

    dns_p = os.path.join(results_root, f"dns_{timestep}", "cube",
                          f"dns_stats_{timestep}.pkl")
    if os.path.exists(dns_p):
        paths["DNS"] = dns_p

    les_p = os.path.join(results_root, f"les_{timestep}",
                          f"les_stats_{timestep}.pkl")
    if os.path.exists(les_p):
        paths["LES"] = les_p

    peps_cands = sorted(glob.glob(
        os.path.join(results_root, f"peps_{timestep}",
                      f"peps_stats_{timestep}*.pkl")))
    if peps_cands:
        pref = [c for c in peps_cands
                if f"_o{fd_order}" in os.path.basename(c)]
        paths["PEPS"] = pref[-1] if pref else peps_cands[-1]

    for label in ("chi93", "4.1e-4"):
        cand = os.path.join(
            results_root, f"mps_{timestep}_sp{sp}", label,
            f"mps_stats_{timestep}_cf{label}_mixed_o{fd_order}_sp{sp}.pkl")
        if os.path.exists(cand):
            paths[label] = cand

    return paths


# -----------------------------------------------------------------------------
# Report writers
# -----------------------------------------------------------------------------

_SOURCE_LABEL = {
    "LES":    "LES-FDF",
    "chi93":  "MPS chi=93",
    "4.1e-4": "MPS 4.1e-4",
    "PEPS":   "PEPS D=9",
}
_SOURCE_ORDER = ["LES", "chi93", "4.1e-4", "PEPS"]


def _write_text_summary(out_path, timestep, sp, fd_order, periodic_xz,
                          fidelity_rows, mean_rows, rms_rows):
    with open(out_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("QC4PDE metrics -- Z field\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"timestep       : {timestep}\n")
        f.write(f"sp             : {sp}\n")
        f.write(f"fd_order       : {fd_order}\n")
        f.write(f"periodic_xz    : {periodic_xz}\n\n")

        # --- 3D field infidelity ---
        f.write("-" * 80 + "\n")
        f.write("3D Z-field infidelity (I = 1 - |<a,b>|^2 / (||a|| ||b||)^2)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'source':>14s}  {'fidelity':>14s}  {'infidelity':>14s}  "
                 f"{'cos(theta)':>12s}  {'||src||':>12s}\n")
        for src, m in fidelity_rows:
            if m is None:
                f.write(f"{_SOURCE_LABEL[src]:>14s}  "
                         f"{'-- skipped --':>44s}\n")
                continue
            f.write(f"{_SOURCE_LABEL[src]:>14s}  "
                     f"{m['fidelity']:>14.6e}  "
                     f"{m['infidelity']:>14.6e}  "
                     f"{m['cos_theta']:>12.6f}  "
                     f"{m['norm_other']:>12.4e}\n")
        f.write("\n")

        # --- Mean profile errors ---
        f.write("-" * 80 + "\n")
        f.write("Mean(Z) profile errors vs DNS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'source':>14s}  {'rel_L2':>14s}  {'RMS_diff':>14s}  "
                 f"{'peak_err':>14s}\n")
        for src, m in mean_rows:
            if m is None:
                f.write(f"{_SOURCE_LABEL[src]:>14s}  {'-- unavailable --':>44s}\n")
                continue
            note = f"  ({m['note']})" if "note" in m else ""
            f.write(f"{_SOURCE_LABEL[src]:>14s}  "
                     f"{m['rel_L2']:>14.6e}  "
                     f"{m['rms_diff']:>14.6e}  "
                     f"{m['peak_err']:>14.6e}{note}\n")
        f.write("\n")

        # --- RMS profile errors ---
        f.write("-" * 80 + "\n")
        f.write("RMS(Z) profile errors vs DNS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'source':>14s}  {'rel_L2':>14s}  {'RMS_diff':>14s}  "
                 f"{'peak_err':>14s}\n")
        for src, m in rms_rows:
            if m is None:
                f.write(f"{_SOURCE_LABEL[src]:>14s}  {'-- unavailable --':>44s}\n")
                continue
            note = f"  ({m['note']})" if "note" in m else ""
            f.write(f"{_SOURCE_LABEL[src]:>14s}  "
                     f"{m['rel_L2']:>14.6e}  "
                     f"{m['rms_diff']:>14.6e}  "
                     f"{m['peak_err']:>14.6e}{note}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Notes:\n")
        f.write("  - Field infidelity uses raw (rescaled) PEPS Z; sign is\n"
                 "    fixed by correlation with DNS. LES is skipped for the\n"
                 "    field metric because LES is on a 64^3 grid vs DNS 512^3.\n")
        f.write("  - Profile errors: LES profile is on 64 y-samples; interp\n"
                 "    to the DNS y_over_H grid before comparing.\n")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    results_root = args[0] if args and not args[0].startswith("--") else "results"
    if args and not args[0].startswith("--"):
        args = args[1:]
    out_dir = args[0] if args and not args[0].startswith("--") else "QC4PDE"
    if args and not args[0].startswith("--"):
        args = args[1:]

    timestep = _parse(args, "--timestep", default="0198")
    dns_data_dir = _parse(args, "--dns-data-dir",
                            default="/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198")
    peps_dir = _parse(args, "--peps-dir",
                        default="/ix/pgivi/moe32/Aidyn_DNS/stats/truncated_0198/PEPS")
    mps_root = _parse(args, "--mps-root",
                        default="/ix/pgivi/moe32/Aidyn_DNS/stats")
    fd_order = int(_parse(args, "--fd-order", default="4"))
    periodic_str = _parse(args, "--periodic-xz", default="false")
    periodic_xz = periodic_str.lower() in ("true", "1", "yes", "on")
    sp = int(_parse(args, "--sp", default="1"))
    skip_field = "--skip-field" in args
    if skip_field:
        args.remove("--skip-field")

    os.makedirs(out_dir, exist_ok=True)
    print(f"[metrics] results_root  = {results_root}")
    print(f"[metrics] out_dir       = {out_dir}")
    print(f"[metrics] timestep      = {timestep}, sp={sp}, fd_order={fd_order}")

    # ---------- Load pickles for profile errors --------------------------
    paths = _discover_pickles(results_root, timestep, fd_order, sp)
    print(f"[metrics] found pickles for: {sorted(paths.keys())}")
    if "DNS" not in paths:
        print("[metrics] ERROR: DNS pickle not found; nothing to compare against")
        sys.exit(1)
    pkls = {k: _load(p) for k, p in paths.items()}
    dns_pkl = pkls["DNS"]
    y_dns = np.asarray(dns_pkl["y_over_H"], dtype=np.float64)
    dns_mean = np.asarray(dns_pkl["profiles"]["mixfrac"]["mean"], dtype=np.float64)
    dns_rms = np.asarray(dns_pkl["profiles"]["mixfrac"]["rms"], dtype=np.float64)

    mean_rows = []
    rms_rows = []
    for src in _SOURCE_ORDER:
        if src not in pkls:
            mean_rows.append((src, None))
            rms_rows.append((src, None))
            continue
        r = pkls[src]
        y_other = np.asarray(r["y_over_H"], dtype=np.float64)
        m_prof = np.asarray(r["profiles"]["mixfrac"]["mean"], dtype=np.float64)
        s_prof = np.asarray(r["profiles"]["mixfrac"]["rms"], dtype=np.float64)
        if len(y_other) != len(y_dns):
            # interp to DNS grid
            m_err = profile_errors_interp(y_dns, dns_mean, y_other, m_prof)
            s_err = profile_errors_interp(y_dns, dns_rms, y_other, s_prof)
            m_err["note"] = f"interp {len(y_other)}->{len(y_dns)}"
            s_err["note"] = f"interp {len(y_other)}->{len(y_dns)}"
        else:
            m_err = profile_errors(dns_mean, m_prof)
            s_err = profile_errors(dns_rms, s_prof)
        mean_rows.append((src, m_err))
        rms_rows.append((src, s_err))
        print(f"[metrics] profile errors for {src}:  "
              f"mean rel_L2={m_err['rel_L2']:.3e}   "
              f"rms rel_L2={s_err['rel_L2']:.3e}")

    # ---------- 3D field infidelity --------------------------------------
    fidelity_rows: List[Tuple[str, Optional[Dict]]] = []
    if skip_field:
        print("[metrics] Skipping 3D field infidelity (--skip-field)")
        for src in _SOURCE_ORDER:
            fidelity_rows.append((src, None))
    else:
        print("[metrics] Loading DNS Z (this may take ~30-60s)...")
        t0 = time.time()
        dns_Z = _load_dns_Z(dns_data_dir, timestep)
        print(f"    DNS Z loaded  shape={dns_Z.shape}  ({time.time()-t0:.1f}s)")
        # ||DNS||
        n_dns = float(np.sqrt(np.sum(dns_Z.astype(np.float64) ** 2)))

        # LES is skipped (different grid)
        fidelity_rows.append(("LES", None))
        print("[metrics] LES field infidelity: skipped (different grid resolution)")

        # MPS cases
        from mps_io import ordering_for_sp
        ordering = ordering_for_sp(sp)
        # locate mps data dir
        for cand in [
            os.path.join(mps_root, f"truncated_{timestep}", f"sp{sp}"),
            os.path.join(mps_root, f"truncated_{timestep}_sp{sp}"),
            os.path.join(mps_root, f"truncated_{timestep}") if sp == 1 else "",
        ]:
            if cand and os.path.isdir(cand):
                mps_data_dir = cand
                break
        else:
            mps_data_dir = os.path.join(mps_root, f"truncated_{timestep}")

        for src, cutoff_str in [("chi93", "chi93"), ("4.1e-4", "0.00041")]:
            if src not in pkls:
                fidelity_rows.append((src, None))
                continue
            t1 = time.time()
            print(f"[metrics] Loading MPS {src} Z...")
            try:
                mps_Z = _load_mps_Z(mps_data_dir, timestep, cutoff_str, ordering)
                m = field_fidelity(dns_Z, mps_Z)
                fidelity_rows.append((src, m))
                del mps_Z
                print(f"    {src}:  fidelity={m['fidelity']:.6e}, "
                      f"infidelity={m['infidelity']:.6e}  "
                      f"({time.time()-t1:.1f}s)")
            except Exception as e:
                print(f"    {src}: failed: {e}")
                fidelity_rows.append((src, None))

        # PEPS (with rescale + sign fix)
        if "PEPS" in pkls:
            t1 = time.time()
            print(f"[metrics] Loading PEPS Z (rescale + sign fix)...")
            try:
                peps_Z = _load_peps_Z(peps_dir, dns_Z)
                m = field_fidelity(dns_Z, peps_Z)
                fidelity_rows.append(("PEPS", m))
                del peps_Z
                print(f"    PEPS:  fidelity={m['fidelity']:.6e}, "
                      f"infidelity={m['infidelity']:.6e}  "
                      f"({time.time()-t1:.1f}s)")
            except Exception as e:
                print(f"    PEPS: failed: {e}")
                fidelity_rows.append(("PEPS", None))
        else:
            fidelity_rows.append(("PEPS", None))

    # ---------- Write summary --------------------------------------------
    out_path = os.path.join(out_dir, "metrics_Z.txt")
    _write_text_summary(out_path, timestep, sp, fd_order, periodic_xz,
                          fidelity_rows, mean_rows, rms_rows)
    print(f"\n[metrics] wrote {out_path}")

    # Also echo the report to stdout for immediate reading
    print("\n" + "=" * 80)
    with open(out_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
