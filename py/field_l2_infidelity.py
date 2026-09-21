"""
field_l2_infidelity.py
----------------------
Full-field (512^3) reconstruction-error metrics of MPS (chi=93) and PEPS (D=9)
against DNS, for the mixture fraction Z. Complements the profile-based L2
errors in Table I with pointwise, whole-cube measures.

Three metrics per source (B = DNS is always the reference):

  L2_sym  = ||A - B|| / (||A|| + ||B||)     (bounded [0,1], symmetric in scale)
  L2_rel  = ||A - B|| / ||B||               (standard relative L2 error)
  infid   = 1 - |<A,B>|^2 / (||A||^2 ||B||^2)   (normalized-overlap infidelity;
                                                 scale/sign independent)

||.|| is the L2 / Frobenius norm over all cells: sqrt(sum X^2).
<A,B> = sum A*B.

Field preparation (matches the infidelity / gradient pipeline):
  * MPS chi=93 : used as reconstructed (original scale), no sign/scale fix.
  * PEPS D=9   : sign-fixed (correlation) and L2-rescaled to DNS, via
                 _peps_scale_and_sign_cube -- the same correction used
                 elsewhere so all metrics are consistent.

Loaders are reused from plot_gradient_stats.py (full-cube DNS/MPS/PEPS), which
in turn use the project's dns_stats / mps_io / peps_io.

Usage
-----
    python field_l2_infidelity.py \
        --dns-data-dir /ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198 \
        --peps-dir /ix/pgivi/moe32/Aidyn_DNS/stats/truncated_0198/PEPS \
        --mps-root /ix/pgivi/moe32/Aidyn_DNS/stats \
        --timestep 0198 --sp 1 \
        [--out QC4PDE/field_l2_infidelity.txt]

Run on a login node with the python module loaded (numpy + project modules).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_gradient_stats import (          # noqa: E402
    _load_dns_cube, _load_mps_cube, _load_peps_cube,
    _peps_scale_and_sign_cube,
)


def _parse(args, name, default=None, cast=str):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        args.pop(i + 1); args.pop(i)
        return cast(v) if cast is not str else v
    return default


def l2_norm(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(x.astype(np.float64) ** 2)))


def metrics(A: np.ndarray, B: np.ndarray) -> dict:
    """All three metrics for field A vs reference B."""
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    nA = l2_norm(A)
    nB = l2_norm(B)
    diff = l2_norm(A - B)
    inner = float(np.sum(A * B))
    l2_sym = diff / (nA + nB) if (nA + nB) > 0 else float("nan")
    l2_rel = diff / nB if nB > 0 else float("nan")
    # normalized-overlap infidelity (scale/sign independent)
    fidelity = (inner ** 2) / (nA ** 2 * nB ** 2) if (nA > 0 and nB > 0) \
        else float("nan")
    infid = 1.0 - fidelity
    return {"norm_A": nA, "norm_B": nB, "norm_diff": diff,
            "L2_sym": l2_sym, "L2_rel": l2_rel, "infid": infid}


def main():
    args = sys.argv[1:]
    dns_data_dir = _parse(args, "--dns-data-dir",
                          default="/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198")
    peps_dir = _parse(args, "--peps-dir",
                      default="/ix/pgivi/moe32/Aidyn_DNS/stats/"
                              "truncated_0198/PEPS")
    mps_root = _parse(args, "--mps-root",
                      default="/ix/pgivi/moe32/Aidyn_DNS/stats")
    timestep = _parse(args, "--timestep", default="0198")
    sp = int(_parse(args, "--sp", default="1"))
    out_txt = _parse(args, "--out", default="QC4PDE/field_l2_infidelity.txt")

    from dns_stats import GridConfig  # noqa: F401  (ensures env is present)

    t0 = time.time()
    print("[l2] loading DNS cube (reference B)...", flush=True)
    dns = _load_dns_cube(dns_data_dir, timestep)
    print(f"    DNS shape={dns.shape}  ||DNS||={l2_norm(dns):.6e} "
          f"({time.time()-t0:.1f}s)", flush=True)

    from mps_io import ordering_for_sp
    ordering = ordering_for_sp(sp)
    mps_dir = os.path.join(mps_root, f"truncated_{timestep}", f"sp{sp}")
    if not os.path.isdir(mps_dir):
        mps_dir = os.path.join(mps_root, f"truncated_{timestep}_sp{sp}")
    if not os.path.isdir(mps_dir) and sp == 1:
        mps_dir = os.path.join(mps_root, f"truncated_{timestep}")

    results = {}

    # MPS chi=93 (reconstructed, no sign/scale correction)
    try:
        print("[l2] loading MPS chi=93 cube...", flush=True)
        mps = _load_mps_cube(mps_dir, timestep, "chi93", ordering=ordering)
        results["MPS chi=93"] = metrics(mps, dns)
        print(f"    done ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"    MPS chi=93 failed: {e}", flush=True)

    # PEPS D=9 (sign-fixed + L2-rescaled to DNS, like infidelity pipeline)
    try:
        print("[l2] loading PEPS D=9 cube (sign+scale corrected)...",
              flush=True)
        peps_raw = _load_peps_cube(peps_dir)
        peps = _peps_scale_and_sign_cube(peps_raw, dns)
        results["PEPS D=9"] = metrics(peps, dns)
        print(f"    done ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"    PEPS D=9 failed: {e}", flush=True)

    # ---- report ---------------------------------------------------------
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w") as fh:
        fh.write("# Full-field (512^3) reconstruction error vs DNS, "
                 "mixture fraction Z\n")
        fh.write(f"# timestep={timestep} sp={sp}\n")
        fh.write("# L2_sym = ||A-B||/(||A||+||B||)\n")
        fh.write("# L2_rel = ||A-B||/||B||       (B = DNS)\n")
        fh.write("# infid  = 1 - |<A,B>|^2/(||A||^2 ||B||^2)\n")
        fh.write("#\n")
        fh.write(f"# {'source':<12s}  {'L2_sym':>10s}  {'L2_rel':>10s}  "
                 f"{'infidelity':>12s}  {'||A-B||':>12s}  {'||A||':>12s}\n")
        for key in ("MPS chi=93", "PEPS D=9"):
            if key not in results:
                continue
            r = results[key]
            fh.write(f"  {key:<12s}  {r['L2_sym']:>10.6f}  "
                     f"{r['L2_rel']:>10.6f}  {r['infid']:>12.6e}  "
                     f"{r['norm_diff']:>12.6e}  {r['norm_A']:>12.6e}\n")

    print(f"\n[l2] wrote {out_txt}")
    print("\n[l2] SUMMARY (B = DNS reference):")
    print(f"  {'source':<12s}  {'L2_sym':>10s}  {'L2_rel':>10s}  "
          f"{'infidelity':>12s}")
    for key in ("MPS chi=93", "PEPS D=9"):
        if key not in results:
            continue
        r = results[key]
        print(f"  {key:<12s}  {r['L2_sym']:>10.6f}  {r['L2_rel']:>10.6f}  "
              f"{r['infid']:>12.6e}")
    print(f"\n[l2] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
