"""
ke_energy_ratio.py
------------------
Box-filter the DNS velocity components u, v, w at several factors
(default 8,16,32,64), cache each filtered cube, and compute the KINETIC
ENERGY ratio of filtered ("ideal LES") to DNS -- combining all three
components into a single energy, then ratioing (NOT averaging three
per-component ratios).

Kinetic energy per cell:  KE = 1/2 (u^2 + v^2 + w^2)

Two scale-independent ratios (each quantity is a per-cell average on its own
grid, so the grid point-count cancels):

  R_mean = < |U_filt|^2 > / < |U_DNS|^2 >
           (resolved KE fraction; INCLUDES the mean flow. The 1/2 cancels.)

  R_var  = [Var(u_f)+Var(v_f)+Var(w_f)] / [Var(u)+Var(v)+Var(w)]
           (resolved TURBULENT KE fraction; mean removed per component.)

Here <|U|^2> = < u^2 + v^2 + w^2 > = (1/N) sum (u^2+v^2+w^2), and Var(.) is the
per-component variance (numpy divides by N). Summing components BEFORE the
ratio is what makes this a kinetic-energy ratio rather than three separate
scalar ratios.

Usage
-----
    python ke_energy_ratio.py \
        --dns-data-dir /ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198 \
        --timestep 0198 \
        --factors 8,16,32,64 \
        --components u,v,w \
        --cache-dir /ix/pgivi/moe32/Aidyn_DNS/stats/filtered_0198 \
        [--out QC4PDE/ke_energy_ratio.txt]

DNS files are expected as jet_<comp>_<timestep>.dat (e.g. jet_u_0198.dat).
Filtered cubes are cached/reused as filtered_<comp>_<timestep>_f<factor>.npy
(same names make_filtered_cube.py / mixfrac_energy_ratio.py use, so existing
caches are reused).

Run on a login node with the python module loaded (needs numpy + dns_stats).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np


def _parse(args, name, default=None, cast=str):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        args.pop(i + 1); args.pop(i)
        return cast(v) if cast is not str else v
    return default


def block_average(cube: np.ndarray, factor: int) -> np.ndarray:
    """Box-filter a cube by averaging each factor^3 block."""
    nx, ny, nz = cube.shape
    cx, cy, cz = (nx // factor) * factor, (ny // factor) * factor, \
                 (nz // factor) * factor
    c = cube[:cx, :cy, :cz]
    c = c.reshape(cx // factor, factor,
                  cy // factor, factor,
                  cz // factor, factor)
    return c.mean(axis=(1, 3, 5))


def main():
    args = sys.argv[1:]
    dns_data_dir = _parse(args, "--dns-data-dir",
                          default="/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198")
    timestep = _parse(args, "--timestep", default="0198")
    factors_str = _parse(args, "--factors", default="8,16,32,64")
    comps_str = _parse(args, "--components", default="u,v,w")
    cache_dir = _parse(args, "--cache-dir",
                       default=f"/ix/pgivi/moe32/Aidyn_DNS/stats/"
                               f"filtered_{timestep}")
    out_txt = _parse(args, "--out", default="QC4PDE/ke_energy_ratio.txt")
    factors = [int(x) for x in factors_str.split(",") if x.strip()]
    comps = [c.strip() for c in comps_str.split(",") if c.strip()]

    os.makedirs(cache_dir, exist_ok=True)
    print(f"[ke] components   = {comps}")
    print(f"[ke] dns_data_dir = {dns_data_dir}")
    print(f"[ke] cache_dir    = {cache_dir}")
    print(f"[ke] factors      = {factors}")

    from dns_stats import GridConfig, load_field
    grid = GridConfig()
    t0 = time.time()

    # ----- DNS: load each component, accumulate KE and per-component var --
    # <|U|^2>_DNS = mean over cells of sum_c u_c^2
    # TKE_DNS     = sum_c Var(u_c)
    dns_meanU2 = 0.0     # < sum_c u_c^2 >  (per-cell)
    dns_tke = 0.0        # sum_c Var(u_c)
    n_dns = None
    comp_dns = {}        # keep DNS components in memory to filter below
    for c in comps:
        p = os.path.join(dns_data_dir, f"jet_{c}_{timestep}.dat")
        print(f"[ke] loading DNS {c} from {p} ...", flush=True)
        arr = np.asarray(load_field(p, grid), dtype=np.float64)
        comp_dns[c] = arr
        n_dns = arr.size
        dns_meanU2 += float(np.mean(arr * arr))
        dns_tke += float(np.var(arr))
        print(f"    {c}: shape={arr.shape}  <{c}^2>={np.mean(arr*arr):.6e}  "
              f"Var={np.var(arr):.6e}", flush=True)
    print(f"[ke] DNS:  <|U|^2>={dns_meanU2:.6e}  TKE(sum Var)={dns_tke:.6e}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ----- Filtered: for each factor, filter each component, accumulate ---
    rows = []
    for fac in factors:
        f_meanU2 = 0.0
        f_tke = 0.0
        n_f = None
        for c in comps:
            cpath = os.path.join(cache_dir,
                                 f"filtered_{c}_{timestep}_f{fac}.npy")
            if os.path.exists(cpath):
                cf = np.load(cpath).astype(np.float64)
                src = "cached"
            else:
                cf = block_average(comp_dns[c], fac)
                np.save(cpath, cf.astype(np.float32))
                src = "saved"
            n_f = cf.size
            f_meanU2 += float(np.mean(cf * cf))
            f_tke += float(np.var(cf))
        R_mean = f_meanU2 / dns_meanU2
        R_var = f_tke / dns_tke
        g = round(n_f ** (1.0 / 3.0))
        rows.append((fac, g, n_f, f_meanU2, f_tke, R_mean, R_var))
        warn = "  [tiny grid]" if g <= 4 else ""
        print(f"    factor {fac:>3d} ({g}^3, {src}): "
              f"<|U|^2>={f_meanU2:.6e}  TKE={f_tke:.6e}  "
              f"R_mean={R_mean:.4f}  R_var={R_var:.4f}{warn}", flush=True)

    # ----- table ---------------------------------------------------------
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w") as fh:
        fh.write("# Kinetic-energy ratio filtered/DNS (scale-independent)\n")
        fh.write(f"# timestep={timestep}  components={','.join(comps)}\n")
        fh.write(f"# DNS: N={n_dns}  <|U|^2>={dns_meanU2:.8e}  "
                 f"TKE(sum_c Var)={dns_tke:.8e}\n")
        fh.write("# R_mean = <|U|^2>_filt / <|U|^2>_DNS   "
                 "(resolved KE fraction, includes mean flow)\n")
        fh.write("# R_var  = sum_c Var(u_c)_filt / sum_c Var(u_c)_DNS   "
                 "(resolved turbulent KE fraction)\n")
        fh.write("#\n")
        fh.write("# factor  grid    Npts        <|U|^2>_filt      "
                 "TKE_filt          R_mean        R_var\n")
        for (fac, g, n_f, fmu2, ftke, R_mean, R_var) in rows:
            fh.write(f"{fac:>7d}  {g:>4d}^3  {n_f:>10d}  {fmu2:>15.6e}  "
                     f"{ftke:>15.6e}  {R_mean:>12.6f}  {R_var:>12.6f}\n")
    print(f"\n[ke] wrote {out_txt}")

    print("\n[ke] SUMMARY (kinetic-energy ratios, filtered/DNS):")
    print("  factor    R_mean(KE)     R_var(TKE)")
    for (fac, g, n_f, fmu2, ftke, R_mean, R_var) in rows:
        note = "   <- coarse" if g <= 4 else ""
        print(f"   {fac:>3d}      {R_mean:>10.4f}     {R_var:>10.4f}{note}")
    print(f"\n[ke] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
