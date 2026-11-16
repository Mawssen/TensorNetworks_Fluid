"""
mixfrac_energy_ratio.py
-----------------------
Filter the 512^3 DNS mixture-fraction cube by block-averaging at several
factors (default 8,16,32,64,128 -> 64^3,32^3,16^3,8^3,4^3), cache each, and
compute SCALE-INDEPENDENT mixture-fraction energy ratios of filtered vs DNS.

Two ratios are reported, both per-cell (so the point-count difference between
grids cancels and the numbers are ~1, decreasing with filter width):

    R_mean = <Z^2>_filt / <Z^2>_DNS     (energy density; raw 2nd moment / N)
    R_var  = Var_filt   / Var_DNS       (resolved fluctuation-energy fraction)

Here <Z^2> = (sum Z^2)/N and Var = <Z^2> - <Z>^2, each computed on the field's
OWN grid. Dividing by N is what makes the ratios scale-independent: a coarse
cube has fewer cells, but the per-cell energy density is comparable to DNS,
so the ratio reflects genuine energy loss from filtering rather than the
change in cell count.

Usage
-----
    python mixfrac_energy_ratio.py \
        --dns-data-dir /ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198 \
        --timestep 0198 \
        --factors 8,16,32,64,128 \
        --cache-dir /ix/pgivi/moe32/Aidyn_DNS/stats/filtered_0198 \
        [--out QC4PDE/mixfrac_energy_ratio.txt]

Note: factor 128 gives a 4^3 cube (only 64 cells) -- its statistics are very
coarse and flagged as such. Run on a login node with the venv loaded.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np


# Self-contained arg parser (inlined so this script depends only on numpy and
# does NOT import plot_qc4pde, which would pull in matplotlib unnecessarily).
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


def energy_sum(cube: np.ndarray) -> float:
    """Raw second moment: sum of Z^2 over all cells."""
    c = cube.astype(np.float64)
    return float(np.sum(c * c))


def main():
    args = sys.argv[1:]
    dns_data_dir = _parse(args, "--dns-data-dir",
                          default="/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198")
    timestep = _parse(args, "--timestep", default="0198")
    # --field: which DNS field to filter. Loads jet_<field>_<ts>.dat and
    # caches filtered_<field>_<ts>_f<factor>.npy. Default mixfrac (back-compat).
    # Examples: mixfrac, u, v, w.
    field = _parse(args, "--field", default="mixfrac")
    factors_str = _parse(args, "--factors", default="8,16,32,64,128")
    cache_dir = _parse(args, "--cache-dir",
                       default=f"/ix/pgivi/moe32/Aidyn_DNS/stats/"
                               f"filtered_{timestep}")
    # Default output name carries the field so u/v/w runs don't overwrite.
    out_txt = _parse(args, "--out",
                     default=f"QC4PDE/{field}_energy_ratio.txt")
    factors = [int(f) for f in factors_str.split(",") if f.strip()]

    os.makedirs(cache_dir, exist_ok=True)
    print(f"[energy] field        = {field}")
    print(f"[energy] dns_data_dir = {dns_data_dir}")
    print(f"[energy] cache_dir    = {cache_dir}")
    print(f"[energy] factors      = {factors}")

    from dns_stats import GridConfig, load_field
    grid = GridConfig()

    t0 = time.time()
    dns_path = os.path.join(dns_data_dir, f"jet_{field}_{timestep}.dat")
    print(f"[energy] loading DNS {field} cube (512^3) from {dns_path} ...",
          flush=True)
    Z_dns = np.asarray(load_field(dns_path, grid), dtype=np.float64)
    n_dns = Z_dns.size
    E_dns = energy_sum(Z_dns)
    meanZ2_dns = E_dns / n_dns
    var_dns = float(np.var(Z_dns))
    print(f"    DNS: shape={Z_dns.shape}  E_sum={E_dns:.6e}  "
          f"<f^2>={meanZ2_dns:.6e}  Var={var_dns:.6e}", flush=True)

    rows = []
    for f in factors:
        # cache (or load) the filtered cube
        cpath = os.path.join(cache_dir,
                             f"filtered_{field}_{timestep}_f{f}.npy")
        if os.path.exists(cpath):
            Zf = np.load(cpath).astype(np.float64)
            src = "cached"
        else:
            Zf = block_average(Z_dns, f)
            np.save(cpath, Zf.astype(np.float32))
            src = "computed+saved"
        n_f = Zf.size
        E_f = energy_sum(Zf)
        meanZ2_f = E_f / n_f
        var_f = float(np.var(Zf))

        R_mean = meanZ2_f / meanZ2_dns      # per-cell energy density ratio
        R_var = var_f / var_dns             # fluctuation-energy fraction
        rows.append((f, Zf.shape[0], n_f, meanZ2_f, R_mean, R_var))
        warn = "  [tiny grid: coarse statistics]" if Zf.shape[0] <= 4 else ""
        print(f"    factor {f:>3d} ({Zf.shape[0]}^3, {src}): "
              f"<Z^2>={meanZ2_f:.6e}  R_mean={R_mean:.4f}  "
              f"R_var={R_var:.4f}{warn}", flush=True)

    # ---- write a small table --------------------------------------------
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w") as fh:
        fh.write(f"# {field} energy ratio (scale-independent)\n")
        fh.write(f"# timestep={timestep}  field={field}\n")
        fh.write(f"# DNS: N={n_dns}  <f^2>={meanZ2_dns:.8e}  "
                 f"Var={var_dns:.8e}\n")
        fh.write("# R_mean = <f^2>_filt / <f^2>_DNS  (per-cell energy "
                 "density; point-count cancels)\n")
        fh.write("# R_var  = Var_filt  / Var_DNS      (resolved fluctuation "
                 "energy fraction)\n")
        fh.write("#\n")
        fh.write("# factor  grid    Npts        <Z^2>_filt        "
                 "R_mean        R_var\n")
        for (f, g, n_f, mZ2, R_mean, R_var) in rows:
            fh.write(f"{f:>7d}  {g:>4d}^3  {n_f:>10d}  {mZ2:>15.6e}  "
                     f"{R_mean:>12.6f}  {R_var:>12.6f}\n")
    print(f"\n[energy] wrote {out_txt}")
    print(f"[energy] done in {time.time()-t0:.1f}s")

    print("\n[energy] SUMMARY (scale-independent energy fractions):")
    print("  factor    R_mean(<Z^2>)    R_var(fluct)")
    for (f, g, n_f, mZ2, R_mean, R_var) in rows:
        note = "   <- 4^3, coarse" if g <= 4 else ""
        print(f"   {f:>3d}       {R_mean:>10.4f}      {R_var:>10.4f}{note}")


if __name__ == "__main__":
    main()
