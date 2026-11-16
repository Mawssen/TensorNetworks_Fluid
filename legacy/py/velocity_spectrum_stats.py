"""
velocity_spectrum_stats.py
--------------------------
Compare DNS, MPS (chi=9), and LES-32dx (box-filtered DNS, factor 32 -> 16^3)
for the jet-flame velocity field:

  (1) 3D energy spectrum E(k)  -- via the provided calculate_spectrum(u,v,w),
      plotted log-log with the 1.6 eps^2/3 k^-5/3 inertial reference line,
      styled like the attached figure.
  (2) Velocity statistics vs DNS: per-component mean / RMS / PDF of u,v,w,
      and the kinetic-energy ratios (R_mean, R_var) already used elsewhere.

Grids
-----
  DNS      : 512^3   (jet_u/v/w_<ts>.dat)
  MPS chi=9: 512^3   (reconstruction; --mps-dir)
  LES-32dx : 16^3    (box-filter DNS by 32; cached filtered_<c>_<ts>_f32.npy)

The spectrum is computed on each field's own grid, so DNS/MPS reach high k
while LES-16^3 only reaches k~8 -- its curve correctly stops there, showing it
carries no fine-scale energy. Stats are per-cell / normalized so they compare
across grids (raw sums would just reflect the point-count difference).

Usage
-----
    python velocity_spectrum_stats.py \
        --dns-data-dir /ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198 \
        --mps-dir      /path/to/mps_chi9_recon \
        --cache-dir    /ix/pgivi/moe32/Aidyn_DNS/stats/filtered_0198 \
        --timestep 0198 \
        [--mps-pattern "jet_{comp}_{timestep}_chi9.dat"] \
        [--out-dir QC4PDE]

The --mps-pattern tells the script how MPS component files are named; edit to
match your reconstruction output. Run with the python module loaded.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 3D energy spectrum -- ported verbatim from the user's calculate_spectrum.
# ---------------------------------------------------------------------------
def calculate_spectrum(u, v, w):
    L = 2 * np.pi
    dim = u.shape[0]
    uu_fft = np.fft.fftn(u)
    vv_fft = np.fft.fftn(v)
    ww_fft = np.fft.fftn(w)
    uu_fft = (np.abs(uu_fft) / dim**3)**2
    vv_fft = (np.abs(vv_fft) / dim**3)**2
    ww_fft = (np.abs(ww_fft) / dim**3)**2

    k_end = int(dim / 2)
    rx = np.array(range(dim)) - dim / 2 + 1
    rx = np.roll(rx, int(dim / 2) + 1)

    r = np.zeros((rx.shape[0], rx.shape[0], rx.shape[0]))
    for i in range(rx.shape[0]):
        for j in range(rx.shape[0]):
            r[i, j, :] = rx[i]**2 + rx[j]**2 + rx[:]**2
    r = np.sqrt(r)

    dx = 2 * np.pi / L
    k = (np.array(range(k_end)) + 1) * dx

    bins = np.zeros((k.shape[0] + 1))
    for N in range(k_end):
        bins[N] = 0 if N == 0 else (k[N] + k[N - 1]) / 2
    bins[-1] = k[-1]

    inds = np.digitize(r * dx, bins, right=True)
    spectrum = np.zeros((k.shape[0]))
    bin_counter = np.zeros((k.shape[0]))
    for N in range(k_end):
        spectrum[N] = (np.sum(uu_fft[inds == N + 1])
                       + np.sum(vv_fft[inds == N + 1])
                       + np.sum(ww_fft[inds == N + 1]))
        bin_counter[N] = np.count_nonzero(inds == N + 1)

    spectrum = spectrum * 2 * np.pi * (k**2) / (bin_counter * dx**3)
    return k, spectrum


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def block_average(cube, factor):
    nx, ny, nz = cube.shape
    cx, cy, cz = (nx // factor) * factor, (ny // factor) * factor, \
                 (nz // factor) * factor
    c = cube[:cx, :cy, :cz].reshape(cx // factor, factor,
                                    cy // factor, factor,
                                    cz // factor, factor)
    return c.mean(axis=(1, 3, 5))


def _parse(args, name, default=None, cast=str):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        args.pop(i + 1); args.pop(i)
        return cast(v) if cast is not str else v
    return default


# ---------------------------------------------------------------------------
# style (matches the attached figure)
# ---------------------------------------------------------------------------
_STYLE = {
    "DNS":  dict(color="black",   linestyle="-",  lw=2.0, label="DNS"),
    "MPS":  dict(color="#9b59b6", linestyle="--", lw=2.2,
                 label=r"MPS ($\chi=9$)"),
    "LES":  dict(color="#16a085", linestyle=":",  lw=2.4,
                 label=r"LES ($32\Delta x$)"),
}
_ORDER = ["DNS", "MPS", "LES"]


def _enable_latex():
    try:
        import io
        plt.rcParams["text.usetex"] = True
        fig = plt.figure(); fig.text(0.5, 0.5, r"$E(k)$")
        fig.savefig(io.BytesIO(), format="png", dpi=50); plt.close(fig)
        has = True
    except Exception:
        has = False
    plt.rcParams.update({
        "text.usetex": bool(has), "font.family": "serif",
        "mathtext.fontset": "cm", "axes.labelsize": 18,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
    })
    if not has:
        print("[spec] LaTeX unavailable; mathtext fallback.")


def plot_u_contour(dns, mps, les, out_dir, grid, comp="u", titles=None,
                   tag=""):
    """Row of x-y mid-z slices of a velocity component, DNS/MPS/LES.

    Shared colour scale (from DNS) and shared physical extent (from DNS grid),
    so the coarse LES panel lines up in (x/H, y/H) with the fine panels and
    shows larger blocks -- same convention as the mixfrac contour figure.
    """
    panels = [("DNS", dns)]
    if mps is not None:
        panels.append(("MPS", mps))
    panels.append(("LES", les))
    n = len(panels)

    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6),
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()

    # mid-z slice of each field's component
    def midslice(fd):
        a = fd[comp]
        return a[:, :, a.shape[2] // 2]

    dns_slice = midslice(dns)
    vmin = float(dns_slice.min())
    vmax = float(dns_slice.max())

    # shared physical extent from the DNS grid, in units of H
    H = grid.H
    dx, dy = grid.dx_m, grid.dy_m
    nxD, nyD = dns_slice.shape
    x_half = (nxD / 2.0) * dx / H
    y_half = (nyD / 2.0) * dy / H
    extent = [-x_half, x_half, -y_half, y_half]

    im = None
    titles = titles or {"DNS": "DNS", "MPS": "MPS", "LES": "LES"}
    for ax, (key, fd) in zip(axes, panels):
        arr = midslice(fd)
        im = ax.imshow(arr.T, origin="lower", cmap="inferno",
                       vmin=vmin, vmax=vmax, extent=extent,
                       aspect="equal", interpolation="nearest")
        ax.set_title(titles[key], fontsize=18)
        ax.set_xlabel(r"$x/H$", fontsize=16)
        if key == panels[0][0]:
            ax.set_ylabel(r"$y/H$", fontsize=16)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=12)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(rf"${comp}$", fontsize=16)
    png = os.path.join(out_dir, f"contour_{comp}_row{tag}.png")
    fig.savefig(png, dpi=200)
    fig.savefig(png.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"[spec] wrote {png} (+ .pdf)")


def main():
    args = sys.argv[1:]
    dns_dir = _parse(args, "--dns-data-dir",
                     default="/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198")
    mps_dir = _parse(args, "--mps-dir", default=None)
    cache_dir = _parse(args, "--cache-dir",
                       default="/ix/pgivi/moe32/Aidyn_DNS/stats/filtered_0198")
    timestep = _parse(args, "--timestep", default="0198")
    mps_pat = _parse(args, "--mps-pattern",
                     default="truncated_jet_{comp}_{timestep}_il_chi9.mat")
    mps_var = _parse(args, "--mps-var", default="u")
    # chi of the MPS reconstruction (label only) and the LES filter factor.
    # NOTE the mapping: factor = 512/les_grid, so
    #   chi=9  <-> LES 16^3 -> factor 32 (32*dx width)
    #   chi=29 <-> LES 32^3 -> factor 16 (16*dx width)
    mps_chi = _parse(args, "--mps-chi", default="9")
    les_factor = int(_parse(args, "--les-factor", default="32"))
    out_dir = _parse(args, "--out-dir", default="QC4PDE")
    comps = ["u", "v", "w"]

    # Labels reflect the actual chi / filter factor for this run.
    run_tag = f"_chi{mps_chi}_f{les_factor}"
    _STYLE["MPS"]["label"] = rf"MPS ($\chi={mps_chi}$)"
    _STYLE["LES"]["label"] = rf"LES (${les_factor}\Delta x$)"
    _TITLES = {"DNS": "DNS",
               "MPS": rf"MPS ($\chi={mps_chi}$)",
               "LES": rf"LES (${les_factor}\Delta x$)"}

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    _enable_latex()

    from dns_stats import GridConfig, load_field
    grid = GridConfig()
    t0 = time.time()

    # ---- load DNS components (512^3) ------------------------------------
    print("[spec] loading DNS u,v,w ...", flush=True)
    dns = {}
    for c in comps:
        dns[c] = np.asarray(
            load_field(os.path.join(dns_dir, f"jet_{c}_{timestep}.dat"), grid),
            dtype=np.float64)
    print(f"    DNS {dns['u'].shape} ({time.time()-t0:.1f}s)", flush=True)

    # ---- MPS chi=9 (512^3) ----------------------------------------------
    # mps_calc.jl's save_to_mat writes a .mat with variable `u` holding the
    # already-cropped 512^3 reconstruction. Read it directly (NOT via
    # load_field, which expects the full 864x1008x576 raw .dat and crops).
    def _read_mat_cube(path, var):
        # MAT.jl (save_to_mat) writes HDF5-based .mat for large arrays, so try
        # h5py FIRST -- it needs only numpy, avoiding the scipy/numpy ABI
        # conflict on some clusters (scipy compiled vs numpy 1.x but numpy 2.x
        # installed -> "_ARRAY_API not found"). Fall back to scipy for old v5.
        try:
            import h5py
            with h5py.File(path, "r") as f:
                key = var if var in f else list(f.keys())[0]
                arr = np.asarray(f[key], dtype=np.float64)
            # HDF5/MATLAB v7.3 stores in Fortran (column-major) order, which
            # h5py reads transposed vs the (x,y,z) we want -> reverse axes.
            return np.ascontiguousarray(arr.T)
        except (OSError, ImportError):
            # not an HDF5 file (old v5 .mat) -> use scipy
            from scipy.io import loadmat
            m = loadmat(path)
            if var in m:
                return np.asarray(m[var], dtype=np.float64)
            keys = [k for k in m if not k.startswith("__")]
            return np.asarray(m[keys[0]], dtype=np.float64)

    mps = None
    if mps_dir:
        try:
            mps = {}
            for c in comps:
                p = os.path.join(mps_dir,
                                 mps_pat.format(comp=c, timestep=timestep))
                arr = _read_mat_cube(p, mps_var)
                mps[c] = arr
            print(f"    MPS chi={mps_chi} {mps['u'].shape} loaded from .mat",
                  flush=True)
        except Exception as e:
            print(f"    MPS load failed: {e}", flush=True)
            mps = None
    else:
        print("    (no --mps-dir given; skipping MPS)", flush=True)

    # ---- LES-32dx (box-filter DNS -> 16^3) ------------------------------
    les = {}
    for c in comps:
        cpath = os.path.join(cache_dir,
                             f"filtered_{c}_{timestep}_f{les_factor}.npy")
        if os.path.exists(cpath):
            les[c] = np.load(cpath).astype(np.float64)
        else:
            les[c] = block_average(dns[c], les_factor)
            np.save(cpath, les[c].astype(np.float32))
    print(f"    LES-{les_factor}dx {les['u'].shape} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ---- spectra --------------------------------------------------------
    spectra = {}
    print("[spec] computing spectra ...", flush=True)
    spectra["DNS"] = calculate_spectrum(dns["u"], dns["v"], dns["w"])
    if mps is not None:
        spectra["MPS"] = calculate_spectrum(mps["u"], mps["v"], mps["w"])
    spectra["LES"] = calculate_spectrum(les["u"], les["v"], les["w"])
    print(f"    done ({time.time()-t0:.1f}s)", flush=True)

    # ---- spectrum figure (styled like the attached) --------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for key in _ORDER:
        if key not in spectra:
            continue
        k, E = spectra[key]
        ax.loglog(k, E, **_STYLE[key])
    # inertial-range reference 1.6 eps^2/3 k^-5/3, scaled to sit on DNS.
    kD, ED = spectra["DNS"]
    fit_mask = (kD >= 2) & (kD <= 10)
    C = np.median(ED[fit_mask] / kD[fit_mask]**(-5.0 / 3.0))
    kref = np.logspace(np.log10(kD.min()), np.log10(kD.max()), 100)
    ax.loglog(kref, C * kref**(-5.0 / 3.0), color="red", lw=1.5,
              label=r"$1.6\,\epsilon^{2/3} k^{-5/3}$")
    ax.set_xlabel(r"$Wavenumber\,(k)$")
    ax.set_ylabel(r"$E(k)$")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower left", framealpha=0.9)
    png = os.path.join(out_dir,
                       f"velocity_energy_spectrum{run_tag}.png")
    fig.savefig(png, dpi=200)
    fig.savefig(png.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"[spec] wrote {png} (+ .pdf)")

    # ---- u-velocity contour row (DNS / MPS / LES) ----------------------
    plot_u_contour(dns, mps, les, out_dir, grid, comp="u", titles=_TITLES,
                   tag=run_tag)

    def comp_stats(field_dict):
        s = {}
        for c in comps:
            a = field_dict[c]
            s[c] = (float(a.mean()), float(a.std()))
        # KE ratios vs DNS computed below
        return s

    stats = {"DNS": comp_stats(dns), "LES": comp_stats(les)}
    if mps is not None:
        stats["MPS"] = comp_stats(mps)

    # KE ratios (combined over components), vs DNS
    def meanU2(fd):
        return sum(float(np.mean(fd[c] * fd[c])) for c in comps)

    def tke(fd):
        return sum(float(np.var(fd[c])) for c in comps)

    dns_mU2, dns_tke = meanU2(dns), tke(dns)
    ke_rows = []
    for key, fd in (("MPS", mps), ("LES", les)):
        if fd is None:
            continue
        ke_rows.append((key, meanU2(fd) / dns_mU2, tke(fd) / dns_tke))

    stats_txt = os.path.join(out_dir, f"velocity_stats{run_tag}.txt")
    with open(stats_txt, "w") as fh:
        fh.write(f"# Velocity statistics: DNS vs MPS(chi={mps_chi}) "
                 f"vs LES-{les_factor}dx\n")
        fh.write(f"# timestep={timestep}\n#\n")
        fh.write("# Per-component mean and RMS (std):\n")
        fh.write(f"# {'source':<6s} {'grid':>6s}  "
                 f"{'mean_u':>10s} {'rms_u':>10s}  "
                 f"{'mean_v':>10s} {'rms_v':>10s}  "
                 f"{'mean_w':>10s} {'rms_w':>10s}\n")
        for key in _ORDER:
            if key not in stats:
                continue
            s = stats[key]
            g = {"DNS": dns, "MPS": mps, "LES": les}[key]["u"].shape[0]
            fh.write(f"  {key:<6s} {g:>5d}^3 "
                     f" {s['u'][0]:>10.4e} {s['u'][1]:>10.4e} "
                     f" {s['v'][0]:>10.4e} {s['v'][1]:>10.4e} "
                     f" {s['w'][0]:>10.4e} {s['w'][1]:>10.4e}\n")
        fh.write("#\n# Kinetic-energy ratios vs DNS "
                 "(combined over u,v,w):\n")
        fh.write("# R_mean = <|U|^2>_src/<|U|^2>_DNS, "
                 "R_var = TKE_src/TKE_DNS\n")
        fh.write(f"# {'source':<6s} {'R_mean':>10s} {'R_var':>10s}\n")
        for (key, rmean, rvar) in ke_rows:
            fh.write(f"  {key:<6s} {rmean:>10.4f} {rvar:>10.4f}\n")
    print(f"[spec] wrote {stats_txt}")

    # ---- per-component PDFs figure --------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2),
                             constrained_layout=True)
    for j, c in enumerate(comps):
        ax = axes[j]
        # shared range from DNS
        lo, hi = np.percentile(dns[c], [0.1, 99.9])
        for key, fd in (("DNS", dns), ("MPS", mps), ("LES", les)):
            if fd is None:
                continue
            h, edges = np.histogram(fd[c].ravel(), bins=80,
                                    range=(lo, hi), density=True)
            ctr = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(ctr, h, **_STYLE[key])
        ax.set_xlabel(rf"${c}$")
        ax.set_ylabel(rf"$P({c})$")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        if j == 0:
            ax.legend(fontsize=10)
    png = os.path.join(out_dir, f"velocity_pdfs{run_tag}.png")
    fig.savefig(png, dpi=200)
    fig.savefig(png.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"[spec] wrote {png} (+ .pdf)")

    print("\n[spec] SUMMARY  KE ratios vs DNS:")
    print(f"  {'source':<6s} {'R_mean':>8s} {'R_var':>8s}")
    for (key, rmean, rvar) in ke_rows:
        print(f"  {key:<6s} {rmean:>8.4f} {rvar:>8.4f}")
    print(f"\n[spec] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
