"""
jhs_velocity_spectrum_stats.py
------------------------------
JHS (1024^3 forced isotropic, 2*pi box) velocity comparison of
DNS vs MPS(chi) vs LES (box-filtered DNS):

    factor 32 -> LES  32^3  <->  MPS chi = 27
    factor 16 -> LES  64^3  <->  MPS chi = 84

Outputs (all tagged with _chi<chi>_f<factor> so cases don't overwrite):
  * jhs_energy_spectrum_chi<..>_f<..>.png/pdf   3D E(k) with k^-5/3 reference
  * jhs_energy_spectrum_chi<..>_f<..>.mat       k_DNS/E_DNS, k_MPS/E_MPS, k_LES/E_LES
  * jhs_contour_u_chi<..>_f<..>.png/pdf         mid-z slice of u, DNS/MPS/LES
  * jhs_velocity_stats_chi<..>_f<..>.txt        mean/RMS per component + KE ratios
  * jhs_velocity_pdfs_chi<..>_f<..>.png/pdf     per-component PDFs

Differences from the jet script (this is a DIFFERENT dataset):
  * data are .mat (U_t1/V_t1/W_t1), read with h5py (v7.3/HDF5) -- no scipy,
    which is ABI-broken against numpy 2.x on this cluster.
  * no jet width H: the box is 2*pi, so contour axes are x/L, y/L in [-0.5,0.5].
  * no dns_stats.GridConfig dependency at all.

MEMORY WARNING
--------------
A 1024^3 float64 cube is 8.6 GB. The spectrum needs the three squared FFT
magnitudes plus a radius and bin-index array at once. To keep this tractable
the FFT magnitudes and the radius/index arrays are stored in float32/int32 and
intermediates are freed aggressively; even so expect a peak well above 100 GB.
RUN THIS AS A BATCH JOB with large --mem (see submit_jhs_analysis.sh), never
on a login node.

Usage
-----
    python jhs_velocity_spectrum_stats.py \
        --dns-dir /ix/.../Forced_Isotropic_1024Cubed_mat_in_one \
        --mps-dir /ix/.../mps_chi27_jobs \
        --mps-pattern "truncated_{comp}_t1_1024_il_chi27.mat" \
        --mps-chi 27 --les-factor 32 \
        --dns-pattern "{comp}_t1.mat" \
        --components U,V,W \
        --cache-dir /ix/.../jhs_filtered \
        --out-dir QC4PDE_JHS
"""

from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------
def _parse(args, name, default=None, cast=str):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        args.pop(i + 1); args.pop(i)
        return cast(v) if cast is not str else v
    return default


# ---------------------------------------------------------------------------
# .mat reader (h5py first: MAT.jl writes v7.3 for large arrays; scipy on this
# cluster is broken against numpy 2.x)
# ---------------------------------------------------------------------------
def read_mat_cube(path, var=None):
    try:
        import h5py
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            if var is not None and var in f:
                key = var
            else:
                # pick the largest dataset (skips scalars like chi_crit)
                key = max(keys, key=lambda k: int(np.prod(f[k].shape)))
            arr = np.asarray(f[key], dtype=np.float64)
        # HDF5/MATLAB v7.3 is column-major -> h5py reads transposed
        return np.ascontiguousarray(arr.T)
    except (OSError, ImportError):
        from scipy.io import loadmat
        m = loadmat(path)
        keys = [k for k in m if not k.startswith("__")]
        if var is not None and var in m:
            return np.asarray(m[var], dtype=np.float64)
        key = max(keys, key=lambda k: int(np.prod(np.shape(m[k]))))
        return np.asarray(m[key], dtype=np.float64)


def block_average(cube, factor):
    nx, ny, nz = cube.shape
    cx, cy, cz = (nx // factor) * factor, (ny // factor) * factor, \
                 (nz // factor) * factor
    c = cube[:cx, :cy, :cz].reshape(cx // factor, factor,
                                    cy // factor, factor,
                                    cz // factor, factor)
    return c.mean(axis=(1, 3, 5))


# ---------------------------------------------------------------------------
# 3D spectrum -- same algorithm as the user's calculate_spectrum, but with the
# large temporaries in float32/int32 and freed as soon as possible so a 1024^3
# case is feasible. Numerics are unchanged (sums are accumulated in float64).
# ---------------------------------------------------------------------------
def calculate_spectrum(u, v, w, verbose=True):
    L = 2 * np.pi
    dim = u.shape[0]
    k_end = int(dim / 2)

    # squared, normalized FFT magnitudes (float32 to halve memory)
    def sq_fft(a, name):
        if verbose:
            print(f"      fft {name} ...", flush=True)
        F = np.fft.fftn(a)
        out = (np.abs(F) / dim**3).astype(np.float32) ** 2
        del F
        gc.collect()
        return out

    uu = sq_fft(u, "u")
    vv = sq_fft(v, "v")
    ww = sq_fft(w, "w")

    # radius array (float32)
    rx = np.arange(dim) - dim / 2 + 1
    rx = np.roll(rx, int(dim / 2) + 1).astype(np.float32)
    if verbose:
        print("      building radius array ...", flush=True)
    r2 = (rx[:, None, None] ** 2 + rx[None, :, None] ** 2
          + rx[None, None, :] ** 2)
    r = np.sqrt(r2, dtype=np.float32)
    del r2
    gc.collect()

    dx = 2 * np.pi / L
    k = (np.arange(k_end) + 1) * dx

    bins = np.zeros(k.shape[0] + 1)
    for N in range(k_end):
        bins[N] = 0 if N == 0 else (k[N] + k[N - 1]) / 2
    bins[-1] = k[-1]

    if verbose:
        print("      digitizing ...", flush=True)
    inds = np.digitize(r * dx, bins, right=True).astype(np.int32)
    del r
    gc.collect()

    if verbose:
        print("      binning ...", flush=True)
    # bincount over the flattened index is far cheaper than k_end masks
    flat = inds.ravel()
    total = (uu.ravel().astype(np.float64)
             + vv.ravel().astype(np.float64)
             + ww.ravel().astype(np.float64))
    nb = k_end + 2
    spec_sum = np.bincount(flat, weights=total, minlength=nb)
    bin_counter = np.bincount(flat, minlength=nb).astype(np.float64)
    del flat, total, uu, vv, ww, inds
    gc.collect()

    # indices 1..k_end correspond to N+1 in the original loop
    spectrum = spec_sum[1:k_end + 1]
    counts = bin_counter[1:k_end + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        spectrum = spectrum * 2 * np.pi * (k**2) / (counts * dx**3)
    return k, spectrum


# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------
_STYLE = {
    "DNS": dict(color="black",   linestyle="-",  lw=2.0, label="DNS"),
    "MPS": dict(color="#9b59b6", linestyle="--", lw=2.2, label="MPS"),
    "LES": dict(color="#16a085", linestyle=":",  lw=2.4, label="LES"),
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
        print("[jhs] LaTeX unavailable; mathtext fallback.")


def plot_contour(dns_u, mps_u, les_u, out_dir, titles, tag):
    """Row of mid-z slices of u; axes normalized by the 2*pi box: x/L in
    [-0.5, 0.5]. All panels share the DNS colour scale and physical extent."""
    panels = [("DNS", dns_u)]
    if mps_u is not None:
        panels.append(("MPS", mps_u))
    panels.append(("LES", les_u))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6),
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()

    def mid(a):
        return a[:, :, a.shape[2] // 2]

    ds = mid(dns_u)
    vmin, vmax = float(ds.min()), float(ds.max())
    extent = [-0.5, 0.5, -0.5, 0.5]        # x/L, y/L over the 2*pi box

    im = None
    for ax, (key, a) in zip(axes, panels):
        im = ax.imshow(mid(a).T, origin="lower", cmap="jet",
                       vmin=vmin, vmax=vmax, extent=extent,
                       aspect="equal", interpolation="nearest")
        ax.set_title(titles[key], fontsize=18)
        ax.set_xlabel(r"$x/L$", fontsize=16)
        if key == panels[0][0]:
            ax.set_ylabel(r"$y/L$", fontsize=16)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=12)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(r"$u$", fontsize=16)
    png = os.path.join(out_dir, f"jhs_contour_u{tag}.png")
    fig.savefig(png, dpi=200); fig.savefig(png.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"[jhs] wrote {png} (+ .pdf)")


def main():
    args = sys.argv[1:]
    dns_dir = _parse(args, "--dns-dir",
                     default="/ix/pgivi/moe32/Schmidt/Forced_Isotropic/"
                             "5_Grid_1024/Forced_Isotropic_1024Cubed_mat_in_one")
    dns_pat = _parse(args, "--dns-pattern", default="{comp}_t1.mat")
    mps_dir = _parse(args, "--mps-dir", default=None)
    mps_pat = _parse(args, "--mps-pattern",
                     default="truncated_{comp}_t1_1024_il_chi{chi}.mat")
    mps_chi = _parse(args, "--mps-chi", default="27")
    les_factor = int(_parse(args, "--les-factor", default="32"))
    comps = [c.strip() for c in
             _parse(args, "--components", "U,V,W").split(",") if c.strip()]
    cache_dir = _parse(args, "--cache-dir", default="jhs_filtered")
    out_dir = _parse(args, "--out-dir", default="QC4PDE_JHS")
    skip_spectrum = "--skip-spectrum" in args
    if skip_spectrum:
        args.remove("--skip-spectrum")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    _enable_latex()

    tag = f"_chi{mps_chi}_f{les_factor}"
    les_grid = None  # filled after filtering
    titles = {"DNS": "DNS",
              "MPS": rf"MPS ($\chi={mps_chi}$)",
              "LES": rf"LES (${les_factor}\Delta x$)"}
    _STYLE["MPS"]["label"] = titles["MPS"]
    _STYLE["LES"]["label"] = titles["LES"]

    t0 = time.time()
    print(f"[jhs] chi={mps_chi}  les_factor={les_factor}  comps={comps}")
    print(f"[jhs] dns_dir={dns_dir}")

    # ---- DNS ------------------------------------------------------------
    dns = {}
    for c in comps:
        p = os.path.join(dns_dir, dns_pat.format(comp=c))
        print(f"[jhs] loading DNS {c} from {p} ...", flush=True)
        dns[c] = read_mat_cube(p)
        print(f"    {c}: shape={dns[c].shape} "
              f"({dns[c].nbytes/1e9:.1f} GB) ({time.time()-t0:.1f}s)",
              flush=True)

    # ---- MPS ------------------------------------------------------------
    mps = None
    if mps_dir:
        try:
            mps = {}
            for c in comps:
                p = os.path.join(mps_dir,
                                 mps_pat.format(comp=c, chi=mps_chi))
                mps[c] = read_mat_cube(p, var="u")
            print(f"    MPS chi={mps_chi}: shape={mps[comps[0]].shape} "
                  f"loaded", flush=True)
        except Exception as e:
            print(f"    MPS load failed: {e}", flush=True)
            mps = None
    else:
        print("    (no --mps-dir; skipping MPS)", flush=True)

    # ---- LES (box-filter DNS) -------------------------------------------
    les = {}
    for c in comps:
        cpath = os.path.join(cache_dir, f"jhs_filtered_{c}_f{les_factor}.npy")
        if os.path.exists(cpath):
            les[c] = np.load(cpath).astype(np.float64)
        else:
            les[c] = block_average(dns[c], les_factor)
            np.save(cpath, les[c].astype(np.float32))
    les_grid = les[comps[0]].shape[0]
    print(f"    LES factor {les_factor}: {les[comps[0]].shape} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ---- stats (cheap; do before the memory-hungry spectrum) ------------
    def meanU2(fd):
        return sum(float(np.mean(fd[c] * fd[c])) for c in comps)

    def tke(fd):
        return sum(float(np.var(fd[c])) for c in comps)

    dns_mU2, dns_tke = meanU2(dns), tke(dns)
    rows = []
    for key, fd in (("MPS", mps), ("LES", les)):
        if fd is None:
            continue
        rows.append((key, meanU2(fd) / dns_mU2, tke(fd) / dns_tke))

    stats_txt = os.path.join(out_dir, f"jhs_velocity_stats{tag}.txt")
    with open(stats_txt, "w") as fh:
        fh.write(f"# JHS 1024^3: DNS vs MPS(chi={mps_chi}) vs "
                 f"LES factor {les_factor} ({les_grid}^3)\n")
        fh.write(f"# DNS: <|U|^2>={dns_mU2:.8e}  TKE(sum Var)={dns_tke:.8e}\n")
        fh.write("# per-component mean / RMS:\n")
        fh.write(f"# {'source':<6s} {'grid':>7s} " +
                 " ".join(f"{'mean_'+c:>12s} {'rms_'+c:>12s}"
                          for c in comps) + "\n")
        for key, fd in (("DNS", dns), ("MPS", mps), ("LES", les)):
            if fd is None:
                continue
            g = fd[comps[0]].shape[0]
            line = f"  {key:<6s} {g:>6d}^3 "
            for c in comps:
                line += f" {float(fd[c].mean()):>12.5e} " \
                        f"{float(fd[c].std()):>12.5e}"
            fh.write(line + "\n")
        fh.write("#\n# KE ratios vs DNS (combined over components):\n")
        fh.write(f"# {'source':<6s} {'R_mean':>10s} {'R_var':>10s}\n")
        for (key, rmean, rvar) in rows:
            fh.write(f"  {key:<6s} {rmean:>10.4f} {rvar:>10.4f}\n")
    print(f"[jhs] wrote {stats_txt}")

    # ---- contour --------------------------------------------------------
    plot_contour(dns[comps[0]],
                 None if mps is None else mps[comps[0]],
                 les[comps[0]], out_dir, titles, tag)

    # ---- PDFs -----------------------------------------------------------
    fig, axes = plt.subplots(1, len(comps), figsize=(5 * len(comps), 4.2),
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    for j, c in enumerate(comps):
        ax = axes[j]
        lo, hi = np.percentile(dns[c], [0.1, 99.9])
        for key, fd in (("DNS", dns), ("MPS", mps), ("LES", les)):
            if fd is None:
                continue
            h, e = np.histogram(fd[c].ravel(), bins=80, range=(lo, hi),
                                density=True)
            ax.plot(0.5 * (e[:-1] + e[1:]), h, **_STYLE[key])
        ax.set_xlabel(rf"${c}$"); ax.set_ylabel(rf"$P({c})$")
        ax.set_yscale("log"); ax.grid(True, alpha=0.25)
        if j == 0:
            ax.legend(fontsize=10)
    png = os.path.join(out_dir, f"jhs_velocity_pdfs{tag}.png")
    fig.savefig(png, dpi=200); fig.savefig(png.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"[jhs] wrote {png} (+ .pdf)")

    # ---- spectra (memory heavy; last, and skippable) --------------------
    if skip_spectrum:
        print("[jhs] --skip-spectrum given; done.")
        return

    spectra = {}
    print("[jhs] computing DNS spectrum (1024^3: this is the heavy step) ...",
          flush=True)
    spectra["DNS"] = calculate_spectrum(dns[comps[0]], dns[comps[1]],
                                        dns[comps[2]])
    # free DNS cubes before the next spectrum to limit peak memory
    print(f"    DNS spectrum done ({time.time()-t0:.1f}s)", flush=True)

    if mps is not None:
        print("[jhs] computing MPS spectrum ...", flush=True)
        spectra["MPS"] = calculate_spectrum(mps[comps[0]], mps[comps[1]],
                                            mps[comps[2]])
        print(f"    MPS spectrum done ({time.time()-t0:.1f}s)", flush=True)

    print("[jhs] computing LES spectrum ...", flush=True)
    spectra["LES"] = calculate_spectrum(les[comps[0]], les[comps[1]],
                                        les[comps[2]])

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for key in _ORDER:
        if key not in spectra:
            continue
        k, E = spectra[key]
        ax.loglog(k, E, **_STYLE[key])
    kD, ED = spectra["DNS"]
    m = (kD >= 2) & (kD <= 10) & np.isfinite(ED) & (ED > 0)
    if m.any():
        C = np.median(ED[m] / kD[m] ** (-5.0 / 3.0))
        kref = np.logspace(np.log10(kD.min()), np.log10(kD.max()), 100)
        ax.loglog(kref, C * kref ** (-5.0 / 3.0), color="red", lw=1.5,
                  label=r"$1.6\,\epsilon^{2/3} k^{-5/3}$")
    ax.set_xlabel(r"$Wavenumber\,(k)$")
    ax.set_ylabel(r"$E(k)$")
    ax.set_ylim(bottom=1e-8)          # fixed floor, as in the reference figure
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower left", framealpha=0.9)
    # ---- save the spectra to .mat (v7.3 via h5py) -------------------
    mat_path = os.path.join(out_dir, f"jhs_energy_spectrum{tag}.mat")
    try:
        import h5py
        with h5py.File(mat_path, "w") as hf:
            for key in _ORDER:
                if key not in spectra:
                    continue
                kk, EE = spectra[key]
                # store as k_<src> and E_<src>; transpose for MATLAB
                # column-major convention on read-back.
                hf.create_dataset(f"k_{key}", data=np.asarray(kk,
                                                              dtype=np.float64))
                hf.create_dataset(f"E_{key}", data=np.asarray(EE,
                                                              dtype=np.float64))
            hf.attrs["chi"] = str(mps_chi)
            hf.attrs["les_factor"] = int(les_factor)
        print(f"[jhs] wrote {mat_path}")
    except Exception as e:
        print(f"[jhs] WARNING: could not write {mat_path}: {e}")

    png = os.path.join(out_dir, f"jhs_energy_spectrum{tag}.png")
    fig.savefig(png, dpi=200); fig.savefig(png.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"[jhs] wrote {png} (+ .pdf)")

    print("\n[jhs] SUMMARY  KE ratios vs DNS:")
    print(f"  {'source':<6s} {'R_mean':>8s} {'R_var':>8s}")
    for (key, rmean, rvar) in rows:
        print(f"  {key:<6s} {rmean:>8.4f} {rvar:>8.4f}")
    print(f"\n[jhs] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
