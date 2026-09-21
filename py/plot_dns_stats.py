"""
plot_dns_stats.py
-----------------
Produce figures matching those in the PeleLM-FDF paper, using only DNS data.
Reads the .pkl saved by run_dns_stats.py.

Figure layout is designed so that MPS curves can be added later simply by
loading a second .pkl and passing it alongside.

Usage
-----
python plot_dns_stats.py <results_pkl> <out_dir>
"""

from __future__ import annotations

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
# Single-sided plotting helpers
# -----------------------------------------------------------------------------

def _fold_profile(y: np.ndarray, prof: np.ndarray, kind: str = "mean"):
    """Fold a y-profile about y=0 so the result spans only y >= 0.

    kind = 'mean' : average the two halves   (use for mean fields, scalar
                    dissipation, anything linear in the field)
    kind = 'rms'  : sqrt(mean of squared halves)  (preserves RMS magnitudes)

    The y-coordinate need not be exactly symmetric; we resample by index
    distance from the center, which is exact when y is uniformly spaced
    and centered at 0 (the case here).
    """
    n = len(y)
    # find the index closest to y = 0
    j0 = int(np.argmin(np.abs(y)))
    # number of cells available on each side
    n_pos = n - j0
    n_neg = j0 + 1
    n_keep = min(n_pos, n_neg)

    y_pos = y[j0:j0 + n_keep]
    p_pos = prof[j0:j0 + n_keep]
    p_neg = prof[j0 - np.arange(n_keep)]   # mirrored

    if kind == "mean":
        folded = 0.5 * (p_pos + p_neg)
    elif kind == "rms":
        folded = np.sqrt(0.5 * (p_pos ** 2 + p_neg ** 2))
    else:
        raise ValueError(f"Unknown kind={kind!r}; use 'mean' or 'rms'.")
    return y_pos, folded


# -----------------------------------------------------------------------------
# Profile / 1-D figures
# -----------------------------------------------------------------------------

def plot_mean_rms(results, out_dir, single_sided: bool = False,
                  xlim_paper: bool = True,
                  ylim_overrides: dict = None):
    """Figure 6 analog: mean + RMS profiles for Z, T, Y_CO, Y_CO2.

    Parameters
    ----------
    single_sided : bool
        If True, fold about y=0 and plot only y/H >= 0 (paper layout).
    xlim_paper : bool
        If True (and single_sided), clamp x to [0, 5] like the paper.
    ylim_overrides : dict, optional
        Per-variable y-limit override, e.g. {"T": (0, 1500)}. Defaults
        below match the paper's Figure 6 at t = 40 t_j.
    """
    y = results["y_over_H"]
    prof = results["profiles"]
    # Paper-matched y-limits (from Fig. 6, last column at t = 40 t_j).
    # Each entry: (key, label, ylim).
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
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax, (key, label) in zip(axes.ravel(), variables):
        if key not in prof:
            ax.set_visible(False)
            continue
        d = prof[key]
        if single_sided:
            yh, mean_p = _fold_profile(y, d["mean"], kind="mean")
            _,  rms_p  = _fold_profile(y, d["rms"],  kind="rms")
        else:
            yh, mean_p, rms_p = y, d["mean"], d["rms"]
        ax.plot(yh, mean_p, "-",  color="C3", label="DNS mean")
        ax.plot(yh, rms_p,  "--", color="C3", label="DNS RMS")
        ax.set_xlabel(r"$y/H$")
        ax.set_ylabel(label)
        if single_sided and xlim_paper:
            ax.set_xlim(0, 5)
        if key in DEFAULT_YLIM and DEFAULT_YLIM[key] is not None:
            ax.set_ylim(*DEFAULT_YLIM[key])
        ax.legend(loc="best", fontsize=9)
    suffix = " (single-sided, paper layout)" if single_sided else ""
    fig.suptitle(f"Reynolds-averaged mean and RMS profiles (DNS){suffix}")
    fig.tight_layout()
    fname = ("fig06_mean_rms_single_sided.png"
             if single_sided else "fig06_mean_rms.png")
    path = os.path.join(out_dir, fname)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_chi_profile(results, out_dir, single_sided: bool = False,
                     xlim_paper: bool = True,
                     ylim: tuple = (0.0, 5000.0)):
    """Figure 9 analog. Default ylim matches the paper's Fig. 9 at
    t = 40 t_j (peak ~4500 1/s)."""
    y = results["y_over_H"]
    chi = results["chi_profile"]
    if single_sided:
        yh, chi_p = _fold_profile(y, chi, kind="mean")
    else:
        yh, chi_p = y, chi
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(yh, chi_p, "-", color="C3", label="DNS")
    ax.set_xlabel(r"$y/H$")
    ax.set_ylabel(r"$\overline{\chi}$  [1/s]")
    ax.set_title("Scalar dissipation rate profile")
    if single_sided and xlim_paper:
        ax.set_xlim(0, 5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    fname = ("fig09_chi_profile_single_sided.png"
             if single_sided else "fig09_chi_profile.png")
    path = os.path.join(out_dir, fname)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_conditional_T(results, out_dir,
                        xlim: tuple = (0.0, 1.0),
                        ylim: tuple = (600.0, 1400.0)):
    """Figure 10 analog. Default limits match the paper's Fig. 10."""
    d = results["conditional_T_on_Z"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(d["psi_Z"], d["E_T"], "o-", color="C3", ms=4, label="DNS")
    ax.axvline(results["z_st"], color="k", ls=":", alpha=0.5,
               label=r"$Z_{st}$")
    ax.set_xlabel(r"$\psi_Z$")
    ax.set_ylabel(r"$E(T \mid Z = \psi_Z)$  [K]")
    ax.set_title("Conditional mean temperature")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "fig10_conditional_T.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_pdf_Z(results, out_dir,
               xlim: tuple = (0.0, 1.0),
               ylim: tuple = (0.0, 6.0)):
    """Figure 12 analog. Default limits match the paper's Fig. 12."""
    d = results["pdf_mixfrac"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(d["psi_Z"], d["pdf"], "-", color="C3", label="DNS")
    ax.set_xlabel(r"$\psi_Z$")
    ax.set_ylabel(r"PDF$(Z)$")
    ax.set_title("Mixture-fraction PDF near centerline")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "fig12_pdf_Z.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_joint_pdf(results, out_dir, log_scale=False,
                   xlim: tuple = (0.0, 1.0),
                   ylim: tuple = (0.0, 0.17),
                   levels=(5, 10, 20, 50, 100, 200),
                   show_peak_marker: bool = True):
    """Figure 13 analog: joint PDF as labeled contour lines.

    By default reproduces the paper's style: contours of the RAW PDF
    p(psi_Z, psi_YCO2) at hand-picked log-spaced values
    (5, 10, 20, 50, 100, 200) with numerical labels on each line.
    Pass log_scale=True to instead plot log(1 + p).

    Parameters
    ----------
    levels : sequence of float
        Specific contour values to draw. Default matches the paper.
        Levels above the data max are silently dropped.
    show_peak_marker : bool
        If True, mark the global peak of the joint PDF with a black
        dot, like the paper's Figure 13.
    """
    d = results["joint_pdf_Z_YCO2"]
    Z_c = d["psi_Z"]
    Y_c = d["psi_YCO2"]
    H = d["joint_pdf"].T

    data = np.log1p(H) if log_scale else H
    vmax = float(data.max())
    if vmax <= 0:
        print("[plot] joint PDF is empty; skipping")
        return

    if log_scale:
        plot_levels = [float(np.log1p(lv)) for lv in levels if lv < vmax]
        label_fmt = "%.2f"
        title_tag = r"$\log(1+p)$"
    else:
        plot_levels = [float(lv) for lv in levels if lv < vmax]
        label_fmt = "%d"
        title_tag = r"$p$"

    if not plot_levels:
        print(f"[plot] joint PDF: all levels above vmax={vmax:.3f}; skipping")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contour(Z_c, Y_c, data, levels=plot_levels,
                    cmap="winter", linewidths=2.0)
    ax.clabel(cs, inline=True, fontsize=9, fmt=label_fmt)

    if show_peak_marker:
        # Mark the GLOBAL peak of the joint PDF, like the paper.
        # H has shape (n_YCO2, n_Z); find the (j, i) pair where it's max.
        j_peak, i_peak = np.unravel_index(int(np.argmax(H)), H.shape)
        ax.plot(Z_c[i_peak], Y_c[j_peak], "ko", ms=8)

    ax.set_xlabel(r"$\psi_Z$")
    ax.set_ylabel(r"$\psi_{Y_{CO_2}}$")
    ax.set_title(f"Joint PDF of $(Z, Y_{{CO_2}})$  [{title_tag}]")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig13_joint_pdf_DNS.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_manifold(results, out_dir,
                  xlim: tuple = (0.0, 1.0),
                  ylim: tuple = (0.0, 0.3),
                  zlim: tuple = (0.0, 2.5),
                  T_range: tuple = (500.0, 1500.0),
                  elev: float = 10.0,
                  azim: float = -75.0,
                  roll: float = 0.0,
                  paper_style: bool = True):
    """Figure 14 analog: 3D scatter of (Z, Y_O2, Y_OH*1000) colored by T.

    paper_style=True (default) renders the figure with the visual style
    of the published paper:
      - solid points (alpha=1.0), larger marker size
      - no title, tight layout
      - "jet" colormap with the full range
      - axis labels inside the cube panels
      - more aggressive subsample (uses up to all available samples)
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    d = results["manifold_scatter"]
    n = d["Z"].size

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if paper_style:
        marker_size = 1.0
        alpha = 1.0
        # Use ALL available samples (the scatter sample was subsampled
        # at stats time; here we just use what we have)
        idx = np.arange(n)
    else:
        marker_size = 0.5
        alpha = 0.6
        idx = np.arange(n)

    sc = ax.scatter(
        d["Z"][idx], d["Y_O2"][idx], d["Y_OH"][idx] * 1000.0,
        c=d["T"][idx], cmap="jet", s=marker_size, alpha=alpha,
        vmin=T_range[0], vmax=T_range[1],
        edgecolors="none",
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.10, label=r"$T$ [K]")
    ax.set_xlabel(r"$Z$")
    ax.set_ylabel(r"$Y_{O_2}$")
    ax.set_zlabel(r"$Y_{OH}\times 10^3$")
    if not paper_style:
        ax.set_title("Compositional manifold (DNS)")

    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    if zlim is not None: ax.set_zlim(*zlim)
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)
        if roll != 0.0:
            print(f"[plot] warning: roll={roll} ignored "
                  f"(matplotlib too old; need >= 3.7)")

    if paper_style:
        # Tighter pane styling, neutral panes (white background)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor("white")
            pane.set_edgecolor("0.6")
            pane.set_alpha(1.0)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig14_manifold_DNS.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


def write_summary(results, out_dir):
    """Text summary of the scalar statistics (delta_Z, extinction)."""
    ext = results["extinction"]
    lines = [
        "DNS scalar statistics (single timestep)",
        "=======================================",
        f"delta_Z / (2H)             : {results['delta_Z_over_2H']:.4f}",
        f"volume-averaged M_ext      : {ext['volume_averaged_marker']:.3e}",
        f"conditional P(extinction)  : {ext['conditional_extinction_prob']:.3e}",
        f"T on stoichiometric surf.  : {ext['T_on_stoich_surface']:.2f} K",
        f"near-stoich cell count     : {ext['n_near_st_cells']:,} / "
        f"{ext['n_total_cells']:,}",
        f"Z_st                       : {results['z_st']}",
    ]
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[write] {path}")


def plot_chi_validation(validation, grid_y_over_H, out_dir):
    """Side-by-side comparison of stored vs. reconstructed chi.

    Shows: (1) x-z-averaged profiles (the quantity in paper Fig. 9),
    (2) log-scale 2D slices. Intended as a sanity check that the
    reconstruction is physically meaningful even if pointwise disagreement
    from finite-difference order is expected.
    """
    if validation is None or validation.get("chi_diagnostics") is None:
        return
    d = validation["chi_diagnostics"]
    prof_ref = d["profile_y_reference"]
    prof_rec = d["profile_y_reconstructed"]
    sl_ref = d["slice_z0_reference"]
    sl_rec = d["slice_z0_reconstructed"]

    fig = plt.figure(figsize=(12, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])

    # Profile comparison
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(grid_y_over_H, prof_ref, "-",  color="k", label="stored (S3D)")
    ax.plot(grid_y_over_H, prof_rec, "--", color="C3",
            label="reconstructed")
    ax.set_xlabel(r"$y/H$")
    ax.set_ylabel(r"$\overline{\chi}$  [1/s]")
    ax.set_title(f"x-z-averaged chi\n"
                 f"profile corr. = {d['profile_correlation']:.4f}")
    ax.legend()

    # Stored slice
    ax = fig.add_subplot(gs[0, 1])
    vmax = float(np.log10(max(sl_ref.max(), sl_rec.max(), 1.0)))
    im = ax.imshow(np.log10(np.maximum(sl_ref.T, 1.0)),
                   origin="lower", aspect="auto", cmap="viridis",
                   vmin=0, vmax=vmax)
    ax.set_title(r"$\log_{10}\chi$ stored (z=0 slice)")
    ax.set_xlabel("x index"); ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax)

    # Reconstructed slice
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(np.log10(np.maximum(sl_rec.T, 1.0)),
                   origin="lower", aspect="auto", cmap="viridis",
                   vmin=0, vmax=vmax)
    ax.set_title(r"$\log_{10}\chi$ reconstructed (z=0 slice)")
    ax.set_xlabel("x index"); ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig09b_chi_validation.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] {path}")


def _plot_field_validation_panel(
    profile_y, prof_ref, prof_rec,
    slice_ref, slice_rec,
    out_path,
    *,
    profile_label: str,
    profile_title: str,
    contour_label: str,
    contour_title_ref: str,
    contour_title_rec: str,
    log_scale: bool,
    profile_corr: float,
    profile_rel_l2: float,
    rec_label: str = "reconstructed",
):
    """Generic 3-panel validation figure: profile + stored contour + reconstructed contour.

    Used for both Z and chi, on either the cube or the full grid.
    """
    fig = plt.figure(figsize=(13, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])

    # Profile comparison
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(profile_y, prof_ref, "-", color="k", lw=2, label="stored")
    ax.plot(profile_y, prof_rec, "--", color="C3",
            label=f"{rec_label}\n(corr {profile_corr:.6f})")
    ax.set_xlabel(r"$y/H$")
    ax.set_ylabel(profile_label)
    ax.set_title(f"{profile_title}\nrel L2 = {profile_rel_l2:.2e}")
    ax.legend()

    # Contour shared color scale
    if log_scale:
        FLOOR = max(slice_ref.max(), slice_rec.max()) * 1e-6
        FLOOR = max(FLOOR, 1e-30)
        data_ref = np.log10(np.maximum(slice_ref.T, FLOOR))
        data_rec = np.log10(np.maximum(slice_rec.T, FLOOR))
    else:
        data_ref = slice_ref.T
        data_rec = slice_rec.T
    vmin = float(min(data_ref.min(), data_rec.min()))
    vmax = float(max(data_ref.max(), data_rec.max()))
    if vmax <= vmin:
        vmax = vmin + 1.0
    levels = np.linspace(vmin, vmax, 16)
    nx, ny = slice_ref.shape
    xi = np.arange(nx); yi = np.arange(ny)

    ax = fig.add_subplot(gs[0, 1])
    cs = ax.contourf(xi, yi, data_ref, levels=levels, cmap="viridis")
    fig.colorbar(cs, ax=ax, label=contour_label)
    ax.set_title(contour_title_ref)
    ax.set_xlabel("x index"); ax.set_ylabel("y index")

    ax = fig.add_subplot(gs[0, 2])
    cs = ax.contourf(xi, yi, data_rec, levels=levels, cmap="viridis")
    fig.colorbar(cs, ax=ax, label=contour_label)
    ax.set_title(contour_title_rec)
    ax.set_xlabel("x index"); ax.set_ylabel("y index")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_Z_validation(validation_full, out_dir):
    """Z validation: cube and full versions if data available.
    Produces fig_Z_validation_cube.png and fig_Z_validation_full.png.
    """
    if validation_full is None:
        return

    # ---- Cube version ----
    Z_rec = validation_full.get("Z_reconstructed_cube")
    Z_ref = validation_full.get("Z_stored_cube")
    if Z_rec is not None and Z_ref is not None:
        y_over_H = validation_full.get(
            "y_over_H", np.arange(Z_rec.shape[1]) - (Z_rec.shape[1] - 1) / 2.0)
        prof_ref = Z_ref.mean(axis=(0, 2))
        prof_rec = Z_rec.mean(axis=(0, 2))
        nz = Z_rec.shape[2]; k0 = nz // 2
        p = validation_full.get("Z_pointwise", {})
        _plot_field_validation_panel(
            y_over_H, prof_ref, prof_rec,
            Z_ref[:, :, k0], Z_rec[:, :, k0],
            os.path.join(out_dir, "fig_Z_validation_cube.png"),
            profile_label=r"$\overline{Z}$",
            profile_title="x-z-averaged Z (cube)",
            contour_label=r"$Z$",
            contour_title_ref=r"$Z$ stored  (z = nz/2, cube)",
            contour_title_rec=r"$Z$ reconstructed  (z = nz/2, cube)",
            log_scale=False,
            profile_corr=p.get("correlation", float("nan")),
            profile_rel_l2=validation_full.get("Z_rel_l2", float("nan")),
        )

    # ---- Full-grid version ----
    Z_rec_f = validation_full.get("Z_reconstructed_full")
    Z_ref_f = validation_full.get("Z_stored_full")
    if Z_rec_f is not None and Z_ref_f is not None:
        y_over_H_f = validation_full.get(
            "y_over_H_full",
            np.arange(Z_rec_f.shape[1]) - (Z_rec_f.shape[1] - 1) / 2.0)
        prof_ref = Z_ref_f.mean(axis=(0, 2))
        prof_rec = Z_rec_f.mean(axis=(0, 2))
        nz = Z_rec_f.shape[2]; k0 = nz // 2
        p = validation_full.get("Z_full_pointwise", {})
        _plot_field_validation_panel(
            y_over_H_f, prof_ref, prof_rec,
            Z_ref_f[:, :, k0], Z_rec_f[:, :, k0],
            os.path.join(out_dir, "fig_Z_validation_full.png"),
            profile_label=r"$\overline{Z}$",
            profile_title="x-z-averaged Z (full grid)",
            contour_label=r"$Z$",
            contour_title_ref=r"$Z$ stored  (z = nz/2, full)",
            contour_title_rec=r"$Z$ reconstructed  (z = nz/2, full)",
            log_scale=False,
            profile_corr=p.get("correlation", float("nan")),
            profile_rel_l2=validation_full.get("Z_full_rel_l2", float("nan")),
        )


def plot_chi_validation_full(validation_full, out_dir):
    """Chi validation: cube and full versions if data available.
    Produces fig_chi_validation_cube.png and fig_chi_validation_full.png.
    """
    if validation_full is None:
        return
    fd_order = validation_full.get("fd_order", "?")

    # ---- Cube version (uses chi_fullgrid_diagnostics: full-grid recon
    # cropped to cube vs stored cube) ----
    d = validation_full.get("chi_fullgrid_diagnostics")
    if d is not None:
        y_over_H = validation_full.get(
            "y_over_H",
            np.arange(len(d["profile_y_reference"]))
            - (len(d["profile_y_reference"]) - 1) / 2.0)
        _plot_field_validation_panel(
            y_over_H,
            d["profile_y_reference"], d["profile_y_reconstructed"],
            d["slice_z0_reference"], d["slice_z0_reconstructed"],
            os.path.join(out_dir, "fig_chi_validation_cube.png"),
            profile_label=r"$\overline{\chi}$  [1/s]",
            profile_title=f"x-z-averaged chi (cube, FD order = {fd_order})",
            contour_label=r"$\log_{10}\chi$",
            contour_title_ref=r"$\chi$ stored  (z = nz/2, cube)",
            contour_title_rec=r"$\chi$ reconstructed  (z = nz/2, cube)",
            log_scale=True,
            profile_corr=d["profile_correlation"],
            profile_rel_l2=d["profile_rel_l2"],
        )

    # ---- Full-grid version ----
    d_full = validation_full.get("chi_full_diagnostics")
    if d_full is not None:
        y_over_H_f = validation_full.get(
            "y_over_H_full",
            np.arange(len(d_full["profile_y_reference"]))
            - (len(d_full["profile_y_reference"]) - 1) / 2.0)
        _plot_field_validation_panel(
            y_over_H_f,
            d_full["profile_y_reference"], d_full["profile_y_reconstructed"],
            d_full["slice_z0_reference"], d_full["slice_z0_reconstructed"],
            os.path.join(out_dir, "fig_chi_validation_full.png"),
            profile_label=r"$\overline{\chi}$  [1/s]",
            profile_title=f"x-z-averaged chi (full grid, FD order = {fd_order})",
            contour_label=r"$\log_{10}\chi$",
            contour_title_ref=r"$\chi$ stored  (z = nz/2, full)",
            contour_title_rec=r"$\chi$ reconstructed  (z = nz/2, full)",
            log_scale=True,
            profile_corr=d_full["profile_correlation"],
            profile_rel_l2=d_full["profile_rel_l2"],
        )


def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_dns_stats.py <results_pkl_or_dir> <out_dir>")
        print("  If first arg is a directory, looks for "
              "<dir>/cube/dns_stats_*.pkl and <dir>/full/dns_stats_*.pkl")
        sys.exit(1)

    inp = sys.argv[1]
    out_dir = sys.argv[2]

    # Identify which regions' pickles exist
    tasks = []  # list of (pkl_path, region_name)
    if os.path.isdir(inp):
        for region in ("cube", "full"):
            rdir = os.path.join(inp, region)
            if os.path.isdir(rdir):
                for f in sorted(os.listdir(rdir)):
                    if f.startswith("dns_stats_") and f.endswith(".pkl"):
                        tasks.append((os.path.join(rdir, f), region))
        # full-grid validation pickle lives at the top-level
        vf_path = None
        for f in sorted(os.listdir(inp)):
            if f.startswith("dns_validation_full_") and f.endswith(".pkl"):
                vf_path = os.path.join(inp, f)
    else:
        tasks.append((inp, "cube"))
        vf_path = inp.replace("dns_stats_", "dns_validation_full_")
        if not os.path.exists(vf_path):
            vf_path = None

    if not tasks:
        print(f"No dns_stats_*.pkl found under {inp}")
        sys.exit(1)

    _setup_mpl()

    # Per-region statistics figures
    for pkl_path, region in tasks:
        region_out = os.path.join(out_dir, region)
        os.makedirs(region_out, exist_ok=True)
        print(f"\n--- plotting {region} figures -> {region_out} ---")
        with open(pkl_path, "rb") as f:
            results = pickle.load(f)

        plot_mean_rms(results, region_out, single_sided=False)
        plot_mean_rms(results, region_out, single_sided=True)
        plot_chi_profile(results, region_out, single_sided=False)
        plot_chi_profile(results, region_out, single_sided=True)
        plot_conditional_T(results, region_out)
        plot_pdf_Z(results, region_out)
        plot_joint_pdf(results, region_out)
        plot_manifold(results, region_out)

        # Cube-only validation plot (if present)
        val_path = pkl_path.replace("dns_stats_", "dns_validation_")
        if os.path.exists(val_path):
            with open(val_path, "rb") as f:
                validation = pickle.load(f)
            plot_chi_validation(validation, results["y_over_H"], region_out)

        write_summary(results, region_out)

    # Full-grid validation figures (shared, go in top-level out_dir)
    if vf_path is not None:
        print(f"\n--- plotting full-grid validation -> {out_dir} ---")
        with open(vf_path, "rb") as f:
            validation_full = pickle.load(f)
        plot_Z_validation(validation_full, out_dir)
        plot_chi_validation_full(validation_full, out_dir)


if __name__ == "__main__":
    main()
