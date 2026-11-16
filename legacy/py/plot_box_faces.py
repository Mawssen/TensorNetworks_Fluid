"""
plot_box_faces.py
-----------------
Reproduce the Aitzhan/Hawkes jet-flame box schematic as a real data figure:
render two faces of the DNS mixture-fraction cube as surfaces in 3D.

  * Front face : the x-y plane at the FIRST z gridpoint  (z = z[0]).
  * Right face : the z-y plane at the LAST  x gridpoint  (x = x[-1]).

Both faces are coloured by the mixture fraction Z on a shared colormap and
normalisation, so they match along their common vertical (y) edge. Axes are
labelled to match the schematic: x streamwise, y cross-streamwise (vertical),
z spanwise.

Coordinates are normalised by the fuel-jet width H (GridConfig.H), i.e. the
axes read x/H, y/H, z/H, consistent with the rest of the QC4PDE figures.

Usage
-----
    python plot_box_faces.py \
        --dns-data-dir /ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198 \
        --timestep 0198 \
        [--out QC4PDE/box_faces.png] \
        [--cmap inferno] [--azim -60] [--elev 18]

Run on the cluster with the venv + python module loaded (login node; the
venv is broken on viz nodes per the pipeline notes).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors


# ---------------------------------------------------------------------------
# tiny arg parser (matches the rest of the codebase style)
# ---------------------------------------------------------------------------

def _parse(args, name, default=None, cast=str):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        args.pop(i + 1); args.pop(i)
        return cast(v) if cast is not str else v
    return default


# ---------------------------------------------------------------------------
# LaTeX-style rendering (same robust check used elsewhere)
# ---------------------------------------------------------------------------

def _latex_actually_works() -> bool:
    import io
    saved = dict(plt.rcParams)
    try:
        plt.rcParams["text.usetex"] = True
        fig = plt.figure()
        fig.text(0.5, 0.5, r"$y/H$")
        fig.savefig(io.BytesIO(), format="png", dpi=50)
        plt.close(fig)
        return True
    except Exception:
        return False
    finally:
        plt.rcParams.update(saved)


def _enable_latex_style():
    has_tex = _latex_actually_works()
    plt.rcParams.update({
        "text.usetex": bool(has_tex),
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
    })
    if not has_tex:
        print("[box] LaTeX not usable; using mathtext (CM) fallback.")
    return bool(has_tex)


# ---------------------------------------------------------------------------
# DNS loader -- same code path as the pipeline
# ---------------------------------------------------------------------------

def load_dns_cube(dns_data_dir: str, timestep: str) -> np.ndarray:
    from dns_stats import GridConfig, load_field
    grid = GridConfig()
    Z = load_field(os.path.join(dns_data_dir, f"jet_mixfrac_{timestep}.dat"),
                   grid)
    return np.asarray(Z, dtype=np.float64)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    dns_data_dir = _parse(args, "--dns-data-dir",
                          default="/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198")
    timestep = _parse(args, "--timestep", default="0198")
    out_png = _parse(args, "--out", default="QC4PDE/box_faces.png")
    cmap_name = _parse(args, "--cmap", default="bluered")
    azim = float(_parse(args, "--azim", default="-60"))
    elev = float(_parse(args, "--elev", default="18"))
    show_cbar = "--colorbar" in args
    if show_cbar:
        args.remove("--colorbar")
    top_y = _parse(args, "--top-y", default="last").lower()
    if top_y not in ("first", "last"):
        print(f"[box] unknown --top-y '{top_y}', using 'last'")
        top_y = "last"

    _enable_latex_style()

    from dns_stats import GridConfig
    g = GridConfig()
    H = g.H

    print(f"[box] loading DNS mixfrac cube (t={timestep})...")
    Zc = load_dns_cube(dns_data_dir, timestep)          # (nx, ny, nz)
    nx, ny, nz = Zc.shape
    print(f"      cube shape = {Zc.shape}")

    # Physical, H-normalised coordinates, centred at 0 (same convention as the
    # contour figure). dx, dy, dz from the DNS grid spacing.
    dx = g.dx_m / H
    dy = getattr(g, "dy_m", g.dx_m) / H
    dz = getattr(g, "dz_m", g.dx_m) / H
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    z = (np.arange(nz) - (nz - 1) / 2.0) * dz

    # ---- the two faces ---------------------------------------------------
    # Front face: x-y plane at the FIRST z gridpoint.
    #   data indexed [:, :, 0]  -> shape (nx, ny)
    front = Zc[:, :, 0]
    # Right face: z-y plane at the LAST x gridpoint.
    #   data indexed [-1, :, :] -> shape (ny, nz)
    right = Zc[-1, :, :]

    # Shared colour normalisation across ALL three faces (front x-y at z[0],
    # right z-y at x[-1], top x-z at the chosen y edge).
    top_probe = Zc[:, -1, :] if top_y != "first" else Zc[:, 0, :]
    vmin = float(min(front.min(), right.min(), top_probe.min()))
    vmax = float(max(front.max(), right.max(), top_probe.max()))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Colormaps:
    #  'bluered' (default): deep blue background (low Z, the oxidiser
    #     co-flow) ramping through to a hot yellow-red flame (high Z, the
    #     fuel-rich jet core). Matches "background blue, flame yellow-red".
    #  'flame': black background -> deep red -> orange -> yellow.
    #  any named matplotlib colormap can be passed via --cmap.
    if cmap_name == "bluered":
        cmap = colors.LinearSegmentedColormap.from_list(
            "bluered",
            [
                (0.00, "#08306b"),   # deep blue  (background / oxidiser)
                (0.20, "#2171b5"),   # mid blue
                (0.38, "#6baed6"),   # light blue
                (0.50, "#f7f7c8"),   # pale yellow transition
                (0.65, "#fee08b"),   # yellow
                (0.82, "#f46d43"),   # orange
                (1.00, "#a50026"),   # deep red   (hottest flame)
            ],
        )
    elif cmap_name == "flame":
        cmap = colors.LinearSegmentedColormap.from_list(
            "flame",
            [
                (0.00, "#000000"),   # black background
                (0.15, "#1a0000"),   # near-black red
                (0.40, "#7f0000"),   # deep red
                (0.65, "#e02200"),   # red-orange
                (0.85, "#ff8800"),   # orange
                (1.00, "#ffee33"),   # hot yellow
            ],
        )
    else:
        cmap = cm.get_cmap(cmap_name)

    fig = plt.figure(figsize=(7.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # The schematic has y (cross-stream) pointing UP. matplotlib 3D uses its
    # z-axis as the vertical, so we map:
    #     screen horizontal-1 (plot x) <- data x  (streamwise)
    #     screen horizontal-2 (plot y) <- data z  (spanwise)
    #     screen vertical    (plot z) <- data y  (cross-stream, UP)
    # Faces are still the requested planes: x-y at first z, z-y at last x.

    # --- Front face: data x-y plane at z = z[0] ---------------------------
    #   varies over data-x and data-y; data-z fixed at z[0].
    #   plot coords: (x, z[0], y)  -> spans plot-x and plot-vertical, at the
    #   near spanwise wall (plot-y = z[0]).
    Xf, Yf = np.meshgrid(x, y, indexing="ij")          # (nx, ny) over x, y
    px = Xf                                            # plot-x  <- data x
    py = np.full_like(Xf, z[0])                        # plot-y  <- data z0
    pz = Yf                                            # plot-z  <- data y
    fc_front = cmap(norm(front))
    ax.plot_surface(px, py, pz, rcount=nx, ccount=ny,
                    facecolors=fc_front, shade=False,
                    linewidth=0, antialiased=False)

    # --- Right face: data z-y plane at x = x[-1] --------------------------
    #   varies over data-z and data-y; data-x fixed at x[-1].
    #   plot coords: (x[-1], z, y) -> spans plot-y and plot-vertical, at the
    #   far streamwise wall (plot-x = x[-1]).
    Zr, Yr = np.meshgrid(z, y, indexing="ij")          # (nz, ny) over z, y
    px2 = np.full_like(Zr, x[-1])                      # plot-x  <- data x_last
    py2 = Zr                                           # plot-y  <- data z
    pz2 = Yr                                           # plot-z  <- data y
    fc_right = cmap(norm(right.T))                     # (nz, ny) over z, y
    ax.plot_surface(px2, py2, pz2, rcount=nz, ccount=ny,
                    facecolors=fc_right, shade=False,
                    linewidth=0, antialiased=False)

    # --- Top face: data x-z plane at a cross-stream edge ------------------
    #   varies over data-x and data-z; data-y fixed at one end. With a centred
    #   cube, y[0] is the most-negative (bottom) edge and y[-1] the top edge.
    #   Default to the top edge so it reads as the "lid" of the box like the
    #   schematic; --top-y first uses y[0] (bottom) instead.
    if top_y == "first":
        y_top_idx, y_top_val = 0, y[0]
    else:
        y_top_idx, y_top_val = -1, y[-1]
    top = Zc[:, y_top_idx, :]                          # (nx, nz) over x, z
    Xt, Zt = np.meshgrid(x, z, indexing="ij")          # (nx, nz) over x, z
    px3 = Xt                                            # plot-x  <- data x
    py3 = Zt                                            # plot-y  <- data z
    pz3 = np.full_like(Xt, y_top_val)                  # plot-z  <- data y_edge
    fc_top = cmap(norm(top))                            # (nx, nz) over x, z
    ax.plot_surface(px3, py3, pz3, rcount=nx, ccount=nz,
                    facecolors=fc_top, shade=False,
                    linewidth=0, antialiased=False)

    # ---- limits & true physical aspect -----------------------------------
    # plot-x = data x (streamwise), plot-y = data z (spanwise),
    # plot-z = data y (cross-stream, vertical).
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(z.min(), z.max())
    ax.set_zlim(y.min(), y.max())
    ax.set_box_aspect((nx * dx, nz * dz, ny * dy))     # true physical shape

    ax.view_init(elev=elev, azim=azim)

    # ---- strip ALL decoration (clean floating box, like the schematic) ---
    ax.set_axis_off()                    # hides axis lines, ticks, tick labels
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticks([])
        # make the background panes fully transparent so no grey walls show
        axis.pane.set_visible(False)
        axis.line.set_visible(False)
    # some mpl versions still draw a faint 3D frame; blank it via margins
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    except Exception:
        pass

    # ---- optional colorbar (off by default; schematic has none) ----------
    if show_cbar:
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(r"$Z$", fontsize=13)
        cbar.ax.tick_params(labelsize=10)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"[box] wrote {out_png}")


if __name__ == "__main__":
    main()
