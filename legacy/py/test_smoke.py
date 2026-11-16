"""Smoke test: run dns_stats on synthetic fields at a tiny grid to make
sure the whole pipeline executes end-to-end without errors.
"""
import os
import sys
import shutil
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dns_stats import (
    GridConfig, load_all_dns_fields, compute_all_stats,
)

def write_field(path, arr_nz_ny_nx):
    # write as big-endian float32, same layout dns_stats expects
    arr_nz_ny_nx.astype(">f4").tofile(path)

def main():
    # Tiny "full" grid and tiny "center cube"
    grid = GridConfig(
        nx_full=32, ny_full=40, nz_full=24, n=16,
        H=1.0, Lx_H_full=12.0, Ly_H_full=14.0, Lz_H_full=8.0,
    )

    tmp = tempfile.mkdtemp(prefix="dns_smoke_")
    try:
        shape = (grid.nz_full, grid.ny_full, grid.nx_full)
        # Build Z that varies smoothly across y (like a jet profile)
        zi, yi, xi = np.indices(shape)
        y_center = (grid.ny_full - 1) / 2.0
        y_norm = (yi - y_center) / y_center  # in [-1, 1]
        Z = np.exp(-3.0 * y_norm**2).astype(">f4")   # peaks at centerline
        T = (500 + 800 * np.exp(-4.0 * (y_norm**2 + 0.1 * np.sin(xi*0.3)**2))).astype(">f4")
        Y_CO = (0.5 * np.exp(-3.0 * y_norm**2)).astype(">f4")
        Y_CO2 = (0.08 * np.exp(-4.0 * y_norm**2)).astype(">f4")
        Y_OH = (0.001 * np.exp(-5.0 * y_norm**2)).astype(">f4")
        Y_O2 = (0.25 * (1.0 - np.exp(-3.0 * y_norm**2))).astype(">f4")
        chi = (50 * np.exp(-6.0 * y_norm**2)).astype(">f4")

        for name, arr in [
            ("mixfrac", Z), ("T", T),
            ("Y_CO", Y_CO), ("Y_CO2", Y_CO2), ("Y_OH", Y_OH), ("Y_O2", Y_O2),
            ("chi", chi),
        ]:
            write_field(os.path.join(tmp, f"jet_{name}_0001.dat"), arr)

        fields = load_all_dns_fields(tmp, "0001", grid, required_only=True)
        print("Loaded fields:", sorted(fields.keys()))
        for k, v in fields.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                  f"range=[{v.min():.3e}, {v.max():.3e}]")

        results = compute_all_stats(fields, grid)

        print("\n--- Summary ---")
        print(f"delta_Z/(2H) = {results['delta_Z_over_2H']:.4f}")
        print(f"chi profile shape: {results['chi_profile'].shape}")
        print(f"mean T profile shape: {results['profiles']['T']['mean'].shape}")
        print(f"conditional T: {results['conditional_T_on_Z']['E_T'].shape}, "
              f"finite count: {np.isfinite(results['conditional_T_on_Z']['E_T']).sum()}")
        print(f"PDF Z: {results['pdf_mixfrac']['pdf'].shape}, "
              f"integral ~ {np.trapezoid(results['pdf_mixfrac']['pdf'], results['pdf_mixfrac']['psi_Z']):.3f}")
        print(f"joint PDF shape: {results['joint_pdf_Z_YCO2']['joint_pdf'].shape}")
        print(f"scatter sample size: {results['manifold_scatter']['Z'].size}")
        ext = results["extinction"]
        print(f"extinction: M_vol={ext['volume_averaged_marker']:.3e}, "
              f"P_cond={ext['conditional_extinction_prob']:.3e}, "
              f"T_st={ext['T_on_stoich_surface']:.2f}, "
              f"n_near_st={ext['n_near_st_cells']}")
        print("\nOK")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
