"""Smoke test for derived_fields.py.

Tests:
1. Stream beta values are sensible (beta_ox < 0 < beta_fuel).
2. When given mass fractions that are linear combinations of pure fuel
   and pure oxidizer streams, Bilger reconstruction recovers the mixing
   parameter with near-zero error.
3. For a Z(y) = tanh profile with constant alpha, rho, the reconstructed
   chi matches the analytical formula to finite-difference accuracy.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from derived_fields import (
    StreamConfig, _mole_to_mass_fractions, _beta_from_mass_fractions,
    compute_Z_bilger, compute_chi,
    relative_l2_error, pointwise_error_stats,
)


def test_stream_betas():
    """beta_ox should be negative (oxygen-rich), beta_fuel should be positive
    (C and H carriers present, no O)."""
    sc = StreamConfig()
    Y_fuel = _mole_to_mass_fractions(sc.fuel_mole_fractions)
    Y_ox = _mole_to_mass_fractions(sc.ox_mole_fractions)
    beta_f = _beta_from_mass_fractions(Y_fuel)
    beta_o = _beta_from_mass_fractions(Y_ox)
    print(f"beta_fuel = {beta_f:+.6e}")
    print(f"beta_ox   = {beta_o:+.6e}")
    assert beta_f > 0, "expected fuel stream beta > 0"
    assert beta_o < 0, "expected oxidizer stream beta < 0"
    # Sanity: sum of mass fractions in each stream = 1
    assert abs(sum(Y_fuel.values()) - 1.0) < 1e-10
    assert abs(sum(Y_ox.values()) - 1.0) < 1e-10
    print("  stream betas OK\n")


def test_bilger_linear_mixing():
    """Pure two-stream mixing: Y = phi * Y_fuel + (1 - phi) * Y_ox.
    Bilger's Z should equal phi exactly (within float64 precision),
    because beta is linear in Y and the formula is its normalization."""
    sc = StreamConfig()
    Y_fuel = _mole_to_mass_fractions(sc.fuel_mole_fractions)
    Y_ox = _mole_to_mass_fractions(sc.ox_mole_fractions)

    # Build a 3D "field" that represents pure mixing with phi varying along x
    shape = (20, 8, 8)
    phi = np.linspace(0.0, 1.0, shape[0])[:, None, None] * np.ones(shape)

    # All species that appear in either stream
    species_list = sorted(set(Y_fuel) | set(Y_ox))
    fields = {}
    for sp in species_list:
        yf = Y_fuel.get(sp, 0.0)
        yo = Y_ox.get(sp, 0.0)
        fields[f"Y_{sp}"] = (phi * yf + (1 - phi) * yo).astype(np.float32)

    Z_rec, info = compute_Z_bilger(fields)
    err = float(np.abs(Z_rec - phi).max())
    print(f"Z_rec range     : [{Z_rec.min():.4f}, {Z_rec.max():.4f}]")
    print(f"Linear-mixing max|Z - phi| = {err:.3e}")
    # With float32 inputs and float64 arithmetic inside, precision ~1e-6 is fine
    assert err < 1e-5, f"Bilger inversion error too large: {err}"
    print("  linear mixing test OK\n")


def test_chi_gradient():
    """Analytical: Z(y) = 0.5*(1 - tanh(k*y/L)), alpha and rho constant.
    chi = 2*D*|grad Z|^2 = 2*alpha*|grad Z|^2

    With 8th-order centered differences on a well-resolved smooth
    profile, we expect error << 1e-5 in the interior.
    """
    nx, ny, nz = 8, 64, 8
    Lx, Ly, Lz = 1.0, 2.0, 1.0
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    y = (np.arange(ny) - (ny - 1) / 2) * dy
    k = 3.0
    Z_prof = 0.5 * (1.0 - np.tanh(k * y / Ly))
    Z3 = np.broadcast_to(Z_prof[None, :, None], (nx, ny, nz)).astype(np.float32)

    alpha0 = 2.5e-5
    rho0 = 1.1
    alpha = np.full((nx, ny, nz), alpha0, dtype=np.float32)
    rho = np.full((nx, ny, nz), rho0, dtype=np.float32)

    dZdy_an = -0.5 * (k / Ly) * (1.0 - np.tanh(k * y / Ly) ** 2)
    chi_an = 2.0 * alpha0 * dZdy_an ** 2

    # Test each order
    for order in (2, 4, 6, 8):
        chi_rec = compute_chi(Z3, alpha, rho, dx, dy, dz,
                              alpha_is="thermal_diffusivity",
                              order=order)
        chi_rec_prof = chi_rec[nx // 2, :, nz // 2]
        skip = max(order, 4)  # skip enough boundary rows
        rel = (np.abs(chi_rec_prof[skip:-skip] - chi_an[skip:-skip])
               / (chi_an[skip:-skip] + 1e-30))
        print(f"  order={order}: interior rel.err max={rel.max():.3e}, "
              f"mean={rel.mean():.3e}")

    # 8th-order should be very accurate
    chi_rec8 = compute_chi(Z3, alpha, rho, dx, dy, dz, order=8)
    chi_rec8_prof = chi_rec8[nx // 2, :, nz // 2]
    rel8 = (np.abs(chi_rec8_prof[8:-8] - chi_an[8:-8])
            / (chi_an[8:-8] + 1e-30))
    assert rel8.max() < 1e-5, f"8th-order chi error too large: {rel8.max()}"
    print("  8th-order chi test OK\n")


def main():
    print("--- derived_fields.py smoke test ---\n")
    test_stream_betas()
    test_bilger_linear_mixing()
    test_chi_gradient()
    print("All tests passed.")


if __name__ == "__main__":
    main()
