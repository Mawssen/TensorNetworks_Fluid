"""
derived_fields.py
-----------------
Reconstruct derived fields from independent DNS variables.

Two fields are reconstructed:

1. Mixture fraction Z via Bilger's formulation, from species mass
   fractions alone.

2. Scalar dissipation rate chi = (2*gamma/rho) * |grad Z|^2, from
   thermal diffusivity, density, and the mixture fraction field.

Each reconstruction can be compared against the precomputed DNS file
(jet_mixfrac_*.dat or jet_chi_*.dat) for validation.

All inputs are 3D numpy arrays with axes (x, y, z) matching the
convention in dns_stats.py. The center cube is assumed periodic in x
and z, and non-periodic in y.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Atomic weights (g/mol) -- element-specific constants drop out of ratios,
# but using standard values lets beta have physical units.
# -----------------------------------------------------------------------------
W_C = 12.011
W_H = 1.008
W_O = 15.999
W_N = 14.007

# Molecular weights of species in the skeletal CO/H2 mechanism.
# Computed from atomic weights.
W_SPECIES: Dict[str, float] = {
    "CO":  W_C + W_O,          # 28.010
    "CO2": W_C + 2 * W_O,      # 44.009
    "H":   W_H,                # 1.008
    "H2":  2 * W_H,            # 2.016
    "H2O": 2 * W_H + W_O,      # 18.015
    "HCO": W_H + W_C + W_O,    # 29.018
    "HO2": W_H + 2 * W_O,      # 33.006
    "O":   W_O,                # 15.999
    "O2":  2 * W_O,            # 31.998
    "OH":  W_O + W_H,          # 17.007
    "N2":  2 * W_N,            # 28.014
}

# Atom counts for each species: (nC, nH, nO)
SPECIES_COMPOSITION: Dict[str, Tuple[int, int, int]] = {
    "CO":  (1, 0, 1),
    "CO2": (1, 0, 2),
    "H":   (0, 1, 0),
    "H2":  (0, 2, 0),
    "H2O": (0, 2, 1),
    "HCO": (1, 1, 1),
    "HO2": (0, 1, 2),
    "O":   (0, 0, 1),
    "O2":  (0, 0, 2),
    "OH":  (0, 1, 1),
    "N2":  (0, 0, 0),  # inert, does not enter beta
}


# -----------------------------------------------------------------------------
# Stream compositions for the Hawkes et al. CO/H2 DNS
# -----------------------------------------------------------------------------
@dataclass
class StreamConfig:
    """Volume (=mole) fractions in the two streams of the planar jet.

    Defaults match the flame described in Hawkes et al. (2007) and the
    paper under study: fuel stream = 50% CO, 10% H2, 40% N2;
    oxidizer stream = 25% O2, 75% N2.
    """
    fuel_mole_fractions: Dict[str, float] = None
    ox_mole_fractions: Dict[str, float] = None

    def __post_init__(self):
        if self.fuel_mole_fractions is None:
            self.fuel_mole_fractions = {"CO": 0.50, "H2": 0.10, "N2": 0.40}
        if self.ox_mole_fractions is None:
            self.ox_mole_fractions = {"O2": 0.25, "N2": 0.75}


def _mole_to_mass_fractions(x: Dict[str, float]) -> Dict[str, float]:
    """Convert species mole fractions to mass fractions using W_SPECIES."""
    mean_W = sum(xi * W_SPECIES[sp] for sp, xi in x.items())
    return {sp: xi * W_SPECIES[sp] / mean_W for sp, xi in x.items()}


def _beta_from_mass_fractions(Y: Dict[str, float]) -> float:
    """Compute Bilger's beta from a species-mass-fraction dictionary.

    Y may be a dict of scalars (stream compositions) or a dict of
    ndarrays (field compositions). Missing species are treated as 0.

    beta = 2*Y_C/W_C + Y_H/(2*W_H) - Y_O/W_O
    where Y_E = sum over species alpha of (n_{E,alpha} * W_E / W_alpha) * Y_alpha
    """
    Y_C = 0.0
    Y_H = 0.0
    Y_O = 0.0
    for sp, Yalpha in Y.items():
        if sp not in SPECIES_COMPOSITION:
            raise KeyError(f"Unknown species '{sp}'; add it to "
                           f"SPECIES_COMPOSITION / W_SPECIES.")
        nC, nH, nO = SPECIES_COMPOSITION[sp]
        Walpha = W_SPECIES[sp]
        if nC:
            Y_C = Y_C + (nC * W_C / Walpha) * Yalpha
        if nH:
            Y_H = Y_H + (nH * W_H / Walpha) * Yalpha
        if nO:
            Y_O = Y_O + (nO * W_O / Walpha) * Yalpha
    return 2.0 * Y_C / W_C + Y_H / (2.0 * W_H) - Y_O / W_O


# -----------------------------------------------------------------------------
# Mixture fraction (Bilger)
# -----------------------------------------------------------------------------

# Map from the filename suffix used in dns_stats.load_all_dns_fields to
# the species name used here. Extend if more species appear.
_FIELD_TO_SPECIES = {
    "Y_CO":  "CO",
    "Y_CO2": "CO2",
    "Y_H":   "H",
    "Y_H2":  "H2",
    "Y_H2O": "H2O",
    "Y_HCO": "HCO",
    "Y_HO2": "HO2",
    "Y_O":   "O",
    "Y_O2":  "O2",
    "Y_OH":  "OH",
}


def compute_Z_bilger(
    fields: Dict[str, np.ndarray],
    streams: StreamConfig = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Reconstruct the mixture fraction field via Bilger's formula.

    Parameters
    ----------
    fields : dict
        Must contain all available Y_* species fields. Missing species
        are treated as zero (and reported in the returned diagnostics).
    streams : StreamConfig, optional
        Stream compositions. Defaults match the paper's flame.

    Returns
    -------
    Z : ndarray of shape (nx, ny, nz)
        Reconstructed mixture fraction, clipped to [0, 1].
    info : dict
        Diagnostics: beta_fuel, beta_ox, list of species used, list of
        species missing, sum-of-Y statistics.
    """
    if streams is None:
        streams = StreamConfig()

    # Assemble species dict and track what we actually used.
    species_Y: Dict[str, np.ndarray] = {}
    missing = []
    for fkey, spname in _FIELD_TO_SPECIES.items():
        if fkey in fields:
            species_Y[spname] = fields[fkey].astype(np.float64)
        else:
            missing.append(spname)

    if not species_Y:
        raise ValueError("No species mass-fraction fields found. Need at "
                         "least Y_CO, Y_CO2, Y_H2, Y_H2O, Y_O2, Y_OH etc.")

    # Pointwise beta field
    beta_field = _beta_from_mass_fractions(species_Y)

    # Stream betas: convert mole -> mass, then apply same formula
    Y_fuel = _mole_to_mass_fractions(streams.fuel_mole_fractions)
    Y_ox = _mole_to_mass_fractions(streams.ox_mole_fractions)
    beta_fuel = _beta_from_mass_fractions(Y_fuel)
    beta_ox = _beta_from_mass_fractions(Y_ox)

    denom = beta_fuel - beta_ox
    if abs(denom) < 1e-12:
        raise ValueError(f"beta_fuel - beta_ox = {denom}, too small to "
                         "normalize. Check stream compositions.")

    Z = (beta_field - beta_ox) / denom
    Z_clipped = np.clip(Z, 0.0, 1.0).astype(np.float32)

    # Diagnostics: does sum_alpha Y_alpha + Y_N2 (if present) hit 1?
    sum_Y = sum(species_Y.values())
    info = {
        "beta_fuel": beta_fuel,
        "beta_ox": beta_ox,
        "beta_denom": denom,
        "species_used": sorted(species_Y.keys()),
        "species_missing": sorted(missing),
        "sumY_mean": float(sum_Y.mean()),
        "sumY_min": float(sum_Y.min()),
        "sumY_max": float(sum_Y.max()),
        "Z_raw_min": float(Z.min()),
        "Z_raw_max": float(Z.max()),
        "Z_clip_fraction": float(((Z < 0) | (Z > 1)).mean()),
    }
    return Z_clipped, info


# -----------------------------------------------------------------------------
# Scalar dissipation rate
# -----------------------------------------------------------------------------

# --- Centered FD coefficients for first derivative ---
# df/dx|_i = (1/h) * sum_{k=1}^{K} a_k * (f_{i+k} - f_{i-k})
_CENTERED_COEFFS = {
    2: [(1, 1.0 / 2.0)],
    4: [(1, 2.0 / 3.0), (2, -1.0 / 12.0)],
    6: [(1, 3.0 / 4.0), (2, -3.0 / 20.0), (3, 1.0 / 60.0)],
    # 8th-order matches S3D's iorder=8
    8: [(1, 4.0 / 5.0), (2, -1.0 / 5.0),
        (3, 4.0 / 105.0), (4, -1.0 / 280.0)],
}

# --- Forward (one-sided) FD coefficients for first derivative ---
# df/dx|_0 = (1/h) * sum_{j} c_j * f_j
# Coefficients give an O(h^p) approximation at i=0 using points 0..p.
# Source: standard explicit FD tables (Fornberg 1988).
_FORWARD_COEFFS = {
    2: [-3.0/2.0, 2.0, -1.0/2.0],
    4: [-25.0/12.0, 4.0, -3.0, 4.0/3.0, -1.0/4.0],
    6: [-49.0/20.0, 6.0, -15.0/2.0, 20.0/3.0, -15.0/4.0,
        6.0/5.0, -1.0/6.0],
    8: [-761.0/280.0, 8.0, -14.0, 56.0/3.0, -35.0/2.0,
        56.0/5.0, -14.0/3.0, 8.0/7.0, -1.0/8.0],
}


def _apply_forward_along_axis(f: np.ndarray, axis: int, i: int,
                              coeffs) -> np.ndarray:
    """Compute forward FD at index `i` along `axis` using point i, i+1, ...
    The slab returned has shape equal to f with `axis` reduced to size 1.
    """
    out = None
    for j, c in enumerate(coeffs):
        sl = [slice(None)] * f.ndim
        sl[axis] = slice(i + j, i + j + 1)
        term = c * f[tuple(sl)]
        out = term if out is None else out + term
    return out


def _apply_backward_along_axis(f: np.ndarray, axis: int, i: int,
                               coeffs) -> np.ndarray:
    """Compute backward FD at index `i` along `axis` using point i, i-1, ...
    Backward stencil = -1 * (forward coefficients applied to reversed indices).
    """
    out = None
    for j, c in enumerate(coeffs):
        sl = [slice(None)] * f.ndim
        sl[axis] = slice(i - j, i - j + 1)
        term = (-c) * f[tuple(sl)]
        out = term if out is None else out + term
    return out


def _deriv_along_axis_nonperiodic(
    f: np.ndarray, h: float, axis: int, order: int,
) -> np.ndarray:
    """Spatial derivative along `axis` with non-periodic BCs.
    Interior: centered scheme of given order. Boundary points within
    K = order/2 cells: forward (low side) or backward (high side)
    one-sided scheme of the SAME order.
    """
    if order not in _CENTERED_COEFFS:
        raise ValueError(f"Unsupported order={order}; use 2, 4, 6, or 8.")
    centered = _CENTERED_COEFFS[order]
    forward = _FORWARD_COEFFS[order]
    K = max(k for k, _ in centered)

    n = f.shape[axis]
    if n < len(forward):
        raise ValueError(f"Axis {axis} has size {n} but order={order} "
                         f"needs at least {len(forward)} points.")
    if n < 2 * K:
        # No room for centered interior; fall back to forward at all points
        out = np.zeros_like(f)
        for i in range(K):
            slab = _apply_forward_along_axis(f, axis, i, forward) / h
            sl = [slice(None)] * f.ndim
            sl[axis] = slice(i, i + 1)
            out[tuple(sl)] = slab
        for i in range(K, n):
            slab = _apply_backward_along_axis(f, axis, i, forward) / h
            sl = [slice(None)] * f.ndim
            sl[axis] = slice(i, i + 1)
            out[tuple(sl)] = slab
        return out

    out = np.zeros_like(f)

    # --- Interior: centered ---
    for k, a in centered:
        sh_pos = np.zeros_like(f)
        sh_neg = np.zeros_like(f)
        sl_p_dst = [slice(None)] * f.ndim; sl_p_src = [slice(None)] * f.ndim
        sl_n_dst = [slice(None)] * f.ndim; sl_n_src = [slice(None)] * f.ndim
        sl_p_dst[axis] = slice(0, n - k)
        sl_p_src[axis] = slice(k, n)
        sl_n_dst[axis] = slice(k, n)
        sl_n_src[axis] = slice(0, n - k)
        sh_pos[tuple(sl_p_dst)] = f[tuple(sl_p_src)]
        sh_neg[tuple(sl_n_dst)] = f[tuple(sl_n_src)]
        out = out + a * (sh_pos - sh_neg)
    out = out / h

    # --- Boundaries: one-sided of same order ---
    for i in range(K):
        slab = _apply_forward_along_axis(f, axis, i, forward) / h
        sl = [slice(None)] * f.ndim
        sl[axis] = slice(i, i + 1)
        out[tuple(sl)] = slab
    for i in range(n - K, n):
        slab = _apply_backward_along_axis(f, axis, i, forward) / h
        sl = [slice(None)] * f.ndim
        sl[axis] = slice(i, i + 1)
        out[tuple(sl)] = slab

    return out


def _gradient_periodic_x_z(
    f: np.ndarray,
    dx: float, dy: float, dz: float,
    order: int = 2,
    periodic_xz: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centered finite-difference gradient.

    With periodic_xz=True (default, full grid): periodic BCs in x and z,
    non-periodic (one-sided at boundaries) in y.

    With periodic_xz=False (cube): non-periodic (one-sided at boundaries)
    in ALL three directions. This is the correct treatment for the
    extracted center cube, since the cube boundaries are not true
    periodic neighbors.

    Input axes: (x, y, z). Returns (df/dx, df/dy, df/dz) with same shape.

    Parameters
    ----------
    order : int
        Spatial order of accuracy. Supported: 2, 4, 6, 8. S3D uses 8.
    periodic_xz : bool
        If True, treat x and z as periodic; if False, use one-sided
        boundary stencils in x and z too.
    """
    if order not in _CENTERED_COEFFS:
        raise ValueError(f"Unsupported order={order}; use 2, 4, 6, or 8.")
    coeffs = _CENTERED_COEFFS[order]

    if periodic_xz:
        def _deriv_periodic(g: np.ndarray, h: float, axis: int) -> np.ndarray:
            out = np.zeros_like(g)
            for k, a in coeffs:
                out = out + a * (np.roll(g, -k, axis=axis)
                                 - np.roll(g, k, axis=axis))
            return out / h
        dfdx = _deriv_periodic(f, dx, axis=0)
        dfdz = _deriv_periodic(f, dz, axis=2)
    else:
        dfdx = _deriv_along_axis_nonperiodic(f, dx, axis=0, order=order)
        dfdz = _deriv_along_axis_nonperiodic(f, dz, axis=2, order=order)

    # y is always non-periodic
    dfdy = _deriv_along_axis_nonperiodic(f, dy, axis=1, order=order)

    return dfdx, dfdy, dfdz


def compute_chi(
    Z: np.ndarray,
    alpha: np.ndarray,
    rho: np.ndarray,
    dx: float, dy: float, dz: float,
    alpha_is: str = "thermal_diffusivity",
    order: int = 8,
    periodic_xz: bool = True,
) -> np.ndarray:
    """Reconstruct the scalar dissipation rate.

        chi = 2 * D * |grad Z|^2      (Peters, Pope, standard form)

    where D is the diffusivity of the mixture fraction. Under unity
    Lewis number, D equals the thermal diffusivity lambda/(rho*cp).

    Note on the paper's notation: the paper writes "chi = 2*gamma/rho
    (grad Z . grad Z) where gamma is the thermal diffusivity
    coefficient." That formula is only dimensionally consistent if
    gamma = lambda/cp (kg/m/s), so that gamma/rho = thermal diffusivity.
    When the diffusivity D is provided directly (m^2/s), no factor of
    1/rho is needed.

    Parameters
    ----------
    Z, alpha, rho : ndarray (nx, ny, nz)
        Mixture fraction, diffusivity field, density.
    dx, dy, dz : float
        Uniform grid spacings in meters.
    alpha_is : str
        'thermal_diffusivity' (D = alpha, m^2/s) or
        'lambda_over_cp'      (D = alpha/rho).
    order : int
        Spatial order for the gradient: 2, 4, 6, or 8. Default 8 to match S3D.
    periodic_xz : bool
        If True (default), x and z gradients use periodic BCs. This is
        correct for the FULL grid since the DNS domain is periodic in
        x and z. For the EXTRACTED CENTER CUBE, pass False so that x and
        z use one-sided boundary stencils -- the cube boundaries are not
        periodic neighbors and assuming periodicity introduces a small
        artifact along the cube's x and z faces.

    Returns
    -------
    chi : ndarray (nx, ny, nz), float32
    """
    Z_f = Z.astype(np.float64)
    alpha_f = alpha.astype(np.float64)
    rho_f = rho.astype(np.float64)

    dZdx, dZdy, dZdz = _gradient_periodic_x_z(
        Z_f, dx, dy, dz, order=order, periodic_xz=periodic_xz)
    grad2 = dZdx * dZdx + dZdy * dZdy + dZdz * dZdz

    if alpha_is == "thermal_diffusivity":
        D = alpha_f
    elif alpha_is == "lambda_over_cp":
        D = alpha_f / np.maximum(rho_f, 1e-30)
    else:
        raise ValueError(f"Unknown alpha_is={alpha_is!r}")

    chi = 2.0 * D * grad2
    return chi.astype(np.float32)


# -----------------------------------------------------------------------------
# Validation: reconstructed vs. precomputed
# -----------------------------------------------------------------------------

def relative_l2_error(
    reconstructed: np.ndarray,
    reference: np.ndarray,
) -> float:
    """||recon - ref||_2 / ||ref||_2  (standard fidelity metric)."""
    ref = reference.astype(np.float64)
    rec = reconstructed.astype(np.float64)
    denom = np.linalg.norm(ref)
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(rec - ref) / denom)


def pointwise_error_stats(
    reconstructed: np.ndarray,
    reference: np.ndarray,
) -> Dict[str, float]:
    """Pointwise absolute-error summary: mean, max, std."""
    diff = (reconstructed.astype(np.float64)
            - reference.astype(np.float64))
    return {
        "abs_mean": float(np.abs(diff).mean()),
        "abs_max":  float(np.abs(diff).max()),
        "abs_std":  float(np.abs(diff).std()),
        "signed_mean": float(diff.mean()),
        "correlation": float(np.corrcoef(
            reconstructed.ravel(), reference.ravel())[0, 1]),
    }


def chi_diagnostics(
    chi_reconstructed: np.ndarray,
    chi_reference: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Extra diagnostics to understand where chi_recon and chi_ref differ.

    Returns x-z-averaged profiles of both (the quantity shown in paper
    Fig. 9), a 2D slice at z = center, and peak ratios.
    """
    # x-z averaged profile (paper's Reynolds-averaged chi(y))
    prof_ref = chi_reference.mean(axis=(0, 2))
    prof_rec = chi_reconstructed.mean(axis=(0, 2))

    # 2D slice at the center z-plane
    nz = chi_reference.shape[2]
    k0 = nz // 2
    slice_ref = chi_reference[:, :, k0]
    slice_rec = chi_reconstructed[:, :, k0]

    return {
        "profile_y_reference": prof_ref,
        "profile_y_reconstructed": prof_rec,
        "profile_rel_l2": relative_l2_error(prof_rec, prof_ref),
        "profile_correlation": float(np.corrcoef(prof_rec, prof_ref)[0, 1]),
        "slice_z0_reference": slice_ref,
        "slice_z0_reconstructed": slice_rec,
        "peak_reference": float(chi_reference.max()),
        "peak_reconstructed": float(chi_reconstructed.max()),
        "peak_ratio": float(chi_reconstructed.max()
                            / max(chi_reference.max(), 1e-30)),
    }


def validate_reconstructions(
    fields: Dict[str, np.ndarray],
    grid,
    streams: StreamConfig = None,
    alpha_is: str = "thermal_diffusivity",
    fd_order: int = 8,
    periodic_xz_cube: bool = False,
) -> Dict:
    """Run both reconstructions and compare against the stored fields.

    Parameters
    ----------
    fd_order : int
        Order of the centered FD scheme. Supported: 2, 4, 6, 8.
    periodic_xz_cube : bool
        Whether to treat the cube as periodic in x and z. Default False
        (cube boundaries are NOT true periodic neighbors -- they're a
        centered subset of the full periodic domain). Set True only as
        a diagnostic, e.g. to measure how much the boundary treatment
        actually matters.
    """
    result = {}

    # --- Z reconstruction ---
    Z_rec, Z_info = compute_Z_bilger(fields, streams=streams)
    result["Z_reconstructed"] = Z_rec
    result["Z_info"] = Z_info

    if "mixfrac" in fields:
        Z_ref = fields["mixfrac"]
        result["Z_rel_l2"] = relative_l2_error(Z_rec, Z_ref)
        result["Z_pointwise"] = pointwise_error_stats(Z_rec, Z_ref)
    else:
        result["Z_rel_l2"] = None
        result["Z_pointwise"] = None

    # --- chi reconstruction on the CUBE ---
    if "alpha" in fields and "rho" in fields and "mixfrac" in fields:
        dx = grid.dx_m
        dy = grid.dy_m
        dz = grid.dz_m
        chi_rec = compute_chi(
            fields["mixfrac"], fields["alpha"], fields["rho"],
            dx, dy, dz, alpha_is=alpha_is, order=fd_order,
            periodic_xz=periodic_xz_cube,
        )
        result["chi_reconstructed"] = chi_rec
        result["chi_fd_order"] = fd_order
        result["chi_periodic_xz"] = periodic_xz_cube

        if "chi" in fields:
            chi_ref = fields["chi"]
            result["chi_rel_l2"] = relative_l2_error(chi_rec, chi_ref)
            result["chi_pointwise"] = pointwise_error_stats(chi_rec, chi_ref)
            result["chi_diagnostics"] = chi_diagnostics(chi_rec, chi_ref)
        else:
            result["chi_rel_l2"] = None
            result["chi_pointwise"] = None
            result["chi_diagnostics"] = None
    else:
        result["chi_reconstructed"] = None
        result["chi_rel_l2"] = None
        result["chi_pointwise"] = None
        missing = [k for k in ("alpha", "rho", "mixfrac") if k not in fields]
        result["chi_skipped_because_missing"] = missing

    return result


def chi_fd_order_sweep(
    fields: Dict[str, np.ndarray],
    grid,
    alpha_is: str = "thermal_diffusivity",
    orders: Tuple[int, ...] = (2, 4, 6, 8),
    periodic_xz_cube: bool = False,
) -> Dict[int, Dict[str, float]]:
    """Run chi reconstruction at several FD orders and report metrics.

    Useful for diagnosing the relationship between FD order and agreement
    with stored chi. The cube is non-periodic by default; pass
    periodic_xz_cube=True to test the periodic-wrap variant.
    """
    if ("chi" not in fields or "mixfrac" not in fields
            or "alpha" not in fields or "rho" not in fields):
        return {}
    chi_ref = fields["chi"]
    dx = grid.dx_m
    dy = grid.dy_m
    dz = grid.dz_m

    out: Dict[int, Dict[str, float]] = {}
    for o in orders:
        chi_rec = compute_chi(
            fields["mixfrac"], fields["alpha"], fields["rho"],
            dx, dy, dz, alpha_is=alpha_is, order=o,
            periodic_xz=periodic_xz_cube,
        )
        d = chi_diagnostics(chi_rec, chi_ref)
        out[o] = {
            "pointwise_rel_l2": relative_l2_error(chi_rec, chi_ref),
            "pointwise_corr": pointwise_error_stats(chi_rec, chi_ref)["correlation"],
            "profile_rel_l2": d["profile_rel_l2"],
            "profile_corr": d["profile_correlation"],
            "peak_ratio": d["peak_ratio"],
        }
    return out


def print_fd_order_sweep(sweep: Dict[int, Dict[str, float]]) -> None:
    """Print a table comparing metrics across FD orders."""
    if not sweep:
        return
    print("\n--- chi vs stored: FD-order sweep ---")
    print(f"  {'order':>5s}  {'pt_L2':>8s}  {'pt_corr':>8s}  "
          f"{'prof_L2':>8s}  {'prof_corr':>10s}  {'peak_ratio':>10s}")
    for o, m in sweep.items():
        print(f"  {o:>5d}  {m['pointwise_rel_l2']:>8.3f}  "
              f"{m['pointwise_corr']:>8.4f}  "
              f"{m['profile_rel_l2']:>8.3f}  "
              f"{m['profile_corr']:>10.6f}  "
              f"{m['peak_ratio']:>10.2f}")
    print()


def validate_reconstructions_full(
    data_dir: str,
    timestep: str,
    grid,
    streams: StreamConfig = None,
    alpha_is: str = "thermal_diffusivity",
    fd_order: int = 8,
    periodic_xz_cube: bool = False,
) -> Dict:
    """Load FULL-GRID fields, reconstruct Z and chi on the full domain
    (with physically-correct periodic BCs in x and z), then crop to the
    512^3 center cube for comparison against the stored cube fields.

    The "cube-only" reconstruction (chi reconstructed directly on the
    cube without going through the full grid) honors periodic_xz_cube:
    set True to test the periodic-wrap variant on cube boundaries.

    Memory cost: loads ~12 full-grid fields at ~2 GB each -> ~24 GB peak.
    """
    import os
    from dns_stats import load_field_full, extract_center_cube

    if streams is None:
        streams = StreamConfig()

    def _path(name):
        return os.path.join(data_dir, f"jet_{name}_{timestep}.dat")

    species_names = list(_FIELD_TO_SPECIES.keys())
    species_full: Dict[str, np.ndarray] = {}
    for name in species_names:
        p = _path(name)
        if os.path.exists(p):
            species_full[name] = load_field_full(p, grid)

    mixfrac_full = load_field_full(_path("mixfrac"), grid)
    alpha_full = load_field_full(_path("alpha"), grid)
    rho_full = load_field_full(_path("rho"), grid)
    chi_full_ref = load_field_full(_path("chi"), grid)

    # --- Z reconstruction on FULL grid ---
    Z_rec_full, Z_info = compute_Z_bilger(species_full, streams=streams)

    # --- chi reconstruction on FULL grid: physically-correct periodic BCs.
    # The full domain IS periodic in x and z, so this is always periodic
    # regardless of the cube setting.
    chi_rec_full = compute_chi(
        mixfrac_full, alpha_full, rho_full,
        grid.dx_m, grid.dy_m, grid.dz_m,
        alpha_is=alpha_is, order=fd_order,
        periodic_xz=True,
    )

    # --- Crop to center cube and compare against stored cube ---
    Z_rec_cube = extract_center_cube(Z_rec_full, grid)
    chi_rec_cube = extract_center_cube(chi_rec_full, grid)
    mixfrac_cube = extract_center_cube(mixfrac_full, grid)
    chi_ref_cube = extract_center_cube(chi_full_ref, grid)

    # Also keep the "cube-only" reconstruction. The user-controllable
    # periodic_xz_cube flag determines BC treatment here.
    chi_rec_cube_only = compute_chi(
        mixfrac_cube, extract_center_cube(alpha_full, grid),
        extract_center_cube(rho_full, grid),
        grid.dx_m, grid.dy_m, grid.dz_m,
        alpha_is=alpha_is, order=fd_order,
        periodic_xz=periodic_xz_cube,
    )

    # Compute full-grid validation metrics too
    chi_full_rel_l2 = relative_l2_error(chi_rec_full, chi_full_ref)
    chi_full_pw = pointwise_error_stats(chi_rec_full, chi_full_ref)
    chi_full_diag = chi_diagnostics(chi_rec_full, chi_full_ref)

    Z_full_rel_l2 = relative_l2_error(Z_rec_full, mixfrac_full)
    Z_full_pw = pointwise_error_stats(Z_rec_full, mixfrac_full)

    result = {
        "Z_info": Z_info,
        # cube comparisons
        "Z_reconstructed_cube": Z_rec_cube,
        "Z_stored_cube": mixfrac_cube,
        "Z_rel_l2": relative_l2_error(Z_rec_cube, mixfrac_cube),
        "Z_pointwise": pointwise_error_stats(Z_rec_cube, mixfrac_cube),
        "chi_reconstructed_cube_fullgrid": chi_rec_cube,
        "chi_reconstructed_cube_only": chi_rec_cube_only,
        "chi_stored_cube": chi_ref_cube,
        "chi_fullgrid_rel_l2": relative_l2_error(chi_rec_cube, chi_ref_cube),
        "chi_fullgrid_pointwise": pointwise_error_stats(chi_rec_cube, chi_ref_cube),
        "chi_fullgrid_diagnostics": chi_diagnostics(chi_rec_cube, chi_ref_cube),
        "chi_cubeonly_rel_l2": relative_l2_error(chi_rec_cube_only, chi_ref_cube),
        "chi_cubeonly_pointwise": pointwise_error_stats(chi_rec_cube_only, chi_ref_cube),
        "chi_cubeonly_diagnostics": chi_diagnostics(chi_rec_cube_only, chi_ref_cube),
        # full-grid comparisons (everything evaluated on 864x1008x576)
        "Z_reconstructed_full": Z_rec_full,
        "Z_stored_full": mixfrac_full,
        "Z_full_rel_l2": Z_full_rel_l2,
        "Z_full_pointwise": Z_full_pw,
        "chi_reconstructed_full": chi_rec_full,
        "chi_stored_full": chi_full_ref,
        "chi_full_rel_l2": chi_full_rel_l2,
        "chi_full_pointwise": chi_full_pw,
        "chi_full_diagnostics": chi_full_diag,
        "fd_order": fd_order,
        "y_over_H": grid.y_over_H(),
        "y_over_H_full": grid.y_over_H_full(),
    }
    return result


def print_validation_full_report(result: Dict) -> None:
    """Report comparing full-grid vs cube-only chi reconstruction, plus
    full-domain validation."""
    print("\n" + "=" * 64)
    print("Full-grid reconstruction validation report")
    print("=" * 64)

    info = result["Z_info"]
    print("\n[Mixture fraction Z via Bilger]")
    print(f"  species used    : {info['species_used']}")
    if info['species_missing']:
        print(f"  species MISSING : {info['species_missing']}")
    print(f"  ---- cube-cropped ----")
    print(f"  rel. L2 error   : {result['Z_rel_l2']:.4e}")
    p = result["Z_pointwise"]
    print(f"  correlation     : {p['correlation']:.6f}")
    print(f"  abs error max   : {p['abs_max']:.4e}")
    if "Z_full_rel_l2" in result:
        print(f"  ---- full grid ----")
        print(f"  rel. L2 error   : {result['Z_full_rel_l2']:.4e}")
        pf = result["Z_full_pointwise"]
        print(f"  correlation     : {pf['correlation']:.6f}")
        print(f"  abs error max   : {pf['abs_max']:.4e}")

    print(f"\n[Scalar dissipation chi, FD order = {result['fd_order']}]")
    print("                             full-grid       cube-only")
    print("                             ---------       ---------")
    dfull = result["chi_fullgrid_diagnostics"]
    dcube = result["chi_cubeonly_diagnostics"]
    pfull = result["chi_fullgrid_pointwise"]
    pcube = result["chi_cubeonly_pointwise"]
    print(f"  pointwise rel. L2 (cube)   : {result['chi_fullgrid_rel_l2']:>10.4e}"
          f"      {result['chi_cubeonly_rel_l2']:>10.4e}")
    print(f"  pointwise correl. (cube)   : {pfull['correlation']:>10.6f}"
          f"      {pcube['correlation']:>10.6f}")
    print(f"  profile rel. L2   (cube)   : {dfull['profile_rel_l2']:>10.4e}"
          f"      {dcube['profile_rel_l2']:>10.4e}")
    print(f"  profile correlation (cube) : {dfull['profile_correlation']:>10.6f}"
          f"      {dcube['profile_correlation']:>10.6f}")
    if "chi_full_rel_l2" in result:
        print(f"  --- whole-domain (full vs full) ---")
        print(f"  pointwise rel. L2          : {result['chi_full_rel_l2']:>10.4e}")
        print(f"  pointwise correlation      : {result['chi_full_pointwise']['correlation']:>10.6f}")
        print(f"  profile rel. L2            : {result['chi_full_diagnostics']['profile_rel_l2']:>10.4e}")
        print(f"  profile correlation        : {result['chi_full_diagnostics']['profile_correlation']:>10.6f}")
    print("=" * 64 + "\n")


def print_validation_report(result: Dict) -> None:
    """Human-readable summary of validate_reconstructions output."""
    print("\n" + "=" * 64)
    print("Derived-field validation report")
    print("=" * 64)

    # Z
    info = result["Z_info"]
    print("\n[Mixture fraction Z via Bilger]")
    print(f"  species used    : {info['species_used']}")
    if info["species_missing"]:
        print(f"  species MISSING : {info['species_missing']}")
    print(f"  beta_fuel       : {info['beta_fuel']:+.6e}")
    print(f"  beta_ox         : {info['beta_ox']:+.6e}")
    print(f"  beta_fuel-beta_ox: {info['beta_denom']:+.6e}")
    print(f"  sum_Y stats     : mean={info['sumY_mean']:.4f}, "
          f"min={info['sumY_min']:.4f}, max={info['sumY_max']:.4f}")
    print(f"  Z raw range     : [{info['Z_raw_min']:+.4f}, "
          f"{info['Z_raw_max']:+.4f}] "
          f"(clipped {100*info['Z_clip_fraction']:.3f}% of cells)")
    if result["Z_rel_l2"] is not None:
        print(f"  rel. L2 error   : {result['Z_rel_l2']:.4e}")
        p = result["Z_pointwise"]
        print(f"  abs error mean  : {p['abs_mean']:.4e}")
        print(f"  abs error max   : {p['abs_max']:.4e}")
        print(f"  correlation     : {p['correlation']:.6f}")
    else:
        print("  (no stored mixfrac field; skipping comparison)")

    # chi
    print("\n[Scalar dissipation chi]")
    if result["chi_reconstructed"] is None:
        missing = result.get("chi_skipped_because_missing", ["?"])
        print(f"  SKIPPED: missing field(s) {missing}")
    else:
        print(f"  FD order used     : {result.get('chi_fd_order', '?')}")
        if result["chi_rel_l2"] is not None:
            print(f"  pointwise rel. L2 : {result['chi_rel_l2']:.4e}")
            p = result["chi_pointwise"]
            print(f"  pointwise correl. : {p['correlation']:.6f}")
            d = result.get("chi_diagnostics")
            if d is not None:
                print(f"  profile rel. L2   : {d['profile_rel_l2']:.4e}"
                      f"   (<-- this is the quantity in paper Fig. 9)")
                print(f"  profile correl.   : {d['profile_correlation']:.6f}")
                print(f"  peak stored       : {d['peak_reference']:.3e}")
                print(f"  peak reconstructed: {d['peak_reconstructed']:.3e}")
                print(f"  peak ratio        : {d['peak_ratio']:.2f}")
        else:
            print("  (no stored chi field; skipping comparison)")
    print("=" * 64 + "\n")
