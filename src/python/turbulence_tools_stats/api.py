"""Preferred, owner-approved statistics defaults with explicit reference fallback."""
from functools import lru_cache
import importlib.util
from pathlib import Path

from .spectrum import calculate_spectrum_binned
from .reductions import compare_fields_chunked, spectral_derivative_slabbed


@lru_cache(maxsize=None)
def _reference_module(filename):
    path = Path(__file__).resolve().parents[3] / "legacy/mps_py_codes" / filename
    spec = importlib.util.spec_from_file_location("_turbulence_reference_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(implementation):
    if implementation not in ("optimized", "reference"):
        raise ValueError("implementation must be 'optimized' or 'reference'")


def calculate_spectrum(u, v, w, *, implementation="optimized"):
    """Return historical (k, spectrum); preferred default is Float64 binning.

    Default inputs: three same-shaped cubic Float64 3D arrays in the caller's
    axis order. No centering or axis conversion is added. Historical bins,
    2*pi domain and normalization are preserved. Select implementation='reference'
    for the unchanged historical Float64 function. No Float32 path is selected.
    """
    _check(implementation)
    if implementation == "reference":
        return _reference_module("calculate_spectrum.py").calculate_spectrum(u, v, w)
    return calculate_spectrum_binned(u, v, w)


def compare_fields(reference, candidate, *, implementation="optimized"):
    """Return {'fidelity', 'relative_l2'}; preferred default is chunked Float64.

    Default inputs: same-shaped Float64 3D arrays in matching axis order.
    No mean subtraction or alignment is inferred. Fidelity is squared normalized
    overlap and relative L2 is norm(candidate-reference)/norm(reference), as in
    the historical functions. Select implementation='reference' for both old
    calculations. Reduction-order differences are documented in the benchmarks.
    """
    _check(implementation)
    if implementation == "reference":
        old = _reference_module("calculate_u_comp_fidelity_l2norm.py")
        return {"fidelity": float(old.calculate_fidelity(reference, candidate)),
                "relative_l2": float(old.calculate_l2norm(reference, candidate))}
    return compare_fields_chunked(reference, candidate)


def spectral_derivative(field, axis, dx, order=1, *, implementation="reference"):
    """Periodic derivative in supplied axis order; original CPU method is default.

    axis is 0/1/2, dx is that axis's spacing, order is the derivative order.
    Shape and physical grid are unchanged. The reference supports the original
    input types. Select implementation='optimized' explicitly for the Float64
    3D slabbed alternative when memory is limiting; it was slower on full JHS.
    """
    _check(implementation)
    if implementation == "reference":
        return _reference_module("calculate_spectral_derivative.py").spectral_derivative(
            field, axis, dx, order=order)
    return spectral_derivative_slabbed(field, axis, dx, order=order)
