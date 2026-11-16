"""Opt-in Float64 kernels preserving historical derivative/comparison formulas."""
import operator
import numpy as np


def _field(field):
    if not isinstance(field, np.ndarray) or field.dtype != np.dtype(np.float64):
        raise TypeError("use a NumPy Float64 array")
    if field.ndim != 3 or min(field.shape) < 1:
        raise ValueError("use a nonempty 3D field")


def _slab_size(value):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("slab_size must be a positive integer")
    value = operator.index(value)
    if value < 1:
        raise ValueError("slab_size must be positive")
    return value


def spectral_derivative_slabbed(matrix, axis, dx, order=1, *, slab_size=8):
    """Historical periodic spectral derivative with bounded FFT workspace.

    Accept a nonempty 3D Float64 field, axis 0/1/2, spacing dx and derivative
    order as in calculate_spectral_derivative.py. Return a Float64 field with
    unchanged shape/axis order. Every FFT line spans the ENTIRE selected axis;
    slabs partition another axis, so neither boundary behavior nor wavenumbers
    change. Input/output are still full arrays. No normalization, mean removal,
    precision reduction or boundary approximation is introduced.
    """
    _field(matrix)
    slab_size = _slab_size(slab_size)
    axis = operator.index(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    n = matrix.shape[axis]
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    shape = [1, 1, 1]
    shape[axis] = n
    multiplier = (1j * k.reshape(shape)) ** order
    partition_axis = (axis + 1) % 3
    output = np.empty(matrix.shape, dtype=np.float64)
    for start in range(0, matrix.shape[partition_axis], slab_size):
        selection = [slice(None)] * 3
        selection[partition_axis] = slice(start, start + slab_size)
        selection = tuple(selection)
        transformed = np.fft.fft(matrix[selection], axis=axis)
        transformed *= multiplier
        output[selection] = np.fft.ifft(transformed, axis=axis).real
    return output


def compare_fields_chunked(reference, candidate, *, slab_size=8):
    """Return historical squared-overlap fidelity and relative L2, without copies.

    Inputs: same-shaped nonempty real Float64 3D arrays in identical axis order.
    No centering, scaling or alignment is inferred. Fidelity is
    abs(vdot(a,b))**2/(vdot(a,a)*vdot(b,b)); relative_l2 is norm(b-a)/norm(a).
    These match calculate_fidelity/calculate_l2norm in the historical helper.
    Slabs bound temporary differences; reduction order changes can affect
    rounding. No acceptance tolerance is chosen. Zero denominators retain
    IEEE NaN/Inf behavior rather than applying an invented epsilon.
    """
    _field(reference)
    _field(candidate)
    if reference.shape != candidate.shape:
        raise ValueError("field shapes must match")
    slab_size = _slab_size(slab_size)
    aa = bb = ab = dd = np.float64(0)
    for start in range(0, reference.shape[0], slab_size):
        a = reference[start:start + slab_size].ravel()
        b = candidate[start:start + slab_size].ravel()
        difference = b - a
        aa += np.vdot(a, a)
        bb += np.vdot(b, b)
        ab += np.vdot(a, b)
        dd += np.vdot(difference, difference)
    with np.errstate(divide="ignore", invalid="ignore"):
        fidelity = np.abs(ab)**2 / (aa * bb)
        relative_l2 = np.abs(np.sqrt(dd) / np.sqrt(aa))
    return {"fidelity": float(fidelity), "relative_l2": float(relative_l2)}
