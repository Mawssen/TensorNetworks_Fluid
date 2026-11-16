"""Float64 acceleration of the 3D spectrum used by the historical 2FI script.

Reference: legacy/mps_py_codes/calculate_spectrum.py:calculate_spectrum.
That implementation and the separate historical Float32 variant are untouched.
"""

import operator

import numpy as np


def _shell_geometry(dim):
    # Keep the original arithmetic, mode order, shell edges and endpoint rule.
    length = 2 * np.pi
    dx = 2 * np.pi / length
    k_end = int(dim / 2)
    rx = np.array(range(dim)) - dim / 2 + 1
    rx = np.roll(rx, int(dim / 2) + 1)
    k = (np.array(range(k_end)) + 1) * dx
    bins = np.zeros(k.shape[0] + 1)
    for index in range(k_end):
        bins[index] = 0 if index == 0 else (k[index] + k[index - 1]) / 2
    bins[-1] = k[-1]
    return rx, k, bins, dx


def _shell_indices(rx, bins, dx, start, stop):
    # Same (x²+y²)+z² evaluation as the historical i,j loops, in bounded slabs.
    radius_squared = rx[start:stop, None, None]**2 + rx[None, :, None]**2
    radius_squared = radius_squared + rx[None, None, :]**2
    radius = np.sqrt(radius_squared)
    return np.digitize(radius * dx, bins, right=True)


def calculate_spectrum_binned(u, v, w, *, slab_size=8):
    """Return ``(k, spectrum)`` using the historical 2FI 3D shell convention.

    Inputs are three same-shaped cubic NumPy Float64 arrays with side >= 2.
    Axes retain their supplied order; callers establish physical axis meanings.
    No centering, unit conversion, filtering, or precision conversion is done.
    As in the reference, the domain length is 2*pi, FFT powers are normalized by
    dim**6, zero/overflow bins are excluded, and the historical shell-count /
    2*pi*k**2 normalization is retained. This is not a new spectrum definition.

    ``slab_size`` is a positive number of first-axis planes used for shell
    accumulation, independent of physical grid spacing. It bounds geometry
    temporaries; it does not make the full 3D FFT an out-of-core calculation.
    Components are transformed and reduced one at a time. Results and all
    floating intermediates stay Float64 (FFT arrays are Complex128).

    Weighted bincount replaces a full-grid scan for every shell. Changing the
    reduction order can change rounding, so bitwise equality is not promised.
    No approximation or scientific acceptance tolerance is selected here.
    """
    fields = (u, v, w)
    for field in fields:
        if not isinstance(field, np.ndarray) or field.dtype != np.dtype(np.float64):
            raise TypeError("each component must be a NumPy Float64 array")
        if field.ndim != 3 or len(set(field.shape)) != 1 or field.shape[0] < 2:
            raise ValueError("components must be cubic 3D arrays with side >= 2")
    if any(field.shape != u.shape for field in fields):
        raise ValueError("all three components must have the same shape")
    if isinstance(slab_size, (bool, np.bool_)):
        raise TypeError("slab_size must be a positive integer")
    slab_size = operator.index(slab_size)
    if slab_size <= 0:
        raise ValueError("slab_size must be positive")

    dim = u.shape[0]
    rx, k, bins, dx = _shell_geometry(dim)
    nbins = len(k) + 2  # include zero and overflow bins until the final slice
    sums = np.zeros((3, nbins), dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)

    for component, field in enumerate(fields):
        transformed = np.fft.fftn(field)
        power = np.abs(transformed)
        del transformed
        # Same operations as (abs(FFT)/dim**3)**2, without extra power cubes.
        power /= dim**3
        np.square(power, out=power)
        for start in range(0, dim, slab_size):
            stop = min(start + slab_size, dim)
            indices = _shell_indices(rx, bins, dx, start, stop).ravel()
            sums[component] += np.bincount(
                indices, weights=power[start:stop].ravel(), minlength=nbins)
            if component == 0:
                counts += np.bincount(indices, minlength=nbins)
        del power

    spectrum = sums[0, 1:-1] + sums[1, 1:-1] + sums[2, 1:-1]
    bin_counter = counts[1:-1].astype(np.float64)
    # Preserve reference zero-count behavior (NaN), rather than invent a fill.
    with np.errstate(invalid="ignore", divide="ignore"):
        spectrum = spectrum * 2 * np.pi * (k**2) / (bin_counter * dx**3)
    return k, spectrum
