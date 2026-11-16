import numpy as np

def spectral_derivative(matrix, axis, dx, order=1): #order = order of derivative = 1st deriv, 2nd deriv,...
    """
    Compute the derivative of a 3D matrix along a specified axis using Fourier spectral differentiation
    with periodic boundary conditions.

    Parameters:
    - matrix: np.array, the 3D input matrix (shape: Nx * Ny * Nz).
    - axis: int, the axis along which to compute the derivative (0, 1, or 2).
    - dx: float, the spacing between points along the given axis.
    - order: int, order of the derivative (default is 1 for first derivative).

    Returns:
    - derivative: np.array, the 3D matrix of derivatives with the same shape as input.
    """
    # Get shape along the specified axis
    N = matrix.shape[axis]

    # Compute wavenumbers (k values) for the given axis
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    # Reshape k for broadcasting along the specified axis
    k_shape = np.ones(3, dtype=int)
    k_shape[axis] = N  # Expand along the specified axis
    k = k.reshape(k_shape)

    # Compute the Fourier transform along the specified axis
    F_k = np.fft.fft(matrix, axis=axis)

    # Apply Fourier differentiation (multiplication by (ik)^order)
    F_k_derivative = (1j * k) ** order * F_k

    # Inverse Fourier transform to get the real-space derivative
    derivative = np.fft.ifft(F_k_derivative, axis=axis).real

    return derivative
    # return F_k_derivative
