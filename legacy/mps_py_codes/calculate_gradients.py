from calculate_spectral_derivative import spectral_derivative
from calculate_finite_difference import central_dif_3d, finite_dif_4th_order_zero, finite_dif_4th_order_periodic, finite_dif_4th_order_periodic_for, central_dif
from calculate_u_comp_fidelity_l2norm import calculate_fidelity
import numpy as np


def calculate_gradients(velocity_field, h, order=4, spectral = False):
    """
    Calculate the velocity gradients of a 3D velocity field.

    Parameters:
    - velocity_field: a tuple of 3 np.arrays (u_x, u_y, u_z), each of shape Nx * Ny * Nz.
    - dx: float, the spacing between grid points (assumed to be the same for all directions).

    Returns:
    - gradients: a dictionary containing the velocity gradients.
    """
    u_x, u_y, u_z = velocity_field
    dx = h[0]
    dy = h[1]
    dz = h[2]
    if spectral:
        gradients = {
        'dux_dx': spectral_derivative(u_x, axis=0, dx=dx),
        'dux_dy': spectral_derivative(u_x, axis=1, dx=dy),
        'dux_dz': spectral_derivative(u_x, axis=2, dx=dz),
        
        'duy_dx': spectral_derivative(u_y, axis=0, dx=dx),
        'duy_dy': spectral_derivative(u_y, axis=1, dx=dy),
        'duy_dz': spectral_derivative(u_y, axis=2, dx=dz),
        
        'duz_dx': spectral_derivative(u_z, axis=0, dx=dx),
        'duz_dy': spectral_derivative(u_z, axis=1, dx=dy),
        'duz_dz': spectral_derivative(u_z, axis=2, dx=dz),
        }
    else:
        gradients = {
            'dux_dx': central_dif(u_x, axis=0, dx=dx, order=order),
            'dux_dy': central_dif(u_x, axis=1, dx=dy, order=order),
            'dux_dz': central_dif(u_x, axis=2, dx=dz, order=order),

            'duy_dx': central_dif(u_y, axis=0, dx=dx, order=order),
            'duy_dy': central_dif(u_y, axis=1, dx=dy, order=order),
            'duy_dz': central_dif(u_y, axis=2, dx=dz, order=order),

            'duz_dx': central_dif(u_z, axis=0, dx=dx, order=order),
            'duz_dy': central_dif(u_z, axis=1, dx=dy, order=order),
            'duz_dz': central_dif(u_z, axis=2, dx=dz, order=order)
            }
    return gradients




def calculate_gradients_fidelities(gradients, gradients_comp):
    """
    Calculate the fidelities between the gradients of the original and compressed flow.

    Parameters:
    - gradients: a dictionary containing the gradients of the original flow (e.g., dux_dx, dux_dy, etc.).
    - gradients_comp: a dictionary containing the gradients of the compressed flow (e.g., dux_dx, dux_dy, etc.).

    Returns:
    - fidelity_dux: a list containing the fidelities between the x-component of the gradients of the original and compressed flow.
    - fidelity_duy: a list containing the fidelities between the y-component of the gradients of the original and compressed flow.
    - fidelity_duz: a list containing the fidelities between the z-component of the gradients of the original and compressed flow.
    """
    fid_dux_dx = calculate_fidelity(gradients['dux_dx'], gradients_comp['dux_dx'])
    fid_dux_dy = calculate_fidelity(gradients['dux_dy'], gradients_comp['dux_dy'])
    fid_dux_dz = calculate_fidelity(gradients['dux_dz'], gradients_comp['dux_dz'])
    fidelity_dux = [fid_dux_dx, fid_dux_dy, fid_dux_dz]

    fid_duy_dx = calculate_fidelity(gradients['duy_dx'], gradients_comp['duy_dx'])
    fid_duy_dy = calculate_fidelity(gradients['duy_dy'], gradients_comp['duy_dy'])
    fid_duy_dz = calculate_fidelity(gradients['duy_dz'], gradients_comp['duy_dz'])
    fidelity_duy = [fid_duy_dx, fid_duy_dy, fid_duy_dz]

    fid_duz_dx = calculate_fidelity(gradients['duz_dx'], gradients_comp['duz_dx'])
    fid_duz_dy = calculate_fidelity(gradients['duz_dy'], gradients_comp['duz_dy'])
    fid_duz_dz = calculate_fidelity(gradients['duz_dz'], gradients_comp['duz_dz'])
    fidelity_duz = [fid_duz_dx, fid_duz_dy, fid_duz_dz]
    
    return fidelity_dux, fidelity_duy, fidelity_duz



def calculate_invariants(M):
    """
    Calculate the divergence (first invariant), second (Q), and third (R) invariants of the gradient tensor.

    Parameters:
    - gradients: a dictionary containing the velocity gradients:
      'dux_dx', 'dux_dy', 'dux_dz',
      'duy_dx', 'duy_dy', 'duy_dz',
      'duz_dx', 'duz_dy', 'duz_dz'.

    Returns:
    - invariants: a dictionary with keys 'div', 'Q', and 'R', each a numpy array of shape Nx * Ny * Nz.
    """

    # M = np.array([[gradients['dux_dx'], gradients['dux_dy'], gradients['dux_dz']],
    #               [gradients['duy_dx'], gradients['duy_dy'], gradients['duy_dz']],
    #               [gradients['duz_dx'], gradients['duz_dy'], gradients['duz_dz']]])
    
    axis1 = 3
    axis2 = 4

    P = -1*np.trace(M,axis1=axis1,axis2=axis2)
    M2 = M @ M
    M3 = M2 @ M

    Q =-1*np.trace(M2,axis1=axis1,axis2=axis2)/2
    R =-1*np.trace(M3,axis1=axis1,axis2=axis2)/3
    return P, Q, R



def calculate_invariants_exact(M):
    """
    Calculate the divergence (first invariant), second (Q), and third (R) invariants of the gradient tensor.

    Parameters:
    - gradients: a dictionary containing the velocity gradients:
      'dux_dx', 'dux_dy', 'dux_dz',
      'duy_dx', 'duy_dy', 'duy_dz',
      'duz_dx', 'duz_dy', 'duz_dz'.

    Returns:
    - invariants: a dictionary with keys 'div', 'Q', and 'R', each a numpy array of shape Nx * Ny * Nz.
    """

    # M = np.array([[gradients['dux_dx'], gradients['dux_dy'], gradients['dux_dz']],
    #               [gradients['duy_dx'], gradients['duy_dy'], gradients['duy_dz']],
    #               [gradients['duz_dx'], gradients['duz_dy'], gradients['duz_dz']]])
    
    axis1 = 3
    axis2 = 4

    P = -1*np.trace(M,axis1=axis1,axis2=axis2)
    M2 = M @ M
    # M3 = M2 @ M

    Q = 1/2*(P**2 - np.trace(M2,axis1=axis1,axis2=axis2))
    R = -np.linalg.det(M)
    # R1 = -1*(P**3 - 3*P*Q + np.trace(M3,axis1=axis1,axis2=axis2))/3
    
    return P, Q, R



def compute_Qw(A):
    """
    Computes Qw (mean vorticity norm) for a 3D velocity gradient tensor.

    Parameters:
    A (numpy.ndarray): Velocity gradient tensor of shape (N, N, N, 3, 3).

    Returns:
    float: Scalar value representing Qw.
    """
    # Compute the antisymmetric part (vorticity tensor W_ij)
    W = 0.5 * (A - np.transpose(A, axes=(0,1,2,4,3)))

    # Compute W_ij W_ij (Frobenius norm squared)
    # W_squared = np.einsum("...ij,...ij->...", W, W)
    # W_squared = np.sum(W.numpy() * W.numpy(), axis=(-2, -1))
    W_squared = np.sum(W* W, axis=(-2, -1))

    # Compute Qw = (1/2) * mean(W_ij W_ij)
    # Qw = 0.5 * np.mean(W_squared)
    Qw = np.sqrt(np.mean(W_squared))

    return Qw


