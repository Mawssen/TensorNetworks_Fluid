import numpy as np

def calculate_unit_vectors(u, v, w):
    """
    Given 3D scalar fields for velocity components (u, v, w),
    return unit vectors (u_hat, v_hat, w_hat) by normalizing 
    at each grid point.

    Parameters
    ----------
    u, v, w : np.ndarray
        Arrays of the same shape (e.g., (N, N, N)) representing 
        the three velocity components.

    epsilon : float, optional
        A small threshold to avoid division by zero for near-zero 
        magnitudes. Defaults to 1e-16.
    
    Returns
    -------
    u_hat, v_hat, w_hat : np.ndarray
        Arrays of the same shape as u, v, w with unit vectors. 
        Where the norm is below epsilon, the output is 0.0.
    """
    epsilon = 1e-16

    # Compute the velocity magnitude at each grid cell
    norm = np.sqrt(u**2 + v**2 + w**2)
    
    # Use np.divide with 'where' to safely handle zeros
    u_hat = np.divide(u, norm, out=np.zeros_like(u), where=(norm > epsilon))
    v_hat = np.divide(v, norm, out=np.zeros_like(v), where=(norm > epsilon))
    w_hat = np.divide(w, norm, out=np.zeros_like(w), where=(norm > epsilon))
    
    return u_hat, v_hat, w_hat


def calculate_theta(u, v, w, u_comp, v_comp, w_comp):
    """
    Compute the angular difference (in radians) between 
    two 3D velocity fields (u, v, w) and (u_comp, v_comp, w_comp).

    Internally uses calculate_unit_vectors() to convert each field
    to its corresponding unit vectors, then computes:

        dot(u_hat, u_hat_comp) = cos(theta)
        theta = arccos(cos(theta))

    Parameters
    ----------
    u, v, w : np.ndarray
        Original velocity components (same shape).

    u_comp, v_comp, w_comp : np.ndarray
        Compressed or approximated velocity components (same shape).

    Returns
    -------
    theta : np.ndarray
        Angular difference at each grid point (in radians), 
        with the same shape as the input arrays.
    """

    # Get unit vectors of the original field
    u_hat, v_hat, w_hat = calculate_unit_vectors(u, v, w)

    # Get unit vectors of the compressed field
    u_comp_hat, v_comp_hat, w_comp_hat = calculate_unit_vectors(u_comp, v_comp, w_comp)
    
    # Dot product of unit vectors gives cos(theta)
    dot = u_hat * u_comp_hat + v_hat * v_comp_hat + w_hat * w_comp_hat
    
    # Clip to [-1, 1] to avoid numerical issues when taking arccos
    dot_clipped = np.clip(dot, -1.0, 1.0)
    
    # Compute theta in radians
    theta = np.arccos(dot_clipped)
    
    return theta
