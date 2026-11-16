import numpy as np
from scipy.ndimage import zoom

def compute_URDNS(u, URDNS_size):
    """
    Compute the Upsampled Reduced DNS (URDNS) for a 3D velocity field `u`.
    
    Parameters
    ----------
    u : np.array
        The original 3D velocity field of shape (Nx, Ny, Nz).
    URDNS_size : int
        The target size for the reduced DNS grid (e.g., 64 for a 64x64x64 grid).
        
    Returns
    -------
    u_URDNS : np.array
        The upsampled URDNS velocity field of the same size as `u`.
    """
    # Get the original size of the velocity field
    Nx, Ny, Nz = u.shape
    
    # Compute the downsampling factor
    d = Nx // URDNS_size
    
    # Downsample u by skipping every d-th point
    u_skipped = u[::d, ::d, ::d]
    
    # Interpolate back to the original size using zoom
    zoom_factors = (Nx / u_skipped.shape[0], Ny / u_skipped.shape[1], Nz / u_skipped.shape[2])
    # print("zoom_factors: ",zoom_factors)
    u_URDNS = zoom(u_skipped, zoom_factors, order=2)  # Linear interpolation (order=1)
    
    return u_URDNS

# u_URDNS_32 = compute_URDNS(u, 32)

def compute_URDNS_uvw(u, v, w, URDNS_size):
    u_URDNS = compute_URDNS(u, URDNS_size)
    v_URDNS = compute_URDNS(v, URDNS_size)
    w_URDNS = compute_URDNS(w, URDNS_size)
    return u_URDNS, v_URDNS, w_URDNS