import numpy as np

def calculate_tkEnergy(velocities, ntime):
    """
    Calculate total kinetic energy (E) of a 3D turbulent flow field.

    Parameters
    ----------
    velocities : tuple of 3 arrays
        A tuple containing the x, y, and z components of the velocity field.
    ntime : int
        Number of time steps in the flow field.

    Returns
    -------
    E : float
        The total kinetic energy of the flow field.

    Notes
    -----
    This function assumes that the input velocities are in the format of a
    tuple of 3 3D arrays, each representing the x, y, and z components of the
    velocity field. The input velocities should have the same shape.
    """
    
    u, v, w = velocities
    nx, ny, nz = u.shape
    E_u = 0.5 * np.sum(u**2) / (nx * ny * nz) / ntime
    E_v = 0.5 * np.sum(v**2) / (nx * ny * nz) / ntime
    E_w = 0.5 * np.sum(w**2) / (nx * ny * nz) / ntime
    E = E_u + E_v + E_w
    return E

