# from save_as_mat import save_as_mat
import numpy as np

def calculate_eps(gradients, nu=0.000185): #this nu is for Forced Isotropic of JHS
    """
    Calculate the total dissipation rate using precomputed velocity gradients.

    Parameters:
    - gradients: a dictionary containing the velocity gradients (e.g., dux_dx, dux_dy, etc.).
    - nu: float, kinematic viscosity of the fluid.

    Returns:
    - dissipation_rate: float, the total dissipation rate across the entire 3D field.
    """
    
    # Extract gradients from the dictionary
    dux_dx = gradients['dux_dx']
    dux_dy = gradients['dux_dy']
    dux_dz = gradients['dux_dz']
    
    duy_dx = gradients['duy_dx']
    duy_dy = gradients['duy_dy']
    duy_dz = gradients['duy_dz']
    
    duz_dx = gradients['duz_dx']
    duz_dy = gradients['duz_dy']
    duz_dz = gradients['duz_dz']
    
    # Symmetric velocity gradients
    S11 = dux_dx + dux_dx  # Symmetric term for ∂u_x/∂x
    S12 = dux_dy + duy_dx  # Symmetric term for ∂u_x/∂y + ∂u_y/∂x
    S13 = dux_dz + duz_dx  # Symmetric term for ∂u_x/∂z + ∂u_z/∂x
    
    S22 = duy_dy + duy_dy  # Symmetric term for ∂u_y/∂y
    S23 = duy_dz + duz_dy  # Symmetric term for ∂u_y/∂z + ∂u_z/∂y
    
    S33 = duz_dz + duz_dz  # Symmetric term for ∂u_z/∂z
    
    # Sij = np.array([[S11, S12, S13], [S12, S22, S23], [S13, S23, S33]])/2
    # Sij = np.transpose(Sij, axes=(2,3,4,0,1))
    
    # print("Sij shape: ", Sij.shape)
    # s_tilde = calculate_s_tilde(Sij)

    # save_as_mat('Sij', {'S11': S11, 'S12': S12, 'S13': S13, 'S22': S22, 'S23': S23, 'S33': S33})

    # Compute dissipation rate using the symmetric velocity gradients
    dissipation_rate = nu/2 * (
        S11**2 + S22**2 + S33**2 + 2*S12**2 + 2*S13**2 + 2*S23**2
    )
    
    # # Total dissipation rate (sum over the entire 3D field)
    # dissipation_rate = np.mean(dissipation_rate_points)
    
    return dissipation_rate
    # return dissipation_rate,s_tilda


def calculate_s_tilde(Sij):
    """
    Computes s_tilde for a 3D strain-rate tensor S_ij of shape (N, N, N, 3, 3).

    Parameters
    ----------
    Sij : np.ndarray
        Strain-rate tensor of shape (N, N, N, 3, 3). 
        Assumed to be real and symmetric. If not, it will be symmetrized.
    eps : float, optional
        Small number for avoiding division by zero, by default 1e-30.

    Returns
    -------
    s_tilda : np.ndarray
        The scalar field s_tilda of shape (N, N, N).
    """

    # -------------------------------------------------------------------------
    # 1) Symmetrize S if not guaranteed symmetric
    # -------------------------------------------------------------------------
    # (If your S is already guaranteed to be symmetric, you can skip this step.)
    # Sij = 0.5 * (Sij + Sij.transpose(0,1,2,4,3))  # shape: (N, N, N, 3, 3)

    # -------------------------------------------------------------------------
    # 2) Compute deviatoric part: S_star = S - (1/3)*trace(S)*I
    # -------------------------------------------------------------------------
    trace_S = np.trace(Sij, axis1=3, axis2=4)            # shape (N, N, N)
    I = np.eye(3).reshape((1, 1, 1, 3, 3))               # shape (1,1,1,3,3)
    trace_S_expanded = trace_S[..., np.newaxis, np.newaxis]  # (N, N, N, 1, 1)

    S_star = Sij - (trace_S_expanded / 3.0) * I          # (N, N, N, 3, 3)

    # -------------------------------------------------------------------------
    # 3) Compute eigenvalues of S_star (vectorized)
    # -------------------------------------------------------------------------
    # Reshape to (N*N*N, 3, 3) for np.linalg.eigh
    N = S_star.shape[0]  # Assuming a cubic domain
    S_star_reshaped = S_star.reshape(-1, 3, 3)

    # w: eigenvalues, v: eigenvectors (not used here)
    w, _ = np.linalg.eigh(S_star_reshaped)  # w shape: (N*N*N, 3)

    # Reshape eigenvalues back to (N, N, N, 3)
    S_star_eigen = w.reshape(N, N, N, 3)

    # -------------------------------------------------------------------------
    # 4) Compute s_tilda = [ -3 sqrt(6) * (alpha beta gamma) ]
    #                      / [ (alpha^2 + beta^2 + gamma^2)^(3/2) ]
    # -------------------------------------------------------------------------
    product_eigs = np.prod(S_star_eigen, axis=-1)           # (N, N, N)
    sum_squares  = np.sum(S_star_eigen**2, axis=-1)         # (N, N, N)
    denominator  = sum_squares ** 1.5                       # (N, N, N)

    alpha = S_star_eigen[:,:,:, 2]
    beta  = S_star_eigen[:,:,:, 1]
    gamma = S_star_eigen[:,:,:, 0]

    # To avoid division by zero, replace zero or near-zero values with eps
    eps=1e-16
    denominator_safe = np.where(denominator < eps, eps, denominator)

    s_tilde = (
        -3.0 * np.sqrt(6.0) * product_eigs
        / denominator_safe
    )  # shape: (N, N, N)

    return s_tilde, alpha, beta, gamma

