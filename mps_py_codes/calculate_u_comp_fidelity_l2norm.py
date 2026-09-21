import numpy as np

def calculate_fidelity(u1,u2):
    """
    Calculate the fidelity between two vectors u1 and u2.

    Parameters
    ----------
    u1, u2 : array_like
        The two vectors to calculate the fidelity for.

    Returns
    -------
    fidelity : float
        The fidelity between u1 and u2.

    Notes
    -----
    The fidelity is a measure of how similar two vectors are.  It is calculated
    as the absolute value of the inner product of the two vectors divided by
    the product of their norms.

    """

    # Compute the fidelity
    u1 = u1.astype(np.float64)
    u2 = u2.astype(np.float64)

    fidelity = np.abs(np.vdot(u1, u2))**2 / (np.vdot(u1, u1) * np.vdot(u2, u2))
    return fidelity


def calculate_l2norm(u1,u2):  
    """
    Calculate the L2 norm (Euclidean distance) between two vectors u1 and u2.

    Parameters
    ----------
    u1, u2 : array_like
        The two vectors to calculate the L2 norm for.

    Returns
    -------
    l2_norm : float
        The L2 norm between u1 and u2.

    Notes
    -----
    The L2 norm is a measure of the Euclidean distance between two vectors.  It
    is calculated as the absolute value of the difference of the two vectors
    divided by the norm of the first vector.

    """
    
    l2_norm = np.abs(np.linalg.norm(u2 - u1)/np.linalg.norm(u1))
    return l2_norm


def calculate_u_comp_fidelity_l2(u_truth, mps_U, chi_crit):

    
    """
    Calculate the fidelity and L2 norm of a compressed MPS representation of a vector.

    Parameters
    ----------
    u_truth : array_like
        The true vector to calculate the fidelity and L2 norm of.
    mps_U : list of array_like
        The MPS representation of the vector to calculate the fidelity and L2 norm of.
    chi_crit : list of integers
        The list of chi values to use for each layer of the MPS.

    Returns
    -------
    u_comp : array_like
        The compressed representation of the vector.
    fidelity : float
        The fidelity of the compressed representation.
    l2_norm : float
        The L2 norm of the compressed representation.

    Notes
    -----
    The fidelity measures the similarity between the two vectors, and is calculated
    as the absolute value of the dot product of the two vectors divided by the product
    of their norms.  The L2 norm is a measure of the Euclidean distance between two
    vectors, and is calculated as the absolute value of the difference of the two vectors
    divided by the norm of the first vector.

    """

    # Initialize u_comp with the first element of mps_U
    u_comp = mps_U[0]

    # Iterate through the layers of mps_U to contract tensors
    for i in range(len(chi_crit)):
        # Dynamically slice A: In each iteration, slice the last axis based on chi_crit
        if i == 0:
            A = u_comp[:, :, 0:chi_crit[i]]  # In the first iteration, slice the 3rd axis
        else:
            # Create slices for the first i+2 dimensions, then slice the last dimension with chi_crit[i]
            slices = [slice(None)] * (i+2) + [slice(0, chi_crit[i])]
            A = u_comp[tuple(slices)]  # Slice A according to the chi_crit[i] value

        # Prepare B by taking all elements except for the last along the first axis
        BB = mps_U[i + 1]
        B = BB[0:chi_crit[i], :, :]

        # Perform the tensor dot product
        u_comp = np.tensordot(A, B, axes=([i+2], [0]))  # Perform the contraction on the appropriate axes

    # Reshape the final u_comp
    u_comp = u_comp.reshape(u_truth.shape)

    # Flatten both u_comp and u_truth for fidelity calculation
    u_comp_flat = u_comp.flatten()
    u_truth_flat = u_truth.flatten()
    
    u_comp_flat = u_comp_flat.astype(np.float64)
    u_truth_flat = u_truth_flat.astype(np.float64)

    # Compute the fidelity
    fidelity = calculate_fidelity(u_truth_flat, u_comp_flat)

    # Compute the L2 norm (Euclidean distance)
    l2_norm = calculate_l2norm(u_truth_flat, u_comp_flat)

    return u_comp, fidelity, l2_norm

