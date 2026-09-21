import numpy as np
from MPS_permute import MPS_permute_uneq

def reshape_and_transpose(u, ordered_primes, axis_powers):
    
    """
    Reshapes a matrix to its ordered prime factors, then transposes it
    according to the powers of the prime factors.

    Parameters
    ----------
    u : numpy array
        The matrix to reshape and transpose
    ordered_primes : tuple of int
        The prime factors of the elements of u.shape, in descending order
    axis_powers : tuple of int
        The powers of the prime factors of the elements of u.shape

    Returns
    -------
    final_u : numpy array
        The reshaped and transposed matrix
    """
    # Reshape the matrix to its Ordered Primes shape
    reshaped_u = np.reshape(u, ordered_primes)
    transpose_tuple = MPS_permute_uneq(axis_powers)
    final_u = np.transpose(reshaped_u, transpose_tuple)
    
    return final_u




def inverse_reshape_and_transpose(u_reshaped, axis_powers, u_shape):

    """
    Inverse of reshape_and_transpose. Given a matrix u_reshaped, reshaped to its
    ordered prime factors, and the axis powers, this function will transpose it
    back to its original shape.

    Parameters
    ----------
    u_reshaped : numpy array
        The matrix to transpose back to its original shape
    axis_powers : tuple of int
        The powers of the prime factors of the elements of u.shape
    u_shape : tuple of int
        The original shape of the matrix

    Returns
    -------
    original_u : numpy array
        The transposed and reshaped matrix
    """
    
    # Generate the transpose tuple from axis_powers
    transpose_tuple = MPS_permute_uneq(axis_powers)
    
    # Find the inverse of the transpose tuple
    inverse_transpose = np.argsort(transpose_tuple)
    # print("inv transpose tuple: ", inverse_transpose)

    # Apply the inverse transpose to u_final
    u_inversed_transpose = np.transpose(u_reshaped, inverse_transpose)
    
    # Reshape the matrix back to the original shape
    original_u = np.reshape(u_inversed_transpose, u_shape)
    
    return original_u

