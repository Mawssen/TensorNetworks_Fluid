import numpy as np

def comp_ratio_uneq(chi, n):
    
    """
    Compute compression ratios for each column of chi, given n.

    Parameters
    ----------
    chi : 2D array
        Array of Schmidt coefficients for each column, with shape (l, s).
    n : 1D array
        Array of physical dimensions, with shape (l,).

    Returns
    -------
    compression_ratios : 2D array
        Array of compression ratios for each column of chi, with shape (s, 1).
    """
    
    # Ensure chi is a NumPy array of type float
    chi = np.array(chi, dtype=float)
    
    # Get the number of columns (s)
    _, c = chi.shape

    # Initialize an array to store the compression ratios for each column
    compression_ratios = np.zeros([c,1])
    
    # Loop over each column of chi
    for col in range(c):
        # Extract the column from chi
        chi_col = chi[:, col]
        
        # Create chi_aug for the current column
        chi_aug = np.concatenate(([1], chi_col, [1]))
        
        # Initialize total_params for the current column
        total_params = 0
        
        # Compute total_params for the current column
        for i in range(len(chi_aug) - 1):
            total_params += n[i] * chi_aug[i] * chi_aug[i + 1]
        
        # Compute inherent_gdof for the current column
        inherent_gdof = np.sum(chi_col ** 2)
        
        # Compute compression ratio for the current column
        numerator = np.prod(n)
        denom = total_params - inherent_gdof
        compression_ratios[col,0] = numerator / denom
    
    return compression_ratios

def comp_ratio_memory(chi, cr_dof=False):
    """
    Computes the memory compression ratio of DNS over truncated MPS
    using bond dimensions.

    Parameters:
    - chi: list of bond dimensions [chi_0, chi_1, ..., chi_N], with chi_0 = chi_N = 1 for open boundary.

    Returns:
    - comp_ratio_memory: memory of DNS / memory of truncated MPS
    """
    #memory of full MPS
    # n_bonds = len(chi)
    #  # Build chi_full: [2, 4, 8, ..., 2^(n/2), ..., 8, 4, 2]
    # if n_bonds %2 == 1:
    #     half = [2**i for i in range(1, int(n_bonds // 2) + 2)]
    #     chi_full = half + half[::-1][1:]
    # else:
    #     half = [2**i for i in range(1, int(n_bonds // 2) + 1)]
    #     chi_full = half + half[::-1][0:]
    # mps_full_memory = sum(chi_full[j] * chi_full[j + 1] for j in range(len(chi_full) - 1))

    N3 = int(len(chi)+1)

    # Compute MPS memory
    chi = np.concatenate(([1], chi, [1]))

    mps_comp_memory = 2*sum(chi[j] * chi[j - 1] for j in range(1,len(chi)))
    # print("mps_comp_memory: ", mps_comp_memory)    
    # Compression ratio: full / compressed
    comp_ratio_memory = 2**(N3) / mps_comp_memory if mps_comp_memory != 0 else float('inf')
    if cr_dof:
        sum2_chi = sum(chi[j]**2 for j in range(1, len(chi)-1))
        # print("sum2_chi: ", sum2_chi)
        comp_ratio_dof = 2**(N3) / (mps_comp_memory - sum2_chi)
        return comp_ratio_memory, comp_ratio_dof
    else:
        return comp_ratio_memory
