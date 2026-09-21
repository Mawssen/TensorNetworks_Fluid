import numpy as np

def CritChis_uneq(SingValsU, desiredAccs, chi_max):
    
    """
    Compute the critical bond dimension chi_c for given desired accuracy in the unequal-bond-dimension case.
    
    Parameters
    ----------
    SingValsU : ndarray
        The singular values of the MPS.
    desiredAccs : ndarray
        The desired accuracy.
    chi_max : ndarray
        The maximum bond dimension.
    
    Returns
    -------
    critChis_U : ndarray
        The critical bond dimension chi_c.
    """
    L = SingValsU.shape[0] + 1
    critChis_U = np.zeros((L - 1, len(desiredAccs)), dtype=int)
    normU = np.sum(SingValsU[0, :] ** 2)

    for l in range(L - 1):
        for i in range(len(desiredAccs)):
            error = 1
            if i == 0:
                chi_Guess = 1
            else:
                chi_Guess = critChis_U[l, i - 1]
            
            desiredAcc = desiredAccs[i]
            
            while error > desiredAcc and (chi_Guess <= chi_max[l]):
                norm1 = np.sum(SingValsU[l, :chi_Guess] ** 2)
                error = 1 - (norm1 / normU)
                chi_Guess += 1
            
            critChis_U[l, i] = chi_Guess - 1
    
    return critChis_U



def CritChis_uneq_guess(SingValsU, desiredAccs, chi_max, chi_guess):
    
    L = SingValsU.shape[0] + 1
    critChis_U = np.zeros((L - 1, len(desiredAccs)), dtype=int)
    normU = np.sum(SingValsU[0, :] ** 2)

    for l in range(L - 1):
        for i in range(len(desiredAccs)):
            error = 1
            chi_Guess = chi_guess[i,l]
            
            desiredAcc = desiredAccs[i]
            
            while error > desiredAcc and (chi_Guess <= chi_max[l]):
                norm1 = np.sum(SingValsU[l, :chi_Guess] ** 2)
                error = 1 - (norm1 / normU)
                chi_Guess += 1
            
            critChis_U[l, i] = chi_Guess - 1
    
    return critChis_U

