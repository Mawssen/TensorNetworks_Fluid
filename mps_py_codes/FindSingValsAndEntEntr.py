import numpy as np
from math import prod
from scipy.sparse.linalg import svds

def FindSingValsAndEntEntr_uneq(u):
    
    """
    Find the singular values and entanglement entropy of the given matrix u.

    Parameters
    ----------
    u : array_like
       matrix u (velocity or pressure, etc) to to be decomposed. u can be with unequal dimensions of any number

    Returns
    -------
    singular_values : array_like
        The singular values of the SVD decomposition of the MPS.
    Entg_entr : array_like
        The entanglement entropy of the MPS.
    chi_max : array_like
        The maximum chi value of the MPS.
    ys : list
        A list of the y matrices for each iteration.

    Notes:
    This function is taking each axis (x,y,z) as a bond.
    """

    Ps = u.shape
    L = len(Ps)
    # print("L = ",L)
    chi_max = np.zeros([L - 1,1])
    Entg_entr = []
    
    # Initialize list to store y matrices for each iteration
    ys = []
    
    # Initialize sv_len array
    sv_len = np.zeros((1, L - 1))

    # Compute sv_len based on the min of two products
    for i in range(1, L):
        first_prod = np.prod(Ps[:i])  # Product from Ps[0] to Ps[i-1]
        second_prod = np.prod(Ps[i:]) # Product from Ps[i] to Ps[L-1]
        sv_len[0, i-1] = min(first_prod, second_prod)

    # Initialize singular_values array with max of sv_len values
    singular_values = np.zeros((L - 1, int(max(sv_len[0]))))
    u_reshaped = u
    first_part = 0
    second_part = 1
    for i in range(L - 1):
        # if i<(L+1)/2:
        if first_part<=second_part:
            # print("i = ", i)
            # First part of shape_u: multiply the elements from 0 to i
            first_part = np.prod(Ps[:i + 1])

            # Second part of shape_u: multiply the elements from i+1 to the end
            second_part = np.prod(Ps[i + 1:])
        
            shape_u = (first_part, second_part)

        else:
            shape_u = (int(u_reshaped.shape[1]*Ps[i]), int(u_reshaped.shape[1]/Ps[i]))

        # print("shape_u: ", shape_u)
        u_reshaped = np.reshape(u_reshaped, shape_u)
        # print("u_reshaped size: ",u_reshaped.shape)
        # Perform SVD
        y, s, vt = np.linalg.svd(u_reshaped, full_matrices=False)
        s_matrix = np.zeros((y.shape[1], vt.shape[0]))
        s_matrix[:len(s), :len(s)] = np.diag(s)
        u_reshaped = np.dot(s_matrix,vt)
        print("y size: ",y.shape)
        print("s size: ",s_matrix.shape)
        print("vt size: ",vt.shape)
        print("s*vt size: ",u_reshaped.shape)
        # Store the y matrix for this iteration
        print("Before y: ",y.shape)
        y_tuple = (int(y.shape[0]/Ps[i]), int(Ps[i]), int(y.shape[1]))
        y = np.reshape(y, y_tuple)
        ys.append(y)
        print("After y: ",y.shape, "\n")
        
        singular_values[i, 0:len(s)] = s[:]
        chi_max[i,0] = int(len(s))

        # Calculate entanglement entropy
        p = s**2 / np.sum(s**2)
        entEnt = -np.sum(p * np.log2(p))
        Entg_entr.append(entEnt)
        # print(" ")
    
    # For the last mps we don't have U, there is just s*vt
    print("Before y: ",u_reshaped.shape)
    y_tuple = (int(y.shape[2]), int(Ps[-1]), 1)
    u_reshaped = np.reshape(u_reshaped, y_tuple)
    print("After y: ",u_reshaped.shape, "\n")
    ys.append(u_reshaped)
    print("-----")
    chi_max = chi_max.astype(int)
    return singular_values, np.array(Entg_entr), chi_max, ys




def FindSingValsAndEntEntr_uneq_1024(u, max_dim):
    
    """
    Find the singular values and entanglement entropy of the given matrix u.

    Parameters
    ----------
    u : array_like
       matrix u (velocity or pressure, etc) to to be decomposed. u can be with unequal dimensions of any number

    Returns
    -------
    singular_values : array_like
        The singular values of the SVD decomposition of the MPS.
    Entg_entr : array_like
        The entanglement entropy of the MPS.
    chi_max : array_like
        The maximum chi value of the MPS.
    ys : list
        A list of the y matrices for each iteration.

    Notes:
    This function is taking each axis (x,y,z) as a bond.
    """

    Ps = u.shape
    L = len(Ps)
    # print("L = ",L)
    chi_max = np.zeros([L - 1,1])
    Entg_entr = []
    
    # Initialize list to store y matrices for each iteration
    ys = []
    
    # Initialize sv_len array
    sv_len = np.zeros((1, L - 1))

    # Compute sv_len based on the min of two products
    for i in range(1, L):
        first_prod = np.prod(Ps[:i])  # Product from Ps[0] to Ps[i-1]
        second_prod = np.prod(Ps[i:]) # Product from Ps[i] to Ps[L-1]
        sv_len[0, i-1] = min(first_prod, second_prod)

    # Initialize singular_values array with max of sv_len values
    singular_values = np.zeros((L - 1, int(max(sv_len[0]))))
    u_reshaped = u
    first_part = 0
    second_part = 1
    max_dim_flag = 0
    for i in range(L - 1):
        # if i<(L+1)/2:
        if first_part<=second_part:
            # print("i = ", i)
            # First part of shape_u: multiply the elements from 0 to i
            first_part = np.prod(Ps[:i + 1])

            # Second part of shape_u: multiply the elements from i+1 to the end
            second_part = np.prod(Ps[i + 1:])
        
            shape_u = (first_part, second_part)

            if max_dim_flag == 1:
                shape_u = (int(u_reshaped.shape[0]*Ps[i]), int(u_reshaped.shape[1]/Ps[i]))
                max_dim_flag = 0

        else:
            shape_u = (int(u_reshaped.shape[1]*Ps[i]), int(u_reshaped.shape[1]/Ps[i]))

        # print("shape_u: ", shape_u)
        u_reshaped = np.reshape(u_reshaped, shape_u)
        # print("u_reshaped size: ",u_reshaped.shape)
        # Perform SVD
        if min(shape_u[0], shape_u[1]) > max_dim:
            y, s, vt = svds(u_reshaped, k=max_dim, which='LM')
            s = s[::-1]
            max_dim_flag = 1
        else:
            y, s, vt = np.linalg.svd(u_reshaped, full_matrices=False)
        s_matrix = np.zeros((y.shape[1], vt.shape[0]))
        s_matrix[:len(s), :len(s)] = np.diag(s)
        u_reshaped = np.dot(s_matrix,vt)
        print("y size: ",y.shape)
        print("s size: ",s_matrix.shape)
        print("vt size: ",vt.shape)
        print("s*vt size: ",u_reshaped.shape)
        # Store the y matrix for this iteration
        print("Before y: ",y.shape)
        y_tuple = (int(y.shape[0]/Ps[i]), int(Ps[i]), int(y.shape[1]))
        y = np.reshape(y, y_tuple)
        ys.append(y)
        print("After y: ",y.shape, "\n")
        
        singular_values[i, 0:len(s)] = s[:]
        chi_max[i,0] = int(len(s))

        # Calculate entanglement entropy
        p = s**2 / np.sum(s**2)
        entEnt = -np.sum(p * np.log2(p))
        Entg_entr.append(entEnt)
        # print(" ")
    
    # For the last mps we don't have U, there is just s*vt
    print("Before y: ",u_reshaped.shape)
    y_tuple = (int(y.shape[2]), int(Ps[-1]), 1)
    u_reshaped = np.reshape(u_reshaped, y_tuple)
    print("After y: ",u_reshaped.shape, "\n")
    ys.append(u_reshaped)
    print("-----")
    chi_max = chi_max.astype(int)
    return singular_values, np.array(Entg_entr), chi_max, ys



def FindSingValsAndEntEntr_eq(u):
    
    """
    This function takes a one-dimensional array of the MPS and reshapes it into
    a matrix for each bond. It then performs a singular value decomposition on
    each bond and stores the singular values and entanglement entropies.

    Parameters:
    u (numpy array): 1D array representing the MPS.

    Returns:
    sv (numpy array): 2D array containing the singular values for each bond.
    Entg_entr (numpy array): 1D array containing the entanglement entropies for
        each bond.

    Notes:
    This function is taking each volume (xyz) as a bond.
    """

    n = int(np.log2(u.size) / 3)
    singular_values = []
    Entg_entr = []

    for i in range(1, n):
        shape_u = (2**(3*i), 2**(3*n-3*i))
        u_reshaped = np.reshape(u, shape_u)
        _, s, _ = np.linalg.svd(u_reshaped, full_matrices=False)
        singular_values.append(s)

        # Calculate entanglement entropy
        p = s**2 / np.sum(s**2)
        entEnt = -np.sum(p * np.log2(p))
        Entg_entr.append(entEnt)

    max_s_length = max(len(s) for s in singular_values)
    sv = np.zeros((n-1, max_s_length))
    
    for i, s in enumerate(singular_values):
        sv[i, :len(s)] = s

    return sv, np.array(Entg_entr)




def FindSingValsAndEntEntr_uneq_1(u):
    # L = int(np.log2(u.size))
    Ps = u.shape
    L = len(Ps)
    # r = int(np.mod(L,3))
    # n = int((L-r)/3)
    # P_2 = Ps.count(2)
    # P_3 = Ps.count(3)
    chi_max = np.zeros(L-1)
    # singular_values = []
    Entg_entr = []

    # Initialize sv_len array
    sv_len = np.zeros((1, L - 1))

    # Compute sv_len based on the min of two products
    for i in range(1, L):
        first_prod = prod(Ps[:i])  # Product from Ps[0] to Ps[i-1]
        second_prod = prod(Ps[i:]) # Product from Ps[i] to Ps[L-1]
        sv_len[0, i-1] = min(first_prod, second_prod)

    # Initialize singular_values array with max of sv_len values
    singular_values = np.zeros((L - 1, int(max(sv_len[0]))))
        
    for i in range(L - 1):
        # First part of shape_u: multiply the elements from 0 to i
        first_part = 1
        for j in range(i + 1):
            first_part *= Ps[j]

        # Second part of shape_u: multiply the elements from i+1 to the end
        second_part = 1
        for j in range(i + 1, L):
            second_part *= Ps[j]
        
        shape_u = (first_part, second_part)
        u_reshaped = np.reshape(u, shape_u)
        s = np.linalg.svd(u_reshaped, full_matrices=False, compute_uv=False)
        # s = sp.linalg.svd(u_reshaped, full_matrices=False, compute_uv=False)
        # print("s shape: ",s.shape)
        singular_values[i,0:len(s)] = s[:]
        chi_max[i] = int(len(s))
        # Calculate entanglement entropy
        p = s**2 / np.sum(s**2)
        entEnt = -np.sum(p * np.log2(p))
        Entg_entr.append(entEnt)

    # print(" ")
    return singular_values, np.array(Entg_entr), chi_max



def FindSingValsAndEntEntr_uneq_pwr2(u):
    L = int(np.log2(u.size))
    r = int(np.mod(L,3))
    n = int((L-r)/3)
    chi_max = np.zeros(n+r-1)
    # singular_values = []
    Entg_entr = []

    i = 1
    j = 0
    sv_row_len = int((L-(np.mod(L,2)))/2)
    singular_values = np.zeros((n+r-1,2**sv_row_len))
    while i < L:
        # print(i)
        if r != 0:
            shape_u = (2**i,2**(L-i))
            r=r-1
            if r == 0:
                i = i+2
        else:
            if i == 1:
                i = i+2
                # print(i)
            shape_u = (2**(i), 2**(L-i))
            i = i+2
        u_reshaped = np.reshape(u, shape_u)
        _, s, _ = np.linalg.svd(u_reshaped, full_matrices=False)
        
        # print("s shape: ",s.shape)
        singular_values[j,0:len(s)] = s[:]
        chi_max[j] = int(len(s))
        # Calculate entanglement entropy
        p = s**2 / np.sum(s**2)
        entEnt = -np.sum(p * np.log2(p))
        Entg_entr.append(entEnt)
        i=i+1
        j=j+1

    return singular_values, np.array(Entg_entr), chi_max



def FindSingValsAndEntEntr_uneq_pwr2_3(u):
    # L = int(np.log2(u.size))
    Ps = u.shape
    L = len(Ps)
    r = int(np.mod(L,3))
    n = int((L-r)/3)
    P_2 = Ps.count(2)
    P_3 = Ps.count(3)
    chi_max = np.zeros(n+r-1)
    # singular_values = []
    Entg_entr = []
    
    sv_len = np.zeros((1,L-1))
    for i in range(3,L,3):
        sv_len[0,i-1] = min(3*2**(i-1), 2**(L-i))
    singular_values = np.zeros((n+r-1,int(max(max(sv_len)))))
        
        # half_L = int((L-(np.mod(L,2)))/2)
        # if P_3 < half_L:
        #     singular_values = np.zeros((n+r-1, 2**(half_L))) #Still it's an over-estimation
        # else:
        #     singular_values = np.zeros((n+r-1,(3**(half_L-P_3))* 2**(half_L-P_3))) #Still it's an over-estimation
    
    i = 3
    j = 0
    while i < L:
        # print(i)
        #needs to be revised for a general case
        # if r != 0:
        #     shape_u = (2**i,2**(L-i))
        #     r=r-1
        #     if r == 0:
        #         i = i+2
        # else:
        #     if i == 1:
        #         i = i+2
        #         # print(i)
        #     shape_u = (2**(i), 2**(L-i))
        #     i = i+2
        shape_u = (3*2**(i-1), 2**(L-i))
        u_reshaped = np.reshape(u, shape_u)
        s = np.linalg.svd(u_reshaped, full_matrices=False, compute_uv=False)
        # s = sp.linalg.svd(u_reshaped, full_matrices=False, compute_uv=False)
        print("s shape: ",s.shape)
        singular_values[j,0:len(s)] = s[:]
        chi_max[j] = int(len(s))
        # Calculate entanglement entropy
        p = s**2 / np.sum(s**2)
        entEnt = -np.sum(p * np.log2(p))
        Entg_entr.append(entEnt)
        i=i+3
        j=j+1

    print(" ")
    return singular_values, np.array(Entg_entr), chi_max




