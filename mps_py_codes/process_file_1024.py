
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import h5py
import time
from sympy import factorint
from math import prod

from velocity_name import velocity_name_ChF, velocity_name_FI
from reshape_and_transpose import inverse_reshape_and_transpose, reshape_and_transpose
from prime_sort import prime_sort
from FindSingValsAndEntEntr import FindSingValsAndEntEntr_uneq, FindSingValsAndEntEntr_uneq_1024
from CritChis_uneq import CritChis_uneq, CritChis_uneq_guess
from comp_ratio_uneq import comp_ratio_uneq
from calculate_u_comp_fidelity_l2norm import calculate_u_comp_fidelity_l2
from save_as_mat import save_as_mat, save_as_mat_hdf5

def process_file_1024(file_case,file_name,file_format,file_path,cutoff=np.array([1e-2, 1e-3, 1e-4])):

    """
    Process a velocity field file from Channelflow or Forced_Isotropic cases.

    Parameters
    ----------
    file_case : str
        Either 'Channelflow' or 'Forced_Isotropic'.
    file_name : str
        The name of the file, without extension.
    file_format : str
        The file format, either '.h5' or '.mat'.
    file_path : str
        The path to the file.
    cutoff : array_like, optional
        The cutoffs for the chi values, default is [1e-2, 1e-3, 1e-4].

    Returns
    -------
    result : dict
        A dictionary containing the results, including:
        - mps_U : the MPS of the velocity field
        - u_comp_e2, u_comp_e3, u_comp_e4 : the compressed velocity fields
        - SingVals_U : the singular values of the velocity field
        - Chi_max : the maximum chi value
        - Chi_U : the chi values
        - co_cr_fid_l2_U : the compression ratio, fidelity, and l2 norm of the compressed velocity fields
        - EntEntr_U : the entropy of the singular values
        - RunTime : the time taken to process the file
    """
    start_time = time.time()
    cutoff = cutoff.reshape(-1,1)

    cr_u = np.zeros([cutoff.shape[0],1])
    u_fidelity = np.zeros([cutoff.shape[0],1])
    u_l2norm = np.zeros([cutoff.shape[0],1])
    
    if 'U' in file_name:
        vel_name = 'U1'
    elif 'V' in file_name:
        vel_name = 'V1'
    elif 'W' in file_name:
        vel_name = 'W1'
    else:
        print(f"Velocity name for file {file_name} not found.")

    # if file_case == 'Channelflow':
    #     vel_name = velocity_name_ChF(file_name)
    # if file_case == 'Forced_Isotropic':
    #     vel_name = velocity_name_FI(file_name)
    # else:
    #     print(f"File case: {file_case} not found.")
    
    
    if file_format == '.h5':
        with h5py.File(file_path, 'r') as file:
            vel = file[vel_name][()]
            # vel = vel[()]
    
    if file_format == '.mat':
        # mat_data = scipy.io.loadmat(file_path)
        # vel = mat_data[vel_name]
        with h5py.File(file_path, 'r') as file:
            u = file[vel_name][()]
    else:
        print(f"File format {file_format} not found.")

    
    # u = vel[0:74, 0:97, 0:38, vel_index-1]
    # u = vel[0:16, 0:16, 0:16, vel_index-1]
    

    # xyz_tuple = (2,1,0)
    # u = np.transpose(u,xyz_tuple)
    
    print("original u shape: ",u.shape)

    fx = factorint(u.shape[0])
    fy = factorint(u.shape[1])
    fz = factorint(u.shape[2])

    # Convert the factor counts to lists
    P_x = list(fx.values())
    P_y = list(fy.values())
    P_z = list(fz.values())
    
    # Compute Lx_P, Ly_P, Lz_P
    Lx_P = P_x[0] + (P_x[1] if len(P_x) > 1 else 0)
    Ly_P = P_y[0] + (P_y[1] if len(P_y) > 1 else 0)
    Lz_P = P_z[0] + (P_z[1] if len(P_z) > 1 else 0)
    L_P = Lx_P + Ly_P + Lz_P
    print(f"Lx_P: {Lx_P}, Ly_P: {Ly_P}, Lz_P: {Lz_P}")

    ordered_primes, axis_p,transpose_primes = prime_sort(u.shape)
    print("ordered primes: ",ordered_primes)
    print("axes powers:",axis_p)

    # Reshape and transpose u
    u_reshaped = reshape_and_transpose(u, ordered_primes, axis_p)
    print("u after reshape and transpose:", u_reshaped.shape)

    # max_dim = 20
    # SingVals_U, EntEntr_U, chi_max,Us = FindSingValsAndEntEntr_uneq_1024(u_reshaped, max_dim)
    SingVals_U, EntEntr_U, chi_max,Us = FindSingValsAndEntEntr_uneq(u_reshaped)

    # chi_guess = np.array([2, 4, 8, 16, 32, 64, 128, 256, 510, 1014, 1990, 3664, 5408, 5250])
    chi_e2 = [2,2,1,1,1]
    chi_e3 = [1,3,2,1,1]
    chi_e4 = [1,1,1,1,1]
    chi_guess = np.array([chi_e2, chi_e3, chi_e4])
                         
    # chi_U = CritChis_uneq(SingVals_U, cutoff, chi_max)
    chi_U = CritChis_uneq_guess(SingVals_U, cutoff, chi_max, chi_guess)

    cr_u = comp_ratio_uneq(chi_U,u_reshaped.shape)


    # Calculate u_comp, fidelity, and l2_norm
    # for i in range(len(cutoff)):
    #     if i == 1:
    #         u_comp_e3, u_fidelity[i,:], u_l2norm[i,:] = calculate_u_comp_fidelity_l2(u_reshaped, Us, chi_U[:,i])
    #     else:
    #         _, u_fidelity[i,:], u_l2norm[i,:] = calculate_u_comp_fidelity_l2(u_reshaped, Us, chi_U[:,i])
    #         # _, u_fidelity[i,:], u_l2norm[i,:] = calculate_u_comp_fidelity_l2(u_reshaped, Us, chi_max[:])
    
    u_comp_e2, u_fidelity[0,:], u_l2norm[0,:] = calculate_u_comp_fidelity_l2(u_reshaped, Us, chi_U[:,0])
    u_comp_e3, u_fidelity[1,:], u_l2norm[1,:] = calculate_u_comp_fidelity_l2(u_reshaped, Us, chi_U[:,1])
    u_comp_e4, u_fidelity[2,:], u_l2norm[2,:] = calculate_u_comp_fidelity_l2(u_reshaped, Us, chi_U[:,2])

    u_comp_e2 = inverse_reshape_and_transpose(u_comp_e2, axis_p, u.shape)
    u_comp_e3 = inverse_reshape_and_transpose(u_comp_e3, axis_p, u.shape)
    u_comp_e4 = inverse_reshape_and_transpose(u_comp_e4, axis_p, u.shape)

    print("u_comp shape: ",u_comp_e3.shape)

    diff_u = u_comp_e3 - u
    # print("difference:", diff_u[:,0,0])
    print("Max abs difference u_comp_e3:", np.max(np.abs(diff_u)))
    print("Max abs u_truth:", np.max(np.abs(u)))
    
    co_cr_fid_l2_u = np.array([cutoff, cr_u, u_fidelity, u_l2norm])

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    result_U = {
        'SingVals_U': SingVals_U,
        'Chi_max': chi_max,
        'Chi_U': chi_U,
        'co_cr_fid_l2_U': co_cr_fid_l2_u,
        'EntEntr_U':EntEntr_U,
        'RunTime': elapsed_time
    }

    u_size_name = f'_{u.shape[0]}x{u.shape[1]}x{u.shape[2]}_'
    output_name = file_case+u_size_name+file_name

    # Save data in MATLAB 7.3 (HDF5) format
    save_as_mat_hdf5(output_name + '_SingVals_U', result_U)
    save_as_mat_hdf5(output_name + '_comp_e2', {'u_comp_e2': u_comp_e2})
    save_as_mat_hdf5(output_name + '_comp_e3', {'u_comp_e3': u_comp_e3})
    save_as_mat_hdf5(output_name + '_comp_e4', {'u_comp_e4': u_comp_e4})
    save_as_mat_hdf5(output_name + '_mps_U', {'Us': Us})
    # S = h5read('Forced_Isotropic_10x10x10_U_t1_mps_U.mat','/Us_0');

    return result_U, Us, u_comp_e2, u_comp_e3, u_comp_e4



def process_file_1(file_name, cutoff=np.array([1e-2, 1e-3, 1e-4])):
    start_time = time.time()
    
    cutoff = cutoff.reshape(-1,1)

    cr_u = np.zeros([cutoff.shape[0],1])
    cr_v = np.zeros([cutoff.shape[0],1])
    cr_w = np.zeros([cutoff.shape[0],1])

    u_fidelity = np.zeros([cutoff.shape[0],1])
    v_fidelity = np.zeros([cutoff.shape[0],1])
    w_fidelity = np.zeros([cutoff.shape[0],1])

    u_l2norm = np.zeros([cutoff.shape[0],1])
    v_l2norm = np.zeros([cutoff.shape[0],1])
    w_l2norm = np.zeros([cutoff.shape[0],1])
    
    # file_path = f'Forced_Isotropic/02 Grid 128/Forced_Isotropic_128Cubed_SubDomain_h5/{file_name}.h5'
    file_path = f'Forced_Isotropic/01 Grid 64/Forced_Isotropic_64Cubed_h5/{file_name}.h5'
    # file_path = f'Forced_Isotropic/3 Grid 256/Forced_Isotropic_256Cubed_SubDomain_h5/{file_name}.h5'
    # file_path = f'Forced_Isotropic/4 Grid 512/Forced_Isotropic_512Cubed_SubDomain_h5/{file_name}.h5'
    # file_path = f'Channelflow/3D/21 Grid 512x512x512_Part1_h5/{file_name}.h5'
    # file_path = f'Channelflow/3D/31 Grid 2048x512x1536_h5/{file_name}.h5'
    # file_path = f'Channelflow/3D/62 Grid 256x64x128_h5/{file_name}.h5'

    # vel_name = velocity_name_ChF(file_name)
    vel_name = velocity_name_FI(file_name)

    if not vel_name:
        print(f"Velocity name for file {file_name} not found.")
        return None

    with h5py.File(file_path, 'r') as file:
        vel = file[vel_name]
        vel = vel[()]

    # u = vel[:, :, :, 0]
    # v = vel[:, :, :, 1]
    # w = vel[:, :, :, 2]

    u = vel[0:32, 0:16, :, 0]
    v = vel[0:32, 0:16, :, 1]
    w = vel[0:32, 0:16, :, 2]

    # u, v, w = combine_parts(file_name)

    xyz_tuple = (2,1,0)
    u = np.transpose(u,xyz_tuple)
    v = np.transpose(v,xyz_tuple)
    w = np.transpose(w,xyz_tuple)


    print("original u shape: ",u.shape)

    fx = factorint(u.shape[0])
    fy = factorint(u.shape[1])
    fz = factorint(u.shape[2])

    # Convert the factor counts to lists
    P_x = list(fx.values())
    P_y = list(fy.values())
    P_z = list(fz.values())
    
    # Compute Lx_P, Ly_P, Lz_P
    Lx_P = P_x[0] + (P_x[1] if len(P_x) > 1 else 0)
    Ly_P = P_y[0] + (P_y[1] if len(P_y) > 1 else 0)
    Lz_P = P_z[0] + (P_z[1] if len(P_z) > 1 else 0)
    L_P = Lx_P + Ly_P + Lz_P
    # print(f"L_P2: {L_P2}, L_P3: {L_P3}")
    print(f"Lx_P: {Lx_P}, Ly_P: {Ly_P}, Lz_P: {Lz_P}")


    # shape_tuple = prime_sort(u.shape)

    # print(shape_tuple)
    # transpose_tuple = MPS_permute(Lx_P, Ly_P, Lz_P)
    
    ordered_primes, axis_p,transpose_primes = prime_sort(u.shape)
    print(ordered_primes)
    print(axis_p)
    # Reshape and transpose u
    u_reshaped = reshape_and_transpose(u, ordered_primes, axis_p)
    v_reshaped = reshape_and_transpose(v, ordered_primes, axis_p)
    w_reshaped = reshape_and_transpose(w, ordered_primes, axis_p)
    print(u.shape)


    SingVals_U, EntEntr_U, chi_max,Us = FindSingValsAndEntEntr_uneq(u_reshaped)
    SingVals_V, EntEntr_V, _, Vs= FindSingValsAndEntEntr_uneq(v_reshaped)
    SingVals_W, EntEntr_W, _, Ws= FindSingValsAndEntEntr_uneq(w_reshaped)

    # print("shape Us: ",Us[1])
    chi_U = CritChis_uneq(SingVals_U, cutoff, chi_max)
    chi_V = CritChis_uneq(SingVals_V, cutoff, chi_max)
    chi_W = CritChis_uneq(SingVals_W, cutoff, chi_max)

    cr_u = comp_ratio_uneq(chi_U,u_reshaped.shape)
    cr_v = comp_ratio_uneq(chi_V,v_reshaped.shape)
    cr_w = comp_ratio_uneq(chi_W,w_reshaped.shape)
    # mps_U = result['mps_U']
    # u_truth = result['u']
    # chi_crit = [2, 4, 5, 4, 2]

    # Calculate u_comp, fidelity, and l2_norm
    for i in range(len(cutoff)):
        u_comp, u_fidelity[i,:], u_l2norm[i,:] = calculate_u_comp_fidelity_l2(u_reshaped, Us, chi_U[:,i])
        v_comp, v_fidelity[i,:], v_l2norm[i,:] = calculate_u_comp_fidelity_l2(v_reshaped, Vs, chi_V[:,i])
        w_comp, w_fidelity[i,:], w_l2norm[i,:] = calculate_u_comp_fidelity_l2(w_reshaped, Ws, chi_W[:,i])

    # print("co_u: ",cutoff.shape)
    # print("cr_u: ",cr_u.shape)
    # print("u_fidelity: ",u_fidelity.shape)
    # print("u_l2norm: ",u_l2norm.shape)
    
    co_cr_fid_l2_u = np.array([cutoff, cr_u, u_fidelity, u_l2norm])
    co_cr_fid_l2_v = np.array([cutoff, cr_v, v_fidelity, v_l2norm])
    co_cr_fid_l2_w = np.array([cutoff, cr_w, w_fidelity, w_l2norm])

    # Print the results
    # print("u Fidelity:", u_fidelity*100)
    # print("u L2 Norm:", u_l2norm)

    # print("v Fidelity:", v_fidelity*100)
    # print("v L2 Norm:", v_l2norm)

    # print("w Fidelity:", w_fidelity*100)
    # print("w L2 Norm:", w_l2norm)
    # print("Comp_u:", cr_u, " Avg comp:", np.mean(cr_u))
    # print("Comp_u:", cr_v, " Avg comp:", np.mean(cr_v))
    # print("Comp_u:", cr_w, " Avg comp:", np.mean(cr_w))

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return {
        'cutoff': cutoff,
        'mps_U':Us,
        'mps_V':Vs,
        'mps_W':Ws,
        'SingVals_U': SingVals_U,
        'SingVals_V': SingVals_V,
        'SingVals_W': SingVals_W,
        'Chi_max': chi_max,
        'Chi_U': chi_U,
        'Chi_V': chi_V,
        'Chi_W': chi_W,
        'co_cr_fid_l2_U': co_cr_fid_l2_u,
        'co_cr_fid_l2_V': co_cr_fid_l2_v,
        'co_cr_fid_l2_W': co_cr_fid_l2_w,
        'EntEntr_U':EntEntr_U,
        'EntEntr_V':EntEntr_V,
        'EntEntr_W':EntEntr_W,
        'RunTime': elapsed_time
    }


def load_JHS_velocity_1024(file_path):
    vel_name = []
    with h5py.File(file_path, 'r') as file:
        for key in file.keys():
            vel_name.append(key)  # Save the key name
            break
        vel = file[vel_name[0]][()]
        # xcoor = file['xcoor'][()]
        # ycoor = file['ycoor'][()]
        # zcoor = file['zcoor'][()]
    vel = np.transpose(vel, (2, 1, 0))
    return vel


def load_JHS_velocity_1024_jul(file_path):
    # if 'U' in file_name:
    #     vel_name = 'U1'
    # elif 'V' in file_name:
    #     vel_name = 'V1'
    # elif 'W' in file_name:
    #     vel_name = 'W1'
    # else:
    #     print(f"Velocity name for file {file_name} not found.")

    vel_name = []
    with h5py.File(file_path, 'r') as file:
        for key in file.keys():
            vel_name.append(key)  
        vel = file[vel_name[1]][()]
        chi_crit = file[vel_name[0]][()]
    vel = np.transpose(vel, (2, 1, 0))
    return vel, chi_crit


def load_JHS_velocity_1024_jul_2(file_path):
    # if 'U' in file_name:
    #     vel_name = 'U1'
    # elif 'V' in file_name:
    #     vel_name = 'V1'
    # elif 'W' in file_name:
    #     vel_name = 'W1'
    # else:
    #     print(f"Velocity name for file {file_name} not found.")

    vel_name = []
    with h5py.File(file_path, 'r') as file:
        for key in file.keys():
            vel_name.append(key)  
        vel = file[vel_name[0]][()]
    vel = np.transpose(vel, (2, 1, 0))
    return vel



def load_velocity_binary(filename, N, skip_bytes=0):
    shape = (N, N, N)
    with open(filename, 'rb') as f:
        f.seek(skip_bytes)
        data = np.fromfile(f, dtype=np.float64)
        try:
            array = data.reshape(shape, order='F')  # Fortran order
        except ValueError as e:
            print(f"Could not reshape array from {filename}: {e}")
            array = None
        return array