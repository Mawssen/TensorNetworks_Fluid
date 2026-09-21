import scipy.io
import numpy as np
from calculate_flow_parameters_comb import calculate_flow_parameters_comb
from process_file import load_JHS_velocity

def compare_org_comp(org_case, org_name, org_path, comp_paths, nu=0.000185, finite_diff_scheme='4th_order' ):  # or finite_dif_scheme == 'central'

    u,v,w,xcoor,ycoor,zcoor = load_JHS_velocity(org_case,org_name, org_path)
    u_p = u-np.mean(u)
    v_p = v-np.mean(v)
    w_p = w-np.mean(w)

    # print(f'u_mean:{np.mean(u)}, v_mean:{np.mean(v)}, w_mean:{np.mean(w)}',)

    # Set grid spacing
    dx = xcoor[1] - xcoor[0]
    dy = ycoor[1] - ycoor[0]
    dz = zcoor[1] - zcoor[0]
    h = np.array([dx, dy, dz])
    # print(f'h:{h}')

    # Load compressed velocity data from .mat files
    components = {'U': {}, 'V': {}, 'W': {}}
    velocity_names = ['U', 'V', 'W']
    comp_names = ['comp_e2', 'comp_e3', 'comp_e4']

    for i, velocity_name in enumerate(velocity_names):
        mat_data = scipy.io.loadmat(comp_paths[i])
        components[velocity_name][comp_names[0]] = mat_data['u_comp_e2']
        components[velocity_name][comp_names[1]] = mat_data['u_comp_e3']
        components[velocity_name][comp_names[2]] = mat_data['u_comp_e4']

    # Calculate flow parameters for each compressed velocity component
    flow_params = {}
    spectrum = {}

    for comp in comp_names:
        u_comp = components[velocity_names[0]][comp]
        v_comp = components[velocity_names[1]][comp]
        w_comp = components[velocity_names[2]][comp]
        u_p_comp = u_comp-np.mean(u_comp)
        v_p_comp = v_comp-np.mean(v_comp)
        w_p_comp = w_comp-np.mean(w_comp)
        # print(f'u_{comp}_mean:{np.mean(u_comp)}, v_{comp}_mean:{np.mean(v_comp)}, w_{comp}_mean:{np.mean(w_comp)}')

        # Call the function to calculate flow parameters (assuming the function exists)
        flow_params[comp],spectrum[comp] = calculate_flow_parameters_comb(
            (u_p, v_p, w_p), (u_p_comp, v_p_comp, w_p_comp), nu, h, finite_diff_scheme)

    return flow_params, spectrum

