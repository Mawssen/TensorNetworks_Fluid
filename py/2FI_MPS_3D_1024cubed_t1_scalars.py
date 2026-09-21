import sys
mps_codes_path = "/ix/pgivi/moe32/Schmidt/Nik_code/code/AnalysingResultsFromDNS_demo/ComputationalSpaces/mps_py_codes"
sys.path.append(mps_codes_path)

###########################################################################################################################

from process_file_1024 import load_JHS_velocity_1024, load_JHS_velocity_1024_jul
from calculate_flow_parameters_comb import calculate_flow_parameters_comb
from save_as_mat import save_as_mat
import numpy as np

cutoff = 1e-2
comp_path_u = f'truncated_u_1024_il_cf0.01.mat'
comp_path_v = f'truncated_v_1024_il_cf0.01.mat'
comp_path_w = f'truncated_w_1024_il_cf0.01.mat'

org_path_u = f'../../../Forced_Isotropic_1024Cubed_mat_in_one/U_t1.mat'
org_path_v = f'../../../Forced_Isotropic_1024Cubed_mat_in_one/V_t1.mat'
org_path_w = f'../../../Forced_Isotropic_1024Cubed_mat_in_one/W_t1.mat'

u = load_JHS_velocity_1024(org_path_u)
v = load_JHS_velocity_1024(org_path_v)
w = load_JHS_velocity_1024(org_path_w)
u = u-np.mean(u)
v = v-np.mean(v)
w = w-np.mean(w)

u_comp, chi_u = load_JHS_velocity_1024_jul(comp_path_u)
v_comp, chi_v = load_JHS_velocity_1024_jul(comp_path_v)
w_comp, chi_w = load_JHS_velocity_1024_jul(comp_path_w)
u_comp = u_comp-np.mean(u_comp)
v_comp = v_comp-np.mean(v_comp)
w_comp = w_comp-np.mean(w_comp)

nu = 0.000185
dx = 0.0061359406
dy = dx
dz = dx

flow_parameters, spectrum = calculate_flow_parameters_comb((u, v, w), (u_comp, v_comp, w_comp), (dx, dy, dz))

save_as_mat(f'flow_parameters_cf_{cutoff}', flow_parameters)
save_as_mat(f'spectrum_cf_{cutoff}', spectrum)

print(f'Flow Parameters for cutoff = {cutoff} components:')
for comp, params in flow_parameters.items():
    converted_params = [float(p) for p in params]
    print(f"{comp}:\n{converted_params} \n")

