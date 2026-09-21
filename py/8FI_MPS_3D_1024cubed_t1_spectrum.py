import sys
mps_codes_path = "/ix/pgivi/moe32/Schmidt/Nik_code/code/AnalysingResultsFromDNS_demo/ComputationalSpaces/mps_py_codes"
sys.path.append(mps_codes_path)

###########################################################################################################################

from process_file_1024 import load_JHS_velocity_1024_jul
from save_as_mat import save_as_mat
from calculate_spectrum import calculate_spectrum
import numpy as np

cutoff = 1e-2
comp_path_u = f'truncated_u_1024_il_cf0.01.mat'
comp_path_v = f'truncated_v_1024_il_cf0.01.mat'
comp_path_w = f'truncated_w_1024_il_cf0.01.mat'

u_comp, chi_u = load_JHS_velocity_1024_jul(comp_path_u)
v_comp, chi_v = load_JHS_velocity_1024_jul(comp_path_v)
w_comp, chi_w = load_JHS_velocity_1024_jul(comp_path_w)
u_comp = u_comp-np.mean(u_comp)
v_comp = v_comp-np.mean(v_comp)
w_comp = w_comp-np.mean(w_comp)


k, spec = calculate_spectrum(u_comp, v_comp, w_comp)

save_as_mat(f'spectrum_{cutoff}_only', {'k': k, 'spec': spec})
