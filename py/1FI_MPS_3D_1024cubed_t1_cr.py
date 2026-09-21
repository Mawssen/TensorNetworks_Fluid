#1_cr

import sys
mps_codes_path = "/ix/pgivi/moe32/Schmidt/mps_py_codes"
sys.path.append(mps_codes_path)

###########################################################################################################################

from calculate_u_comp_fidelity_l2norm import calculate_fidelity, calculate_l2norm
from comp_ratio_uneq import comp_ratio_uneq, comp_ratio_memory
from process_file_1024 import load_velocity_binary, load_JHS_velocity_1024_jul, load_JHS_velocity_1024
from save_as_mat import save_as_mat
import numpy as np
import json

cutoff = 1e-2
comp_path_u = f'truncated_U_1024_il_cf0.01.mat'
comp_path_v = f'truncated_V_1024_il_cf0.01.mat'
comp_path_w = f'truncated_W_1024_il_cf0.01.mat'

###########################################################################################################################
## Binary data - Daniel

# comp_path_u = f'truncated_U_1024_il_cf0.0001.mat'
# comp_path_v = f'truncated_V_1024_il_cf0.0001.mat'
# comp_path_w = f'truncated_W_1024_il_cf0.0001.mat'

# N = 32
# config_path = "config_FI.json"
# with open(config_path, "r") as f:
#     config = json.load(f)
# org_path = config.get("32_binary_dns", {})


# # N = 1024
# # config_path = mps_codes_path + "/config_FI.json"
# # with open(config_path, "r") as f:
# #     config = json.load(f)
# # org_path = config.get("full_binary_dns", {})


# u = load_velocity_binary(org_path['vel']['U'], N, skip_bytes=192)
# v = load_velocity_binary(org_path['vel']['V'], N)
# w = load_velocity_binary(org_path['vel']['W'], N)
# u = u-np.mean(u)
# v = v-np.mean(v)
# w = w-np.mean(w)


###########################################################################################################################
## JHS data

# N = 128
# comp_path_u = f'jhs_truncated_u_1024_il_cf0.0001.mat'
# comp_path_v = f'jhs_truncated_v_1024_il_cf0.0001.mat'
# comp_path_w = f'jhs_truncated_w_1024_il_cf0.0001.mat'
# config_path = "config_FI.json"
# with open(config_path, "r") as f:
#     config = json.load(f)
# org_path = config.get("128_dns", {})

N = 1024
config_path = mps_codes_path + "/config_FI.json"
with open(config_path, "r") as f:
    config = json.load(f)
org_path = config.get("full_dns", {})

u = load_JHS_velocity_1024(org_path['vel']['U'])
v = load_JHS_velocity_1024(org_path['vel']['V'])
w = load_JHS_velocity_1024(org_path['vel']['W'])
# # u = u-np.mean(u)
# # v = v-np.mean(v)
# # w = w-np.mean(w)


###########################################################################################################################

u_comp, chi_u = load_JHS_velocity_1024_jul(comp_path_u)
v_comp, chi_v = load_JHS_velocity_1024_jul(comp_path_v)
w_comp, chi_w = load_JHS_velocity_1024_jul(comp_path_w)
# u_comp = u_comp-np.mean(u_comp)
# v_comp = v_comp-np.mean(v_comp)
# w_comp = w_comp-np.mean(w_comp)

#comp ratio dof - Nik's thesis
# chi_u = np.expand_dims(chi_u, axis=1)
# chi_v = np.expand_dims(chi_v, axis=1)
# chi_w = np.expand_dims(chi_w, axis=1)
# u_size = (2,) *3* int(np.log2(u.shape[0]))
# cr_u = comp_ratio_uneq(chi_u,u_size)
# cr_v = comp_ratio_uneq(chi_v,u_size)
# cr_w = comp_ratio_uneq(chi_w,u_size)

#comp ratio of memory
cr_u = comp_ratio_memory(chi_u, cr_dof=True) # both cr_memory and cr_dof
cr_v = comp_ratio_memory(chi_v, cr_dof=True)
cr_w = comp_ratio_memory(chi_w, cr_dof=True)
fidelity_u = calculate_fidelity(u, u_comp)
fidelity_v = calculate_fidelity(v, v_comp)
fidelity_w = calculate_fidelity(w, w_comp)
l2_norm_u = calculate_l2norm(u, u_comp)
l2_norm_v = calculate_l2norm(v, v_comp)
l2_norm_w = calculate_l2norm(w, w_comp)

print("cutoff: ", cutoff)
print("cr_u: ", cr_u, "cr_v: ", cr_v, "cr_w: ", cr_w)
print("fidelity_u: ", fidelity_u, "fidelity_v: ", fidelity_v, "fidelity_w: ", fidelity_w)
print("l2_norm_u: ", l2_norm_u, "l2_norm_v: ", l2_norm_v, "l2_norm_w: ", l2_norm_w)

output = {
    'cutoff': cutoff,
    'chi_u': chi_u,
    'chi_v': chi_v,
    'chi_w': chi_w,
    'cr_u': cr_u,
    'cr_v': cr_v,
    'cr_w': cr_w,
    'fidelity_u': fidelity_u,
    'fidelity_v': fidelity_v,
    'fidelity_w': fidelity_w,
    'l2_norm_u': l2_norm_u,
    'l2_norm_v': l2_norm_v,
    'l2_norm_w': l2_norm_w}

save_as_mat(f'chi_cr_fidelity_l2norm_cf_{cutoff}', output)
