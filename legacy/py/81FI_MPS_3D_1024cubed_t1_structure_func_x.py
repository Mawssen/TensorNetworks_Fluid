#8_structure_functions

import sys
mps_codes_path = "/ix/pgivi/moe32/Schmidt/mps_py_codes"
sys.path.append(mps_codes_path)

###########################################################################################################################


import numpy as np
from process_file_1024 import load_velocity_binary, load_JHS_velocity_1024_jul, load_JHS_velocity_1024
from process_file_1024 import load_JHS_velocity_1024_jul
from structure_functions import safe_struct
from save_as_mat import save_as_mat
import json


# cutoff = 1e-4
# comp_path_u = f'truncated_U_1024_il_cf0.0001.mat'
# comp_path_v = f'truncated_V_1024_il_cf0.0001.mat'
# comp_path_w = f'truncated_W_1024_il_cf0.0001.mat'
direction = "x"

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
# u = u-np.mean(u)
# v = v-np.mean(v)
# w = w-np.mean(w)

###########################################################################################################################

# u_comp, _ = load_JHS_velocity_1024_jul(comp_path_u)
# v_comp, _ = load_JHS_velocity_1024_jul(comp_path_v)
# w_comp, _ = load_JHS_velocity_1024_jul(comp_path_w)


# D = safe_struct((u_comp, v_comp, w_comp), direction, chkpt_freq=64, dtype=np.float64)
# save_as_mat(f'sf_cf_{cutoff}_dir_{direction}_comp', D)

D = safe_struct((u, v, w), direction, chkpt_freq=64, dtype=np.float64)
save_as_mat(f'sf_DNS_dir_{direction}_comp', D)
