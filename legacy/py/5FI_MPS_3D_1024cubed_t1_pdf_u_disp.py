#5_pdf_u_disp

import sys
mps_codes_path = "/ix/pgivi/moe32/Schmidt/mps_py_codes"
sys.path.append(mps_codes_path)

###########################################################################################################################

#PDF velocity and dissipation rate

import numpy as np
import h5py
from process_file_1024 import load_velocity_binary, load_JHS_velocity_1024_jul, load_JHS_velocity_1024
from plotting import plot_pdf, plot_pdf_eps, plot_scatter
from process_file_1024 import load_JHS_velocity_1024_jul
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
# # u = u-np.mean(u)
# # v = v-np.mean(v)
# # w = w-np.mean(w)

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


with h5py.File(org_path['disp'], 'r') as file:
    eps = np.array(file['eps'])


u_comp, _ = load_JHS_velocity_1024_jul(comp_path_u)
v_comp, _ = load_JHS_velocity_1024_jul(comp_path_v)
w_comp, _ = load_JHS_velocity_1024_jul(comp_path_w)
with h5py.File('dissipation_comp.h5', 'r') as file:
    eps_comp = np.array(file['eps_comp'])


#Daniel's data 
# pdf_min = 1e-9
# pdf_max = 1
# x_max = 6.1
# x_min = -x_max
# norm_dist = True

#JHS data 
pdf_min = 1e-8
pdf_max = 1
x_max = 4
x_min = -x_max
norm_dist = True
log_norm_dist = True
x_max_disp = 22
x_min_disp = -1
pdf_max_disp = 10
pdf_min_disp = 1e-8
ticks_min = 0     #for dissipation 
ticks_max = 20  #for dissipation

# log_scale = False
# x_max_disp = 40
# x_min_disp = 1e-8
# pdf_max_disp = 1
# pdf_min_disp = 1e-6

# x_max_disp = 0.02
# pdf_max_disp = 200
# pdf_min_disp = 1


# pdf_min = None
# pdf_max = None
# x_max = None
# x_min = None
# norm_dist = False
# log_norm_dist = False

plot_pdf({'DNS':u, f'Comp cutoff {cutoff}':u_comp}, title=r'$u$',
        file_name='Velocity_u_1024Cubed_SameLimits', pdf_min=pdf_min, pdf_max=pdf_max,
        x_min=x_min, x_max=x_max, norm_dist=norm_dist)

plot_pdf({'DNS':v, f'Comp cutoff {cutoff}':v_comp}, title=r'$v$',
        file_name='Velocity_v_1024Cubed_SameLimits', pdf_min=pdf_min, pdf_max=pdf_max,
        x_min=x_min, x_max=x_max, norm_dist=norm_dist)

plot_pdf({'DNS':w, f'Comp cutoff {cutoff}':w_comp}, title=r'$w$',
        file_name='Velocity_w_1024Cubed_SameLimits', pdf_min=pdf_min, pdf_max=pdf_max,
        x_min=x_min, x_max=x_max, norm_dist=norm_dist)

# plot_pdf_eps({'DNS': eps, f'Comp cutoff {cutoff}': eps_comp}, log_norm_dist=log_norm_dist)

plot_pdf_eps({'DNS': eps, f'Comp cutoff {cutoff}': eps_comp}, log_norm_dist=log_norm_dist, 
             x_min=x_min_disp, x_max=x_max_disp, pdf_min=pdf_min_disp, pdf_max=pdf_max_disp)


# plot_scatter(eps.T, eps_comp, cutoff, title='Dissipation', label=r'$\varepsilon$', file_name='Dissipation_scatter_1',
#              ticks_min=ticks_min, ticks_max=ticks_max)
