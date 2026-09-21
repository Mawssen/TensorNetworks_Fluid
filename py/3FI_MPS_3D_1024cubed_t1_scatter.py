#3_scatters

import sys
mps_codes_path = "/ix/pgivi/moe32/Schmidt/mps_py_codes"
sys.path.append(mps_codes_path)

###########################################################################################################################

from process_file_1024 import load_JHS_velocity_1024_jul, load_JHS_velocity_1024, load_velocity_binary
from plotting import plot_scatter, plot_contour
from calculate_flow_parameters_comb import rel_error
from calculate_orientation import calculate_unit_vectors
from save_as_mat import save_as_mat
import numpy as np
import matplotlib.pyplot as plt
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

# v_max = 3.6
# v_min = -4.2

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

v_max = 2
v_min = -3

###########################################################################################################################

u_comp, chi_u = load_JHS_velocity_1024_jul(comp_path_u)
v_comp, chi_v = load_JHS_velocity_1024_jul(comp_path_v)
w_comp, chi_w = load_JHS_velocity_1024_jul(comp_path_w)
# u_comp = u_comp-np.mean(u_comp)
# v_comp = v_comp-np.mean(v_comp)
# w_comp = w_comp-np.mean(w_comp)


# u_hat, v_hat, w_hat = calculate_unit_vectors(u, v, w)
# u_comp_hat, v_comp_hat, w_comp_hat = calculate_unit_vectors(u_comp, v_comp, w_comp)

# plot_scatter(u, u_comp, cutoff, title='Velocity u', label='u', file_name='Velocity_u')
# plot_scatter(v, v_comp, cutoff, title='Velocity v', label='v', file_name='Velocity_v')
# plot_scatter(w, w_comp, cutoff, title='Velocity w', label='w', file_name='Velocity_w')

# plot_scatter(u_hat, u_comp_hat, cutoff, title='Orientation of Velocity u', label='u hat', file_name='Orientation_of_Velocity_u')
# plot_scatter(v_hat, v_comp_hat, cutoff, title='Orientation of Velocity v', label='v hat', file_name='Orientation_of_Velocity_v')
# plot_scatter(w_hat, w_comp_hat, cutoff, title='Orientation of Velocity w', label='w hat', file_name='Orientation_of_Velocity_w')

# plot_scatter(u*u, u_comp*u_comp, cutoff, title='Reynolds Stress u\'u\'', label='u\'u\'', file_name='Reynolds_Stress_uu')
# plot_scatter(v*v, v_comp*v_comp, cutoff, title='Reynolds Stress v\'v\'', label='v\'v\'', file_name='Reynolds_Stress_vv')
# plot_scatter(w*w, w_comp*w_comp, cutoff, title='Reynolds Stress w\'w\'', label='w\'w\'', file_name='Reynolds_Stress_ww')

#plot_scatter(u*v, u_comp*v_comp, cutoff, title='Reynolds Stress u\'v\'', label='u\'v\'', file_name='Reynolds_Stress_uv')
#plot_scatter(u*w, u_comp*w_comp, cutoff, title='Reynolds Stress u\'w\'', label='u\'w\'', file_name='Reynolds_Stress_uw')
#plot_scatter(v*w, v_comp*w_comp, cutoff, title='Reynolds Stress v\'w\'', label='v\'w\'', file_name='Reynolds_Stress_vw')

# E_k = 0.5 * (u**2+v**2+w**2)
# E_k_comp = 0.5 * (u_comp**2+v_comp**2+w_comp**2)
# plot_scatter(E_k, E_k_comp, cutoff, title='Kinetic Energy', label=r'$E_{k}$', file_name='Kinetic_Energy')

# print(f'DNS:  mean(u) = {np.mean(u)}, max(u) = {np.max(u)}, min(u) = {np.min(u)}')
# print(f'Comp: mean(u) = {np.mean(u_comp)}, max(u) = {np.max(u_comp)}, min(u) = {np.min(u_comp)}')
# print(f'DNS:  mean(v) = {np.mean(v)}, max(v) = {np.max(v)}, min(v) = {np.min(v)})')
# print(f'Comp: mean(v) = {np.mean(v_comp)}, max(v) = {np.max(v_comp)}, min(v) = {np.min(v_comp)}')
# print(f'DNS:  mean(w) = {np.mean(w)}, max(w) = {np.max(w)}, min(w) = {np.min(w)}')
# print(f'Comp: mean(w) = {np.mean(w_comp)}, max(w) = {np.max(w_comp)}, min(w) = {np.min(w_comp)}')

k = N // 2
print("max u at k = ", k, " is ", np.max(u[:, :, k]))
print("min u at k = ", k, " is ", np.min(u[:, :, k]))

print("\nmax u_comp at k = ", k, " is ", np.max(u_comp[:, :, k]))
print("min u_comp at k = ", k, " is ", np.min(u_comp[:, :, k]))

c_map = 'cmr.iceburn'


plot_contour(u, c_map=c_map, v_min=v_min, v_max=v_max)
plot_contour(u_comp, cutoff=cutoff, c_map=c_map, v_min=v_min, v_max=v_max)


# u = u-np.mean(u)
# v = v-np.mean(v)
# w = w-np.mean(w)
# u_comp = u_comp-np.mean(u_comp)
# v_comp = v_comp-np.mean(v_comp)
# w_comp = w_comp-np.mean(w_comp)

# # mean_rs[_][0] = truth, mean_rs[_][1] = comp, mean_rs[_][2] = rel_error
# mean_rs = {'uu': [0, 0, 0], 'vv': [0, 0, 0], 'ww': [0, 0, 0], 'uv': [0, 0, 0], 'uw': [0, 0, 0], 'vw': [0, 0, 0]}
# mean_rs['uu'][0]= np.mean(u*u)
# mean_rs['uu'][1]= np.mean(u_comp*u_comp)
# mean_rs['uu'][2]= rel_error(mean_rs['uu'][0], mean_rs['uu'][1])

# mean_rs['vv'][0]= np.mean(v*v)
# mean_rs['vv'][1]= np.mean(v_comp*v_comp)
# mean_rs['vv'][2]= rel_error(mean_rs['vv'][0], mean_rs['vv'][1])

# mean_rs['ww'][0]= np.mean(w*w)
# mean_rs['ww'][1]= np.mean(w_comp*w_comp)
# mean_rs['ww'][2]= rel_error(mean_rs['ww'][0], mean_rs['ww'][1])

# mean_rs['uv'][0]= np.mean(u*v)
# mean_rs['uv'][1]= np.mean(u_comp*v_comp)
# mean_rs['uv'][2]= rel_error(mean_rs['uv'][0], mean_rs['uv'][1])

# mean_rs['uw'][0]= np.mean(u*w)
# mean_rs['uw'][1]= np.mean(u_comp*w_comp)
# mean_rs['uw'][2]= rel_error(mean_rs['uw'][0], mean_rs['uw'][1])

# mean_rs['vw'][0]= np.mean(v*w)
# mean_rs['vw'][1]= np.mean(v_comp*w_comp)
# mean_rs['vw'][2]= rel_error(mean_rs['vw'][0], mean_rs['vw'][1])

# save_as_mat('mean_rs', mean_rs)

# print('mean E_k: ',np.mean(E_k))
# print('mean u\'u\': ',mean_rs['uu'])
# print('mean v\'v\': ',mean_rs['vv'])
# print('mean w\'w\': ',mean_rs['ww'])
# print('mean u\'v\': ',mean_rs['uv'])
# print('mean u\'w\': ',mean_rs['uw'])
# print('mean v\'w\': ',mean_rs['vw'])
