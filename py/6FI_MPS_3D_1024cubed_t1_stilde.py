#6_stilde

import sys
mps_codes_path = "/ix/pgivi/moe32/Schmidt/mps_py_codes"
sys.path.append(mps_codes_path)

###########################################################################################################################

#PDF of stilde 
import numpy as np
from save_as_mat import load_h5
from plotting import plot_pdf
from calculate_eps_stilde import calculate_s_tilde
from save_as_mat import save_as_h5
import json

cutoff = 1e-2


###########################################################################################################################
## Binary data - Daniel

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


###########################################################################################################################
## JHS data

# N = 128
# config_path = "config_FI.json"
# with open(config_path, "r") as f:
#     config = json.load(f)
# org_path = config.get("128_dns", {})

N = 1024
config_path = mps_codes_path + "/config_FI.json"
with open(config_path, "r") as f:
    config = json.load(f)
org_path = config.get("full_dns", {})


###########################################################################################################################

## when calc for DNS is done
alpha = load_h5(org_path['strain_rate']['alpha'])
beta  = load_h5(org_path['strain_rate']['beta'])
gamma = load_h5(org_path['strain_rate']['gamma'])
s_tilde = load_h5(org_path['strain_rate']['s_tilde'])

## when calc for comp is done
alpha_comp = load_h5('Strain_rate_eigen_values_alpha_comp.h5')
beta_comp = load_h5('Strain_rate_eigen_values_beta_comp.h5')
gamma_comp = load_h5('Strain_rate_eigen_values_gamma_comp.h5')
s_tilde_comp = load_h5('Strain_rate_eigen_values_stilde_comp.h5')


# grad_dux_dx = load_h5(org_path['grad_vel']['dux_dx'])
# grad_dux_dy = load_h5(org_path['grad_vel']['dux_dy'])
# grad_dux_dz = load_h5(org_path['grad_vel']['dux_dz'])
# grad_duy_dx = load_h5(org_path['grad_vel']['duy_dx'])
# grad_duy_dy = load_h5(org_path['grad_vel']['duy_dy'])
# grad_duy_dz = load_h5(org_path['grad_vel']['duy_dz'])
# grad_duz_dx = load_h5(org_path['grad_vel']['duz_dx'])
# grad_duz_dy = load_h5(org_path['grad_vel']['duz_dy'])
# grad_duz_dz = load_h5(org_path['grad_vel']['duz_dz'])


# # Symmetric velocity gradients
# S11 = grad_dux_dx + grad_dux_dx
# S12 = grad_dux_dy + grad_duy_dx 
# S13 = grad_dux_dz + grad_duz_dx
# S22 = grad_duy_dy + grad_duy_dy
# S23 = grad_duy_dz + grad_duz_dy
# S33 = grad_duz_dz + grad_duz_dz

# Sij = np.array([[S11, S12, S13], 
#                 [S12, S22, S23],
#                 [S13, S23, S33]])/2
# Sij = np.transpose(Sij, axes=(2,3,4,0,1))

# print("Sij shape: ", Sij.shape)
# s_tilde, alpha, beta, gamma = calculate_s_tilde(Sij)



# grad_dux_dx_comp = load_h5("grad_dux_dx_comp.h5")
# grad_dux_dy_comp = load_h5("grad_dux_dy_comp.h5")
# grad_dux_dz_comp = load_h5("grad_dux_dz_comp.h5")
# grad_duy_dx_comp = load_h5("grad_duy_dx_comp.h5")
# grad_duy_dy_comp = load_h5("grad_duy_dy_comp.h5")
# grad_duy_dz_comp = load_h5("grad_duy_dz_comp.h5")
# grad_duz_dx_comp = load_h5("grad_duz_dx_comp.h5")
# grad_duz_dy_comp = load_h5("grad_duz_dy_comp.h5")
# grad_duz_dz_comp = load_h5("grad_duz_dz_comp.h5")


# # Symmetric velocity gradients
# S11_comp = grad_dux_dx_comp + grad_dux_dx_comp
# S12_comp = grad_dux_dy_comp + grad_duy_dx_comp
# S13_comp = grad_dux_dz_comp + grad_duz_dx_comp
# S22_comp = grad_duy_dy_comp + grad_duy_dy_comp
# S23_comp = grad_duy_dz_comp + grad_duz_dy_comp
# S33_comp = grad_duz_dz_comp + grad_duz_dz_comp

# Sij_comp = np.array([[S11_comp, S12_comp, S13_comp], 
#                      [S12_comp, S22_comp, S23_comp],
#                      [S13_comp, S23_comp, S33_comp]])/2
# Sij_comp = np.transpose(Sij_comp, axes=(2,3,4,0,1))

# print("Sij_comp shape: ", Sij_comp.shape)
# s_tilde_comp, alpha_comp, beta_comp, gamma_comp = calculate_s_tilde(Sij_comp)

##when calc is done, comment these lines
# save_as_h5('Strain_rate_eigen_values_stilde', 'stilde', s_tilde)
# save_as_h5('Strain_rate_eigen_values_alpha', 'alpha', alpha)
# save_as_h5('Strain_rate_eigen_values_beta', 'beta', beta)
# save_as_h5('Strain_rate_eigen_values_gamma', 'gamma', gamma)

# save_as_h5('Strain_rate_eigen_values_stilde_comp', 'stilde', s_tilde_comp)
# save_as_h5('Strain_rate_eigen_values_alpha_comp', 'alpha', alpha_comp)
# save_as_h5('Strain_rate_eigen_values_beta_comp', 'beta', beta_comp)
# save_as_h5('Strain_rate_eigen_values_gamma_comp', 'gamma', gamma_comp)


norm_dist = False
log_Y = False

plot_pdf({rf'$\tilde{{s}}^*$ DNS':s_tilde, rf'$\tilde{{s}}^*_{{comp}}$ cutoff {cutoff}':s_tilde_comp}, 
         title=r'$\tilde{s}^*$', log_Y=log_Y, norm_dist=norm_dist, 
         file_name='Strain_rate_eigen_values_stilde', x_min=-1.1, x_max=1.1, 
         pdf_min=0, pdf_max=1.6)

plot_pdf({rf'$\alpha$ DNS':alpha, rf'$\alpha_{{comp}}$ cutoff {cutoff}':alpha_comp},
         title=r'$\alpha$', norm_dist=norm_dist,
         file_name='Strain_rate_eigen_values_alpha', x_min=-5, x_max=125,
         pdf_min=1e-8, pdf_max=1)

plot_pdf({rf'$\beta$ DNS':beta, rf'$\beta_{{comp}}$ cutoff {cutoff}':beta_comp},
         title=r'$\beta$', norm_dist=norm_dist,
         file_name='Strain_rate_eigen_values_beta', x_min=-60, x_max=60,
         pdf_min=1e-8, pdf_max=1)

plot_pdf({rf'$\gamma$ DNS':gamma, rf'$\gamma_{{comp}}$ cutoff {cutoff}':gamma_comp},
         title=r'$\gamma$', norm_dist=norm_dist,
         file_name='Strain_rate_eigen_values_gamma', x_min=-160, x_max=10, 
         pdf_min=1e-8, pdf_max=1)
