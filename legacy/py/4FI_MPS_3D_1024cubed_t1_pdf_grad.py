#4_pdf_grad

import sys
mps_codes_path = "/ix/pgivi/moe32/Schmidt/mps_py_codes"
sys.path.append(mps_codes_path)

###########################################################################################################################

#PDF of Gradients

import numpy as np
from save_as_mat import load_h5
from plotting import plot_pdf
import json


cutoff = 1e-2
grad_dux_dx_comp = load_h5("grad_dux_dx.h5")
grad_dux_dy_comp = load_h5("grad_dux_dy.h5")
grad_dux_dz_comp = load_h5("grad_dux_dz.h5")
grad_duy_dx_comp = load_h5("grad_duy_dx.h5")
grad_duy_dy_comp = load_h5("grad_duy_dy.h5")
grad_duy_dz_comp = load_h5("grad_duy_dz.h5")
grad_duz_dx_comp = load_h5("grad_duz_dx.h5")
grad_duz_dy_comp = load_h5("grad_duz_dy.h5")
grad_duz_dz_comp = load_h5("grad_duz_dz.h5")


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


grad_dux_dx = load_h5(org_path['grad_vel']['dux_dx'])
grad_dux_dy = load_h5(org_path['grad_vel']['dux_dy'])
grad_dux_dz = load_h5(org_path['grad_vel']['dux_dz'])
grad_duy_dx = load_h5(org_path['grad_vel']['duy_dx'])
grad_duy_dy = load_h5(org_path['grad_vel']['duy_dy'])
grad_duy_dz = load_h5(org_path['grad_vel']['duy_dz'])
grad_duz_dx = load_h5(org_path['grad_vel']['duz_dx'])
grad_duz_dy = load_h5(org_path['grad_vel']['duz_dy'])
grad_duz_dz = load_h5(org_path['grad_vel']['duz_dz'])


#Daniel's data 
# pdf_min = 1e-11
# pdf_max = 1e-1
# x_max = 550
# # x_max_longitudinal = 300
# # x_min_longitudinal = -x_max_longitudinal
# x_min = -x_max
# gen_norm_dist = True

#JHS data 
pdf_min = 1e-8
pdf_max = 1e-0
x_max = 200
x_min = -x_max
laplace_dist = True


# pdf_min = None
# pdf_max = None
# x_max = None
# x_min = None
# laplace_dist = True

plot_pdf({'DNS':grad_dux_dx, f'Comp cutoff {cutoff}':grad_dux_dx_comp}, 
        title=r'$\partial u / \partial x$', file_name='grad_dux_dx_SameLimits', 
         pdf_min=pdf_min, pdf_max=pdf_max, x_min=x_min, x_max=x_max, 
         laplace_dist=laplace_dist)

plot_pdf({'DNS':grad_duy_dy, f'Comp cutoff {cutoff}':grad_duy_dy_comp}, 
         title=r'$\partial v / \partial y$', file_name='grad_duy_dy_SameLimits', 
         pdf_min=pdf_min, pdf_max=pdf_max, x_min=x_min, x_max=x_max, 
         laplace_dist=laplace_dist)

plot_pdf({'DNS':grad_duz_dz, f'Comp cutoff {cutoff}':grad_duz_dz_comp}, 
         title=r'$\partial w / \partial z$', file_name='grad_duz_dz_SameLimits', 
         pdf_min=pdf_min, pdf_max=pdf_max, x_min=x_min, x_max=x_max, 
         laplace_dist=laplace_dist)

plot_pdf({'DNS':grad_dux_dy, f'Comp cutoff {cutoff}':grad_dux_dy_comp}, 
         title=r'$\partial u / \partial y$', file_name='grad_dux_dy_SameLimits', 
         pdf_min=pdf_min, pdf_max=pdf_max, x_min=x_min, x_max=x_max, 
         laplace_dist=laplace_dist)

plot_pdf({'DNS':grad_dux_dz, f'Comp cutoff {cutoff}':grad_dux_dz_comp}, 
         title=r'$\partial u / \partial z$', file_name='grad_dux_dz_SameLimits', 
         pdf_min=pdf_min, pdf_max=pdf_max, x_min=x_min, x_max=x_max, 
         laplace_dist=laplace_dist)

plot_pdf({'DNS':grad_duy_dx, f'Comp cutoff {cutoff}':grad_duy_dx_comp}, 
         title=r'$\partial v / \partial x$', file_name='grad_duy_dx_SameLimits', 
         pdf_min=pdf_min, pdf_max=pdf_max, x_min=x_min, x_max=x_max, 
         laplace_dist=laplace_dist)

plot_pdf({'DNS':grad_duy_dz, f'Comp cutoff {cutoff}':grad_duy_dz_comp}, 
         title=r'$\partial v / \partial z$', file_name='grad_duy_dz_SameLimits', 
         pdf_min=pdf_min, pdf_max=pdf_max, x_min=x_min, x_max=x_max, 
         laplace_dist=laplace_dist)

plot_pdf({'DNS':grad_duz_dx, f'Comp cutoff {cutoff}':grad_duz_dx_comp}, 
         title=r'$\partial w / \partial x$', file_name='grad_duz_dx_SameLimits', 
         pdf_min=pdf_min, pdf_max=pdf_max, x_min=x_min, x_max=x_max, 
         laplace_dist=laplace_dist)

plot_pdf({'DNS':grad_duz_dy, f'Comp cutoff {cutoff}':grad_duz_dy_comp}, 
         title=r'$\partial w / \partial y$', file_name='grad_duz_dy_SameLimits', 
         pdf_min=pdf_min, pdf_max=pdf_max, x_min=x_min, x_max=x_max, 
         laplace_dist=laplace_dist)
