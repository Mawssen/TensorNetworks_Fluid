#7_joint_pdf_QR

import sys
mps_codes_path = "/ix/pgivi/moe32/Schmidt/mps_py_codes"
sys.path.append(mps_codes_path)


###########################################################################################################################
#Joint PDF plot

import matplotlib.pyplot as plt
import numpy as np
from plotting import plot_QR_joint_pdf
from calculate_gradients import calculate_invariants, calculate_invariants_exact, compute_Qw
from save_as_mat import save_as_h5, load_h5
from plotting import plot_pdf
import json


cutoff = 1e-2
Qw_comp = 9.171738021735438

# N = 32
# config_path = "config_FI.json"
# with open(config_path, "r") as f:
#     config = json.load(f)
# org_path = config.get("32_binary_dns", {})
# Qw =  2.353024608021444

N = 1024
config_path = mps_codes_path + "/config_FI.json"
with open(config_path, "r") as f:
    config = json.load(f)

#Daniel
# org_path = config.get("full_binary_dns", {})
# Qw = 32.59979212200147

#JHS
org_path = config.get("full_dns", {})
Qw = 15.756560377380076

## when calc is done
div_exact_comp = load_h5('first_invariant_div_spectral.h5')
Q_exact_comp = load_h5('second_invariant_Q_spectral.h5')
R_exact_comp = load_h5('third_invariant_R_spectral.h5')
# Qw_comp = Qw

## when calc is done
div_exact = load_h5(org_path['invars']['div'])
Q_exact = load_h5(org_path['invars']['Q'])
R_exact = load_h5(org_path['invars']['R'])

# grad_dux_dx = load_h5(org_path['grad_vel']['dux_dx'])
# grad_dux_dy = load_h5(org_path['grad_vel']['dux_dy'])
# grad_dux_dz = load_h5(org_path['grad_vel']['dux_dz'])
# grad_duy_dx = load_h5(org_path['grad_vel']['duy_dx'])
# grad_duy_dy = load_h5(org_path['grad_vel']['duy_dy'])
# grad_duy_dz = load_h5(org_path['grad_vel']['duy_dz'])
# grad_duz_dx = load_h5(org_path['grad_vel']['duz_dx'])
# grad_duz_dy = load_h5(org_path['grad_vel']['duz_dy'])
# grad_duz_dz = load_h5(org_path['grad_vel']['duz_dz'])

# grad = np.array([[grad_dux_dx, grad_dux_dy, grad_dux_dz],
#                  [grad_duy_dx, grad_duy_dy, grad_duy_dz],
#                  [grad_duz_dx, grad_duz_dy, grad_duz_dz]])

# grad = np.transpose(grad, (2, 3, 4, 0, 1))  # Rearrange dimensions

# div_inexact, Q_inexact, R_inexact = calculate_invariants(grad)
# div_exact, Q_exact, R_exact = calculate_invariants_exact(grad)
# Qw = compute_Qw(grad)
# print("Qw = ", Qw)
# print("\nnorm(div_inexact-div_exact) = ", np.linalg.norm(div_inexact-div_exact))
# print("norm(Q_inexact-Q_exact) = ", np.linalg.norm(Q_inexact-Q_exact))
# print("norm(R_inexact-R_exact) = ", np.linalg.norm(R_inexact-R_exact))

# save_as_h5('first_invariant_div_spectral_inexact','div', div_inexact)
# save_as_h5('second_invariant_Q_spectral_inexact','Q', Q_inexact)
# save_as_h5('third_invariant_R_spectral_inexact','R', R_inexact)

# save_as_h5('first_invariant_div_spectral_exact','div', div_exact)
# save_as_h5('second_invariant_Q_spectral_exact','Q', Q_exact)
# save_as_h5('third_invariant_R_spectral_exact','R', R_exact)

# # double comment
# grad_dux_dx_comp = load_h5("grad_dux_dx_comp.h5")
# grad_dux_dy_comp = load_h5("grad_dux_dy_comp.h5")
# grad_dux_dz_comp = load_h5("grad_dux_dz_comp.h5")
# grad_duy_dx_comp = load_h5("grad_duy_dx_comp.h5")
# grad_duy_dy_comp = load_h5("grad_duy_dy_comp.h5")
# grad_duy_dz_comp = load_h5("grad_duy_dz_comp.h5")
# grad_duz_dx_comp = load_h5("grad_duz_dx_comp.h5")
# grad_duz_dy_comp = load_h5("grad_duz_dy_comp.h5")
# grad_duz_dz_comp = load_h5("grad_duz_dz_comp.h5")

# grad_comp = np.array([[grad_dux_dx_comp, grad_dux_dy_comp, grad_dux_dz_comp],
#                       [grad_duy_dx_comp, grad_duy_dy_comp, grad_duy_dz_comp],
#                       [grad_duz_dx_comp, grad_duz_dy_comp, grad_duz_dz_comp]])

# grad_comp = np.transpose(grad_comp, (2, 3, 4, 0, 1))  

# div_inexact_comp, Q_inexact_comp, R_inexact_comp = calculate_invariants(grad_comp)
# div_exact_comp, Q_exact_comp, R_exact_comp = calculate_invariants_exact(grad_comp)
# Qw_comp = compute_Qw(grad_comp)


# print("\nnorm(div_inexact_comp-div_exact_comp) = ", np.linalg.norm(div_inexact_comp-div_exact_comp))
# print("norm(Q_inexact_comp-Q_exact_comp) = ", np.linalg.norm(Q_inexact_comp-Q_exact_comp))
# print("norm(R_inexact_comp-R_exact_comp) = ", np.linalg.norm(R_inexact_comp-R_exact_comp))

# save_as_h5('first_invariant_div_spectral_inexact_comp','div_comp', div_inexact_comp)
# save_as_h5('second_invariant_Q_spectral_inexact_comp','Q_comp', Q_inexact_comp)
# save_as_h5('third_invariant_R_spectral_inexact_comp','R_comp', R_inexact_comp)

# save_as_h5('first_invariant_div_spectral_exact_comp','div_comp', div_exact_comp)
# save_as_h5('second_invariant_Q_spectral_exact_comp','Q_comp', Q_exact_comp)
# save_as_h5('third_invariant_R_spectral_exact_comp','R_comp', R_exact_comp)

# print("Qw: ", Qw, "\tQw_comp: ", Qw_comp)

x_max = 200
x_min = -x_max
pdf_max = 1
pdf_min = 1e-8

# x_max = None
# x_min = None
# pdf_max = None
# pdf_min = None


# # plot_pdf({'DNS':div_exact}, title='Divergence (First Invariant)',
# #          file_name='div_DNS_1024Cubed_spectral', x_min=x_min, 
# #          x_max=x_max, pdf_min=pdf_min, pdf_max=pdf_max)

plot_pdf({f'Comp cutoff {cutoff}':div_exact_comp}, 
         title=r'$\nabla \cdot \mathbf{U}$',file_name='div_1024Cubed_spectral',
         x_min=x_min, x_max=x_max, pdf_min=pdf_min, pdf_max=pdf_max, my_color='blue')

# plot_pdf({f'Comp cutoff {cutoff}':div_exact_comp}, my_color='blue', 
#          title='Divergence',file_name='div_1024Cubed_spectral_SameLimits',
#          x_min=x_min, x_max=x_max, ticks_min=-100, ticks_max=100,
#          pdf_min=pdf_min, pdf_max=pdf_max)

# # plot_QR_joint_pdf(Q_exact, R_exact, Qw,
# #                   label_1 = 'DNS (No Assumption)',
# #                   file_name=f'PDF_QR_Joint_DNS_exact')

# # plot_QR_joint_pdf(Q_inexact, R_inexact, Qw, 
# #                   label_1 = rf'DNS (Assume $\nabla \cdot U = 0$)',
# #                   file_name=f'PDF_QR_Joint_DNS_inexact')

# plot_QR_joint_pdf(Q_exact_comp, R_exact_comp, Qw_comp,
#                   label_1 = rf'Comp cutoff {cutoff} (No Assumption)',
#                   file_name=f'PDF_QR_Joint_cutoff_{cutoff}_exact')

# plot_QR_joint_pdf(Q_inexact_comp, R_inexact_comp, Qw_comp, 
#                   label_1 = rf'Comp cutoff {cutoff} (Assume $\nabla \cdot U = 0$)',
#                   file_name=f'PDF_QR_Joint_cutoff_{cutoff}_inexact')


# # plot_QR_joint_pdf(Q_exact, R_exact, Qw, Q_inexact, R_inexact, Qw,
# #                   label_1=r'DNS (No Assumption)', colormesh=False,
# #                   label_2=r'DNS (Assume $\nabla \cdot {U} = 0$)',
# #                   file_name=f'PDF_QR_Joint_DNS_contourlines')

# plot_QR_joint_pdf(Q_exact_comp, R_exact_comp, Qw_comp, Q_inexact_comp, R_inexact_comp, Qw_comp,
#                   label_1=rf'Comp cutoff {cutoff} (No Assumption)', colormesh=False,
#                   label_2=rf'Comp cutoff {cutoff} (Assume $\nabla \cdot U = 0$)',
#                   file_name=f'PDF_QR_Joint_cutoff_{cutoff}_contourlines')

plot_QR_joint_pdf(Q_exact, R_exact, Qw, Q_exact_comp, R_exact_comp, Qw_comp,
                  label_1=r'DNS', colormesh=False,
                  label_2=rf'Comp cutoff {cutoff}',
                  file_name=f'PDF_QR_Joint_cutoff_{cutoff}_exact_contourlines')

# # plot_QR_joint_pdf(Q_inexact, R_inexact, Qw, Q_inexact_comp, R_inexact_comp, Qw_comp,
# #                   label_1=r'DNS (Assume $\nabla \cdot U = 0$)', colormesh=False,
# #                   label_2=rf'Comp cutoff {cutoff} (Assume $\nabla \cdot U = 0$)',
# #                   file_name=f'PDF_QR_Joint_cutoff_{cutoff}_inexact_contourlines')
