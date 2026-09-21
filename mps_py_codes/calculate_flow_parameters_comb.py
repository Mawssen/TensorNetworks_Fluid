
import numpy as np
from calculate_tkEnergy import calculate_tkEnergy
from calculate_gradients import calculate_gradients
from calculate_gradients import calculate_gradients_fidelities
from calculate_eps_stilde import calculate_eps
from calculate_u_comp_fidelity_l2norm import calculate_fidelity
# from scipy.integrate import simps
from calculate_spectrum import calculate_spectrum
from save_as_mat import save_as_mat, save_as_mat_hdf5, save_as_h5

def rel_error(s1,s2):
    return 100*np.abs(s1-s2)/np.abs(s1)

def calculate_flow_parameters_comb(velocities, velocities_comp, h, nu=0.000185, order=4, spectral = True): 
    """
    Calculate various flow parameters from the given velocity field, 
    and the same parameters but from the compressed velocity field.

    Parameters
    ----------
    velocities : tuple of 3D arrays
        The three components of the velocity field.
    velocities_comp : tuple of 3D arrays
        The three components of the compressed velocity field.
    nu : float
        The kinematic viscosity.
    h : float
        The grid spacing.

    Returns
    -------
    dict
        A dictionary containing the following flow parameters:
        - E: total kinetic energy
        - eps: mean dissipation rate
        - u_prime: r.m.s. velocity fluctuation
        - Lambda: Taylor micro length scale
        - Rey_Taylor: Taylor micro-scale Reynolds number
        - tau: Kolmogorov time scale
        - eta: Kolmogorov length scale
        - fidelity_dux: fidelity for dux
        - fidelity_duy: fidelity for duy
        - fidelity_duz: fidelity for duz
    """

    u, v, w = velocities
    u_comp, v_comp, w_comp = velocities_comp

    # Calculate velocity gradients for both the original and compressed flow
    gradients = calculate_gradients(velocities, h, order = order, spectral = spectral)
    gradients_comp = calculate_gradients(velocities_comp, h, order = order ,spectral = spectral)
    
    # save_as_h5('grad_dux_dx', 'dux_dx', gradients['dux_dx'])
    # save_as_h5('grad_dux_dy', 'dux_dy', gradients['dux_dy'])
    # save_as_h5('grad_dux_dz', 'dux_dz', gradients['dux_dz'])
    # save_as_h5('grad_duy_dx', 'duy_dx', gradients['duy_dx'])
    # save_as_h5('grad_duy_dy', 'duy_dy', gradients['duy_dy'])
    # save_as_h5('grad_duy_dz', 'duy_dz', gradients['duy_dz'])
    # save_as_h5('grad_duz_dx', 'duz_dx', gradients['duz_dx'])
    # save_as_h5('grad_duz_dy', 'duz_dy', gradients['duz_dy'])
    # save_as_h5('grad_duz_dz', 'duz_dz', gradients['duz_dz'])

    save_as_h5('grad_dux_dx_comp', 'dux_dx', gradients_comp['dux_dx'])
    save_as_h5('grad_dux_dy_comp', 'dux_dy', gradients_comp['dux_dy'])
    save_as_h5('grad_dux_dz_comp', 'dux_dz', gradients_comp['dux_dz'])
    save_as_h5('grad_duy_dx_comp', 'duy_dx', gradients_comp['duy_dx'])
    save_as_h5('grad_duy_dy_comp', 'duy_dy', gradients_comp['duy_dy'])
    save_as_h5('grad_duy_dz_comp', 'duy_dz', gradients_comp['duy_dz'])
    save_as_h5('grad_duz_dx_comp', 'duz_dx', gradients_comp['duz_dx'])
    save_as_h5('grad_duz_dy_comp', 'duz_dy', gradients_comp['duz_dy'])
    save_as_h5('grad_duz_dz_comp', 'duz_dz', gradients_comp['duz_dz'])
    print(" ")

    # Calculate the fidelities for dux, duy, duz in all directions
    fidelity_dux, fidelity_duy, fidelity_duz = calculate_gradients_fidelities(gradients, gradients_comp)
    
    # Calculate total kinetic energy (E)
    nx, ny, nz = u.shape
    ntime = 1                                   # Assuming ntime=1, modify if different
    E = 0.691891015576730 #FD and Spectral
    # E = calculate_tkEnergy(velocities, ntime)
    E_comp = calculate_tkEnergy(velocities_comp, ntime)

    # Calculate the dissipation rates for original and compressed flow
    dissipation_rate = calculate_eps(gradients, nu)
    dissipation_rate_comp = calculate_eps(gradients_comp, nu)

    # save_as_h5('dissipation', 'eps', dissipation_rate)
    save_as_h5('dissipation_comp', 'eps_comp', dissipation_rate_comp)

    # Mean dissipation rate
    # eps = 0.088651667902893 #FD
    eps = 0.091859602123993 #Spectral
    # eps = np.sum(dissipation_rate) / (nx * ny * nz) / ntime
    eps_comp = np.sum(dissipation_rate_comp) / (nx * ny * nz) / ntime
    
    # r.m.s. velocity fluctuation
    u_prime = np.sqrt(2 / 3 * E)
    u_prime_comp = np.sqrt(2 / 3 * E_comp)

    # Taylor micro length scale
    Lambda = np.sqrt(15 * nu * u_prime**2 / eps)
    Lambda_comp = np.sqrt(15 * nu * u_prime**2 / eps_comp)
    # Lambda = np.sqrt(15 * nu * u_prime**2 / eps_truth)

    # Taylor micro-scale Reynolds number
    Rey_Taylor = u_prime * Lambda / nu
    Rey_Taylor_comp = u_prime * Lambda_comp / nu

    # Kolmogorov time scale and length scale
    tau = np.sqrt(nu / eps)
    tau_comp = np.sqrt(nu / eps_comp)

    # Kolmogorov time scale and length scale
    eta = nu**(0.75) * eps**(-0.25)
    eta_comp = nu**(0.75) * eps_comp**(-0.25)

    # k, spec = calculate_spectrum(u, v, w)
    k,spec_comp = calculate_spectrum(u_comp,v_comp,w_comp)
    
    spectrum = {
        "k": k,
        # "spec": spec,
        "spec_comp": spec_comp
    }
    # # Numerically integrate E(k)/k with respect to k using the trapezoidal rule
    # integral_value = simps(spec / k, k)
    # integral_value_comp = simps(spec_comp / k, k)
    

    # integral_value = np.trapz(spec / k, k)
    integral_value_comp = np.trapz(spec_comp / k, k)
    
    # Integral scale
    L = 0.680765445960042 #FD and Spectral
    # L = (np.pi/(2*u_prime**2))*integral_value
    L_comp = (np.pi/(2*u_prime_comp**2))*integral_value_comp

    # Large eddy turnover time
    T_L = L / u_prime
    T_L_comp = L_comp / u_prime_comp

    compare = {
        "E":[E,E_comp,rel_error(E,E_comp)],
        "mean_eps": [eps, eps_comp, rel_error(eps,eps_comp)],
        "u_prime": [u_prime,u_prime_comp, rel_error(u_prime,u_prime_comp)],
        "Lambda": [Lambda,Lambda_comp,rel_error(Lambda,Lambda_comp)],
        "Rey_Taylor": [Rey_Taylor,Rey_Taylor_comp,rel_error(Rey_Taylor,Rey_Taylor_comp)],
        "tau": [tau, tau_comp, rel_error(tau,tau_comp)],
        "eta": [eta, eta_comp, rel_error(eta,eta_comp)],
        "L": [L, L_comp, rel_error(L,L_comp)],
        "T_L": [T_L, T_L_comp, rel_error(T_L,T_L_comp)],
        "fidelity_eps": [calculate_fidelity(dissipation_rate, dissipation_rate_comp)],
        "fidelity_dux": fidelity_dux,
        "fidelity_duy": fidelity_duy,
        "fidelity_duz": fidelity_duz
    }
    
    return compare, spectrum
    # return compare



def calculate_flow_parameters_comb_dani(velocities, velocities_comp, h, nu=0.0005, order=4, spectral = True): 
    """
    Calculate various flow parameters from the given velocity field, 
    and the same parameters but from the compressed velocity field.

    Parameters
    ----------
    velocities : tuple of 3D arrays
        The three components of the velocity field.
    velocities_comp : tuple of 3D arrays
        The three components of the compressed velocity field.
    nu : float
        The kinematic viscosity.
    h : float
        The grid spacing.

    Returns
    -------
    dict
        A dictionary containing the following flow parameters:
        - E: total kinetic energy
        - eps: mean dissipation rate
        - u_prime: r.m.s. velocity fluctuation
        - Lambda: Taylor micro length scale
        - Rey_Taylor: Taylor micro-scale Reynolds number
        - tau: Kolmogorov time scale
        - eta: Kolmogorov length scale
        - fidelity_dux: fidelity for dux
        - fidelity_duy: fidelity for duy
        - fidelity_duz: fidelity for duz
    """

    u, v, w = velocities
    u_comp, v_comp, w_comp = velocities_comp

    # Calculate velocity gradients for both the original and compressed flow
    gradients = calculate_gradients(velocities, h, order = order, spectral = spectral)
    gradients_comp = calculate_gradients(velocities_comp, h, order = order ,spectral = spectral)
    
    # # Save gradients
    # save_as_h5('grad_dux_dx', 'dux_dx', gradients['dux_dx'])
    # save_as_h5('grad_dux_dy', 'dux_dy', gradients['dux_dy'])
    # save_as_h5('grad_dux_dz', 'dux_dz', gradients['dux_dz'])
    # save_as_h5('grad_duy_dx', 'duy_dx', gradients['duy_dx'])
    # save_as_h5('grad_duy_dy', 'duy_dy', gradients['duy_dy'])
    # save_as_h5('grad_duy_dz', 'duy_dz', gradients['duy_dz'])
    # save_as_h5('grad_duz_dx', 'duz_dx', gradients['duz_dx'])
    # save_as_h5('grad_duz_dy', 'duz_dy', gradients['duz_dy'])
    # save_as_h5('grad_duz_dz', 'duz_dz', gradients['duz_dz'])
    # print(" ")

    save_as_h5('grad_dux_dx_comp', 'dux_dx', gradients_comp['dux_dx'])
    save_as_h5('grad_dux_dy_comp', 'dux_dy', gradients_comp['dux_dy'])
    save_as_h5('grad_dux_dz_comp', 'dux_dz', gradients_comp['dux_dz'])
    save_as_h5('grad_duy_dx_comp', 'duy_dx', gradients_comp['duy_dx'])
    save_as_h5('grad_duy_dy_comp', 'duy_dy', gradients_comp['duy_dy'])
    save_as_h5('grad_duy_dz_comp', 'duy_dz', gradients_comp['duy_dz'])
    save_as_h5('grad_duz_dx_comp', 'duz_dx', gradients_comp['duz_dx'])
    save_as_h5('grad_duz_dy_comp', 'duz_dy', gradients_comp['duz_dy'])
    save_as_h5('grad_duz_dz_comp', 'duz_dz', gradients_comp['duz_dz'])
    print(" ")


    # Calculate the fidelities for dux, duy, duz in all directions
    fidelity_dux, fidelity_duy, fidelity_duz = calculate_gradients_fidelities(gradients, gradients_comp)
    
    # Calculate total kinetic energy (E)
    nx, ny, nz = u.shape
    ntime = 1                                   # Assuming ntime=1, modify if different
    E = calculate_tkEnergy(velocities, ntime)
    E_comp = calculate_tkEnergy(velocities_comp, ntime)

    # Calculate the dissipation rates for original and compressed flow
    dissipation_rate = calculate_eps(gradients, nu)
    dissipation_rate_comp = calculate_eps(gradients_comp, nu)

    # save_as_h5('dissipation', 'eps', dissipation_rate)
    save_as_h5('dissipation_comp', 'eps_comp', dissipation_rate_comp)

    # Mean dissipation rate
    eps = np.sum(dissipation_rate) / (nx * ny * nz) / ntime
    eps_comp = np.sum(dissipation_rate_comp) / (nx * ny * nz) / ntime
    
    # r.m.s. velocity fluctuation
    u_prime = np.sqrt(2 / 3 * E)
    u_prime_comp = np.sqrt(2 / 3 * E_comp)

    # Taylor micro length scale
    Lambda = np.sqrt(15 * nu * u_prime**2 / eps)
    Lambda_comp = np.sqrt(15 * nu * u_prime**2 / eps_comp)
    # Lambda = np.sqrt(15 * nu * u_prime**2 / eps_truth)

    # Taylor micro-scale Reynolds number
    Rey_Taylor = u_prime * Lambda / nu
    Rey_Taylor_comp = u_prime * Lambda_comp / nu

    # Kolmogorov time scale and length scale
    tau = np.sqrt(nu / eps)
    tau_comp = np.sqrt(nu / eps_comp)

    # Kolmogorov time scale and length scale
    eta = nu**(0.75) * eps**(-0.25)
    eta_comp = nu**(0.75) * eps_comp**(-0.25)

    # k, spec = calculate_spectrum(u, v, w)
    k,spec_comp = calculate_spectrum(u_comp,v_comp,w_comp)
    
    spectrum = {
        "k": k,
    #     "spec": spec,
        "spec_comp": spec_comp
    }
    
    # # Numerically integrate E(k)/k with respect to k using the trapezoidal rule
    # integral_value = simps(spec / k, k)
    # integral_value_comp = simps(spec_comp / k, k)

    # integral_value = np.trapz(spec / k, k)
    integral_value_comp = np.trapz(spec_comp / k, k)

    # Integral scale
    if nx == 1024: L = 0.534727275943382   #Spectral - Dani 1024
    elif nx == 32: L = 0.8282468665396511  #Spectral - Dani 512

    # L = (np.pi/(2*u_prime**2))*integral_value
    L_comp = (np.pi/(2*u_prime_comp**2))*integral_value_comp

    # Large eddy turnover time
    T_L = L / u_prime
    T_L_comp = L_comp / u_prime_comp

    compare = {
        "E":[E,E_comp,rel_error(E,E_comp)],
        "mean_eps": [eps, eps_comp, rel_error(eps,eps_comp)],
        "u_prime": [u_prime,u_prime_comp, rel_error(u_prime,u_prime_comp)],
        "Lambda": [Lambda,Lambda_comp,rel_error(Lambda,Lambda_comp)],
        "Rey_Taylor": [Rey_Taylor,Rey_Taylor_comp,rel_error(Rey_Taylor,Rey_Taylor_comp)],
        "tau": [tau, tau_comp, rel_error(tau,tau_comp)],
        "eta": [eta, eta_comp, rel_error(eta,eta_comp)],
        "L": [L, L_comp, rel_error(L,L_comp)],
        "T_L": [T_L, T_L_comp, rel_error(T_L,T_L_comp)],
        "fidelity_eps": [calculate_fidelity(dissipation_rate, dissipation_rate_comp)],
        "fidelity_dux": fidelity_dux,
        "fidelity_duy": fidelity_duy,
        "fidelity_duz": fidelity_duz
    }
    
    return compare, spectrum
    # return compare


def calculate_flow_parameters(velocities, nu, h, finite_diff_scheme='4th_order_periodic'):
    """
    Calculate various flow parameters from the given velocity field.
    
    Parameters
    ----------
    velocities : tuple of 3D arrays
        The three components of the velocity field (u, v, w).
    nu : float
        The kinematic viscosity.
    h : float
        The grid spacing.
    finite_diff_scheme : str, optional
        The finite difference scheme to use ('4th_order' or 'central'), default is '4th_order'.
        
    Returns
    -------
    compare : dict
        A dictionary containing the following flow parameters:
        - E: total kinetic energy
        - eps: mean dissipation rate
        - u_prime: r.m.s. velocity fluctuation
        - Lambda: Taylor micro length scale
        - Rey_Taylor: Taylor micro-scale Reynolds number
        - tau: Kolmogorov time scale
        - eta: Kolmogorov length scale
        - L: Integral scale
        - T_L: Large eddy turnover time
    spectrum : dict
        A dictionary containing the wave number (k) and spectrum (spec).
    """

    u, v, w = velocities

    # Calculate velocity gradients
    gradients = calculate_gradients(velocities, h, finite_diff_scheme) # type: ignore

    # Calculate total kinetic energy (E)
    nx, ny, nz = u.shape
    ntime = 1  # Assuming single time snapshot for now
    E = calculate_tkEnergy(velocities, ntime)

    # Calculate dissipation rate (eps)
    dissipation_rate = calculate_eps(gradients, nu)
    # eps = np.sum(dissipation_rate) / (nx * ny * nz) / ntime
    if finite_diff_scheme == '4th_order_periodic' or finite_diff_scheme == 'central':
        eps = np.sum(dissipation_rate) / (nx * ny * nz) / ntime
   
    elif finite_diff_scheme == '4th_order_zero':
        nx_i = nx - 4 # x interior points without boundary points
        ny_i = ny - 4 # y interior points without boundary points
        nz_i = nz - 4 # z interior points without boundary points

        eps = np.sum(dissipation_rate) / (nx_i * ny_i * nz_i) / ntime

    # r.m.s. velocity fluctuation (u')
    u_prime = np.sqrt(2 / 3 * E)

    # Taylor microscale (Lambda)
    Lambda = np.sqrt(15 * nu * u_prime**2 / eps)

    # Taylor microscale Reynolds number (Re_Taylor)
    Rey_Taylor = u_prime * Lambda / nu

    # Kolmogorov time scale (tau) and length scale (eta)
    tau = np.sqrt(nu / eps)
    eta = nu**(0.75) * eps**(-0.25)

    # # Spectrum and wave number (k)
    # k, spec = calculate_spectrum(u, v, w)

    # # Numerically integrate E(k)/k to get the integral scale
    # integral_value = simps(spec / k, k)
    # integral_value = np.trapezoid(spec / k, k)
    # L = (np.pi / (2 * u_prime**2)) * integral_value
    
    # # Large eddy turnover time (T_L)
    # T_L = L / u_prime

    # Return parameters
    parameters = {
        "E": E,
        "eps": eps,
        "u_prime": u_prime,
        "Lambda": Lambda,
        "Rey_Taylor": Rey_Taylor,
        "tau": tau,
        "eta": eta,
        # "L": L,
        # "T_L": T_L
    }

    # spectrum = {
    #     "k": k,
    #     "spec": spec
    # }

    return parameters#, spectrum

