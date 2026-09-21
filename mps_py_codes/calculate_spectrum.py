import numpy as np

def calculate_spectrum(u,v,w):
    
    L=2*np.pi

    # Perform Fourier Transform on the velocity components
    dim = u.shape[0]
    uu_fft = np.fft.fftn(u)
    vv_fft = np.fft.fftn(v)
    ww_fft = np.fft.fftn(w)

    uu_fft = (np.abs(uu_fft) / dim**3)**2
    vv_fft = (np.abs(vv_fft) / dim**3)**2
    ww_fft = (np.abs(ww_fft) / dim**3)**2

    k_end = int(dim / 2)
    rx = np.array(range(dim)) - dim / 2 + 1
    rx = np.roll(rx, int(dim / 2) + 1)

    r = np.zeros((rx.shape[0], rx.shape[0], rx.shape[0]))
    for i in range(rx.shape[0]):
        for j in range(rx.shape[0]):
            r[i, j, :] = rx[i]**2 + rx[j]**2 + rx[:]**2
    r = np.sqrt(r)

    dx = 2 * np.pi / L
    k = (np.array(range(k_end)) + 1) * dx

    bins = np.zeros((k.shape[0] + 1))
    # print("bins:", bins.shape)
    for N in range(k_end):
        if N == 0:
            bins[N] = 0
        else:
            bins[N] = (k[N] + k[N - 1]) / 2
    bins[-1] = k[-1]

    inds = np.digitize(r * dx, bins, right=True)
    spectrum = np.zeros((k.shape[0]))
    bin_counter = np.zeros((k.shape[0]))

    for N in range(k_end):
        spectrum[N] = np.sum(uu_fft[inds == N + 1]) + np.sum(vv_fft[inds == N + 1]) + np.sum(ww_fft[inds == N + 1])
        bin_counter[N] = np.count_nonzero(inds == N + 1)

    # C = 1/(2*u.shape[0]**6)
    # spectrum = spectrum * C
    spectrum = spectrum * 2 * np.pi * (k**2) / (bin_counter * dx**3)

    return k, spectrum




def calculate_1d_spectrum(u):
    # Number of grid points
    dim = u.shape[0]
    L = 2 * np.pi  # Physical size of the domain

    # Perform Fourier Transform on the velocity component
    u_fft = np.fft.fft(u)
    u_fft = (np.abs(u_fft) / dim)**2  # Normalize and square to get energy

    # Create wave numbers
    k_end = dim // 2
    k = np.fft.fftfreq(dim, d=L / dim)
    k = np.fft.fftshift(k)  # Shift zero frequency to center
    k = k[k >= 0]  # Take only positive wave numbers

    # Compute the spectrum
    spectrum = u_fft[:k_end]  # Take first half (positive frequencies)

    return k, spectrum



def calculate_2d_spectrum(u, v):
    L = 2 * np.pi  # Physical domain size

    # Grid dimensions
    dim = u.shape[0]

    # Perform Fourier Transform on the velocity components
    uu_fft = np.fft.fft2(u)
    vv_fft = np.fft.fft2(v)

    # Normalize and compute energy
    uu_fft = (np.abs(uu_fft) / dim**2)**2
    vv_fft = (np.abs(vv_fft) / dim**2)**2

    # Create radial wave number array
    rx = np.fft.fftfreq(dim) * dim
    kx, ky = np.meshgrid(rx, rx, indexing='ij')
    r = np.sqrt(kx**2 + ky**2)

    # Define wave number bins
    k_end = dim // 2
    dx = 2 * np.pi / L
    k = (np.arange(k_end) + 1) * dx
    bins = np.zeros(k.shape[0] + 1)
    for N in range(k_end):
        bins[N] = 0 if N == 0 else (k[N] + k[N - 1]) / 2
    bins[-1] = k[-1]

    # Digitize radial distances into bins
    inds = np.digitize(r * dx, bins, right=True)

    # Initialize spectrum
    spectrum = np.zeros(k.shape[0])
    bin_counter = np.zeros(k.shape[0])

    # Sum energy in each radial bin
    for N in range(k_end):
        spectrum[N] = np.sum(uu_fft[inds == N + 1]) + np.sum(vv_fft[inds == N + 1])
        bin_counter[N] = np.count_nonzero(inds == N + 1)

    # Normalize spectrum
    spectrum = spectrum * 2 * np.pi * k / (bin_counter * dx**2)

    return k, spectrum



