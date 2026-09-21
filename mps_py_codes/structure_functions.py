import numpy as np
import os
import numba as nb
import time

# # Same function but with for loops instead of vectorized operations
# def struct_func_loop(vel_fields, direction):
#     """Compute vlcorr, vstruct2, vstruct3 using explicit for-loops."""
#     ux, uy, uz = vel_fields
#     NXT, NYT, NZT = ux.shape
#     max_lag = ux.shape[{"x": 0, "y": 1, "z": 2}[direction]]

#     vlcorr = np.zeros(max_lag)
#     vstruct2 = np.zeros(max_lag)
#     vstruct3 = np.zeros(max_lag)

#     for l in range(max_lag):
#         for i in range(NXT):
#             for j in range(NYT):
#                 for k in range(NZT):
#                     u1, u2, u3 = ux[i,j,k], uy[i,j,k], uz[i,j,k]
#                     if direction == "x":
#                         q = (i + l) % NXT
#                         uq1, uq2, uq3 = ux[q,j,k], uy[q,j,k], uz[q,j,k]
#                     elif direction == "y":
#                         q = (j + l) % NYT
#                         uq1, uq2, uq3 = ux[i,q,k], uy[i,q,k], uz[i,q,k]
#                     elif direction == "z":
#                         q = (k + l) % NZT
#                         uq1, uq2, uq3 = ux[i,j,q], uy[i,j,q], uz[i,j,q]
#                     else:
#                         raise ValueError("Invalid direction")

#                     # Longitudinal (first component)
#                     du = u1 - uq1
#                     vlcorr[l] += u1 * uq1
#                     vstruct2[l] += du**2
#                     vstruct3[l] += du**3

#                     # Transverse components
#                     for u, uq in [(u2, uq2), (u3, uq3)]:
#                         du = u - uq
#                         vlcorr[l] += u * uq
#                         vstruct2[l] += du**2
#                         vstruct3[l] += du**3

#     norm = NXT * NYT * NZT
#     vlcorr /= norm
#     vstruct2 /= norm
#     vstruct3 /= norm

#     u2_mean = (np.mean(ux**2) + np.mean(uy**2) + np.mean(uz**2)) / 3
#     vlcorr /= u2_mean

#     return vlcorr, vstruct2, vstruct3


# def struct_func(vel_fields, direction):
#     """
#     Compute the two-point velocity correlation and structure functions 
#     (second- and third-order) in a specified direction using vectorized NumPy operations.

#     Parameters
#     ----------
#     vel_fields : tuple of ndarray
#         A tuple containing three 3D numpy arrays (ux, uy, uz) representing 
#         the components of a velocity field on a uniform grid. Each array should 
#         have the same shape (NXT, NYT, NZT).

#     direction : str
#         The spatial direction in which to compute the structure functions.
#         Must be one of {"x", "y", "z"} corresponding to axes 0, 1, or 2 respectively.

#     Returns
#     -------
#     vlcorr : ndarray
#         One-dimensional array of two-point correlation values as a function of spatial lag r.
#         Normalized by the average kinetic energy of the field.

#     vstruct2 : ndarray
#         One-dimensional array of second-order structure function values, i.e.,
#         ⟨ [u(x) - u(x + r)]² ⟩ for all three velocity components.

#     vstruct3 : ndarray
#         One-dimensional array of third-order structure function values, i.e.,
#         ⟨ [u(x) - u(x + r)]³ ⟩ for all three velocity components.

#     Notes
#     -----
#     - The function uses periodic boundary conditions via np.roll to shift fields by lag r.
#     - The output is averaged over all spatial positions and over all three components of velocity.
#     - Useful for analyzing turbulent flow statistics in an isotropic domain.

#     Example
#     -------
#     >>> ux = np.random.randn(32, 32, 32)
#     >>> uy = np.random.randn(32, 32, 32)
#     >>> uz = np.random.randn(32, 32, 32)
#     >>> vlcorr, vstruct2, vstruct3 = sf_vectorized((ux, uy, uz), "x")
#     """

#     ux, uy, uz = vel_fields
#     NXT, NYT, NZT = ux.shape
#     max_lag = ux.shape[{"x": 0, "y": 1, "z": 2}[direction]]

#     vlcorr = np.zeros(max_lag)
#     vstruct2 = np.zeros(max_lag)
#     vstruct3 = np.zeros(max_lag)

#     for l in range(max_lag):
#         shift_axis = {"x": 0, "y": 1, "z": 2}[direction]

#         shifted_x = np.roll(ux, -l, axis=shift_axis)
#         shifted_y = np.roll(uy, -l, axis=shift_axis)
#         shifted_z = np.roll(uz, -l, axis=shift_axis)

#         # Longitudinal and Transverse
#         for u, uq in [(ux, shifted_x), (uy, shifted_y), (uz, shifted_z)]:
#             du = u - uq
#             vlcorr[l] += np.sum(u * uq)
#             vstruct2[l] += np.sum(du**2)
#             vstruct3[l] += np.sum(du**3)

#     norm = NXT * NYT * NZT
#     vlcorr /= norm
#     vstruct2 /= norm
#     vstruct3 /= norm

#     u2_mean = (np.mean(ux**2) + np.mean(uy**2) + np.mean(uz**2)) / 3
#     vlcorr /= 3*u2_mean

#     return vlcorr, vstruct2, vstruct3


# def struct_func64(vel_fields, direction):
#     """
#     Compute longitudinal, transverse, and mixed (LTT) structure functions and 
#     two-point longitudinal correlation in the specified direction using vectorized operations.

#     Parameters
#     ----------
#     vel_fields : tuple of np.ndarray
#         Tuple of (ux, uy, uz), each of shape (NXT, NYT, NZT).
#     direction : str
#         One of 'x', 'y', or 'z'. Determines direction of separation.

#     Returns
#     -------
#     dict : dict of arrays
#         Keys include:
#             - "r": separation distances
#             - "corr2pt_L": normalized two-point correlation of u_L
#             - "LL": second-order longitudinal structure function
#             - "LLL": third-order longitudinal structure function
#             - "TT": second-order transverse structure function
#             - "TTT": third-order transverse structure function
#             - "LTT": mixed third-order structure function
#     """

#     ux, uy, uz = vel_fields
#     shape = ux.shape
#     axis = {"x": 0, "y": 1, "z": 2}[direction]
#     max_lag = shape[axis]

#     # Initialize arrays
#     corr2pt_L = np.zeros(max_lag)
#     S_LL = np.zeros(max_lag)
#     S_LLL = np.zeros(max_lag)
#     S_TT = np.zeros(max_lag)
#     S_TTT = np.zeros(max_lag)
#     S_LTT = np.zeros(max_lag)

#     # Select longitudinal and transverse components
#     comp = {"x": ux, "y": uy, "z": uz}
#     u_L = comp[direction]
#     u_T = [v for k, v in comp.items() if k != direction]

#     for l in range(max_lag):
#         u_L_shift = np.roll(u_L, -l, axis=axis)
#         du_L = u_L - u_L_shift

#         corr2pt_L[l] = np.sum(u_L * u_L_shift)
#         S_LL[l] = np.sum(du_L**2)
#         S_LLL[l] = np.sum(du_L**3)

#         # Transverse terms
#         du_T_squared_sum = 0.0
#         for u in u_T:
#             u_shift = np.roll(u, -l, axis=axis)
#             du_T = u - u_shift
#             S_TT[l] += np.sum(du_T**2)
#             S_TTT[l] += np.sum(du_T**3)
#             du_T_squared_sum += du_T**2  # accumulate for mixed term

#         S_LTT[l] = np.sum(du_L * du_T_squared_sum)

#     # Normalize
#     N3 = np.prod(ux.shape)
#     corr2pt_L /= N3
#     S_LL  /= N3
#     S_LLL /= N3
#     S_TT  /= N3
#     S_TTT /= N3
#     S_LTT /= N3

#     u2_L_mean = np.mean(u_L**2)
#     corr2pt_L /= u2_L_mean
#     print(f"mean of u_L^2 in {direction}: {u2_L_mean}")

#     r = 2 * np.pi / max_lag * np.arange(max_lag)

#     return {
#         "r": r,
#         "corr2pt_L": corr2pt_L,
#         "LL": S_LL,
#         "LLL": S_LLL,
#         "TT": S_TT,
#         "TTT": S_TTT,
#         "LTT": S_LTT
#     }



# def struct_func32(vel_fields, direction):
#     """
#     Compute longitudinal, transverse, and mixed (LTT) structure functions and
#     two-point longitudinal correlation in the specified direction.

#     Everything is evaluated in float32 to cut memory-bandwidth in half and
#     speed up arithmetic on modern CPUs / GPUs.
#     """

#     # ------------------------------------------------------------------
#     # 1.  Make sure the raw fields themselves are float32 (no copy if
#     #     they already are).  This alone slashes the RAM footprint.
#     # ------------------------------------------------------------------
#     ux, uy, uz = (np.array(f, dtype=np.float32, order='C', copy=False) for f in vel_fields)


#     shape  = ux.shape
#     axis   = {"x": 0, "y": 1, "z": 2}[direction]
#     maxlag = shape[axis]

#     # ------------------------------------------------------------------
#     # 2.  Pre-allocate result arrays as float32
#     # ------------------------------------------------------------------
#     zeros  = lambda: np.zeros(maxlag, dtype=np.float32)
#     corr2pt_L, S_LL, S_LLL, S_TT, S_TTT, S_LTT = (zeros() for _ in range(6))

#     # longitudinal & transverse components
#     comp  = {"x": ux, "y": uy, "z": uz}
#     u_L   = comp[direction]                           # float32
#     u_T   = [v for k, v in comp.items() if k != direction]

#     # ------------------------------------------------------------------
#     # 3.  Lag loop (still pure NumPy, now all float32 math)
#     # ------------------------------------------------------------------
#     for l in range(maxlag):
#         u_L_shift = np.roll(u_L, -l, axis=axis)
#         du_L      = u_L - u_L_shift

#         # second- and third-order longitudinal
#         corr2pt_L[l] = np.sum(u_L * u_L_shift, dtype=np.float32)
#         S_LL[l]      = np.sum(du_L**2,          dtype=np.float32)
#         S_LLL[l]     = np.sum(du_L**3,          dtype=np.float32)

#         # transverse terms
#         du_T_sq_sum = np.float32(0.0)
#         for u in u_T:
#             u_shift  = np.roll(u, -l, axis=axis)
#             du_T     = u - u_shift
#             S_TT[l]  += np.sum(du_T**2, dtype=np.float32)
#             S_TTT[l] += np.sum(du_T**3, dtype=np.float32)
#             du_T_sq_sum += du_T**2      # mixed term

#         S_LTT[l] = np.sum(du_L * du_T_sq_sum, dtype=np.float32)

#     # ------------------------------------------------------------------
#     # 4.  Normalisation (still float32)
#     # ------------------------------------------------------------------
#     N3 = np.float32(np.prod(shape))
#     corr2pt_L /= N3
#     S_LL  /= N3
#     S_LLL /= N3
#     S_TT  /= N3
#     S_TTT /= N3
#     S_LTT /= N3

#     u2_L_mean  = np.mean(u_L**2, dtype=np.float32)
#     corr2pt_L /= u2_L_mean
#     print(f"mean of u_L^2 in {direction}: {u2_L_mean}")

#     # ------------------------------------------------------------------
#     # 5.  Separation distance array (also float32)
#     # ------------------------------------------------------------------
#     r = (np.arange(maxlag, dtype=np.float32) *
#          (np.float32(2*np.pi) / np.float32(maxlag)))

#     return {
#         "r":         r,
#         "corr2pt_L": corr2pt_L,
#         "LL":        S_LL,
#         "LLL":       S_LLL,
#         "TT":        S_TT,
#         "TTT":       S_TTT,
#         "LTT":       S_LTT,
#     }


## Same functions but run time is different in different directions (z much longer)
# @nb.njit(inline='always')
# def _roll_1d_front(a, shift):
#     """
#     Roll *a* along its first axis by -shift (positive shift → forward).
#     Works in nopython mode and allocates only one new array.
#     """
#     n = a.shape[0]
#     shift %= n
#     if shift == 0:
#         return a                     # nothing to do
#     return np.concatenate((a[shift:], a[:shift]), axis=0)

# # ----------------------------------------------------------------------
# # 1.  core – computes only the requested lags, dtype taken from inputs
# # ----------------------------------------------------------------------
# @nb.njit(parallel=True, fastmath=True)
# def _struct_core(uL, uT1, uT2, start, stop, step):
#     """
#     Arrays uL, uT1, uT2 are 2-D and share the same dtype (float32|64).
#     Returns six 1-D arrays of that same dtype.
#     """
#     dtype  = uL.dtype
#     n_lags = (stop - start + step - 1) // step

#     corr2pt_L = np.zeros(n_lags, dtype=dtype)
#     S_LL  = np.zeros(n_lags, dtype=dtype)
#     S_LLL = np.zeros(n_lags, dtype=dtype)
#     S_TT  = np.zeros(n_lags, dtype=dtype)
#     S_TTT = np.zeros(n_lags, dtype=dtype)
#     S_LTT = np.zeros(n_lags, dtype=dtype)

#     for i in nb.prange(n_lags):
#         l = start + i * step
#         uL_s  = _roll_1d_front(uL , l)
#         uT1_s = _roll_1d_front(uT1, l)
#         uT2_s = _roll_1d_front(uT2, l)

#         duL  = uL  - uL_s
#         duT1 = uT1 - uT1_s
#         duT2 = uT2 - uT2_s
#         duT2sum = duT1**2 + duT2**2

#         corr2pt_L[i] = (uL * uL_s).sum()
#         S_LL[i]      = (duL**2).sum()
#         S_LLL[i]     = (duL**3).sum()
#         S_TT[i]      = duT2sum.sum()
#         S_TTT[i]     = (duT1**3 + duT2**3).sum()
#         S_LTT[i]     = (duL * duT2sum).sum()

#     return corr2pt_L, S_LL, S_LLL, S_TT, S_TTT, S_LTT

# # ----------------------------------------------------------------------
# # 2.  public wrapper – now has **dtype** argument
# # ----------------------------------------------------------------------
# def struct_func(vel_fields, direction, *, lag_range=None,
#                 dtype=np.float32):
#     """
#     vel_fields : (ux, uy, uz) – any float/shape
#     direction  : 'x' | 'y' | 'z'
#     lag_range  : None or (start, stop, step)
#     dtype      : np.float32 (default) or np.float64
#     """
#     axis = {'x':0, 'y':1, 'z':2}[direction]
#     ux, uy, uz = (np.ascontiguousarray(f, dtype=dtype) for f in vel_fields)

#     def flatten(a):
#         a = np.moveaxis(a, axis, 0)      # (N, Ny, Nz)
#         return a.reshape(a.shape[0], -1) # (N, M)
    
#     # def flatten(a):
#     #     a = np.moveaxis(a, axis, 0)          # (N, Ny, Nz) view
#     #     a = np.ascontiguousarray(a)          # REAL copy once, always
#     #     return a.reshape(a.shape[0], -1)     # now guaranteed contiguous
    
#     uL  = flatten({'x':ux,'y':uy,'z':uz}[direction])
#     uT1,uT2 = (flatten(f) for k,f in {'x':ux,'y':uy,'z':uz}.items()
#                if k != direction)

#     N = uL.shape[0]
#     start, stop, step = (0, N, 1) if lag_range is None else lag_range

#     (cLL, SLL, SLLL,
#      STT, STTT, SLTT) = _struct_core(uL, uT1, uT2, start, stop, step)

#     norm = uL.size
#     cLL  /= norm;  SLL  /= norm;  SLLL /= norm
#     STT  /= norm;  STTT /= norm;  SLTT /= norm
#     cLL  /= (uL**2).mean()

#     lag_idx = np.arange(start, stop, step, dtype=dtype)
#     r = (2*np.pi/N) * lag_idx

#     return {"r":r, "corr2pt_L":cLL,
#             "LL":SLL, "LLL":SLLL,
#             "TT":STT, "TTT":STTT,
#             "LTT":SLTT}

# # ----------------------------------------------------------------------
# # 3.  checkpoint driver – same API, extra **dtype**
# # ----------------------------------------------------------------------
# def safe_struct(vel_fields, direction, *, lag_step=1,
#                 chkpt_freq=100, outdir="chkpt",
#                 dtype=np.float32):
#     """
#     Runs struct_func in blocks, saving to <outdir>/lag_<dir>_<idx>.npz.
#     dtype controls whether the whole run uses float32 or float64.
#     """
#     os.makedirs(outdir, exist_ok=True)
#     axis = {"x":0,"y":1,"z":2}[direction]
#     Nlag = vel_fields[0].shape[axis]

#     # -------- look for latest checkpoint ------------------------------
#     done_to = -1
#     for f in os.listdir(outdir):
#         if f.startswith(f"lag_{direction}_") and f.endswith(".npz"):
#             try: done_to = max(done_to, int(f.split('_')[-1][:-4]))
#             except ValueError: pass

#     if done_to >= 0:
#         print(f"• Resuming from lag {done_to+1}")
#         acc = dict(np.load(f"{outdir}/lag_{direction}_{done_to}.npz",
#                            allow_pickle=True))
#         for k,a in acc.items():
#             if a.size < Nlag:             # pad to full length
#                 pad = np.zeros(Nlag, dtype=a.dtype)
#                 pad[:a.size] = a
#                 acc[k] = pad
#     else:
#         print("• Starting fresh run")
#         acc = {k: np.zeros(Nlag, dtype=dtype)
#                for k in ("r","corr2pt_L","LL","LLL","TT","TTT","LTT")}

#     # -------- process remaining lags ---------------------------------
#     for start in range(done_to+1, Nlag, chkpt_freq*lag_step):
#         stop = min(start + chkpt_freq*lag_step, Nlag)
#         print(f"  → lags [{start}:{stop}) …", end="", flush=True)

#         part = struct_func(vel_fields, direction,
#                            lag_range=(start, stop, lag_step),
#                            dtype=dtype)

#         for k in acc: acc[k][start:stop] = part[k]
#         np.savez(f"{outdir}/lag_{direction}_{stop-1}.npz", **acc)
#         print("saved", flush=True)

#     return acc


# ----------------------------------------------------------------------
# 1.  core – zero-copy, equal speed for x/y/z
# ----------------------------------------------------------------------
@nb.njit(parallel=True, fastmath=True)
# def _struct_core(uL, uT1, uT2, start, stop, step):
#     """
#     uL, uT1, uT2 : 2-D (N, M) float32|64, contiguous.
#     Computes lags in range(start, stop, step) *without* rolling/copying.
#     """
#     N, M     = uL.shape
#     n_lags   = (stop - start + step - 1) // step
#     dtype    = uL.dtype

#     corr2pt_L = np.zeros(n_lags, dtype=dtype)
#     S_LL  = np.zeros(n_lags, dtype=dtype)
#     S_LLL = np.zeros(n_lags, dtype=dtype)
#     S_TT  = np.zeros(n_lags, dtype=dtype)
#     S_TTT = np.zeros(n_lags, dtype=dtype)
#     S_LTT = np.zeros(n_lags, dtype=dtype)

#     for p in nb.prange(n_lags):          # thread over lag indices
#         l = start + p * step

#         cLL  = 0.0
#         sLL  = 0.0; sLLL = 0.0
#         sTT  = 0.0; sTTT = 0.0; sLTT = 0.0

#         for n in range(N):               # walk along separation axis
#             j = n + l
#             if j >= N:                   # cheap modulo-N
#                 j -= N

#             # contiguous 1-D views of length M
#             uLn  = uL[n];   uLj  = uL[j]
#             t1n  = uT1[n];  t1j  = uT1[j]
#             t2n  = uT2[n];  t2j  = uT2[j]

#             duL   = uLn - uLj
#             duT1  = t1n - t1j
#             duT2  = t2n - t2j
#             duT2s = duT1 * duT1 + duT2 * duT2   # element-wise

#             # reduction over M (NumPy inside Numba is fine)
#             cLL  += np.dot(uLn, uLj)
#             sLL  += (duL * duL      ).sum()
#             sLLL += (duL * duL * duL).sum()
#             sTT  += duT2s.sum()
#             sTTT += (duT1 * duT1 * duT1 + duT2 * duT2 * duT2).sum()
#             sLTT += (duL * duT2s).sum()

#         corr2pt_L[p] = cLL
#         S_LL[p]      = sLL
#         S_LLL[p]     = sLLL
#         S_TT[p]      = sTT
#         S_TTT[p]     = sTTT
#         S_LTT[p]     = sLTT

#     return corr2pt_L, S_LL, S_LLL, S_TT, S_TTT, S_LTT


def _struct_core(uL, uT1, uT2, start, stop, step):
    """
    uL, uT1, uT2 : 2-D (N, M) float32|64, contiguous.
    Computes lags in range(start, stop, step) *without* rolling/copying.
    """
    N, M     = uL.shape
    n_lags   = (stop - start + step - 1) // step
    dtype    = uL.dtype

    corr2pt_L = np.zeros(n_lags, dtype=dtype)
    S_LL  = np.zeros(n_lags, dtype=dtype)
    S_LLL = np.zeros(n_lags, dtype=dtype)
    S_TT  = np.zeros(n_lags, dtype=dtype)
    S_TTT = np.zeros(n_lags, dtype=dtype)
    S_LTT = np.zeros(n_lags, dtype=dtype)
    S_LTT_f = np.zeros(n_lags, dtype=dtype)

    for p in nb.prange(n_lags):          # thread over lag indices
        l = start + p * step

        cLL  = 0.0
        sLL  = 0.0; sLLL = 0.0
        sTT  = 0.0; sTTT = 0.0; sLTT = 0.0
        sLTT_f =0.0
        for n in range(N):               # walk along separation axis
            j = n + l + 1
            if j >= N:                   # cheap modulo-N
                j -= N

            # contiguous 1-D views of length M
            uLn  = uL[n];   uLj  = uL[j]
            t1n  = uT1[n];  t1j  = uT1[j]
            t2n  = uT2[n];  t2j  = uT2[j]

            duL   = uLn - uLj
            duT1  = t1n - t1j
            duT2  = t2n - t2j
            duT2s = duT1 * duT1 + duT2 * duT2   # element-wise
            # reduction over M (NumPy inside Numba is fine)
            # cLL  += np.dot(uLn, uLj)
            # sLL  += (duL * duL).sum()
            # sLLL += (duL * duL * duL).sum()
            # sTT  += duT2s.sum()
            # sTTT += (duT1 * duT1 * duT1 + duT2 * duT2 * duT2).sum()
            sLTT += (duL * duT2s).sum()
            sLTT_f += ((duL * duT1 * duT1) + (duL * duT2 * duT2) + (duT1 * duL * duL) + (duT1 * duT2 * duT2) + (duT2 * duL * duL) + (duT2 * duT1 * duT1)).sum()


        # corr2pt_L[p] = cLL
        # S_LL[p]      = sLL
        # S_LLL[p]     = sLLL
        # S_TT[p]      = sTT
        # S_TTT[p]     = sTTT
        S_LTT[p]     = sLTT
        S_LTT_f[p]   = sLTT_f
    
    # return corr2pt_L, S_LL, S_LLL, S_TT, S_TTT, S_LTT, S_LTT_f
    return S_LTT, S_LTT_f

#
# ----------------------------------------------------------------------
# 2.  struct_func – only one tiny tweak: flatten() enforces contiguity
# ----------------------------------------------------------------------
def struct_func(vel_fields, direction, *, lag_range=None,
                dtype=np.float32):
    axis = {'x':0,'y':1,'z':2}[direction]
    ux, uy, uz = (np.ascontiguousarray(f, dtype=dtype) for f in vel_fields)

    def flatten(a):
        a = np.moveaxis(a, axis, 0)         # (N, Ny, Nz) view
        a = np.ascontiguousarray(a)         # REAL contiguous copy once
        return a.reshape(a.shape[0], -1)    # (N, M)

    uL  = flatten({'x':ux,'y':uy,'z':uz}[direction])
    uT1,uT2 = (flatten(f) for k,f in {'x':ux,'y':uy,'z':uz}.items()
               if k != direction)

    N = uL.shape[0]
    start, stop, step = (0, N, 1) if lag_range is None else lag_range

    # cLL, SLL, SLLL, STT, STTT, SLTT, S_LTT_f = _struct_core(
    #     uL, uT1, uT2, start, stop, step)

    SLTT, S_LTT_f = _struct_core(
        uL, uT1, uT2, start, stop, step)

    norm = uL.size
    # cLL  /= norm; SLL /= norm; SLLL /= norm
    # STT  /= norm; STTT /= norm; 
    # cLL  /= (uL * uL).mean()
    SLTT /= norm; S_LTT_f /= norm

    lag_idx = np.arange(start, stop, step, dtype=dtype)
    # r = (2*np.pi / N) * lag_idx
    r = (2*np.pi / N) * lag_idx + (2*np.pi / N)

    # return {"r": r, "corr2pt_L": cLL,
    #         "LL": SLL, "LLL": SLLL,
    #         "TT": STT, "TTT": STTT,
    #         "LTT": SLTT, "LTT_f": S_LTT_f}
    return {"r": r,
            "LTT": SLTT, "LTT_f": S_LTT_f}
# ----------------------------------------------------------------------
# 3.  checkpoint driver – same API, extra **dtype**
# ----------------------------------------------------------------------
def safe_struct(vel_fields, direction, *, lag_step=1,
                chkpt_freq=100, outdir="chkpt",
                dtype=np.float32):
    """
    Runs struct_func in blocks, saving to <outdir>/lag_<dir>_<idx>.npz.
    dtype controls whether the whole run uses float32 or float64.
    """
    os.makedirs(outdir, exist_ok=True)
    axis = {"x":0,"y":1,"z":2}[direction]
    Nlag = vel_fields[0].shape[axis]

    # -------- look for latest checkpoint ------------------------------
    done_to = -1
    for f in os.listdir(outdir):
        if f.startswith(f"lag_{direction}_") and f.endswith(".npz"):
            try: done_to = max(done_to, int(f.split('_')[-1][:-4]))
            except ValueError: pass

    if done_to >= 0:
        print(f"• Resuming from lag {done_to+1}")
        acc = dict(np.load(f"{outdir}/lag_{direction}_{done_to}.npz",
                           allow_pickle=True))
        for k,a in acc.items():
            if a.size < Nlag:             # pad to full length
                pad = np.zeros(Nlag, dtype=a.dtype)
                pad[:a.size] = a
                acc[k] = pad
    else:
        print("• Starting fresh run")
        acc = {k: np.zeros(Nlag, dtype=dtype)
            #    for k in ("r","corr2pt_L","LL","LLL","TT","TTT","LTT", "LTT_f")}
            for k in ("r","LTT", "LTT_f")}
    # -------- process remaining lags ---------------------------------
    for start in range(done_to+1, Nlag, chkpt_freq*lag_step):
        stop = min(start + chkpt_freq*lag_step, Nlag)
        print(f"  → lags [{start}:{stop}) …", end="", flush=True)

        part = struct_func(vel_fields, direction,
                           lag_range=(start, stop, lag_step),
                           dtype=dtype)

        for k in acc: acc[k][start:stop] = part[k]
        np.savez(f"{outdir}/lag_{direction}_{stop-1}.npz", **acc)
        print("saved", flush=True)

    return acc

