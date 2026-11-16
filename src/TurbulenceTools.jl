# Historical imports configure MKL/BLAS at load time. Execute those settings at
# each import rather than serializing that process state into a precompile cache.
__precompile__(false)

"""Reusable MPS and explicit LES operators, backed by shared historical kernels.

Case scripts own datasets, parameters, grid metadata, output paths and CRC setup.
Import retains the historical MPS MKL backend and BLAS thread count of 10.
"""
module TurbulenceTools

include(joinpath(@__DIR__, "..", "Julia", "TurbulenceMPS.jl"))
include(joinpath(@__DIR__, "..", "Julia", "TurbulenceLES.jl"))
using .TurbulenceMPS: load_centered_be_float32, compress_reconstruct, write_mps_mat
using .TurbulenceLES: block_average, periodic_box_smooth

export load_centered_be_float32, compress_reconstruct, write_mps_mat,
       block_average, periodic_box_smooth
end
