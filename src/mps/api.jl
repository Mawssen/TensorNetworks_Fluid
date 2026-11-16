"""Small MPS API backed by the historical implementation; no case or CRC setup.

Loading retains historical MKL selection and BLAS thread count of 10.
"""
module TurbulenceMPS

module Historical
include(joinpath(@__DIR__, "..", "..", "legacy", "Julia", "0mps_calc.jl"))
end

export load_centered_be_float32, compress_reconstruct, write_mps_mat

"""Read x-fastest big-endian Float32 data with shape `(nx,ny,nz)` in `(x,y,z)`
order. Return a centered `crop_size³` array in the same order, without scaling.
Explicit `full_shape` and `crop_size` allow other loaders' binary fixtures;
defaults preserve the historical Jet Flame reader and ignored trailing bytes.
"""
load_centered_be_float32(path::String; full_shape::NTuple{3,Int}=(864,1008,576),
    crop_size::Int=512) = Historical.load_centered_be_float32(path;
        full_shape=full_shape, crop_size=crop_size)

"""Compress and reconstruct an in-memory floating field, returning
`(field, link_dimensions)`. Input is an equal-sided power-of-two 2D or 3D array;
axes retain caller order (for Jet Flame: x,y,z). No normalization is applied.
`chi::Int` is passed as maxdim with cutoff=0.0 exactly as in the historical CLI.
Alternatively pass `cutoff::Float64` to use the historical cutoff-only branch,
without a bond-dimension cap. Supply exactly one of chi or cutoff.
`ordering::Int` must be explicit: 1 interleaved, 2 sequential, 3 comb1, 4 combn.
Bit ordering and inverse reconstruction are delegated unchanged to history.
No scientific cap range or accuracy criterion is imposed by this wrapper.
"""
function compress_reconstruct(field::AbstractArray{<:AbstractFloat};
        chi::Union{Int,Nothing}=nothing, cutoff::Union{Float64,Nothing}=nothing,
        ordering::Int)
    ordering in 1:4 || throw(ArgumentError("ordering must be 1..4"))
    (isnothing(chi) == isnothing(cutoff)) &&
        throw(ArgumentError("supply exactly one of chi or cutoff"))
    if !isnothing(cutoff)
        return Historical.mps_truncate(field; cutoff=cutoff, split=ordering)
    end
    return Historical.mps_truncate(field; χ=chi, cutoff=0.0, split=ordering)
end

"""Write reconstructed `field` (axes and shape unchanged) and link-dimension
vector `bonds` as historical MAT keys `u` and `chi_crit`. Returns nothing.
`path` is caller-selected; refuses an existing path rather than overwriting.
"""
function write_mps_mat(path::String, field, bonds)
    ispath(path) && throw(ArgumentError("output exists: $path"))
    islink(path) && throw(ArgumentError("output is a symlink: $path"))
    Historical.save_to_mat(path, field, bonds)
    return nothing
end
end
