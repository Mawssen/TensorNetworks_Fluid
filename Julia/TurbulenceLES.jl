"""In-memory historical LES operators with explicit, distinct scientific methods.
No loading, output, plotting, spacing conversion, or CRC setup is performed.
"""
module TurbulenceLES
module Kernels
include("les_block_kernel.jl")
include("les_periodic_kernel.jl")
end
export block_average, periodic_box_smooth

"""
    block_average(field::Array{Float64,3}; block_size::Int)

Average disjoint `block_size³` blocks of a cubic `N³` field; return a new
Float64 `(N÷block_size)³` array. `block_size` must be positive and divide N.
Array axes 1,2,3 remain in the caller's order: no physical axis mapping is
inferred. For a caller's `(x,y,z)` array, output axes remain `(x,y,z)`.
Blocks start at index 1 on each axis; no halo, wrapping, padding, or partial
edge blocks are used. Values are unweighted arithmetic means, without scaling.
For a uniform input grid with spacings `(dx,dy,dz)`, coarse spacings are
`block_size .* (dx,dy,dz)`. Each coarse value represents its whole input block;
if plotted at a block centroid, its first coordinate is the input's first
coordinate plus `(block_size-1)/2` input spacings per axis. Coordinates and units
remain the caller's responsibility. Delegates to the historical threaded/SIMD
kernel; floating reduction order is preserved.
"""
function block_average(field::Array{Float64,3}; block_size::Int)
    block_size > 0 || throw(ArgumentError("block_size must be positive"))
    return Kernels.les_box_filter(field, block_size)
end

"""
    periodic_box_smooth(field::AbstractArray{T,3}; radii::NTuple{3,Int}) where T<:Real

Return a new array of the same shape/element type and axis order as `field`.
Axes 1,2,3 correspond to the supplied radii `(rx,ry,rz)` (historical labels
x,y,z); callers must establish their dataset's physical mapping. Nonnegative
radii give odd stencil widths `(2rx+1,2ry+1,2rz+1)`. Every stencil index wraps
periodically with mod1, including across edges/corners; wide stencils may visit
the same sample multiple times. There is no downsampling, halo or padding.
The output stays at the original grid coordinates and spacings. On a uniform
grid the sampled stencil spans those widths in grid cells; this is not a
disjoint-block width or a change in physical grid spacing.
Uses the historical direct 3D sum/divide implementation, not the separable
cumsum or optional center-sampling variant. Prefer floating inputs: historical
same-type storage can raise InexactError for integer arrays. No dtype conversion
or normalization beyond division by stencil volume is added.
"""
function periodic_box_smooth(field::AbstractArray{T,3}; radii::NTuple{3,Int}) where {T<:Real}
    all(r -> r >= 0, radii) || throw(ArgumentError("radii must be nonnegative"))
    return Kernels.box_filter_3d(field, radii)
end
end
