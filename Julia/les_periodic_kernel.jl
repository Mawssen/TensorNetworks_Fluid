# Numerical implementation moved verbatim from Julia/LES_filter.jl.
function box_filter_3d(u::AbstractArray{T,3}, r::NTuple{3,Int}) where {T<:Real}
    nx, ny, nz = size(u)
    rx, ry, rz = r
    u_f = similar(u)
    norm = (2rx + 1) * (2ry + 1) * (2rz + 1)

    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        s = zero(T)
        for dk in -rz:rz, dj in -ry:ry, di in -rx:rx
            ii = mod1(i + di, nx)  # periodic in x
            jj = mod1(j + dj, ny)  # periodic in y
            kk = mod1(k + dk, nz)  # periodic in z
            s += u[ii, jj, kk]
        end
        u_f[i, j, k] = s / norm
    end

    return u_f
end
