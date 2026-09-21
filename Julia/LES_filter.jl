using LinearAlgebra
using Statistics
import Plots  

##############################
# 3D box filter (LES-style)  #
##############################

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


###############################
# 1D periodic box filter      # gives same output as box_filter_3d but is faster because of vectorization!
###############################
import Base.@propagate_inbounds


@propagate_inbounds function box1d_periodic!(
    out::AbstractVector{T},
    x::AbstractVector{T},
    r::Int;
    org_size::Bool = true
) where {T<:Real}

    n = length(x)
    w = 2r + 1
    @assert n ≥ 1
    @assert r ≥ 0
    @assert n % w == 0 || org_size  # exact blocking required if reduced size

    if r == 0
        if org_size
            copyto!(out, x)
        else
            @inbounds for i in 1:length(out)
                out[i] = x[1 + (i-1)*w]
            end
        end
        return out
    end

    # periodic extension
    tmp = similar(x, n + 2r)
    @inbounds begin
        tmp[1:r] .= x[n-r+1:n]
        tmp[r+1:r+n] .= x
        tmp[r+n+1:end] .= x[1:r]
    end

    c = cumsum(tmp)

    if org_size
        @inbounds for i in 1:n
            s = c[i+w-1] - (i == 1 ? zero(T) : c[i-1])
            out[i] = s / w
        end
    else
        nc = div(n, w)
        @inbounds for I in 1:nc
            i = 1 + (I-1)*w
            s = c[i+w-1] - (i == 1 ? zero(T) : c[i-1])
            out[I] = s / w
        end
    end

    return out
end


function box_filter_3d_sep(
    u::AbstractArray{T,3},
    r::NTuple{3,Int};
    org_size::Bool = true
) where {T<:Real}

    nx, ny, nz = size(u)
    rx, ry, rz = r

    tmp1 = similar(u)
    tmp2 = similar(u)

    # 1) filter along x
    @inbounds for j in 1:ny, k in 1:nz
        box1d_periodic!(@view(tmp1[:, j, k]), @view(u[:, j, k]), rx)
    end

    # 2) filter along y
    @inbounds for i in 1:nx, k in 1:nz
        box1d_periodic!(@view(tmp2[i, :, k]), @view(tmp1[i, :, k]), ry)
    end

    # 3) filter along z (result stays in tmp1)
    @inbounds for i in 1:nx, j in 1:ny
        box1d_periodic!(@view(tmp1[i, j, :]), @view(tmp2[i, j, :]), rz)
    end

    # --- return full-size or coarse-grained field ---
    if org_size
        return tmp1
    else
        wx, wy, wz = (2rx + 1, 2ry + 1, 2rz + 1)

        # int(N/(2*rk+1)) = floor division in Julia
        nx_c = fld(nx, wx)
        ny_c = fld(ny, wy)
        nz_c = fld(nz, wz)

        out = Array{T,3}(undef, nx_c, ny_c, nz_c)

        # sample at the center of each box of size (2*rk+1)
        cx, cy, cz = div(wx, 2), div(wy, 2), div(wz, 2)

        @inbounds for I in 1:nx_c, J in 1:ny_c, K in 1:nz_c
            i = 1 + cx + (I-1)*wx
            j = 1 + cy + (J-1)*wy
            k = 1 + cz + (K-1)*wz
            out[I, J, K] = tmp1[i, j, k]
        end

        return out
    end
end


###################################
# δ-operator using separable box  #
###################################

function delta_operator_3d(u::AbstractArray{T,3}, r::NTuple{3,Int}) where {T<:Real}
    u_f = box_filter_3d(u, r)
    return u_f .- u
end

function plot_smoothing_planes(u::AbstractArray{<:Real,3},
    k_slice::Int,
    r_list::Vector{NTuple{3,Int}})
    nx, ny, nz = size(u)
    @assert 1 ≤ k_slice ≤ nz

    @views umin = minimum(u[:, :, k_slice])
    @views umax = maximum(u[:, :, k_slice])

    nplots = length(r_list) + 1  # heatmaps only

    # tick positions and labels
    xticks = ( [1, 1+nx÷2, nx], ["0", "π", "2π"] )
    yticks = ( [1, 1+ny÷2, ny], ["0", "π", "2π"] )

    # heatmaps share 0.9, colorbar gets 0.1
    w_plots = 0.9
    w_plot = w_plots / nplots
    widths = [fill(w_plot, nplots)..., 1-w_plots]

    layout = Plots.grid(1, nplots + 1; widths = widths)
    plt = Plots.plot(layout = layout, size = (500 * (nplots), 500))

    # -------- DNS (original) --------
    @views Plots.heatmap!(plt[1],
    u[:, :, k_slice],
    clim = (umin, umax),
    title = "DNS",
    colorbar = false,
    xticks = xticks,
    yticks = yticks,
    )

    # -------- filtered fields --------
    for (idx, r) in enumerate(r_list)
    u_f = box_filter_3d_sep(u, r)
    @views Plots.heatmap!(plt[idx + 1],
    u_f[:, :, k_slice],
    clim = (umin, umax),
    title = "kernel size = ($(2*r[1]+1),$(2*r[2]+1),$(2*r[3]+1))",
    colorbar = false,
    xticks = xticks,
    yticks = yticks,
    )
    end

    # -------- colorbar-only panel --------
    dummy = fill(NaN, 2, 2)
    Plots.heatmap!(plt[nplots + 1],
    dummy,
    clim = (umin, umax),
    colorbar = true,
    xticks = false,
    yticks = false,
    framestyle = :none,
    )

    return plt
end

function load_velocity_1024(file_path::String)

    file = matopen(file_path)
    vel_name = collect(keys(file))[1]

    Velocity = read(file, vel_name)
    

    close(file)

    u = Velocity
    u = Float64.(u)

    return u
end


#######################
# Example usage       #
#######################

# # Example: synthetic 3D velocity component
# nx, ny, nz = 64, 64, 64
# # u = randn(nx, ny, nz)  # e.g. one velocity component

# # Choose a z-plane and a set of filter sizes
# k_slice = 16
# # r_list = [(1,1,1), (2,2,2), (3,3,3)]
# r_list = [(1,1,1)]

# # Plot original and smoothed planes
# plot_smoothing_planes(u, k_slice, r_list)