# Numerical implementation moved verbatim from LES_interface/LES_filter.jl.
function les_box_filter(u::Array{Float64,3}, δ::Int)
    N = size(u, 1)
    @assert size(u) == (N, N, N) "Input must be a cubic array N×N×N"
    @assert N % δ == 0 "Filter size δ must evenly divide N"

    M = N ÷ δ
    u_f = zeros(Float64, M, M, M)
    inv_vol = 1.0 / (δ * δ * δ)

    Threads.@threads for K in 1:M
        k0 = (K - 1) * δ
        for J in 1:M
            j0 = (J - 1) * δ
            for I in 1:M
                i0 = (I - 1) * δ
                s = 0.0
                @inbounds for dk in 1:δ
                    kk = k0 + dk
                    for dj in 1:δ
                        jj = j0 + dj
                        @simd for di in 1:δ
                            s += u[i0 + di, jj, kk]
                        end
                    end
                end
                u_f[I, J, K] = s * inv_vol
            end
        end
    end

    return u_f
end
