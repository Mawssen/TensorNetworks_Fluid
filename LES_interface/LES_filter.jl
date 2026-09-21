

function load_velocity_1024(file_path::String)

    file = matopen(file_path)
    vel_name = collect(keys(file))[1]
    u = Float64.(read(file, vel_name))
    close(file)
    
    return u
end



# ── Naive version (simple for loops with mod1) ──
function les_box_filter_naive(u::Array{Float64,3}, δ::Int)
    N = size(u, 1)
    @assert size(u) == (N, N, N) "Input must be a cubic array N×N×N"
    @assert N % δ == 0 "Filter size δ must evenly divide N"

    M = N ÷ δ
    u_f = zeros(Float64, M, M, M)
    inv_vol = 1.0 / δ^3

    @inbounds for K in 1:M, J in 1:M, I in 1:M
        s = 0.0
        for dk in 1:δ, dj in 1:δ, di in 1:δ
            ii = mod1((I-1)*δ + di, N)
            jj = mod1((J-1)*δ + dj, N)
            kk = mod1((K-1)*δ + dk, N)
            s += u[ii, jj, kk]
        end
        u_f[I, J, K] = s * inv_vol
    end

    return u_f
end

# ── Optimized version (no mod1, @threads, @simd, cache-friendly) ──
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

function les_box_filter_vector(ux::Array{Float64,3}, uy::Array{Float64,3},
                               uz::Array{Float64,3}, δ::Int)
    return les_box_filter(ux, δ), les_box_filter(uy, δ), les_box_filter(uz, δ)
end
