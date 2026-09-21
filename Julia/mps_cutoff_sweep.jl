using ITensors, ITensorMPS
using LinearAlgebra
using MAT
using MKL
using Printf
BLAS.set_num_threads(20)
ITensors.set_warn_order(40)

# ---------------------------------------------------------------------------
# Reused verbatim from mps_calc.jl (same conventions/orderings).
# ---------------------------------------------------------------------------

function load_velocity_1024(file_path::String)
    file = matopen(file_path)
    vel_name = collect(keys(file))[1]
    Velocity = read(file, vel_name)
    close(file)
    u = Float64.(Velocity)
    return u
end

function perm_tuple_3D(n)
    a = Int[]
    for i in 0:(3*n - 1)
        index = (i % 3 == 0) * div(i, 3) +
                (i % 3 == 1) * (n + div(i + 1, 3)) +
                (i % 3 == 2) * (2*n - 1 + div(i + 1, 3)) + 1
        push!(a, index)
    end
    return tuple(a...)
end

# ---------------------------------------------------------------------------
# Build the interleaved (split=1) MPS at a given cutoff and return ONLY the
# critical bond dimensions. No contraction / reconstruction -> fast, so a
# whole cutoff list is cheap. Mirrors mps_truncate(split=1) exactly up to the
# MPS(...) call.
# ---------------------------------------------------------------------------
function linkdims_for_cutoff(u::Array{Float64}, cutoff::Float64)
    N = Int(log2(size(u, 1)))
    ndim = Int(ndims(u))
    size_u_split = ntuple(_ -> 2, ndim * N)

    u_reshaped = reshape(u, size_u_split)
    perm = perm_tuple_3D(N)
    u_perm = permutedims(u_reshaped, perm)

    s = siteinds("Qubit", ndim * N)
    u_vector = reshape(u_perm, prod(size(u_perm)))

    ψ = MPS(u_vector, s, cutoff=cutoff)
    return linkdims(ψ)            # [chi_1, ..., chi_{3N-1}]
end

# ---------------------------------------------------------------------------
# Faithful Julia port of comp_ratio_memory (memory definition).
#   chi        : interior bond dims [chi_1 ... chi_{3N-1}]  (length 3N-1)
#   N3         : number of MPS sites = len(chi)+1 = 3N       (so 2^N3 = cells)
#   full mem   : 2^N3   (one Float per cell of the 2^N3-element cube)
#   mps mem    : 2 * sum_{j} chi[j]*chi[j-1]  with chi padded by 1 at both ends
#   CR_memory  : 2^N3 / mps_mem
#   CR_dof     : 2^N3 / (mps_mem - sum_{interior} chi_j^2)   [optional]
# ---------------------------------------------------------------------------
function comp_ratio_memory(chi::Vector{Int}; cr_dof::Bool=false)
    N3 = length(chi) + 1
    chi_pad = vcat(1, chi, 1)                       # [1, chi..., 1]
    mps_mem = 2 * sum(chi_pad[j] * chi_pad[j-1] for j in 2:length(chi_pad))
    cr_mem = mps_mem != 0 ? (2.0^N3) / mps_mem : Inf
    if cr_dof
        sum2 = sum(chi_pad[j]^2 for j in 2:(length(chi_pad)-1))  # interior
        cr_dof_val = (2.0^N3) / (mps_mem - sum2)
        return cr_mem, cr_dof_val
    end
    return cr_mem
end

# ---------------------------------------------------------------------------
# CLI (same key=value style as mps_calc.jl)
# ---------------------------------------------------------------------------
function get_arg(key::AbstractString, default::AbstractString)
    for arg in ARGS
        if startswith(arg, "$(key)=")
            return split(arg, "=", limit=2)[2]
        end
    end
    return default
end

# Parse a comma-separated cutoff list, e.g. cutoffs=1e-2,7e-3,5e-3,3e-3,2e-3,1.5e-3
function parse_cutoffs(s::AbstractString)
    return [parse(Float64, strip(x)) for x in split(s, ",") if length(strip(x)) > 0]
end

total_time = @elapsed begin
    file_name = get_arg("file", "U_t1")
    data_dir  = get_arg("dir",  "../../../../Forced_Isotropic_1024Cubed_mat_in_one")
    target    = parse(Float64, get_arg("target", "512"))
    out_file  = get_arg("out",  "cutoff_sweep_$(file_name).txt")
    cutoffs   = parse_cutoffs(get_arg("cutoffs",
                    "1e-2,7e-3,5e-3,4e-3,3e-3,2.5e-3,2e-3,1.5e-3"))

    println("=== mps_cutoff_sweep.jl (split=1 interleaved) ===")
    println("  file    = $file_name")
    println("  dir     = $data_dir")
    println("  target  = CR >= $target")
    println("  cutoffs = $cutoffs")
    println("=================================================")

    file_path = joinpath(data_dir, "$(file_name).mat")
    println("Loading $file_path ...")
    u = load_velocity_1024(file_path)
    N = Int(log2(size(u, 1)))
    println("  cube: size=$(size(u)), N=$N (3N=$(3N) sites, target cells=2^$(3N))")

    # Results table
    results = Tuple{Float64, Float64, Int, Int}[]  # (cutoff, CR, chi_max, n_bonds)

    open(out_file, "w") do io
        println(io, "# MPS cutoff sweep, split=1 (interleaved), file=$file_name")
        println(io, "# N=$N, sites=3N=$(3N), full memory = 2^$(3N)")
        println(io, "# target CR_memory >= $target")
        println(io, "#")
        println(io, "# cutoff        CR_memory        chi_max   n_bonds")
        for cf in cutoffs
            dt = @elapsed chi = linkdims_for_cutoff(u, cf)
            chi_int = Int.(chi)
            cr = comp_ratio_memory(chi_int)
            chimax = maximum(chi_int)
            nb = length(chi_int)
            push!(results, (cf, cr, chimax, nb))
            flag = cr >= target ? "  >= target" : "  < target"
            line = @sprintf("%-12.4e  %-15.4f  %-8d  %-6d%s",
                            cf, cr, chimax, nb, flag)
            println(line, "   (", round(dt, digits=1), "s)")
            println(io, @sprintf("%-12.4e  %-15.6f  %-8d  %-6d", cf, cr, chimax, nb))
            flush(io)
        end
    end

    # Report the smallest cutoff that still meets the target.
    passing = filter(r -> r[2] >= target, results)
    println("\n=== SUMMARY ===")
    if isempty(passing)
        println("No cutoff in the list reached CR >= $target.")
        println("Every CR was below target -> use a LARGER cutoff (truncate more).")
    else
        # smallest cutoff among those that pass
        best = passing[argmin([r[1] for r in passing])]
        println("Smallest cutoff with CR >= $target :")
        println(@sprintf("  cutoff = %.4e   CR = %.4f   chi_max = %d",
                         best[1], best[2], best[3]))
        # also note the next-smaller cutoff that FAILED, to bracket it
        failing = filter(r -> r[2] < target, results)
        below = filter(r -> r[1] < best[1], failing)
        if !isempty(below)
            nb = below[argmax([r[1] for r in below])]
            println(@sprintf("  (next smaller tried, cutoff = %.4e, gave CR = %.4f < target)",
                             nb[1], nb[2]))
            println("  -> the true threshold cutoff lies between these two.")
        end
    end
    println("Wrote table to: $out_file")
end

println("\nTotal elapsed time: ", total_time, " seconds")
