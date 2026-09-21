using ITensors
using LinearAlgebra
using MAT
using HDF5
using Plots
using JSON
using Printf

# using PyPlot
using MKL 
BLAS.set_num_threads(20)  
ITensors.set_warn_order(40) 


function load_velocity_mat(file_path::String)
    file = matopen(file_path)
    
    # Extract the first variable name (velocity data) and relevant coordinate variables
    vel_name = collect(names(file))[1]
    Vel = read(file, vel_name)
    xcoor = read(file, "xcoor")
    ycoor = read(file, "ycoor")
    zcoor = read(file, "zcoor")
    
    # Permute the dimensions to get the correct orientation
    perm = (3, 2, 1)
    u = permutedims(Vel[:, :, :, 1], perm)  
    v = permutedims(Vel[:, :, :, 2], perm) 
    w = permutedims(Vel[:, :, :, 3], perm) 
    
    # Convert velocity components to Float64 if needed
    u = Float64.(u)
    v = Float64.(v)
    w = Float64.(w)
    
    # Extract coordinate values
    x = xcoor[1, :]
    y = ycoor[1, :]
    z = zcoor[1, :]
    
    # Close the file
    close(file)
    
    # Return the velocity components and coordinates
    return u, v, w, x, y, z
end

    
function load_velocity_h5(file_path::String)
    file = h5open(file_path, "r")
    vel_name = collect(keys(file))[1]
    
    # Extract the velocity and coordinate variables
    Velocity = read(file, vel_name)
    xcoor = read(file, "xcoor")
    ycoor = read(file, "ycoor")
    zcoor = read(file, "zcoor")
    
    close(file)
   
    u = Velocity[1, :, :, :] 
    v = Velocity[2, :, :, :] 
    w = Velocity[3, :, :, :] 

    # Convert velocity components to Float64 if needed
    u = Float64.(u)
    v = Float64.(v)
    w = Float64.(w)
    
    # Extract coordinate values
    x = xcoor[:, 1]
    y = ycoor[:, 1]
    z = zcoor[:, 1]
    
    # Return the velocity components and coordinates
    return u, v, w, x, y, z
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


function perm_tuple_3D(n) #interleaved permutation
    a = Int[]  # Initialize an empty array of integers
    for i in 0:(3*n - 1)
        index = (i % 3 == 0) * div(i, 3) + 
                (i % 3 == 1) * (n + div(i + 1, 3)) + 
                (i % 3 == 2) * (2*n - 1 + div(i + 1, 3))+ 1
        push!(a, index) 
    end
    return tuple(a...)
end


function inv_perm_tuple_3D(n) #interleaved inverse permutation 
    a = Int[]  # Initialize an empty array of integers
    for i in 1:(n)
        index = 3*i-2
        push!(a, index)
    end
    for i in 1:(n)
        index = 3*i-1
        push!(a, index)
    end
    for i in 1:(n)
        index = 3*i
        push!(a, index)
    end
    return tuple(a...)
end


function cont_split_perm(N)
    if N < 1
        return ()
    end

    part1 = collect(1:N)
    part2 = collect(2*N:-1:N+1)
    part3 = collect(2*N+1:3*N)

    return Tuple(vcat(part1, part2, part3));
end

function n_middle_perm(n::Int)
    # Original indices:
    # x: 1:n, y: n+1:2n, z: 2n+1:3n
    x = collect(1:n)
    y = collect(n+1:2n)
    z = collect(2n+1:3n)

    # Prepare triplets: [(x1,y1,z1), ..., (xn,yn,zn)]
    triplets = [(x[i], y[i], z[i]) for i in 1:n]
    # println(triplets)
    # Interleave based on your pattern:
    # Order: 1,3,...,n-1, n, n-2,...,4,2
    if n % 2 == 0
        order = vcat(1:2:n-1, n, (n-2):-2:2)
    else
        order = vcat(1:2:n-2, n, (n-1):-2:2)
    end

    # Flatten selected triplets into one tuple
    return Tuple(Iterators.flatten(triplets[i] for i in order))
end

function inv_n_middle_perm(n::Int)
    perm = n_middle_perm(n)
    inv_perm = Vector{Int}(undef, length(perm))

    for (i, p) in enumerate(perm)
        inv_perm[p] = i
    end

    return Tuple(inv_perm)
end


function one_middle_perm(n::Int)
    # Original indices
    x = collect(1:n)
    y = collect(n+1:2n)
    z = collect(2n+1:3n)

    # Create triplets
    triplets = [(x[i], y[i], z[i]) for i in 1:n]
    # println(triplets)
    # Separate even and odd indices (excluding 1)
    left_indices  = reverse(2:2:n)   # even before 1, descending
    center_index  = 1
    right_indices = 3:2:n           # odd after 1, ascending

    # Compose final order
    order = vcat(left_indices, center_index, right_indices)

    # Flatten the triplets in this custom order
    return Tuple(Iterators.flatten(triplets[i] for i in order))
end

function inv_one_middle_perm(n::Int)
    perm = one_middle_perm(n)
    inv_perm = Vector{Int}(undef, length(perm))

    for (i, p) in enumerate(perm)
        inv_perm[p] = i
    end

    return Tuple(inv_perm)
end

function ordering(u::Array{Float64}, split::Int)
    N = Int(log2(size(u, 1)));
    ndim = Int(ndims(u));  #The number of dimensions of u    
    size_u_split = ntuple(_ -> 2, ndim * N) #the desired qubit indices format: (2,2,2,...)
    
    if split == 2 #2 = split: x1x2x3 y1y2y3 z1z2z3
        u_perm = u;  # No permutation, use original u
    else
        u_reshaped = reshape(u, size_u_split);
        if split == 1  #1 = interleaved: x1x2x3 y1y2y3 z1z2z3 -> x1y1z1 x2y2z2 ...
            perm = perm_tuple_3D(N);
        elseif split == 3 #3 = continuous split: x1x2x3 y1y2y3 z1z2z3 -> x1x2x3 y3y2y1 z1z2z3
            perm = cont_split_perm(N)    
        elseif split == 4 #4 = n_middle: x1x2x3 y1y2y3 z1z2z3 -> x1y1z1 x3y3z3 x2y2z2
            perm = n_middle_perm(N)
        elseif split == 5 #5 = one_middle: x1x2x3 y1y2y3 z1z2z3 -> x2y2z2 x1y1z1 x3y3z3
            perm = one_middle_perm(N)
        end
        u_perm = permutedims(u_reshaped, perm);
    end
end


function inv_ordering(J_trunc::Array{Float64}, split::Int, size_u)
    N = Int(log2(size_u[1]));
    
    if split == 2 #from split to u
        u_rec = reshape(J_trunc, size_u);  # No permutation needed, reshape directly
    
    else
        if split == 1     #from interleaved to u
            inv_perm = inv_perm_tuple_3D(N);
        elseif split == 3 #from continuous split to u
            inv_perm = cont_split_perm(N) #inv_perm = perm            
        elseif split == 4 #from n_middle to u
            inv_perm = inv_n_middle_perm(N)
        elseif split == 5 #from one_middle to u
            inv_perm = inv_one_middle_perm(N)
        end
        u_inv_perm = permutedims(J_trunc, inv_perm);
        u_rec = reshape(u_inv_perm, size_u);
    end
end


function mps_truncate(u::Array{Float64}; χ::Union{Int, Nothing} = nothing, cutoff::Union{Float64, Nothing} = nothing, split::Int)
    # Determine N based on the first dimension of u
    N = Int(log2(size(u, 1)));
    ndim = Int(ndims(u));  #The number of dimensions of u    
    # size_u_split = ntuple(_ -> 2, ndim * N) #the desired qubit indices format: (2,2,2,...)
    
    u_perm = ordering(u, split)
    
    # Create Qubit indices for each dimension
    s = siteinds("Qubit", ndim * N)
    
    # Reshape u_perm into a 1D vector compatible with ITensors
    u_vector = reshape(u_perm, :)

    # Perform MPS truncation based on the given cutoff or χ using explicit if-else
    if χ == nothing && cutoff == nothing
        ψ = MPS(u_vector, s, cutoff=0)  # No truncation
            
    elseif χ == nothing && cutoff !== nothing
        ψ = MPS(u_vector, s, cutoff=cutoff)  # Truncate based on cutoff
            
    elseif χ !== nothing && cutoff == nothing
        ψ = MPS(u_vector, s, maxdim=χ, cutoff=0)  # Truncate based on bond dimension χ
            
    else
        ψ = MPS(u_vector, s, cutoff=cutoff, maxdim=χ)  # Truncate based on both
            
    end
    
    chi_crit = linkdims(ψ)
        
    # Recover the original tensor from the MPS
    J_trunc = Array(contract(ψ),s)
    
    # Perform inverse permutation
    u_rec = inv_ordering(J_trunc, split, size(u))
    
    return u_rec, chi_crit
end

    
function save_to_mat(filename::String, u_data, n_data)
    # Open an HDF5 file and save u and n data
    matfile = matopen(filename, "w")  # "w" means write mode
    
    # Save variables to the .mat file
    write(matfile, "u", u_data)  # Save the 'u_c30_e4' data
    write(matfile, "chi_crit", n_data)  # Save the 'n_c30_e4' data
    
    close(matfile)  # Always close the file after writing
    println("u_data .mat file saved!")
end

function load_velocity_binary(filename::String, shape::NTuple{3, Int}, skip_bytes::Int=0)
    num_elements = prod(shape)
    field = nothing  # pre-define to avoid UndefVarError

    open(filename, "r") do io
        if skip_bytes > 0
            header_values = Vector{Float64}(undef, skip_bytes ÷ 8)
            read!(io, header_values)
            println("Header (", length(header_values), " float64 values):")
            for (i, val) in enumerate(header_values)
                print(rpad("[$(i-1)] = $(@sprintf("%.6e", val))", 30))
                if i % 4 == 0 println() end
            end
            println()
        end

        buffer = Vector{Float64}(undef, num_elements)
        read!(io, buffer)

        try
            field = reshape(buffer, shape...)  # Julia is Fortran-order by default
        catch e
            println("❌ Could not reshape array from $filename: $e")
        end
    end

    return field
end


function bond_singular_values(A::AbstractArray; p::Real = 2)
    # ---- make sure every dimension is 2^n ---------------------------------
    nbits = Int[]
    for (d,L) in enumerate(size(A))
        n = round(Int, log2(L))
        @assert 2^n == L  "dimension $d (=$L) is not a power of two"
        push!(nbits, n)
    end
    m = sum(nbits)                      # total physical sites

    # ---- initial matrix: 2  × 2^{m−1} -------------------------------------
    v = reshape(A, 2^m)
    L, R   = 2, 2^(m-1)
    M      = reshape(v, L, R)

    mid    = (m - 1) ÷ 2                # last step of the “left half”
    svals  = Vector{Vector{Float64}}(undef, m-1)

    # -----------------------------------------------------------------------
    # sweep canonical centre across every bond
    # -----------------------------------------------------------------------
    for step in 1:m-1
        F          = svd(M; full = false)        # U · Diag(S) · Vᵀ
        σ_norm     = F.S ./ norm(F.S, p)
        svals[step] = σ_norm                     # save (normalised) Σ
        # svals[step] = F.S                      # save Σ

        # ---- decide the shape of the next matrix --------------------------
        if step ≤ mid                         # left half
            Lnext, Rnext = L*2, R÷2
            core        = Diagonal(F.S) * F.Vt
        elseif step == mid + 1                # *first* step after middle
            Lnext, Rnext = L,   R÷2
            core        = Diagonal(F.S) * F.Vt
        else                                  # remaining right half
            Lnext, Rnext = L÷2, R÷2
            core        = Diagonal(F.S) * F.Vt
        end

        # core is χ × R ; we may have to drop or pad elements so that
        # reshape(core, Lnext, Rnext) is possible
        needed = Lnext * Rnext
        flat   = vec(core)
        if length(flat) < needed               # pad with zeros if needed
            M = reshape([flat; zeros(eltype(core), needed - length(flat))],
                         Lnext, Rnext)
        else
            M = reshape(view(flat, 1:needed),   Lnext, Rnext)
        end
        L, R = Lnext, Rnext
    end
    return svals
end



########################################################################

#JHS .mat data
    
# total_time = @elapsed begin
#     cf = 1e-3  # Cutoff value for MPS truncation
#     sp = 2    # Split mode: 1 = interleaved, 2 = split, 3 = continuous split, 4 = n_middle, 5 = one_middle
#     file_name = "U_t1"

#     file_path_1024 = "../Forced_Isotropic/02 Grid 128/$file_name.mat"
#     # file_path_1024 = "../../../../../Forced_Isotropic_1024Cubed_mat_in_one/$file_name.mat"
#     u = load_velocity_1024(file_path_1024)
    
#     @time u_rec, chi_c = mps_truncate(u, cutoff=cf, split=sp)
    
#     output_file = "truncated_$(file_name)_1024_csp_cf$(cf).mat"
#     save_to_mat(output_file, u_rec, chi_c)
# end

# println("Total elapsed time: ", total_time, " seconds");


######################################################################## 
#Daniel .bin data
    
# total_time = @elapsed begin
#     cf = 1e-4  # Cutoff value for MPS truncation
#     sp = 1     # Split mode: 1 = interleaved, 2 = split
#     file_name = "U"
#     N = 32

#     if N == 32
#         dataset = "32_binary_dns"
#         config_path = "../Massen_Lib/config_FI.json"
#     elseif N == 1024
#         dataset = "full_binary_dns"
#         config_path = joinpath(mps_codes_path, "config_FI.json")
#     end

#     config = open(config_path, "r") do f
#         JSON.parse(read(f, String))
#     end

#     org_path = get(config, dataset, Dict())
#     shape = (N, N, N)

#     skip_byte = file_name == "U" ? 192 : 0

#     @time u = load_velocity_binary(org_path["vel"][file_name], shape, skip_byte)
# #     @time u = load_velocity_binary(org_path[file_name], shape, skip_byte)
#     println("✅ u shape: ", isnothing(u) ? "Error" : size(u));

#     @time u_rec, chi_c = mps_truncate(u, cutoff=cf, split=sp)
    
#     output_file = "truncated_$(file_name)_1024_sp_cf$(cf).mat"
#     save_to_mat(output_file, u_rec, chi_c)
# end

# println("\nTotal elapsed time: ", total_time, " seconds")

##############################################################################

#JHS singular values

total_time = @elapsed begin
    sp = 1    # Split mode: 1 = interleaved, 2 = split, 3 = continuous split, 4 = n_middle, 5 = one_middle
    file_name = "V_t1"

    # file_path_1024 = "../Forced_Isotropic/02 Grid 128/$file_name.mat"
    file_path_1024 = "../../../../Forced_Isotropic_1024Cubed_mat_in_one/$file_name.mat"
    u = load_velocity_1024(file_path_1024)
    u_perm = ordering(u, sp)
    @time sv = bond_singular_values(u_perm)
    
    matwrite("full_singular_values_$(file_name).mat", Dict("sv" => sv))
end
println("sizes of Σ for each bond: ", length.(sv))
println("Total elapsed time: ", total_time, " seconds");
