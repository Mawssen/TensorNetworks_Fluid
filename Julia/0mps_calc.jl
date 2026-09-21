using ITensors, ITensorMPS
using LinearAlgebra
using MAT
using HDF5
using Random
using JSON
using Printf
#using Plots
#using PyPlot

using MKL 
BLAS.set_num_threads(10)  
ITensors.set_warn_order(40) 


function mutual_info_two_sites(psi::AbstractVector{<:AbstractFloat}, i::Int, j::Int, d::Int=2)
    """
    Calculate mutual information between sites i and j in state psi (real numbers)
    """
    N = Int(round(log(d, length(psi))))
    @assert 1 <= i <= N && 1 <= j <= N "Site indices out of bounds"
    
    if i == j
        # For diagonal: I(i:i) = 2*S(i) - S(ii) = S(i) since S(ii) = S(i) for single site
        rho_i = reduced_density_matrix(psi, [i], d, N)
        S_i = von_neumann_entropy(rho_i)
        return S_i
    else
        # For off-diagonal: I(i:j) = S_i + S_j - S_ij
        rho_ij = reduced_density_matrix(psi, [i,j], d, N)
        rho_i = reduced_density_matrix(psi, [i], d, N) 
        rho_j = reduced_density_matrix(psi, [j], d, N)
        
        S_ij = von_neumann_entropy(rho_ij)
        S_i = von_neumann_entropy(rho_i)
        S_j = von_neumann_entropy(rho_j)
        
        return S_i + S_j - S_ij
    end
end

function von_neumann_entropy(rho::AbstractMatrix{<:AbstractFloat})
    """Calculate von Neumann entropy S = -Tr(ρ log ρ) for real matrices"""
    # Ensure symmetric for numerical stability
    ρ_sym = Symmetric(rho)
    eigvals = eigen(ρ_sym).values
    # Remove numerically zero eigenvalues and avoid log(0)
    eigvals = eigvals[eigvals .> 1e-12]
    if isempty(eigvals)
        return 0.0
    end
    entropy = -sum(eigvals .* log2.(eigvals))
    return entropy
end

function reduced_density_matrix(psi::AbstractVector{<:AbstractFloat}, sites::Vector{Int}, d::Int, N::Int)
    """
    Compute reduced density matrix for given sites by tracing out others
    """
    keep_dims = d ^ length(sites)
    traced_dims = d ^ (N - length(sites))
    
    # Reshape state into tensor form
    dims_tuple = ntuple(i -> d, N)
    psi_tensor = reshape(psi, dims_tuple)
    
    # Create permutation: kept sites first, then traced sites
    perm = [sites..., setdiff(1:N, sites)...]
    psi_permuted = permutedims(psi_tensor, perm)
    
    # Reshape and compute reduced density matrix
    psi_reshaped = reshape(psi_permuted, keep_dims, traced_dims)
    rho = psi_reshaped * psi_reshaped'
    
    return rho
end

function mutual_info_all(psi::AbstractVector{<:AbstractFloat}, d::Int=2)
    """
    Calculate mutual information between ALL pairs of sites (including diagonal)
    """
    psi ./= norm(psi)
    N = Int(round(log(d, length(psi))))
    corr_matrix = zeros(N, N)
    
    for i in 1:N
        for j in 1:N
            corr_matrix[i,j] = mutual_info_two_sites(psi, i, j, d)
        end
    end
    return corr_matrix
end


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


# --- helpers ---
# Guard for supported dimensionalities
_check_d(d) = (d==2 || d==3) || error("Only d=2 or d=3 supported")

# Build block start indices for d blocks of length n
_block_starts(n, d) = [(k-1)*n + 1 for k in 1:d]

# Interleave by picking index i from each block in order
function _interleave_indices(n, d)
    starts = _block_starts(n, d)
    v = Int[]
    for i in 0:n-1
        for b in 1:d
            push!(v, starts[b] + i)
        end
    end
    return v
end

# Inverse of the above interleaving: positions of each block in the interleaved array
function _inverse_interleave_indices(n, d)
    v = Int[]
    for b in 1:d
        for i in 1:n
            push!(v, d*(i-1) + b)
        end
    end
    return v
end

# --- 1) interleaved permutation ---
function perm_tuple(n::Int, d::Int=3)  # interleaved permutation
    _check_d(d)
    return Tuple(_interleave_indices(n, d))
end

# --- 2) inverse interleaved permutation ---
function inv_perm_tuple(n::Int, d::Int=3)  # interleaved inverse permutation
    _check_d(d)
    return Tuple(_inverse_interleave_indices(n, d))
end

# --- 3) contiguous+split permutation ---
function cont_split_perm(N::Int, d::Int=3)
    _check_d(d)
    if N < 1
        return ()
    end
    if d == 3
        part1 = collect(1:N)
        part2 = collect(2N:-1:N+1)      # middle block reversed
        part3 = collect(2N+1:3N)
        return Tuple(vcat(part1, part2, part3))
    else # d == 2
        part1 = collect(1:N)
        part2 = collect(2N:-1:N+1)      # second block reversed
        return Tuple(vcat(part1, part2))
    end
end

# --- 4) "cont_split_reverse" contiguous+split reverse ---
function cont_split_reverse_perm(n::Int, d::Int=3)
    _check_d(d)  # helper from earlier
    if n < 1
        return ()
    end
    # build blocks
    blocks = [collect((k-1)*n+1 : k*n) for k in 1:d]

    if d == 2
        blocks[1] = reverse(blocks[1])            # reverse first block only
    elseif d == 3
        blocks[1] = reverse(blocks[1])            # reverse first block
        blocks[end] = reverse(blocks[end])        # reverse last block
    end

    return Tuple(vcat(blocks...))
end


# --- 5) "n_middle" interleave pattern ---
function n_middle_perm(n::Int, d::Int=3)
    _check_d(d)
    # indices per block
    blocks = [collect((k-1)*n+1 : k*n) for k in 1:d]

    # order of the i=1..n positions: 1,3,5,...,n,(n-2),(n-4),...,2  (with correct parity handling)
    order = if n % 2 == 0
        vcat(1:2:n-1, n, (n-2):-2:2)
    else
        vcat(1:2:n-2, n, (n-1):-2:2)
    end

    # flatten (for each i in order, take the i-th element from each block in block order)
    out = Int[]
    for i in order
        for b in 1:d
            push!(out, blocks[b][i])
        end
    end
    return Tuple(out)
end

function inv_n_middle_perm(n::Int, d::Int=3)
    perm = n_middle_perm(n, d)
    invp = Vector{Int}(undef, length(perm))
    for (i, p) in enumerate(perm)
        invp[p] = i
    end
    return Tuple(invp)
end

# --- 6) "one_middle" pattern (centered at i=1 with left evens reversed, right odds ascending) ---
function one_middle_perm(n::Int, d::Int=3)
    _check_d(d)
    blocks = [collect((k-1)*n+1 : k*n) for k in 1:d]

    left_indices  = collect(reverse(2:2:n))  # evens down to 2
    center_index  = 1
    right_indices = collect(3:2:n)           # odds from 3 up
    order = vcat(left_indices, center_index, right_indices)

    out = Int[]
    for i in order
        for b in 1:d
            push!(out, blocks[b][i])
        end
    end
    return Tuple(out)
end

function inv_one_middle_perm(n::Int, d::Int=3)
    perm = one_middle_perm(n, d)
    invp = Vector{Int}(undef, length(perm))
    for (i, p) in enumerate(perm)
        invp[p] = i
    end
    return Tuple(invp)
end

function ordering(u::AbstractArray{<:AbstractFloat}, split::Int)
    N = Int(log2(size(u, 1)));
    ndim = Int(ndims(u));  #The number of dimensions of u    
    size_u_split = ntuple(_ -> 2, ndim * N) #the desired qubit indices format: (2,2,2,...)
    
    if split == 2 #2 = split: x1x2x3 y1y2y3 z1z2z3
        u_perm = u;  # No permutation, use original u
    else
        u_reshaped = reshape(u, size_u_split);
        if split == 1  #1 = interleaved: x1x2x3 y1y2y3 z1z2z3 -> x1y1z1 x2y2z2 ...
            perm = perm_tuple(N, ndim);
        elseif split == 3 #3 = continuous split: x1x2x3 y1y2y3 z1z2z3 -> x1x2x3 y3y2y1 z1z2z3
            perm = cont_split_perm(N, ndim)
        elseif split == 4 #4 = cont_split_reverse: x1x2x3 y1y2y3 z1z2z3 -> x3x2x1 y1y2y3 z3z2z1
            perm = cont_split_reverse_perm(N, ndim)
        elseif split == 5 #5 = n_middle: x1x2x3 y1y2y3 z1z2z3 -> x1y1z1 x3y3z3 x2y2z2
            perm = n_middle_perm(N, ndim)
        elseif split == 6 #6 = one_middle: x1x2x3 y1y2y3 z1z2z3 -> x2y2z2 x1y1z1 x3y3z3
            perm = one_middle_perm(N, ndim)
        end
        u_perm = permutedims(u_reshaped, perm);
    end
end


function inv_ordering(J_trunc::AbstractArray{<:AbstractFloat}, split::Int, size_u, ndim)
    N = Int(log2(size_u[1]));
    
    if split == 2 #from split to u
        u_rec = reshape(J_trunc, size_u);  # No permutation needed, reshape directly
    
    else
        if split == 1     #from interleaved to u
            inv_perm = inv_perm_tuple(N, ndim);
        elseif split == 3 #from continuous split to u
            inv_perm = cont_split_perm(N, ndim) #inv_perm = perm            
        elseif split == 4 #from cont_split_reverse to u
            inv_perm = cont_split_reverse_perm(N, ndim)
        elseif split == 5 #from n_middle to u
            inv_perm = inv_n_middle_perm(N, ndim)
        elseif split == 6 #from one_middle to u
            inv_perm = inv_one_middle_perm(N, ndim)
        end
        u_inv_perm = permutedims(J_trunc, inv_perm);
        u_rec = reshape(u_inv_perm, size_u);
    end
end


function mps_truncate(u::AbstractArray{<:AbstractFloat}; χ::Union{Int, Nothing} = nothing,
                      cutoff::Union{Float64, Nothing} = nothing, split::Int, mutual_info::Bool = false)
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
    u_rec = inv_ordering(J_trunc, split, size(u), ndim)
    
    if mutual_info == true
        # MI = mutual_information_all(ψ)
        MI = mutual_info_all(vec(J_trunc))
        return u_rec, chi_crit, MI
    else
        return u_rec, chi_crit
    end
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
        println(svals[step])
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



function make_labels(sp::Int, N::Int, dim::Int)
    # label names by dimension
    dims = if dim == 2
        ["x","y"]
    elseif dim == 3
        ["x","y","z"]
    else
        error("Only dim=2 or dim=3 supported")
    end

    labels = String[]

    # -------- shared helpers --------
    up()   = 1:N
    down() = N:-1:1

    # Interleaved across dimensions: x1,y1,(z1), x2,y2,(z2), ...
    interleaved!() = begin
        for i in 1:N
            for d in 1:dim
                push!(labels, "$(dims[d])$(i)")
            end
        end
    end

    # Block by dimension with per-dimension direction
    # dirs is a Vector of functions: [up, down, up] etc.
    block_by_dim!(dirs) = begin
        @assert length(dirs) == dim
        for d in 1:dim
            rng = dirs[d]()
            append!(labels, ["$(dims[d])$(i)" for i in rng])
        end
    end

    # -------- cases --------
    if dim == 2
        if sp == 1
            interleaved!()                                  # x1,y1,x2,y2,...
        elseif sp == 2
            block_by_dim!([up, up])                         # x 1:N, then y 1:N
        elseif sp == 3
            block_by_dim!([up, down])                       # x 1:N, then y N:1
        elseif sp == 4
            block_by_dim!([down, up])                       # x N:1, then y 1:N
        elseif sp == 5
            # center xN,yN ; N-1,N-3,... to left; N-2,N-4,... to right
            left  = String[]; right = String[]
            middle = ["x$(N)","y$(N)"]
            for i in (N-1):-1:1
                if isodd(N - i)  # alternate left/right
                    append!(left,  ["x$(i)","y$(i)"])
                else
                    append!(right, ["x$(i)","y$(i)"])
                end
            end
            labels = vcat(reverse(left), middle, right)
        elseif sp == 6
            # center x1,y1 ; evens to left; odds>1 to right
            left  = String[]; right = String[]
            middle = ["x1","y1"]
            for i in 2:N
                if iseven(i)
                    append!(left,  ["x$(i)","y$(i)"])
                else
                    append!(right, ["x$(i)","y$(i)"])
                end
            end
            labels = vcat(reverse(left), middle, right)
        else
            error("Unknown sp=$sp for dim=2")
        end

    elseif dim == 3
        if sp == 1
            interleaved!()                                  # x1,y1,z1, x2,y2,z2, ...
        elseif sp == 2
            block_by_dim!([up, up, up])                     # x 1:N, y 1:N, z 1:N
        elseif sp == 3
            block_by_dim!([up, down, up])                   # x 1:N, y N:1, z 1:N
        elseif sp == 4
            block_by_dim!([down, up, down])                 # x N:1, y 1:N, z N:1
        else
            error("For dim=3, only sp=1..4 defined by your spec")
        end
    end

    return labels
end


########################################################################

# JHS .mat data
    
# total_time = @elapsed begin
#     cf = 1e-2  # Cutoff value for MPS truncation
#     sp = 1    # Split mode: 1 = interleaved, 2 = split, 3 = continuous split, 4 = n_middle, 5 = one_middle
#     file_name = "U_t1"

#     file_path_1024 = "../Forced_Isotropic/02 Grid 128/$file_name.mat"
#     # file_path_1024 = "../Massen_Lib/u_odd.mat"
#     # file_path_1024 = "../../../../Forced_Isotropic_512Cubed_WholeDomain_mat/2Even/$file_name.mat"
#     # file_path_1024 = "../../../../../Forced_Isotropic_1024Cubed_mat_in_one/$file_name.mat"
#     u = load_velocity_1024(file_path_1024)
    
#     # file_path_512 = "../Forced_Isotropic/01 Grid 64/Forced_Isotropic_64Cubed_mat/U_t1.mat"
#     # file_path_512 = "../../Forced_Isotropic_512Cubed_SubDomain_mat/U_t1.mat"
#     # vel = load_velocity_1024(file_path_512)
#     # u = vel[:, :, :, 1]

#     @time u_rec, chi_c = mps_truncate(u, cutoff=cf, split=sp) 
#     output_file = "truncated_$(file_name)_512_il_cf$(cf).mat"
#     save_to_mat(output_file, u_rec, chi_c)
# end

# println("Total elapsed time: ", total_time, " seconds");


######################################################################## 
#Daniel .bin data
    
# total_time = @elapsed begin
#     cf = 1e-6  # Cutoff value for MPS truncation
#     # chi = 333
#     sp = 1     # Split mode: 1 = interleaved, 2 = split, 3 = c-split(comb1), 4 = c-split-rev(comb2), 5 = n_middle, 6 = 1_middle
#     file_name = "u"
#     # N = 4096

#     # if N == 32
#     #     dataset = "32_binary_dns"
#     #     config_path = "../Massen_Lib/config_FI.json"
#     # elseif N == 1024
#     #     dataset = "full_binary_dns"
#     #     config_path = joinpath(mps_codes_path, "config_FI.json")
#     # end

#     # config = open(config_path, "r") do f
#     #     JSON.parse(read(f, String))
#     # end

#     # org_path = get(config, dataset, Dict())
#     # shape = (N, N, N)

#     # skip_byte = file_name == "U" ? 192 : 0

#     # @time u = load_velocity_binary(org_path["vel"][file_name], shape, 0)
# #     @time u = load_velocity_binary(org_path[file_name], shape, skip_byte)
    

#     println("✅ u shape: ", isnothing(u) ? "Error" : size(u));

#     @time u_rec, chi_c = mps_truncate(u, cutoff=cf, split=sp)

#     output_file = "truncated_$(file_name)_1024_sp_cf$(cf).mat"
#     save_to_mat(output_file, u_rec, chi_c)
# end

# println("\nTotal elapsed time: ", total_time, " seconds")

##############################################################################

# #JHS singular values

# total_time = @elapsed begin
#     sp = 1    # Split mode: 1 = interleaved, 2 = split, 3 = continuous split, 4 = n_middle, 5 = one_middle
#     file_name = "V_t1"
#     file_path_1024 = "../Forced_Isotropic/02 Grid 128/$file_name.mat"
#     # file_path_1024 = "../../../../../Forced_Isotropic_1024Cubed_mat_in_one/$file_name.mat"
#     u = load_velocity_1024(file_path_1024)
#     u_perm = ordering(u, sp)
#     @time sv = bond_singular_values(u_perm)
    
#     matwrite("full_singular_values_$(file_name)_$(sp).mat", Dict("sv" => sv))
# end
# println("sizes of Σ for each bond: ", length.(sv))
# println("Total elapsed time: ", total_time, " seconds");



##############################################################################
#Mutual information

# cf = 1e-3
# n_sp = 1    # Split mode: 1 = interleaved, 2 = split, 3 = c-split(comb1), 4 = c-split-rev(comb2), 5 = n_middle, 6 = 1_middle,
# file_name = "U_t1"
# file_path_1024 = "../Forced_Isotropic/02 Grid 128/$file_name.mat"
# # file_path_1024 = "../../../../../Forced_Isotropic_1024Cubed_mat_in_one/$file_name.mat"
# u = load_velocity_1024(file_path_1024)
# d  = Int(ndims(u))

# # Define split mode names for clarity
# sp_names = [
#     "interleaved",
#     "split",
#     "c-split(comb1)",
#     "c-split-rev(comb2)",
#     "n_middle",
#     "1_middle"
# ]

# for sp in n_sp:n_sp
#     # Compute truncated MPS and mutual information
#     mi_time = @elapsed begin
#         u_rec, chi_c, MIu = mps_truncate(u, cutoff=cf, split=sp, mutual_info=true)
#     end
#     println("MI time sp$(sp)_$(sp_names[sp])_cf_$(cf): ", mi_time, " seconds");
#     # Determine N and label names
#     N = Int(size(MIu, 2) / d)
#     labels = make_labels(sp, N, d)

#     # Make heatmap title (LaTeX + text)
#     title_str = "\$U\\ \\mathrm{MI}\$ — " * sp_names[sp]

#     # Create heatmap
#     p = heatmap(MIu,
#         xlabel = "site",
#         ylabel = "site",
#         title  = title_str,
#         size = (1200,1200),
#         xticks = (1:d*N, labels),
#         yticks = (1:d*N, labels),
#         colormap = :magma,
#         right_margin = 5Plots.mm
#     )

#     # Add text annotations (rounded to 2 decimals)
#     for i in 1:(d*N)
#         for j in 1:(d*N)
#             val = round(MIu[i, j]; digits=2)
#             annotate!(p, j, i, text(string(val), 6, :white, :center))
#         end
#     end

#     matwrite("MI_U_sp$(sp)_$(sp_names[sp])_cf_$(cf).mat", Dict("MI" => MIu))
#     # Save figure
#     savefig(p, "MI_U_sp$(sp)_$(sp_names[sp])_cf_$(cf).png")
#     println("Saved MI_U_sp$(sp)_$(sp_names[sp])_cf_$(cf).png")
# end

##############################################################################

# total_time = @elapsed begin
#     input_path = ARGS[1]
#     cf         = parse(Float64, ARGS[2])
#     sp         = parse(Int,     ARGS[3])
#     filename   = basename(input_path)
#     timestep   = match(r"_(\d+)\.dat$", filename).captures[1]   
#     output_dir = "truncated_$(timestep)"
#     mkpath(output_dir)                                          
#     nx = 864
#     ny = 1008
#     nz = 576
#     ntn = 512
#     n = nx * ny * nz

#     open(input_path, "r") do f
#         raw = read(f, 4 * n)
#         data_u32 = reinterpret(UInt32, raw)
#         data = reinterpret(Float32, ntoh.(data_u32))
#         raw = nothing
#         data_grid = reshape(data, nx, ny, nz)

#         start_y = div(ny, 2) - div(ntn, 2) + 1
#         start_x = div(nx, 2) - div(ntn, 2) + 1
#         start_z = div(nz, 2) - div(ntn, 2) + 1

#         u = data_grid[start_x:start_x+ntn-1, start_y:start_y+ntn-1, start_z:start_z+ntn-1]
#         data_grid = nothing
#         GC.gc()

#         @time u_rec, chi_c = mps_truncate(u, cutoff=cf, split=sp)

#         tag = sp == 1 ? "il"    :
#               sp == 2 ? "seq"   :
#               sp == 3 ? "comb1" :
#               sp == 4 ? "combn" :
#               error("Unknown sp=$sp; valid: 1, 2, 3, 4")
#         output_file = joinpath(output_dir,
#                                "truncated_$(filename)_512_$(tag)_cf$(cf).mat")
#         save_to_mat(output_file, u_rec, chi_c)
#     end
# end

# println("Total elapsed time: ", total_time, " seconds")




##############################################################################

total_time = @elapsed begin
    input_path = ARGS[1]
    mode_arg   = ARGS[2]   # either a cutoff like "3e-4" OR a chi spec like "chi=512"
    sp         = parse(Int, ARGS[3])

    # Decide whether we got a cutoff or a chi (bond-dim) target.
    use_chi   = startswith(lowercase(mode_arg), "chi=")
    chi_val   = nothing
    cf_val    = nothing
    if use_chi
        chi_val = parse(Int, split(mode_arg, "=")[2])
    else
        cf_val = parse(Float64, mode_arg)
    end

    filename   = basename(input_path)
    timestep   = match(r"_(\d+)\.dat$", filename).captures[1]
    output_dir = "truncated_$(timestep)"
    mkpath(output_dir)
    nx = 864
    ny = 1008
    nz = 576
    ntn = 512
    n = nx * ny * nz

    open(input_path, "r") do f
        raw = read(f, 4 * n)
        data_u32 = reinterpret(UInt32, raw)
        data = reinterpret(Float32, ntoh.(data_u32))
        raw = nothing
        data_grid = reshape(data, nx, ny, nz)

        start_y = div(ny, 2) - div(ntn, 2) + 1
        start_x = div(nx, 2) - div(ntn, 2) + 1
        start_z = div(nz, 2) - div(ntn, 2) + 1

        u = data_grid[start_x:start_x+ntn-1, start_y:start_y+ntn-1, start_z:start_z+ntn-1]
        data_grid = nothing
        GC.gc()

        if use_chi
            @time u_rec, chi_c = mps_truncate(u, χ=chi_val, cutoff=0.0, split=sp)
        else
            @time u_rec, chi_c = mps_truncate(u, cutoff=cf_val, split=sp)
        end

        tag = sp == 1 ? "il"    :
              sp == 2 ? "seq"   :
              sp == 3 ? "comb1" :
              sp == 4 ? "combn" :
              error("Unknown sp=$sp; valid: 1, 2, 3, 4")

        # Filename: cf<value> for cutoff mode, chi<value> for chi mode.
        if use_chi
            label = "chi$(chi_val)"
        else
            label = "cf$(cf_val)"
        end
        output_file = joinpath(output_dir,
                               "truncated_$(filename)_512_$(tag)_$(label).mat")
        save_to_mat(output_file, u_rec, chi_c)
    end
end

println("Total elapsed time: ", total_time, " seconds")
