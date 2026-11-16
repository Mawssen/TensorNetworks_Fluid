using ITensors, ITensorMPS
using LinearAlgebra
using MAT
using HDF5
using Plots
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


"""
    load_velocity_dat(file_path)

Load a jet-flame raw `.dat` field EXACTLY as dns_stats.load_field does, so the
cube fed to the MPS is byte-identical to what the Python pipeline analyzes:

  * on disk: big-endian float32 (">f4"), flat length nx*ny*nz
  * reshape to (nz, ny, nx) = (576, 1008, 864)          [C-order]
  * center-crop to n^3 = 512^3 in all three directions
  * transpose (2,1,0) -> (x, y, z)

Returns a Float64 (512,512,512) array.
"""
function load_velocity_dat(file_path::String;
                           nx::Int=864, ny::Int=1008, nz::Int=576, n::Int=512)
    raw = read(file_path)                       # raw bytes
    nfloats = nx * ny * nz
    @assert length(raw) == 4 * nfloats "Unexpected file size: got $(length(raw)) bytes, expected $(4*nfloats) (=4*$nx*$ny*$nz)"

    # Interpret as big-endian Float32, then convert to native.
    be = reinterpret(Float32, raw)              # length nfloats (still BE bytes)
    data_f32 = ntoh.(be)                         # big-endian -> host order

    # Python did np.fromfile(...).reshape((nz,ny,nx)) in C (row-major) order.
    # Julia is column-major, so to index the SAME element we build an array
    # whose Julia dims are reversed (nx,ny,nz) and read C-order data into it:
    # element (iz,iy,ix) in Python C-order == linear index
    #   iz*(ny*nx) + iy*nx + ix
    # We create A[ix,iy,iz] (column-major) by reshaping to (nx,ny,nz) since
    # column-major over (nx,ny,nz) has fastest index ix, matching C's fastest
    # index (last axis = nx). So reshape(data,(nx,ny,nz)) gives A[ix,iy,iz].
    A = reshape(Vector{Float64}(data_f32), (nx, ny, nz))   # A[ix,iy,iz]

    # Center-crop to n in each dimension (0-based sx in Python -> 1-based here)
    sx = div(nx, 2) - div(n, 2) + 1
    sy = div(ny, 2) - div(n, 2) + 1
    sz = div(nz, 2) - div(n, 2) + 1
    sub = A[sx:sx+n-1, sy:sy+n-1, sz:sz+n-1]     # (n,n,n) in (x,y,z)

    # Python returns transpose(sub,(2,1,0)) applied to a (nz,ny,nx) array,
    # yielding (x,y,z). Our `sub` is already (x,y,z) by construction above,
    # so no further permute is needed.
    return Array{Float64}(sub)
end


function perm_tuple_3D(n)
    a = Int[]  # Initialize an empty array of integers
    for i in 0:(3*n - 1)
        index = (i % 3 == 0) * div(i, 3) + 
                (i % 3 == 1) * (n + div(i + 1, 3)) + 
                (i % 3 == 2) * (2*n - 1 + div(i + 1, 3))+ 1
        push!(a, index) 
    end
    return tuple(a...)
end


function inv_perm_tuple_3D(n)
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


function mps_truncate(u::Array{Float64}; χ::Union{Int, Nothing} = nothing, cutoff::Union{Float64, Nothing} = nothing, split::Int)
    # Validate split argument
    if split != 1 && split != 2
        throw(ArgumentError("Invalid 'split' argument. Use 1 for interleaved, 2 for split."))
    end
    
    # Determine N based on the first dimension of u
    N = Int(log2(size(u, 1)));
    ndim = Int(ndims(u));  # The number of dimensions of u    
    size_u_split = ntuple(_ -> 2, ndim * N)
    
    if split == 1
        # Reshape u to match the desired qubit indices format: (2,2,2,...)
        u_reshaped = reshape(u, size_u_split);
        
        # Interleave x1x2x3 y1y2y3 z1z2z3 -> x1y1z1 x2y2z2 ...
        perm = perm_tuple_3D(N);
        u_perm = permutedims(u_reshaped, perm);
    
    elseif split == 2
        u_perm = u;  # No permutation, use original u
    end
    
    # Create Qubit indices for each dimension
    s = siteinds("Qubit", ndim * N)
    
    # Reshape u_perm into a 1D vector compatible with ITensors
    u_vector = reshape(u_perm, prod(size(u_perm)))

    # Perform MPS truncation based on the given cutoff or χ using explicit if-else
    if χ == nothing && cutoff == nothing
        ψ = MPS(u_vector, s)  # No truncation
            
    elseif χ == nothing && cutoff !== nothing
        ψ = MPS(u_vector, s, cutoff=cutoff)  # Truncate based on cutoff
            
    elseif χ !== nothing && cutoff == nothing
        ψ = MPS(u_vector, s, maxdim=χ)  # Truncate based on bond dimension χ
            
    else
        ψ = MPS(u_vector, s, cutoff=cutoff, maxdim=χ)  # Truncate based on both
            
    end
    
    chi_crit = linkdims(ψ)
        
    # Recover the original tensor from the MPS
    Big_ψ = contract(ψ)
    
    # Convert the ITensor back to a Julia array
    J_trunc = Array(Big_ψ, s)
    
    # Perform inverse permutation if the split was interleaved
    if split == 1
        inv_perm = inv_perm_tuple_3D(N);
        u_inv_perm = permutedims(J_trunc, inv_perm);
        u_rec = reshape(u_inv_perm, size(u));
            
    elseif split == 2
        u_rec = reshape(J_trunc, size(u));  # No permutation needed, reshape directly
    end
    
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

    
########################################################################
# Command-line argument parsing
#
# Supports `key=value` args. All are optional and fall back to defaults,
# so the script still runs as `julia mps_calc.jl` with no args.
#
# Examples:
#   julia mps_calc.jl chi=256
#   julia mps_calc.jl chi=400 file=U_t1 split=1 cutoff=0.0
#   julia mps_calc.jl chi=512 dir=../../../../Forced_Isotropic_1024Cubed_mat_in_one
#
function get_arg(key::String, default::String)
    for arg in ARGS
        if startswith(arg, "$(key)=")
            return split(arg, "=", limit=2)[2]
        end
    end
    return default
end

total_time = @elapsed begin
    # Parameters (override from the command line; otherwise use these defaults)
    chi       = parse(Int,     get_arg("chi",    "400"))
    cf        = parse(Float64, get_arg("cutoff", "0.0"))
    sp        = parse(Int,     get_arg("split",  "1"))   # 1 = interleaved, 2 = split
    file_name = get_arg("file", "U_t1")
    data_dir  = get_arg("dir",  "../../../../Forced_Isotropic_1024Cubed_mat_in_one")
    # Optional: file extension (default .mat) and a full path override. This
    # lets the jet velocity (converted to .mat by dat_to_mat.py) be read with
    # the same load_velocity_1024 loader, e.g.
    #   julia mps_calc.jl chi=9 file=jet_u_0198 dir=/path/to/jet_0198_mat
    ext       = get_arg("ext", "mat")
    filepath  = get_arg("filepath", "")

    println("=== mps_calc.jl ===")
    println("  chi    = $chi")
    println("  cutoff = $cf")
    println("  split  = $sp")
    println("  file   = $file_name")
    println("  dir    = $data_dir")
    println("===================")

    file_path_1024 = length(filepath) > 0 ? filepath :
                     joinpath(data_dir, "$(file_name).$(ext)")
    println("  loading: $file_path_1024")
    if endswith(file_path_1024, ".dat")
        u = load_velocity_dat(file_path_1024)
    else
        u = load_velocity_1024(file_path_1024)
    end
    # Print basic stats so the Julia-side load can be checked against Python's
    # dns_stats.load_field(...) (min/mean/max should match to float32 precision).
    println("  loaded cube: size=$(size(u))  min=$(minimum(u))  " *
            "mean=$(sum(u)/length(u))  max=$(maximum(u))")

    @time u_rec, chi_c = mps_truncate(u, χ=chi, cutoff=cf, split=sp)

    split_tag   = sp == 1 ? "il" : "sp"
    output_file = "truncated_$(file_name)_1024_$(split_tag)_chi$(chi).mat"
    @time save_to_mat(output_file, u_rec, chi_c)
    println("Output saved to: $output_file")
end

println("Total elapsed time: ", total_time, " seconds")
