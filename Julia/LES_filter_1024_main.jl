#LES_filter_1024

using MAT
include("LES_filter.jl")

file_name = "U_t1"

r_k = 1
r = (r_k,r_k,r_k) #how many neighbor points in each side r=(1,1,1) means (2r+1 x 2r+1 x 2r+1) = 3x3x3 points 
file_path_1024 = "../../../../Forced_Isotropic_1024Cubed_mat_in_one/$file_name.mat"
# file_path_1024 = "../Forced_Isotropic/02 Grid 128/$file_name.mat"
u = load_velocity_1024(file_path_1024)
# u = u .+ 0.8
u_f = box_filter_3d_sep(u, r)
println("relative norm = ", norm(u_f .- u) / norm(u))
# MAT.matwrite("filtered_$(file_name)_fbox_$(2*r[1]+1)x$(2*r[2]+1)x$(2*r[3]+1).mat", Dict("u" => u_f))

# u_coarse = box_filter_3d_sep(u, r; org_size = false)  # size Int(N/(2*rk+1))
# N_c = Int(floor(size(u)[1]/(2*r_k+1)))
# MAT.matwrite("filtered_$(file_name)_fbox_$(N_c)x$(N_c)x$(N_c).mat", Dict("u" => u_coarse))

r_total = mean(u.*u) - mean(u)^2
R = mean(u_f.*u_f) - mean(u_f)^2
tau = box_filter_3d_sep(u.*u, r) - u_f.*u_f;

println("r = ", r_total)
println("R = ", R)
println("tau = ", mean(tau))
println("r_total - R - tau = ", r_total - R - mean(tau))
