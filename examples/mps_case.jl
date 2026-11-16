# Two synthetic case recipes, sharing the identical API and output handling.
# EDIT CASE VALUES HERE (the CLI can still select either synthetic recipe).
case = length(ARGS) >= 1 ? ARGS[1] : "constant"
output = length(ARGS) >= 2 ? ARGS[2] : "$(case)-mps.mat"
input_path = nothing  # Set an approved binary path to use the existing loader.
full_shape, crop_size = (864,1008,576), 512
ordering = 1
constant_shape, constant_chi = (4,4,4), 1
pattern_shape, pattern_chi = (4,4,4), 3
# For a binary dataset, replace the synthetic field below with:
# field = load_centered_be_float32(input_path; full_shape=approved_shape,
#                                  crop_size=approved_crop)
using Sockets: gethostname
isempty(get(ENV, "SLURM_JOB_ID", "")) && error("use a compute allocation")
startswith(gethostname(), "login") && error("use a compute node")
using TurbulenceTools
length(ARGS) <= 2 || error("usage: mps_case.jl [constant|pattern] [new-output.mat]")
if input_path !== nothing
    field = load_centered_be_float32(input_path; full_shape=full_shape, crop_size=crop_size)
    chi = constant_chi  # Set the approved dataset cap in the editable block above.
elseif case == "constant"
    field, chi = ones(Float32, constant_shape), constant_chi
elseif case == "pattern"
    field, chi = reshape(Float32.(1:prod(pattern_shape)), pattern_shape), pattern_chi
else
    error("unknown synthetic case: $case")
end
reconstructed, bonds = compress_reconstruct(field; chi=chi, ordering=ordering)
write_mps_mat(output, reconstructed, bonds)
println("output=", abspath(output), " shape=", size(reconstructed), " bonds=", bonds)
