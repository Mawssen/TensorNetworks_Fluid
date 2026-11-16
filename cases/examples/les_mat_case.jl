# Adapt this case's loader after confirming dataset axes, units and coordinates.
# MAT variable must ALREADY be a Float64 array in the approved axis order.
length(ARGS) == 8 || error("usage: les_mat_case.jl <input.mat> <variable> <block|periodic> <delta|rx,ry,rz> <dx> <dy> <dz> <new-output.mat>")
# EDIT CASE VALUES HERE, or supply all eight values on the command line.
input, variable, operator, parameter, dx, dy, dz, output = ARGS
spacing = parse.(Float64, [dx,dy,dz])
using Sockets: gethostname
isempty(get(ENV, "SLURM_JOB_ID", "")) && error("use a compute allocation")
startswith(gethostname(), "login") && error("use a compute node")
using MAT
using TurbulenceTools
operator in ("block", "periodic") || error("choose block or periodic explicitly")
(ispath(output) || islink(output)) && error("output exists: $output")
all(x -> isfinite(x) && x > 0, spacing) || error("spacings must be finite and positive")
field = matread(input)[variable]  # No implicit dtype conversion or axis permutation.
if operator == "block"
    delta = parse(Int, parameter)
    filtered = block_average(field; block_size=delta)
    output_spacing = delta .* spacing
else
    radii = Tuple(parse.(Int, split(parameter, ',')))
    length(radii) == 3 || error("periodic parameter must be rx,ry,rz")
    filtered = periodic_box_smooth(field; radii=radii)
    output_spacing = spacing
end
matwrite(output, Dict("u" => filtered, "input_spacing" => spacing,
    "output_spacing" => output_spacing, "operator" => operator,
    "parameter" => parameter))
println("output=", abspath(output), " shape=", size(filtered), " spacing=", output_spacing)
