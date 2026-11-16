# COPY THIS CASE: synthetic by default; replace the loader only after approval.
input_path = nothing                 # Or an approved MAT file path.
variable = "u"                      # Explicit variable in that MAT file.
operator = :block                    # Explicit choice: :block or :periodic.
block_size = 2
radii = (1,0,0)
input_spacing = (1.0,1.0,1.0)         # Synthetic units; set approved dataset units.
output = isempty(ARGS) ? "les-case.mat" : only(ARGS) # Optional new output path.

using Sockets: gethostname
isempty(get(ENV, "SLURM_JOB_ID", "")) && error("use a compute allocation")
startswith(gethostname(), "login") && error("use a compute node")
using TurbulenceTools
using MAT
operator in (:block, :periodic) || error("choose :block or :periodic explicitly")
(ispath(output) || islink(output)) && error("output exists: $output")
# Dataset values must already have the approved dtype and axis order.
field = isnothing(input_path) ?
    [100.0*i + 10.0*j + k for i in 1:4, j in 1:4, k in 1:4] :
    matread(input_path)[variable]
filtered = operator == :block ? block_average(field; block_size=block_size) :
    periodic_box_smooth(field; radii=radii)
output_spacing = operator == :block ? block_size .* input_spacing : input_spacing
matwrite(output, Dict("u" => filtered, "operator" => string(operator),
    "input_spacing" => collect(input_spacing), "output_spacing" => collect(output_spacing),
    "block_size" => block_size, "radii" => collect(radii)))
println("output=", abspath(output), " shape=", size(filtered), " spacing=", output_spacing)
