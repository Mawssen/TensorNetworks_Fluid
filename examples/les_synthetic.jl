# Explicit operator selection; synthetic parameters only, no dataset or file I/O.
# EDIT CASE VALUES HERE; operator remains an explicit required CLI argument.
block_size = 2
radii = (1,0,0)
block_shape = (4,4,4)
periodic_shape = (5,4,3)
using Sockets: gethostname
isempty(get(ENV, "SLURM_JOB_ID", "")) && error("use a compute allocation")
startswith(gethostname(), "login") && error("use a compute node")
using TurbulenceTools
length(ARGS) == 1 || error("usage: les_synthetic.jl <block|periodic>")
if ARGS[1] == "block"
    field = [100.0*i + 10.0*j + k for i in 1:block_shape[1], j in 1:block_shape[2], k in 1:block_shape[3]]
    filtered = block_average(field; block_size=block_size)
    println("block means (shape ", size(filtered), "): ", filtered)
elseif ARGS[1] == "periodic"
    field = zeros(Float64, periodic_shape)
    field[1,1,1] = 3.0
    filtered = periodic_box_smooth(field; radii=radii)
    println("periodic x line (wraparound at index 5): ", filtered[:,1,1])
else
    error("operator must be explicitly block or periodic")
end
