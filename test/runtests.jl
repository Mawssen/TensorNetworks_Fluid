# Run with julia --project=. test/runtests.jl inside a compute allocation.
# Each existing comparison gets its own namespace to isolate oracle modules.
using Sockets: gethostname
isempty(get(ENV,"SLURM_JOB_ID","")) && error("numerical tests require a compute allocation")
startswith(gethostname(),"login") && error("numerical tests require a compute node")

module MPSChiComparison
include("julia/compare_mps_api.jl")
end

module MPSCutoffComparison
include("julia/compare_mps_cutoff.jl")
end

module LESComparison
include("julia/compare_les_api.jl")
end
