#!/usr/bin/env julia
# Compute-node-only structural smoke check; no dataset or MAT write.
using Sockets: gethostname
startswith(first(split(gethostname(), '.')), "login") && error("run on a compute node")
isempty(get(ENV, "SLURM_JOB_ID", "")) && error("run within a compute allocation")
length(ARGS) == 2 || error("usage: mps_v1_smoke.jl <integer chi> <ordering code 1..4>")
chi = parse(Int, ARGS[1])
sp = parse(Int, ARGS[2])
sp in 1:4 || error("ordering must be 1..4")
include(joinpath(@__DIR__, "..", "Julia", "0mps_calc.jl"))
u = ones(Float32, 2, 2, 2)
reconstructed, bonds = mps_truncate(u; χ=chi, cutoff=0.0, split=sp)
@assert size(reconstructed) == size(u)
@assert length(bonds) == 2
@assert all(==(1), bonds)
@assert all(isfinite, reconstructed)
println("structural smoke passed: shape=", size(reconstructed),
        " dtype=", eltype(reconstructed), " chi_crit=", bonds)
