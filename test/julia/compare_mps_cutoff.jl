# Compute-only regression check for the newly exposed cutoff-only API.
using Sockets: gethostname
using Test
isempty(get(ENV,"SLURM_JOB_ID","")) && error("use a compute allocation")
startswith(gethostname(),"login") && error("use a compute node")
using TurbulenceTools
const ROOT = normpath(joinpath(@__DIR__,"..",".."))
const BASELINE = "28149867b368f5b576fc41e82b4ba1425d38a21d"
source = read(`git -C $ROOT show $(BASELINE * ":Julia/0mps_calc.jl")`,String)
# Load only original definitions, without the fixed-size production CLI.
prefix = first(split(source,"\ntotal_time = @elapsed begin"; limit=2))
const Oracle = Module(:CutoffOracle)
Base.include_string(Oracle,prefix,"historical_mps_functions.jl")
@testset "cutoff-only historical regression" begin
    for u in (ones(Float64,4,4,4),reshape(Float64.(1:64),4,4,4))
        actual, bonds = compress_reconstruct(u;cutoff=1e-4,ordering=1)
        expected, expected_bonds = Oracle.mps_truncate(u;cutoff=1e-4,split=1)
        @test isequal(actual,expected)
        @test bonds == expected_bonds
    end
    u = ones(Float64,2,2,2)
    @test_throws ArgumentError compress_reconstruct(u;ordering=1)
    @test_throws ArgumentError compress_reconstruct(u;chi=1,cutoff=1e-4,ordering=1)
end
