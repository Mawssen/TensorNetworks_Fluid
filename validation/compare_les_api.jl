# Tiny synthetic numerical checks; execute inside an existing compute allocation.
using Sockets: gethostname
using Test
isempty(get(ENV, "SLURM_JOB_ID", "")) && error("use a compute allocation")
startswith(gethostname(), "login") && error("use a compute node")
const ROOT = normpath(joinpath(@__DIR__, ".."))
using TurbulenceTools

# Independent pre-extraction implementations from the current base commit.
const BASELINE = "28149867b368f5b576fc41e82b4ba1425d38a21d"
block_source = read(`git -C $ROOT show $(BASELINE * ":LES_interface/LES_filter.jl")`, String)
periodic_source = read(`git -C $ROOT show $(BASELINE * ":Julia/LES_filter.jl")`, String)
const BlockOracle = Module(:BlockOracle)
Base.include_string(BlockOracle, block_source, "historical_block.jl")
# Include only the literal direct kernel; the historical file's plotting import
# is unrelated and should not force Plots installation for a numerical check.
direct = match(r"(?ms)^function box_filter_3d\(.*?^end$", periodic_source)
isnothing(direct) && error("historical periodic function not found")
const PeriodicOracle = Module(:PeriodicOracle)
Base.include_string(PeriodicOracle, direct.match, "historical_periodic.jl")
println("LES oracle commit: ", BASELINE)

@testset "disjoint block averages" begin
    u = [100.0*i + 10.0*j + k for i in 1:4, j in 1:4, k in 1:4]
    untouched = copy(u)
    result = block_average(u; block_size=2)
    expected = [100.0*(2I-0.5) + 10.0*(2J-0.5) + (2K-0.5)
                for I in 1:2, J in 1:2, K in 1:2]
    @test size(result) == (2,2,2)
    @test result[1,1,1] == 166.5
    @test result == expected
    @test isequal(result, BlockOracle.les_box_filter(u, 2))
    @test result == BlockOracle.les_box_filter_naive(u, 2)
    @test block_average(u; block_size=1) == u
    @test u == untouched
    @test_throws AssertionError block_average(u; block_size=3)
    @test_throws ArgumentError block_average(u; block_size=0)
end

@testset "periodic centered smoothing" begin
    u = zeros(Float64, 5,4,3)
    u[1,1,1] = 3.0
    untouched = copy(u)
    result = periodic_box_smooth(u; radii=(1,0,0))
    @test size(result) == size(u)
    @test result[:,1,1] == [1.0,1.0,0.0,0.0,1.0]
    @test sum(result) == 3.0
    @test isequal(result, PeriodicOracle.box_filter_3d(u, (1,0,0)))
    # Wraparound on all axes at once, including the far corner.
    u[1,1,1] = 27.0
    corner = periodic_box_smooth(u; radii=(1,1,1))
    @test corner[5,4,3] == 1.0
    @test isequal(corner, PeriodicOracle.box_filter_3d(u, (1,1,1)))
    u[1,1,1] = 3.0
    @test u == untouched
    @test periodic_box_smooth(u; radii=(0,0,0)) == u
    @test_throws ArgumentError periodic_box_smooth(u; radii=(-1,0,0))
end

module LegacyBlock
include(joinpath(@__DIR__, "..", "LES_interface", "LES_filter.jl"))
end

@testset "historical block entry points still callable" begin
    legacy = LegacyBlock
    u = reshape(Float64.(1:64), 4,4,4)
    @test isequal(legacy.les_box_filter(u,2), block_average(u; block_size=2))
    @test legacy.les_box_filter_vector(u,u,u,2) ==
        (block_average(u; block_size=2), block_average(u; block_size=2), block_average(u; block_size=2))
end
