# Lightweight numerical comparison, to be run ONLY in a compute allocation.
using Sockets: gethostname
using Test
isempty(get(ENV, "SLURM_JOB_ID", "")) && error("use a compute allocation")
startswith(gethostname(), "login") && error("use a compute node")
const ROOT = normpath(joinpath(@__DIR__, ".."))
using TurbulenceTools
using MAT

# Freeze the pre-refactor CLI from the base commit, not the modified worktree.
const BASELINE = "28149867b368f5b576fc41e82b4ba1425d38a21d"
baseline = read(`git -C $ROOT show $(BASELINE * ":Julia/0mps_calc.jl")`, String)
println("Historical oracle: ", BASELINE)
# Only test dimensions change: the historical entry point cannot take tiny files.
# Retain all decoding/crop/ordering/truncation/output expressions verbatim.
for (old, new) in (("    nx = 864", "    nx = 6"),
                   ("    ny = 1008", "    ny = 8"),
                   ("    nz = 576", "    nz = 10"),
                   ("    ntn = 512", "    ntn = 4"))
    active_line = Regex("(?m)^" * old * "\$")
    length(findall(active_line, baseline)) == 1 || error("historical oracle changed: $old")
    global baseline = replace(baseline, active_line => new)
end

mktempdir() do dir
    input = joinpath(dir, "synthetic_0007.dat")
    full = reshape(Float32.(1:480), 6, 8, 10)
    open(input, "w") do io
        write(io, hton.(reinterpret(UInt32, vec(full))))
    end
    field = load_centered_be_float32(input; full_shape=(6,8,10), crop_size=4)
    @test field == full[2:5,3:6,4:7]
    @test size(field) == (4,4,4)
    @test eltype(field) == Float32
    for ordering in 1:4, chi in (1,3)
        case_dir = joinpath(dir, "$(ordering)_$(chi)")
        mkdir(case_dir)
        oracle = Module(gensym(:HistoricalOracle))
        Core.eval(oracle, :(const ARGS = $([input, "chi=$chi", string(ordering)])))
        cd(case_dir) do
            Base.include_string(oracle, baseline, "historical_oracle.jl")
        end
        tag = ("il", "seq", "comb1", "combn")[ordering]
        historical = matread(joinpath(case_dir, "truncated_0007",
            "truncated_synthetic_0007.dat_512_$(tag)_chi$(chi).mat"))
        rec, bonds = compress_reconstruct(field; chi=chi, ordering=ordering)
        # Exact equality is a regression probe, not an invented tolerance.
        # A mismatch needs review; no automatic approximate fallback.
        @test isequal(rec, historical["u"])
        @test vec(historical["chi_crit"]) == bonds
        @test all(bonds .<= chi)
        output = joinpath(case_dir, "api.mat")
        write_mps_mat(output, rec, bonds)
        @test isequal(matread(output)["u"], historical["u"])
        @test_throws ArgumentError write_mps_mat(output, rec, bonds)
    end
end
println("Historical CLI/API exact comparison passed (synthetic dimensions only)")
