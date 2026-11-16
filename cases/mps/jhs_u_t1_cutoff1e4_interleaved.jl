# Approved scientific values. CRC resources are selected separately.
input_path = "/ix/pgivi/moe32/Schmidt/Forced_Isotropic/5_Grid_1024/Forced_Isotropic_1024Cubed_mat_in_one/U_t1.mat"
variable = "U1"
expected_shape = (1024,1024,1024)
cutoff = 1e-4
ordering = 1

using Dates
using TOML
using Sockets: gethostname
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
run_id = Dates.format(now(UTC), "yyyymmddTHHMMSSsss") * "-" * string(getpid())
run_directory = get(ENV, "TURBULENCE_RUN_DIRECTORY",
    joinpath(ROOT, "artifacts", "runs", "mps", "jhs_u_t1", "1024cubed",
        "interleaved", "cutoff_1e-4", run_id))
output_path = joinpath(run_directory, "truncated_U_t1_1024_il_cf$(cutoff).mat")
record = Dict{String,Any}("input_path"=>input_path, "variable"=>variable,
    "shape"=>collect(expected_shape), "dtype"=>"Float64", "crop"=>"none",
    "axes"=>"preserve historical MAT.jl loader order; no permutation",
    "normalization"=>"none", "cutoff"=>cutoff, "ordering"=>ordering,
    "run_directory"=>run_directory, "output_path"=>output_path,
    "host"=>gethostname(), "slurm_job_id"=>get(ENV,"SLURM_JOB_ID","unset"),
    "state"=>"PLANNED")
if ARGS == ["--plan"]
    TOML.print(stdout, record; sorted=true)
    exit(0)
end
isempty(ARGS) || error("usage: julia --project=. $PROGRAM_FILE [--plan]")
isempty(get(ENV,"SLURM_JOB_ID","")) && error("run inside an approved compute allocation")
startswith(gethostname(), "login") && error("numerical runs require a compute node")

using TurbulenceTools
using MAT
mkpath(dirname(run_directory))
mkdir(run_directory)  # Exclusive leaf: never overwrite another run.
function save_record(record)
    open(joinpath(run_directory,"case.toml"), "w") do io
        TOML.print(io, record; sorted=true)
    end
end
record["state"] = "RUNNING"
record["git_commit"] = readchomp(`git -C $ROOT rev-parse HEAD`)
record["git_status"] = readchomp(`git -C $ROOT status --porcelain`)
save_record(record)
open(joinpath(run_directory,"run.log"), "w") do log
    redirect_stdout(log) do
        redirect_stderr(log) do
            started = time()
            try
                println("Loading full JHS field: ", input_path); flush(log)
                # Same MAT read and Float64 conversion as load_velocity_1024.
                field = matopen(input_path) do file
                    Float64.(read(file, variable))
                end
                size(field) == expected_shape || error("unexpected input shape: $(size(field))")
                println("Compressing: shape=",size(field)," cutoff=",cutoff," split=",ordering); flush(log)
                reconstructed, bonds = compress_reconstruct(field; cutoff=cutoff, ordering=ordering)
                write_mps_mat(output_path, reconstructed, bonds)
                record["state"] = "SUCCEEDED"
                record["link_dimensions"] = bonds
                record["output_bytes"] = filesize(output_path)
                println("Output: ", output_path, " bonds=", bonds)
            catch error
                record["state"] = "FAILED"
                record["error"] = sprint(showerror,error)
                rethrow()
            finally
                record["elapsed_seconds"] = time()-started
                save_record(record)
            end
        end
    end
end
println("Run directory: ", run_directory)
