#!/usr/bin/env julia
#
# run_les.jl — LES box filter for CRC/SLURM
#
# Usage:
#   MAT: julia -t auto run_les.jl <path> <delta> <out_dir> mat [0 0 <out_name>]
#   BIN: julia -t auto run_les.jl <path> <delta> <out_dir> bin <N> <skip_bytes> [<out_name>]

using Dates

include("LES_filter.jl")

mean(x) = sum(x) / length(x)

function load_velocity_binary(filename::String, shape::NTuple{3,Int}, skip_bytes::Int=0)
    num_elements = prod(shape)
    open(filename, "r") do io
        if skip_bytes > 0
            skip(io, skip_bytes)
        end
        buffer = Vector{Float64}(undef, num_elements)
        read!(io, buffer)
        return reshape(buffer, shape...)
    end
end

# ── Parse arguments ──
if length(ARGS) < 4
    println("Usage:")
    println("  MAT: julia -t auto run_les.jl <path> <delta> <out_dir> mat [0 0 <out_name>]")
    println("  BIN: julia -t auto run_les.jl <path> <delta> <out_dir> bin <N> <skip> [<out_name>]")
    exit(1)
end

file_path = ARGS[1]
delta = parse(Int, ARGS[2])
out_dir = ARGS[3]
file_type = lowercase(ARGS[4])
custom_out_name = ""

if !isfile(file_path)
    println("ERROR: File not found: $file_path")
    exit(1)
end

mkpath(out_dir)

file_name = splitext(basename(file_path))[1]
# Handle double extensions like rstrt.0020.bin -> rstrt.0020
if occursin(".", file_name)
    # keep as is, it's a valid name
end

# ── Status file for progress polling ──
status_path = joinpath(out_dir, "status_$(file_name)_delta$(delta).json")
function write_status(step, pct)
    open(status_path, "w") do f
        write(f, """{"step":"$step","pct":$pct,"time":"$(Dates.format(now(),"yyyy-mm-dd HH:MM:SS"))"}""")
    end
end

println("="^60)
println(" LES Box Filter -- CRC Job")
println(" $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("="^60)
println("File       : $file_path")
println("File type  : $file_type")
println("Filter (d) : $delta")
println("Output dir : $out_dir")
println("Threads    : $(Threads.nthreads())")
println()

# ── Load ──
write_status("Loading file...", 10)
println("[1/4] Loading file...")

if file_type == "mat"
    t_load = @elapsed u = load_velocity_1024(file_path)
    if length(ARGS) >= 7
        custom_out_name = ARGS[7]
    end

elseif file_type == "bin"
    if length(ARGS) < 6
        println("ERROR: bin format requires <N> and <skip_bytes> arguments")
        exit(1)
    end
    N_grid = parse(Int, ARGS[5])
    skip_bytes = parse(Int, ARGS[6])
    shape = (N_grid, N_grid, N_grid)
    println("  Grid: $(N_grid)^3, skip: $skip_bytes bytes")
    t_load = @elapsed u = load_velocity_binary(file_path, shape, skip_bytes)
    if length(ARGS) >= 7
        custom_out_name = ARGS[7]
    end

else
    println("ERROR: Unknown file type '$file_type'. Use 'mat' or 'bin'.")
    exit(1)
end

N = size(u, 1)
println("  Loaded $(N)^3 in $(round(t_load, digits=2)) s")

if N % delta != 0
    println("ERROR: delta=$delta does not evenly divide N=$N")
    write_status("ERROR: delta=$delta does not divide N=$N", 100)
    exit(1)
end
M = N ÷ delta

# ── Filter ──
write_status("Filtering u ($(N)^3 to $(M)^3)...", 30)
println("[2/4] Filtering u ($(N)^3 -> $(M)^3)...")
t_filter = @elapsed u_f = les_box_filter(u, delta)
println("  Done in $(round(t_filter, digits=2)) s")

# ── Physics checks ──
write_status("Computing physics checks...", 60)
println("[3/4] Computing physics checks...")
mean_u = mean(u)
mean_uf = mean(u_f)
r_total = mean(u .* u) - mean_u^2
R_f = mean(u_f .* u_f) - mean_uf^2

write_status("Computing SGS stress...", 70)
println("  Computing SGS stress...")
t_tau = @elapsed tau_f = les_box_filter(u .* u, delta) .- u_f .* u_f
mean_tau = mean(tau_f)
println("  Done in $(round(t_tau, digits=2)) s")

residual = r_total - R_f - mean_tau
ratio = R_f / r_total

# ── Save ──
write_status("Saving filtered field...", 90)
if !isempty(custom_out_name)
    out_name = endswith(custom_out_name, ".mat") ? custom_out_name : custom_out_name * ".mat"
else
    out_name = "filtered_$(file_name)_$(M)x$(M)x$(M).mat"
end
out_path = joinpath(out_dir, out_name)
println("[4/4] Saving to $out_path ...")
MAT.matwrite(out_path, Dict("u" => u_f))

t_total = t_load + t_filter + t_tau

# ── Final report JSON ──
slurm_job_id = get(ENV, "SLURM_JOB_ID", "local")
report_path = joinpath(out_dir, "report_$(slurm_job_id).json")
open(report_path, "w") do f
    write(f, """{
  "status": "done",
  "file_path": "$(replace(file_path, "\\" => "/"))",
  "file_type": "$(file_type)",
  "delta": $(delta),
  "N": $(N),
  "M": $(M),
  "threads": $(Threads.nthreads()),
  "t_load": $(round(t_load, digits=3)),
  "t_filter": $(round(t_filter, digits=3)),
  "t_tau": $(round(t_tau, digits=3)),
  "t_total": $(round(t_total, digits=3)),
  "mean_u": $(mean_u),
  "mean_uf": $(mean_uf),
  "mean_diff": $(mean_u - mean_uf),
  "r_total": $(r_total),
  "R_f": $(R_f),
  "mean_tau": $(mean_tau),
  "residual": $(residual),
  "ratio": $(ratio),
  "out_file": "$(out_path)",
  "timestamp": "$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))"
}""")
end

rm(status_path, force=true)

println()
println("="^60)
println(" RESULTS")
println("="^60)
println("  Original  : $(N)^3  ->  Filtered : $(M)^3")
println("  Time      : $(round(t_total, digits=3)) s")
println("  Mean(u)   : $mean_u")
println("  Mean(<u>) : $mean_uf")
println("  r_total   = $r_total")
println("  R_f       = $R_f")
println("  mean(tau) = $mean_tau")
println("  R_f/r     = $ratio")
println("  Output    : $out_path")
println("  Report    : $report_path")
println("="^60)