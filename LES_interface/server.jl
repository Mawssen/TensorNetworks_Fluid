using HTTP
using JSON3
using MAT
using Dates
include("LES_filter.jl")

const PORT = 8080
const HISTORY_FILE = joinpath(@__DIR__, "les_history.json")

mean(x) = sum(x) / length(x)

# ── Job tracking ──
mutable struct Job
    id::String
    file_path::String
    delta::Int
    status::String
    progress::String
    result::Dict
    start_time::Float64
end

const JOBS = Dict{String, Job}()
const JOBS_LOCK = ReentrantLock()

function new_job_id()
    return string(round(Int, time() * 1000)) * "_" * string(rand(1000:9999))
end

# ── History persistence ──
function load_history()
    if isfile(HISTORY_FILE)
        try
            return JSON3.read(read(HISTORY_FILE, String), Vector{Any})
        catch e
            println("[HISTORY] Failed to load: $e")
            return []
        end
    end
    return []
end

function save_history(history)
    try
        open(HISTORY_FILE, "w") do f
            write(f, JSON3.write(history))
        end
    catch e
        println("[HISTORY] Failed to save: $e")
    end
end

function append_history(entry::Dict)
    history = load_history()
    pushfirst!(history, entry)
    # Keep last 200 entries
    if length(history) > 200
        history = history[1:200]
    end
    save_history(history)
end

# ── Serve the HTML frontend ──
function serve_frontend(req)
    html = read(joinpath(@__DIR__, "index.html"), String)
    return HTTP.Response(200, ["Content-Type" => "text/html"], body=html)
end

# ── Start a filter job ──
function start_filter(req)
    try
        params = JSON3.read(String(req.body))
        file_path = String(params[:file_path])
        δ = Int(params[:delta])

        job_id = new_job_id()
        job = Job(job_id, file_path, δ, "running", "Starting...", Dict(), time())

        lock(JOBS_LOCK) do
            JOBS[job_id] = job
        end

        Threads.@spawn run_filter_job(job)

        return json_response(Dict("job_id" => job_id, "status" => "running"))
    catch e
        return json_response(Dict("error" => string(e)), 500)
    end
end

# ── Background computation ──
function run_filter_job(job::Job)
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    try
        file_path = job.file_path
        δ = job.delta

        job.progress = "Loading file..."
        println("\n[JOB $(job.id)] Loading: $file_path")
        t_load = @elapsed u = load_velocity_1024(file_path)
        N = size(u, 1)
        println("[JOB $(job.id)] Loaded $(N)³ field in $(round(t_load, digits=2)) s")

        if N % δ != 0
            job.status = "error"
            job.result = Dict("error" => "δ=$δ does not evenly divide N=$N")
            append_history(Dict(
                "job_id" => job.id, "file_path" => file_path, "delta" => δ,
                "status" => "error", "error" => "δ=$δ does not evenly divide N=$N",
                "timestamp" => timestamp
            ))
            return
        end

        M = N ÷ δ

        job.progress = "Filtering u (δ=$δ)..."
        println("[JOB $(job.id)] Filtering with δ=$δ ...")
        t_filter = @elapsed u_f = les_box_filter(u, δ)
        println("[JOB $(job.id)] Filtered in $(round(t_filter, digits=2)) s")

        job.progress = "Computing physics checks..."
        mean_u = mean(u)
        mean_uf = mean(u_f)
        r_total = mean(u .* u) - mean_u^2
        R_f = mean(u_f .* u_f) - mean_uf^2

        job.progress = "Computing SGS stress..."
        println("[JOB $(job.id)] Computing SGS stress ...")
        t_tau = @elapsed tau_f = les_box_filter(u .* u, δ) .- u_f .* u_f
        mean_tau = mean(tau_f)

        residual = r_total - R_f - mean_tau
        ratio = R_f / r_total

        job.progress = "Saving file..."
        file_name = splitext(basename(file_path))[1]
        out_name = "filtered_$(file_name)_$(M)x$(M)x$(M).mat"
        out_path = joinpath(@__DIR__, out_name)
        MAT.matwrite(out_path, Dict("u" => u_f))
        println("[JOB $(job.id)] Saved: $out_path")

        result = Dict(
            "success"     => true,
            "N"           => N,
            "M"           => M,
            "delta"       => δ,
            "t_load"      => round(t_load, digits=3),
            "t_filter"    => round(t_filter, digits=3),
            "t_tau"       => round(t_tau, digits=3),
            "t_total"     => round(t_load + t_filter + t_tau, digits=3),
            "threads"     => Threads.nthreads(),
            "mean_u"      => mean_u,
            "mean_uf"     => mean_uf,
            "mean_diff"   => mean_u - mean_uf,
            "r_total"     => r_total,
            "R_f"         => R_f,
            "mean_tau"    => mean_tau,
            "residual"    => residual,
            "ratio"       => ratio,
            "out_file"    => out_path
        )

        job.status = "done"
        job.result = result
        println("[JOB $(job.id)] Done! Total: $(result["t_total"]) s")

        # Save to persistent history
        append_history(Dict(
            "job_id"    => job.id,
            "file_path" => file_path,
            "delta"     => δ,
            "status"    => "done",
            "timestamp" => timestamp,
            "result"    => result
        ))

    catch e
        errmsg = string(e)
        job.status = "error"
        job.result = Dict("error" => errmsg)
        println("[JOB $(job.id)] ERROR: $e")

        append_history(Dict(
            "job_id"    => job.id,
            "file_path" => job.file_path,
            "delta"     => job.delta,
            "status"    => "error",
            "error"     => errmsg,
            "timestamp" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        ))
    end
end

# ── Poll job status ──
function get_status(req)
    params = HTTP.queryparams(HTTP.URI(req.target))
    job_id = get(params, "id", "")

    job = lock(JOBS_LOCK) do
        get(JOBS, job_id, nothing)
    end

    if job === nothing
        return json_response(Dict("error" => "Job not found"), 404)
    end

    resp = Dict(
        "job_id"   => job.id,
        "status"   => job.status,
        "progress" => job.progress,
        "elapsed"  => round(time() - job.start_time, digits=1),
    )

    if job.status == "done" || job.status == "error"
        resp["result"] = job.result
    end

    return json_response(resp)
end

# ── List active jobs ──
function list_jobs(req)
    jobs = lock(JOBS_LOCK) do
        [Dict(
            "job_id"     => j.id,
            "file"       => basename(j.file_path),
            "delta"      => j.delta,
            "status"     => j.status,
            "progress"   => j.progress,
            "elapsed"    => round(time() - j.start_time, digits=1),
            "has_result"  => j.status == "done"
        ) for j in values(JOBS)]
    end
    sort!(jobs, by=j->j["job_id"], rev=true)
    return json_response(Dict("jobs" => jobs))
end

# ── Get persistent history ──
function get_history(req)
    history = load_history()
    return json_response(Dict("history" => history))
end

# ── Clear history ──
function clear_history(req)
    save_history([])
    return json_response(Dict("cleared" => true))
end

# ── Stop a running job ──
function stop_job(req)
    params = JSON3.read(String(req.body))
    job_id = String(params[:job_id])

    job = lock(JOBS_LOCK) do
        get(JOBS, job_id, nothing)
    end

    if job === nothing
        return json_response(Dict("error" => "Job not found"), 404)
    end

    job.status = "error"
    job.result = Dict("error" => "Stopped by user")
    println("[JOB $(job.id)] Stopped by user")

    append_history(Dict(
        "job_id"    => job.id,
        "file_path" => job.file_path,
        "delta"     => job.delta,
        "status"    => "stopped",
        "error"     => "Stopped by user",
        "timestamp" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    ))

    return json_response(Dict("stopped" => true))
end

function json_response(data, status=200)
    body = JSON3.write(data)
    headers = [
        "Content-Type"                 => "application/json",
        "Access-Control-Allow-Origin"  => "*",
        "Access-Control-Allow-Headers" => "Content-Type",
    ]
    return HTTP.Response(status, headers, body=body)
end

function handle_cors(req)
    headers = [
        "Access-Control-Allow-Origin"  => "*",
        "Access-Control-Allow-Methods" => "POST, GET, OPTIONS, DELETE",
        "Access-Control-Allow-Headers" => "Content-Type",
    ]
    return HTTP.Response(204, headers)
end

# ── Router ──
const ROUTER = HTTP.Router()
HTTP.register!(ROUTER, "GET",     "/",           serve_frontend)
HTTP.register!(ROUTER, "POST",    "/run",        start_filter)
HTTP.register!(ROUTER, "GET",     "/status",     get_status)
HTTP.register!(ROUTER, "GET",     "/jobs",       list_jobs)
HTTP.register!(ROUTER, "GET",     "/history",    get_history)
HTTP.register!(ROUTER, "DELETE",  "/history",    clear_history)
HTTP.register!(ROUTER, "POST",    "/stop",       stop_job)
HTTP.register!(ROUTER, "OPTIONS", "/run",        handle_cors)
HTTP.register!(ROUTER, "OPTIONS", "/stop",       handle_cors)
HTTP.register!(ROUTER, "OPTIONS", "/history",    handle_cors)

# ── JIT warmup ──
println("Warming up JIT ...")
let u_test = randn(64, 64, 64)
    les_box_filter(u_test, 8)
end
println("Warmup done.")

# Show history count
hist = load_history()
println("Loaded $(length(hist)) history entries from les_history.json\n")

println("="^50)
println(" LES Filter Server running on http://localhost:$PORT")
println(" Open this URL in your browser.")
println(" Press Ctrl+C to stop.")
println("="^50)

HTTP.serve(ROUTER, "0.0.0.0", PORT)