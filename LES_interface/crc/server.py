#!/usr/bin/env python3
"""
LES Filter — Local server that manages CRC jobs via SSH.
Run on your Mac: python3 server.py
Opens browser to http://localhost:8080
"""

import http.server
import json
import subprocess
import os
import time
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PORT = 8080
DIR = Path(__file__).parent
HISTORY_FILE = DIR / "les_history.json"
ACTIVE_JOBS_FILE = DIR / "les_active_jobs.json"

# ── CRC Config ──
CRC_USER = "moe32"
CRC_HOST = "h2p.crc.pitt.edu"
CRC_SCRIPT_DIR = "/ix/pgivi/moe32/LES_filter"  # where run_les.jl lives on CRC

# ── Active jobs being tracked ──
active_jobs = {}
jobs_lock = threading.Lock()


def save_active_jobs():
    """Persist active jobs to disk so they survive restarts."""
    with jobs_lock:
        serializable = {}
        for jid, j in active_jobs.items():
            serializable[jid] = {
                "job_id": j.get("job_id"),
                "file_path": j.get("file_path"),
                "fname": j.get("fname"),
                "delta": j.get("delta"),
                "out_dir": j.get("out_dir"),
                "status": j.get("status"),
                "slurm_status": j.get("slurm_status", ""),
                "progress": j.get("progress", ""),
                "pct": j.get("pct", 0),
                "start_time": j.get("start_time", 0),
                "result": j.get("result"),
                "error": j.get("error"),
            }
    try:
        ACTIVE_JOBS_FILE.write_text(json.dumps(serializable, indent=2))
    except Exception as e:
        print(f"[WARN] Could not save active jobs: {e}")


def load_active_jobs():
    """Restore active jobs from disk and resume monitoring."""
    if not ACTIVE_JOBS_FILE.exists():
        return
    try:
        data = json.loads(ACTIVE_JOBS_FILE.read_text())
    except:
        return
    for jid, j in data.items():
        status = j.get("status", "")
        if status in ("done", "error"):
            continue
        j["start_time"] = j.get("start_time", time.time())
        with jobs_lock:
            active_jobs[jid] = j
        print(f"  Restored job {jid} ({j.get('fname')}, d={j.get('delta')})")


def background_poller():
    """Single background thread that checks ALL active jobs via SSH."""
    while True:
        time.sleep(10)

        # Collect jobs that need checking
        with jobs_lock:
            to_check = [
                (jid, dict(j))
                for jid, j in active_jobs.items()
                if j.get("status") in ("running", "queued", "unknown")
            ]

        if not to_check:
            continue

        for jid, j in to_check:
            out_dir = j.get("out_dir", "")
            fname = j.get("fname", "")
            delta = j.get("delta", 0)
            if not (out_dir and fname and delta):
                continue

            try:
                # Check progress files
                ps, pd = check_job_progress(out_dir, fname, delta, jid)

                if ps == "done":
                    with jobs_lock:
                        if jid in active_jobs:
                            active_jobs[jid]["status"] = "done"
                            active_jobs[jid]["result"] = pd
                    save_active_jobs()
                    append_history({
                        "job_id": jid, "file_path": j.get("file_path", ""),
                        "delta": delta, "out_dir": out_dir,
                        "status": "done", "result": pd,
                        "timestamp": pd.get("timestamp", ""),
                    })
                    print(f"[POLLER] Job {jid} done!")

                elif ps == "error":
                    err_msg = pd.get("error", "Unknown error")
                    with jobs_lock:
                        if jid in active_jobs:
                            active_jobs[jid]["status"] = "error"
                            active_jobs[jid]["error"] = err_msg
                    save_active_jobs()
                    append_history({
                        "job_id": jid, "file_path": j.get("file_path", ""),
                        "delta": delta, "out_dir": out_dir,
                        "status": "error", "error": err_msg,
                        "timestamp": pd.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
                    })
                    print(f"[POLLER] Job {jid} error: {err_msg[:60]}")

                elif ps == "running":
                    with jobs_lock:
                        if jid in active_jobs:
                            active_jobs[jid]["status"] = "running"
                            active_jobs[jid]["progress"] = pd.get("step", "...")
                            active_jobs[jid]["pct"] = pd.get("pct", 0)
                    save_active_jobs()

                else:
                    # Unknown — check SLURM status
                    slurm_st = check_slurm_status(jid)
                    with jobs_lock:
                        if jid in active_jobs:
                            active_jobs[jid]["slurm_status"] = slurm_st

                    if slurm_st in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                        # Give it a moment for files to flush
                        time.sleep(3)
                        ps2, pd2 = check_job_progress(out_dir, fname, delta, jid)
                        if ps2 == "done":
                            with jobs_lock:
                                if jid in active_jobs:
                                    active_jobs[jid]["status"] = "done"
                                    active_jobs[jid]["result"] = pd2
                            save_active_jobs()
                            append_history({
                                "job_id": jid, "file_path": j.get("file_path", ""),
                                "delta": delta, "out_dir": out_dir,
                                "status": "done", "result": pd2,
                                "timestamp": pd2.get("timestamp", ""),
                            })
                            print(f"[POLLER] Job {jid} done (after SLURM {slurm_st})!")
                        elif ps2 == "error":
                            err_msg = pd2.get("error", f"SLURM: {slurm_st}")
                            with jobs_lock:
                                if jid in active_jobs:
                                    active_jobs[jid]["status"] = "error"
                                    active_jobs[jid]["error"] = err_msg
                            save_active_jobs()
                            append_history({
                                "job_id": jid, "file_path": j.get("file_path", ""),
                                "delta": delta, "out_dir": out_dir,
                                "status": "error", "error": err_msg,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            })
                            print(f"[POLLER] Job {jid} error after {slurm_st}")
                        else:
                            # No report at all
                            err_msg = f"SLURM {slurm_st} but no report found"
                            with jobs_lock:
                                if jid in active_jobs:
                                    active_jobs[jid]["status"] = "error"
                                    active_jobs[jid]["error"] = err_msg
                            save_active_jobs()
                            append_history({
                                "job_id": jid, "file_path": j.get("file_path", ""),
                                "delta": delta, "out_dir": out_dir,
                                "status": "error", "error": err_msg,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            })
                            print(f"[POLLER] Job {jid}: {err_msg}")

            except Exception as e:
                print(f"[POLLER] Error checking job {jid}: {e}")


def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except:
            return []
    return []


def save_history(history):
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def append_history(entry):
    h = load_history()
    h.insert(0, entry)
    if len(h) > 200:
        h = h[:200]
    save_history(h)


def ssh_cmd(cmd, timeout=15):
    """Run a command on CRC via SSH."""
    full = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
            f"{CRC_USER}@{CRC_HOST}", cmd]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", "SSH timeout", 1
    except Exception as e:
        return "", str(e), 1


def submit_job(file_path, delta, out_dir, cpus=16, mem="64G", wall="02:00:00",
               grid_size=0, out_name=""):
    """Generate SLURM script, upload, and submit."""
    fname = Path(file_path).stem
    job_name = f"{fname}_d{delta}"

    # Auto-detect file type from extension
    ext = Path(file_path).suffix.lower()
    if ext == ".bin":
        file_type = "bin"
        basename_lower = Path(file_path).name.lower()
        skip_bytes = 192 if "daniel_u" in basename_lower else 0
        args = f'"{file_path}" {delta} "{out_dir}" bin {grid_size} {skip_bytes}'
        if out_name:
            args += f' "{out_name}"'
    else:
        file_type = "mat"
        args = f'"{file_path}" {delta} "{out_dir}" mat'
        if out_name:
            args += f' 0 0 "{out_name}"'

    # Read template
    template = (DIR / "les_filter.slurm.template").read_text()
    slurm_script = template \
        .replace("__JOBNAME__", job_name) \
        .replace("__CPUS__", str(cpus)) \
        .replace("__MEM__", mem) \
        .replace("__TIME__", wall) \
        .replace("__FILEPATH__", file_path) \
        .replace("__DELTA__", str(delta)) \
        .replace("__OUTDIR__", out_dir) \
        .replace("__SCRIPTDIR__", CRC_SCRIPT_DIR) \
        .replace("__ARGS__", args)

    # Write temp slurm file locally
    local_slurm = DIR / f"_temp_{job_name}.slurm"
    local_slurm.write_text(slurm_script)

    # Upload slurm script
    scp_result = subprocess.run(
        ["scp", str(local_slurm), f"{CRC_USER}@{CRC_HOST}:{CRC_SCRIPT_DIR}/"],
        capture_output=True, text=True, timeout=30
    )
    if scp_result.returncode != 0:
        local_slurm.unlink(missing_ok=True)
        return None, f"SCP failed: {scp_result.stderr}"

    remote_slurm_name = local_slurm.name

    # Submit
    stdout, stderr, rc = ssh_cmd(f"cd {CRC_SCRIPT_DIR} && sbatch {remote_slurm_name}")

    # Clean up temp slurm files (local and remote)
    local_slurm.unlink(missing_ok=True)
    ssh_cmd(f"rm -f {CRC_SCRIPT_DIR}/{remote_slurm_name}")
    # Also clean any leftover _temp files
    ssh_cmd(f"rm -f {CRC_SCRIPT_DIR}/_temp_*.slurm")

    if rc != 0:
        return None, f"sbatch failed: {stderr}"

    # Parse job ID
    # "Submitted batch job 12345 on cluster smp"
    parts = stdout.split()
    job_id = None
    for i, w in enumerate(parts):
        if w == "job" and i + 1 < len(parts):
            job_id = parts[i + 1]
            break

    if not job_id:
        return None, f"Could not parse job ID from: {stdout}"

    return job_id, None


def check_slurm_status(job_id):
    """Check if SLURM job is still running."""
    stdout, _, rc = ssh_cmd(f"squeue -M smp -j {job_id} -h -o '%T' 2>/dev/null")
    if rc != 0 or not stdout.strip():
        return "COMPLETED"
    return stdout.strip()


def check_job_progress(out_dir, fname, delta, slurm_job_id=None):
    """Check status/report files on CRC — searches out_dir and subdirs."""
    # Search for report_<jobid>.json
    if slurm_job_id:
        cmd = f"find {out_dir} -maxdepth 2 -name 'report_{slurm_job_id}.json' -type f 2>/dev/null"
        stdout, _, rc = ssh_cmd(cmd)
        if rc == 0 and stdout.strip():
            report_file = stdout.strip().split("\n")[0].strip()
            content, _, rc2 = ssh_cmd(f"cat '{report_file}'")
            if rc2 == 0 and content:
                try:
                    data = json.loads(content)
                    st = data.get("status", "")
                    if st in ("done", "error"):
                        return st, data
                except:
                    pass

    # Check for status file (in-progress)
    cmd2 = f"find {out_dir} -maxdepth 2 -name 'status_{fname}_delta{delta}.json' -type f 2>/dev/null"
    stdout, _, rc = ssh_cmd(cmd2)
    if rc == 0 and stdout.strip():
        status_file = stdout.strip().split("\n")[0].strip()
        content, _, rc2 = ssh_cmd(f"cat '{status_file}'")
        if rc2 == 0 and content:
            try:
                status = json.loads(content)
                step = status.get("step", "")
                if step.startswith("ERROR"):
                    return "error", {"error": step}
                return "running", status
            except:
                pass

    return "unknown", {}


def monitor_job(job_info):
    """Background thread that monitors a CRC job."""
    job_id = job_info["job_id"]
    out_dir = job_info["out_dir"]
    fname = job_info["fname"]
    delta = job_info["delta"]

    while True:
        time.sleep(15)

        with jobs_lock:
            if job_id not in active_jobs:
                return
            if active_jobs[job_id].get("status") in ("done", "error"):
                return

        # Check SLURM status
        slurm_status = check_slurm_status(job_id)

        # Check progress files
        progress_status, progress_data = check_job_progress(out_dir, fname, delta)

        with jobs_lock:
            if job_id not in active_jobs:
                return

            active_jobs[job_id]["slurm_status"] = slurm_status

            if progress_status == "done":
                active_jobs[job_id]["status"] = "done"
                active_jobs[job_id]["result"] = progress_data
                save_active_jobs()
                # Save to history
                entry = {
                    "job_id": job_id,
                    "file_path": active_jobs[job_id]["file_path"],
                    "delta": delta,
                    "out_dir": out_dir,
                    "status": "done",
                    "result": progress_data,
                    "timestamp": progress_data.get("timestamp", ""),
                }
                append_history(entry)
                return

            elif progress_status == "running":
                active_jobs[job_id]["progress"] = progress_data.get("step", "...")
                active_jobs[job_id]["pct"] = progress_data.get("pct", 0)
                save_active_jobs()

            elif slurm_status == "COMPLETED":
                # SLURM says done but no report — check one more time
                time.sleep(5)
                ps, pd = check_job_progress(out_dir, fname, delta)
                if ps == "done":
                    active_jobs[job_id]["status"] = "done"
                    active_jobs[job_id]["result"] = pd
                    append_history({
                        "job_id": job_id,
                        "file_path": active_jobs[job_id]["file_path"],
                        "delta": delta, "out_dir": out_dir,
                        "status": "done", "result": pd,
                        "timestamp": pd.get("timestamp", ""),
                    })
                else:
                    # Check for SLURM error log
                    err_out, _, _ = ssh_cmd(
                        f"ls -t {out_dir}/slurm_*.err 2>/dev/null | head -1"
                    )
                    err_msg = "Job completed but no report found."
                    if err_out:
                        err_content, _, _ = ssh_cmd(f"tail -20 '{err_out.strip()}'")
                        if err_content:
                            err_msg = err_content
                    active_jobs[job_id]["status"] = "error"
                    active_jobs[job_id]["error"] = err_msg
                    save_active_jobs()
                    append_history({
                        "job_id": job_id,
                        "file_path": active_jobs[job_id]["file_path"],
                        "delta": delta, "out_dir": out_dir,
                        "status": "error", "error": err_msg,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                return

            elif slurm_status in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                err_out, _, _ = ssh_cmd(
                    f"ls -t {out_dir}/slurm_*.err 2>/dev/null | head -1"
                )
                err_msg = f"SLURM status: {slurm_status}"
                if err_out:
                    err_content, _, _ = ssh_cmd(f"tail -20 '{err_out.strip()}'")
                    if err_content:
                        err_msg += f"\n{err_content}"
                active_jobs[job_id]["status"] = "error"
                active_jobs[job_id]["error"] = err_msg
                append_history({
                    "job_id": job_id,
                    "file_path": active_jobs[job_id]["file_path"],
                    "delta": delta, "out_dir": out_dir,
                    "status": "error", "error": err_msg,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                return


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence logs

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            html = (DIR / "index.html").read_text()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())

        elif path == "/history":
            self.send_json({"history": load_history()})

        elif path == "/jobs":
            # Return cached state immediately — no SSH calls here
            with jobs_lock:
                jobs = []
                for jid, j in active_jobs.items():
                    entry = {
                        "job_id": jid,
                        "file_path": j.get("file_path", ""),
                        "delta": j.get("delta"),
                        "out_dir": j.get("out_dir", ""),
                        "status": j.get("status", "unknown"),
                        "slurm_status": j.get("slurm_status", ""),
                        "progress": j.get("progress", ""),
                        "pct": j.get("pct", 0),
                        "elapsed": round(time.time() - j.get("start_time", time.time()), 1),
                    }
                    if j.get("status") == "done":
                        entry["result"] = j.get("result", {})
                    if j.get("status") == "error":
                        entry["error"] = j.get("error", "")
                    jobs.append(entry)
            jobs.sort(key=lambda x: x["job_id"], reverse=True)
            self.send_json({"jobs": jobs})

        elif path == "/status":
            params = parse_qs(parsed.query)
            jid = params.get("id", [""])[0]
            with jobs_lock:
                j = active_jobs.get(jid)
            if not j:
                self.send_json({"error": "Job not found"}, 404)
                return
            resp = {
                "job_id": jid,
                "status": j.get("status", "unknown"),
                "slurm_status": j.get("slurm_status", ""),
                "progress": j.get("progress", ""),
                "pct": j.get("pct", 0),
                "elapsed": round(time.time() - j.get("start_time", time.time()), 1),
            }
            if j.get("status") == "done":
                resp["result"] = j.get("result", {})
            if j.get("status") == "error":
                resp["error"] = j.get("error", "")
            self.send_json(resp)

        elif path == "/test-ssh":
            stdout, stderr, rc = ssh_cmd("echo OK && hostname")
            if rc == 0:
                self.send_json({"connected": True, "host": stdout})
            else:
                self.send_json({"connected": False, "error": stderr})

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if path == "/run":
            file_path = body.get("file_path", "").strip()
            delta = int(body.get("delta", 0))
            out_dir = body.get("out_dir", "").strip()
            cpus = int(body.get("cpus", 16))
            mem = body.get("mem", "64G").strip()
            wall = body.get("walltime", "02:00:00").strip()
            grid_size = int(body.get("grid_size", 0))
            out_name = body.get("out_name", "").strip()

            if not file_path or not delta or not out_dir:
                self.send_json({"error": "file_path, delta, and out_dir required"}, 400)
                return

            ext = Path(file_path).suffix.lower()
            if ext == ".bin" and grid_size <= 0:
                self.send_json({"error": "Grid size N required for .bin files"}, 400)
                return

            # Submit to CRC
            job_id, err = submit_job(file_path, delta, out_dir, cpus, mem, wall,
                                     grid_size, out_name)
            if err:
                self.send_json({"error": err}, 500)
                return

            fname = Path(file_path).stem

            with jobs_lock:
                active_jobs[job_id] = {
                    "job_id": job_id,
                    "file_path": file_path,
                    "fname": fname,
                    "delta": delta,
                    "out_dir": out_dir,
                    "status": "queued",
                    "slurm_status": "PENDING",
                    "progress": "Job submitted to SLURM",
                    "pct": 5,
                    "start_time": time.time(),
                }

            # Background poller will pick it up
            save_active_jobs()

            self.send_json({"job_id": job_id, "status": "queued"})

        elif path == "/cancel":
            jid = body.get("job_id", "")
            ssh_cmd(f"scancel -M smp {jid}")
            with jobs_lock:
                if jid in active_jobs:
                    active_jobs[jid]["status"] = "error"
                    active_jobs[jid]["error"] = "Cancelled by user"
            append_history({
                "job_id": jid, "status": "cancelled",
                "error": "Cancelled by user",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_path": active_jobs.get(jid, {}).get("file_path", ""),
                "delta": active_jobs.get(jid, {}).get("delta", ""),
                "out_dir": active_jobs.get(jid, {}).get("out_dir", ""),
            })
            self.send_json({"cancelled": True})

        else:
            self.send_response(404)
            self.end_headers()

    def do_DELETE(self):
        if self.path == "/history":
            save_history([])
            self.send_json({"cleared": True})
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    # Test SSH
    print("Testing SSH connection to CRC...")
    out, err, rc = ssh_cmd("echo OK")
    if rc == 0:
        print(f"  Connected to CRC!")
    else:
        print(f"  WARNING: SSH failed: {err}")
        print(f"  Make sure you have SSH keys set up or VPN is connected.")

    # Restore jobs from previous session
    print("\nRestoring active jobs from previous session...")
    load_active_jobs()

    # Start background poller
    poller = threading.Thread(target=background_poller, daemon=True)
    poller.start()
    print("Background poller started (checks every 10s)")

    print()
    print("=" * 50)
    print(f" LES Filter Server running on http://localhost:{PORT}")
    print(f" Open this URL in your browser.")
    print(f" Press Ctrl+C to stop.")
    print("=" * 50)

    # Open browser
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()

    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()