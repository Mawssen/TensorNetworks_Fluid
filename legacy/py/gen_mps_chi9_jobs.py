#!/usr/bin/env python3
"""
gen_mps_chi9_jobs.py
--------------------
Generate one SLURM script per velocity component (u, v, w) that runs
mps_calc.jl at chi=9 to reconstruct that component. Separate jobs = they run
in parallel and one failing does not stop the others.

Uses the environment fixes established for this cluster: source ~/.bashrc,
call the juliaup 1.12 binary directly (NOT `module load julia`), and give each
job a private Julia compile cache.

Usage
-----
    python gen_mps_chi9_jobs.py \
        --rundir /path/to/dir/with/mps_calc.jl \
        --datadir /ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198 \
        --timestep 0198 \
        [--components u,v,w] [--chi 9] [--split 1] \
        [--jobdir mps_chi9_jobs] \
        [--mem 200G] [--cpus 8] [--time 3:00:00] [--qos normal]

Then:
    bash mps_chi9_jobs/submit_all.sh
"""

import os
import sys


def _parse(args, name, default=None):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        args.pop(i + 1); args.pop(i)
        return v
    return default


JOB = """#!/bin/bash
#SBATCH --job-name=mps9_{comp}
#SBATCH --cluster={cluster}
#SBATCH --partition={cluster}
#SBATCH --qos={qos}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={jobdir}/mps9_{comp}_%j.out

set -euo pipefail

set +eu
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
set -eu

JULIA="$HOME/.juliaup/bin/julia"
echo "Using Julia: $JULIA"; "$JULIA" --version

JOBCACHE="${{SLURM_SCRATCH:-$HOME/.julia_jobcache/$SLURM_JOB_ID}}/julia_depot"
mkdir -p "$JOBCACHE"
export JULIA_DEPOT_PATH="$JOBCACHE:$HOME/.julia"

cd {rundir}

echo "[mps chi={chi}] component file = jet_{comp}_{timestep}"
"$JULIA" mps_calc.jl \\
    chi={chi} \\
    cutoff=0.0 \\
    split={split} \\
    file=jet_{comp}_{timestep} \\
    dir={datadir} \\
    ext=dat

echo "[mps chi={chi}] jet_{comp}_{timestep} done."
"""


def main():
    args = sys.argv[1:]
    rundir = os.path.abspath(_parse(args, "--rundir", os.getcwd()))
    datadir = _parse(args, "--datadir",
                     "/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198")
    timestep = _parse(args, "--timestep", "0198")
    comps = [c.strip() for c in
             _parse(args, "--components", "u,v,w").split(",") if c.strip()]
    chi = _parse(args, "--chi", "9")
    split = _parse(args, "--split", "1")
    jobdir = os.path.abspath(_parse(args, "--jobdir", "mps_chi9_jobs"))
    mem = _parse(args, "--mem", "200G")
    cpus = _parse(args, "--cpus", "8")
    walltime = _parse(args, "--time", "3:00:00")
    qos = _parse(args, "--qos", "normal")
    cluster = _parse(args, "--cluster", "smp")

    os.makedirs(jobdir, exist_ok=True)
    generated = []
    for comp in comps:
        script = JOB.format(
            comp=comp, cluster=cluster, qos=qos, cpus=cpus, mem=mem,
            time=walltime, jobdir=jobdir, rundir=rundir, chi=chi,
            split=split, timestep=timestep, datadir=datadir,
        )
        path = os.path.join(jobdir, f"job_mps9_{comp}.sh")
        with open(path, "w") as fh:
            fh.write(script)
        os.chmod(path, 0o755)
        generated.append(path)
        print(f"[gen] wrote {path}")

    submit_all = os.path.join(jobdir, "submit_all.sh")
    with open(submit_all, "w") as fh:
        fh.write("#!/bin/bash\nset -euo pipefail\n")
        for p in generated:
            fh.write(f"sbatch {p}\n")
    os.chmod(submit_all, 0o755)
    print(f"[gen] wrote {submit_all}")
    print(f"\n[gen] {len(generated)} jobs in {jobdir}/")
    print(f"[gen] submit: bash {submit_all}")
    print("[gen] or run one at a time: "
          f"sbatch {jobdir}/job_mps9_u.sh")


if __name__ == "__main__":
    main()
