#!/usr/bin/env python3
"""
gen_cutoff_jobs.py
------------------
Generate one SLURM script per cutoff so each cutoff's MPS build runs as an
independent parallel job on CRC. Each job runs mps_cutoff_sweep.jl with a
single-element cutoffs= list and its own output file (no collisions).

Also writes submit_all.sh that sbatch-submits every generated job.

Usage
-----
    python gen_cutoff_jobs.py \
        [--cutoffs 1e-2,7e-3,5e-3,4e-3,3e-3,2e-3] \
        [--file U_t1] \
        [--dir ../../../../Forced_Isotropic_1024Cubed_mat_in_one] \
        [--target 512] \
        [--jobdir cutoff_jobs] \
        [--julia-module julia] \
        [--mem 180G] [--cpus 20] [--time 6:00:00] [--cluster smp]

Then on CRC:
    bash cutoff_jobs/submit_all.sh          # submit all in parallel
    # ... after they finish:
    cat cutoff_jobs/cutoff_*_result.txt     # collected CR per cutoff
"""

import os
import sys
import re


def _parse(args, name, default=None):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        args.pop(i + 1); args.pop(i)
        return v
    return default


def _tag(cf: str) -> str:
    """Filesystem-safe tag for a cutoff string, e.g. 7e-3 -> 7e-3, 1e-2 -> 1e-2."""
    return re.sub(r"[^0-9a-zA-Z.+-]", "_", cf.strip())


JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=cf_{tag}
#SBATCH --cluster={cluster}
#SBATCH --partition={cluster}{qos_line}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={jobdir}/cutoff_{tag}_%j.out

set -euo pipefail

# --- Reproduce the interactive environment --------------------------------
# Batch jobs run in a NON-interactive shell that does not source ~/.bashrc,
# so the setup that makes `julia` work by hand (here: the juliaup shim on
# PATH) is missing. Source it, relaxing set -e/-u since typical .bashrc files
# reference unbound vars / return nonzero.
set +eu
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
set -eu

# --- Select the Julia binary ----------------------------------------------
# The interactive Julia that HAS ITensorMPS is juliaup's 1.12 at
# ~/.juliaup/bin/julia -- NOT the system `module load julia` (a different,
# older version whose environment lacks ITensorMPS). Use the juliaup binary
# directly.{julia_setup}
JULIA="{julia_bin}"
echo "Using Julia: $JULIA"; "$JULIA" --version

# --- Per-job private Julia compile cache -----------------------------------
# Give each job its OWN *compiled* cache dir (placed FIRST so Julia WRITES new
# precompile output there, isolating parallel jobs from each other), while
# still READING installed packages from the main $HOME/.julia depot (second).
JOBCACHE="${{SLURM_SCRATCH:-$HOME/.julia_jobcache/$SLURM_JOB_ID}}/julia_depot"
mkdir -p "$JOBCACHE"
export JULIA_DEPOT_PATH="$JOBCACHE:$HOME/.julia"

# Run from the directory that holds the Julia script + the data-dir relative
# path, so `dir=` and `out=` resolve the same way as a manual `julia` run.
cd {rundir}

# Single-cutoff evaluation. Writes its own table so jobs never collide.
"$JULIA" {julia_flags}{julia_script} \\
    file={file} \\
    dir={datadir} \\
    target={target} \\
    cutoffs={cf} \\
    out={jobdir}/cutoff_{tag}_result.txt

echo "cutoff {cf} done -> {jobdir}/cutoff_{tag}_result.txt"

# Clean up the per-job cache if it was placed under $HOME (scratch is auto-cleaned).
if [ -z "${{SLURM_SCRATCH:-}}" ]; then
  rm -rf "$HOME/.julia_jobcache/$SLURM_JOB_ID"
fi
"""


def main():
    args = sys.argv[1:]
    cutoffs_str = _parse(args, "--cutoffs", "1e-2,7e-3,5e-3,4e-3,3e-3,2e-3")
    file_name = _parse(args, "--file", "U_t1")
    datadir = _parse(args, "--dir",
                     "../../../../Forced_Isotropic_1024Cubed_mat_in_one")
    target = _parse(args, "--target", "512")
    jobdir = _parse(args, "--jobdir", "cutoff_jobs")
    julia_module = _parse(args, "--julia-module", "")
    # Julia binary to use. Default = juliaup's julia (the interactive 1.12
    # that has ITensorMPS). A `module load julia` pulls a DIFFERENT/older
    # version whose environment lacks ITensorMPS, so we call this directly.
    julia_bin = _parse(args, "--julia-bin", "$HOME/.juliaup/bin/julia")
    # Optional explicit Julia project (the env that has ITensorMPS). If your
    # .bashrc already activates it, leave empty. Else pass e.g.
    # --julia-project @. or --julia-project /path/to/Project_dir
    julia_project = _parse(args, "--julia-project", "")
    julia_script = _parse(args, "--julia-script", "mps_cutoff_sweep.jl")
    mem = _parse(args, "--mem", "180G")
    cpus = _parse(args, "--cpus", "20")
    walltime = _parse(args, "--time", "1:00:00")
    cluster = _parse(args, "--cluster", "smp")
    # QOS, e.g. --qos short  (priority-13, <=1h on this cluster). Empty -> omit.
    qos = _parse(args, "--qos", "short")
    # Directory that holds mps_cutoff_sweep.jl and from which `dir=` resolves.
    # Default: the current working directory when you run the generator.
    rundir = _parse(args, "--rundir", os.getcwd())
    rundir = os.path.abspath(rundir)

    cutoffs = [c.strip() for c in cutoffs_str.split(",") if c.strip()]
    jobdir_abs = os.path.abspath(jobdir)
    os.makedirs(jobdir_abs, exist_ok=True)

    # QOS header line (newline-prefixed so it sits on its own #SBATCH line),
    # or empty string to omit entirely.
    qos_line = f"\n#SBATCH --qos={qos}" if qos.strip() else ""

    # Julia flags: --project=... (with trailing space) if a project was given.
    julia_flags = f"--project={julia_project} " if julia_project.strip() else ""

    # Optional module load, only if the user explicitly asked for one.
    julia_setup = (f"\nmodule load {julia_module} 2>/dev/null || true"
                   if julia_module.strip() else "")

    generated = []
    for cf in cutoffs:
        tag = _tag(cf)
        script = JOB_TEMPLATE.format(
            tag=tag, cluster=cluster, cpus=cpus, mem=mem, time=walltime,
            jobdir=jobdir_abs, rundir=rundir, julia_bin=julia_bin,
            julia_setup=julia_setup, julia_script=julia_script,
            file=file_name, datadir=datadir, target=target, cf=cf,
            qos_line=qos_line, julia_flags=julia_flags,
        )
        path = os.path.join(jobdir_abs, f"job_cutoff_{tag}.sh")
        with open(path, "w") as fh:
            fh.write(script)
        os.chmod(path, 0o755)
        generated.append(path)
        print(f"[gen] wrote {path}  (cutoff={cf})")

    # submit_all.sh
    submit_all = os.path.join(jobdir_abs, "submit_all.sh")
    with open(submit_all, "w") as fh:
        fh.write("#!/bin/bash\n")
        fh.write("# Submit every per-cutoff job in parallel.\n")
        fh.write("set -euo pipefail\n")
        for p in generated:
            fh.write(f"sbatch {p}\n")
    os.chmod(submit_all, 0o755)
    print(f"[gen] wrote {submit_all}")

    # collect helper
    collect = os.path.join(jobdir_abs, "collect_results.sh")
    with open(collect, "w") as fh:
        fh.write("#!/bin/bash\n")
        fh.write("# Concatenate all per-cutoff results into one sorted table.\n")
        fh.write(f'cd "{jobdir_abs}"\n')
        fh.write('echo "# cutoff        CR_memory        chi_max   n_bonds"\n')
        fh.write("grep -h -v '^#' cutoff_*_result.txt 2>/dev/null | "
                 "sort -g -k1\n")
    os.chmod(collect, 0o755)
    print(f"[gen] wrote {collect}")

    print(f"\n[gen] {len(generated)} jobs generated in {jobdir}/")
    print(f"[gen] submit all:   bash {submit_all}")
    print(f"[gen] collect after: bash {collect}")


if __name__ == "__main__":
    main()
