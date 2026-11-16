#!/bin/bash
set -euo pipefail
sbatch /ix/pgivi/moe32/Aidyn_DNS/stats/mps_chi27_jobs/job_mps27_U.sh
sbatch /ix/pgivi/moe32/Aidyn_DNS/stats/mps_chi27_jobs/job_mps27_V.sh
sbatch /ix/pgivi/moe32/Aidyn_DNS/stats/mps_chi27_jobs/job_mps27_W.sh
