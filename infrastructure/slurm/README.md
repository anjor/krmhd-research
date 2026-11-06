# Slurm runs of the P1 driver

Scripts to run `paper/phase-velocity-slices/scripts/run_forced_turbulence.py` on a Slurm cluster with one GPU per job,
and to copy every output back to the laptop. Only the driver, the two modules it imports, the config and its restart
file go to the cluster. Analysis runs on the laptop.

| File | Runs on | What it does |
|------|---------|--------------|
| `submit.sh` | laptop | Copies the code, the config, the restart file named by `init.path` and these scripts (rsync over ssh, key login only), then runs `cluster_submit.sh` there. `--setup` runs `setup_env.sh` instead. `--dry-run` only prints. |
| `fetch.sh` | laptop | Copies the run folders, the Slurm logs and the ledger back. `--follow` repeats this while jobs are queued or running. |
| `setup_env.sh` | login node | Builds the Python environment (`requirements.txt`: GANDALF 0.8.1, JAX 0.9.1 with CUDA wheels, NumPy and SciPy as in the earlier GPU runs). |
| `cluster_submit.sh` | login node, or inside a job | Checks `limits.conf`, then calls `sbatch`. |
| `job.sh` | compute node | Runs one config with `--max-wall-hours` set a little under the job's time limit. On exit status 3 (paused) it resubmits itself, and the next job continues the run in place with `--resume`. |
| `limits.conf` | login node | Jobs at once, automatic resubmissions per run, GPU hours per week. |
| `site.env.example` | both | Template of the site settings: ssh alias, paths, `sbatch` options, time limit. |
| `tests/test_chain.sh` | laptop | Tests the pause, resubmission and limits with stand-ins for `sbatch` and `squeue` and the real driver on the CPU. |

## Use

```bash
export SITE_ENV=/path/outside/git/site.env     # filled-in copy of site.env.example
infrastructure/slurm/submit.sh --setup          # once: build the environment on the cluster
infrastructure/slurm/submit.sh --time 00:30:00 --pause-after 0.04 paper/phase-velocity-slices/configs/check_beta0p3_m64.yaml
infrastructure/slurm/fetch.sh --follow 10       # copy back after each job
uv run python paper/phase-velocity-slices/analysis/compare_runs.py \
  paper/phase-velocity-slices/data/P1_check-beta0p3-m64_<timestamp> \
  paper/phase-velocity-slices/data/P1_forced-beta0p3-m64_20261018_144656
```

`--pause-after H` makes the first job pause after H hours. It tests the pause, the resubmission and the resume on the cluster before a long run depends on them.

The config, the run code and these scripts must be committed and unchanged before `submit.sh` sends them. Then the commit identifies what ran. `submit.sh` writes it to `COMMIT` on the cluster, and every job prints it.

## Limits

`cluster_submit.sh` checks `limits.conf` before every job, including each automatic resubmission. If a job would break a limit, it is not submitted and the script prints `REFUSED: <reason>`. The defaults allow one job at a time and one automatic resubmission per run, and set no weekly GPU-hour limit. The ledger (`ledger.tsv` on the cluster, copied back by `fetch.sh`) has one `submit` line per job (hours booked = time limit) and one `end` line (hours used, exit status).

## Hardware checks

`configs/check_beta0p3_m64.yaml` and `configs/check_beta0p3_n128.yaml` are the production configs cut to the first 0.1 and 0.2 τ_A, with their own labels. Every setting that affects the dynamics is the same, so on any hardware they must reproduce the earlier Modal A100 runs to round-off. `analysis/compare_runs.py` checks this sample by sample. The progress lines, printed every 20 steps, give the cost per step after compilation.
