# TCOM Comment 2.1 benchmark

This is an opt-in extension; existing experiments and defaults are unchanged.
Run from the repository root with the existing scientific Python environment.
The driver freezes the April 10, 2026 policy-gradient checkpoint and July 16,
2025 Starlink catalogue by path and records their SHA-256 hashes.

```sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 MPLBACKEND=Agg
python test_script/test_tcom_multishell_selection.py
python sim_alg_v1_res/test_tcom_multishell.py --stage geometry --output sim_alg_v1_res/test_tcom_multishell/revision-ail
python sim_alg_v1_res/test_tcom_multishell.py --stage pilot --output sim_alg_v1_res/test_tcom_multishell/revision-ail
python sim_alg_v1_res/test_tcom_multishell.py --stage full --output sim_alg_v1_res/test_tcom_multishell/revision-ail
python sim_alg_v1_res/figures/plot_test_tcom_multishell.py --results sim_alg_v1_res/test_tcom_multishell/revision-ail --output sim_alg_v1_res/figures/plot_test_tcom_multishell.pdf
```

All 54 geometry checks must pass before the pilots; all 15 pilot evaluations
must pass before the full stage. The full stage reuses those same-run pilot
rows. Interrupted stages resume from atomic per-case JSON outputs. Changed
source or inputs require a new output directory. For parallel execution,
partition a stage using `--config shell53|shell43|mixed` and `--selection 0|1|2`;
never run two workers for the same partition.

The 53.22-degree/15.087-rev-per-day and 43-degree/15.024-rev-per-day groups are
TLE-derived clusters. Each single-cluster case has 1,000 satellites; the mixture
has 500 from each. Satellite validity is checked at every evaluation epoch,
not the date the script is run. This driver uses UTC Julian dates for SGP4 and
checks transformed positions and velocities against Skyfield. Older simulation
entry points retain their existing propagation behavior.

Within each configuration, three circular RAAN-block selections are propagated
to six offsets (0, 30, 60, 90, 120, 180 minutes). Their overlapping memberships
are recorded in the manifest. The plotted ranges describe selection sensitivity,
not confidence intervals or independent temporal replications. All five methods
receive identical snapshot inputs, verified by array fingerprints. DuJo uses
500 fixed-traffic updates and final-iterate conversion; DRL uses a fixed seed
and checkpoint without retraining. Only DRL inference uses CUDA; graph algorithms
and LP solves run on the CPU.

The raw records include input fingerprints, throughput, residual offered demand,
served-demand ratio, elapsed evaluation time, and feasible/selected unique
inter-cluster satellite-pair counts. They measure independently optimized
snapshot performance, not continuous schedule feasibility or long-term precession.
Elapsed times include setup and validation and are not a replacement for the
manuscript's dedicated computing-time experiment.

## Manuscript presentation

Fig. 5(a)/(b) stack the single-shell and two-shell plots in one column. Both
show one fixed population at six times, with no averaging or shaded ranges.
The two-shell plot uses predetermined selection 0: the initial minimum-span
500-satellite RAAN block from each cluster, before any selection shifts.
This is 30 method evaluations over six snapshots. Selection is by the existing
protocol order, not by performance. All 270 evaluations, including unfavorable
results in the other selections, remain archived. No experiment was rerun.

```sh
python sim_alg_v1_res/figures/plot_test_tcom_two_shell.py --results sim_alg_v1_res/test_tcom_multishell/revision-ail --output sim_alg_v1_res/figures/plot_test_tcom_two_shell.pdf
```
