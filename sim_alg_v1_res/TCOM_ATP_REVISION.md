# Comment 1.1 ATP replay extension

The revision uses 37 snapshots at 0,10,...,360 seconds and ATP times
0,10,20,30,50,70,90 seconds, with DuJo, DRL, SaTE, MRate, and +Grid.
The saved `sweep-20260430_210444-ail/shell_metadata.json` pins the 1000 satellite
names and their order. The loader checks finite propagation throughout the replay
instead of filtering historical TLEs against the current date.

Protocol: active user fraction 0.0001, traffic seed 0 at evaluation, 500 DuJo
iterations, legacy per-iteration traffic updates, final iterate, frozen April 10
policy-gradient checkpoint. The original ATP protocol is preserved; this differs
from the fixed-traffic multi-shell experiment.

`test_tcom_atp_transition.py` accepts `TCOM_REVISION_POPULATION_METADATA` to pin
membership and `TCOM_REVISION_ATP_TIMES_SEC` to select the sweep. The plotter now
accepts the same sweep variable. The default sweep now includes all seven requested delays; the physical settings,
optimizer settings, snapshot interval, and replay horizon remain unchanged.

`precompute_atp_dujo.py` computes independent DuJo snapshots in parallel using the
above protocol, writing trusted local pickle caches to `TCOM_REVISION_SNAPSHOT_CACHE`.
It retains all dual updates but omits unused intermediate primal/dual objective
diagnostics; the final objective and decision conversion still run. A five-step
legacy-protocol comparison verified exact multiplier, matching, route, rate,
objective, and input equality. Cached replay inputs also passed serialization and
zero-ATP throughput checks. Replay the caches chronologically using the same
cache variable and `TCOM_REVISION_METHODS=DuJo`; never load untrusted pickle files.
`TCOM_REVISION_WORKERS` controls precompute concurrency (default 8).

Initial links are pre-acquired. The initial snapshot is excluded from averages;
there is no additional warm-up and no cold-start experiment. A retained terminal
pair carries its timer across snapshots. Dropping it clears the timer; reselection
restarts acquisition. Each snapshot's matching and input hash are shared across
all seven ATP values. Selected terminal pairs are archived in compressed NumPy
files. The chronological validation reconstructs timers independently.

Execution is isolated at `/home/zhouyou/tcom-comment11-ail` on the RTX 5090 host.
DuJo optimization is CPU-based; DRL inference uses the available GPU. Raw results,
metadata, checkpoint/source hashes, and validation records belong under
`test_tcom_atp_transition/comment-1-1-20260924-ail/` (ignored experiment artifacts).
No experimental data directory is added to the manuscript repository.

Comparison with saved zero-ATP means: SaTE, MRate, and +Grid reproduce the original values exactly. DRL is rerun with its existing stochastic policy-gradient inference (`torch.normal`); its new mean differs from the saved run. All seven ATP settings reuse this run's same sampled decisions, rather than mixing old and new measurements.

To regenerate the revised figure from the validated aggregate results:

```sh
TCOM_REVISION_RESULTS_DIR=sim_alg_v1_res/test_tcom_atp_transition/comment-1-1-20260924-ail \
TCOM_REVISION_ATP_TIMES_SEC=0,10,20,30,50,70,90 \
python sim_alg_v1_res/figures/plot_test_tcom_atp_transition.py
```

The plotter rejects incomplete sweeps, keeps the two panels without shading, and
uses 10-point main fonts with enough vertical headroom for the endpoint markers.
The general readability restyler preserves this regenerated Fig. 10 instead of
restoring the archived four-point figure.

Completed and validated on 2026-09-25: 185 snapshot decisions, 1,295 replay evaluations, and 35 summary rows. Independent timer reconstruction, cross-method input hashes, throughput bounds, and exclusion of the initial snapshot passed. All 37 DuJo final objectives and zero-delay throughputs exactly reproduce the saved run. `validation.json`, `original_comparison.json`, `replay_comparison.json`, and `completion.json` record the checks and executed-source hashes. Large snapshot pickle caches remain on the remote workstation.
