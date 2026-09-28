# Comment 3.6: installed LCT count

Separate from Fig. 8's independent availability masks, compare fully available
2/3/4-terminal layouts: front/back; front/back/right; front/back/right/left.
The lateral directions are the normalized cross product of the radial and
velocity directions and its negative, as already defined by update_arrows.
The four-slot internal representation is retained, with absent slots disabled.
This changes no existing experiment defaults.

Use the saved 1000-satellite single-shell population at 2025-07-16 16:00 UTC,
100 original seed-0 gateways, traffic seed 0, active-user fraction 0.0001, the
same physical parameters and field of regard. Run all five methods fresh at
all three counts (15 evaluations). DuJo uses 500 fixed-traffic updates and
final-iterate conversion. DRL uses the frozen April 10 PG checkpoint, without
retraining; this tests that checkpoint on the added-terminal geometry.
No ATP overhead, temporal average, or placement average is included.

From /home/zhouyou/tcom-comment11-ail on the RTX 5090 host:

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=4
/home/zhouyou/tcom-comment21-ail/.venv/bin/python sim_alg_v1_res/test_tcom_lct_count.py --atp-root /home/zhouyou/tcom-comment11-ail --output /home/zhouyou/tcom-comment11-ail/lct-results --seeds 0 --prepare
/home/zhouyou/tcom-comment21-ail/.venv/bin/python sim_alg_v1_res/test_tcom_lct_count.py --atp-root /home/zhouyou/tcom-comment11-ail --output /home/zhouyou/tcom-comment11-ail/lct-results --seeds 0 --workers 4
```

DuJo runs on workstation CPUs and DRL uses CUDA. The preflight verifies matching
physical/traffic/gateway inputs across counts, nested candidate links, mounting
orientations, fully available installed terminals, and equality of the two-LCT
input fingerprint to the original 100-gateway case. Per-method checks enforce
matched inputs, feasible terminal matching and routes, finite nonnegative rates,
and throughput bounds. Save NPZ routes/rates, terminal masks/directions,
candidate/selected links, JSON metrics, settings and hashes. Completed cases
resume individually. Results belong under ignored test_tcom_lct_count/.
The manuscript reports the results in an appendix table, as requested; no new figure.

Completed 2026-09-28: all 15 cases passed independent NPZ audits for matched
inputs, nested candidate sets, installed masks, feasible matching/routes,
throughput/served ratios and per-source/per-target capacity bounds. The two-LCT
DuJo result reproduces the 100-gateway benchmark (147.17083544254766 Gbps).
DuJo achieves 257.3519638386407 / 218.2877427079096 Gbps with 3/4 terminals;
MRate is highest at four terminals (272.9350383906864 Gbps). Retain these
nonmonotonic and unfavorable outcomes. Raw results and provenance are stored
locally in test_tcom_lct_count/comment-3-6-20260928/ and remotely in lct-results/.


## Fixed-pair correction (2026-09-28)

The initial comparison above reselected the nearest gateway sources when the
candidate topology changed. Its numbers are superseded for the controlled LCT
comparison. Freeze the ordered source-target arrays from the two-LCT reference
for all counts, including every traffic refresh and final DuJo conversion.
Use a separate `lct-fixed-pairs` output, preserving the original `lct-results`.
Locally the corrected directory is `test_tcom_lct_count/comment-3-6-fixed-pairs-20260928/`.
The first correction attempt passed all baseline runs but final conversion reset
the DuJo source assignments; the input fingerprint assertion caught this and
rejected those two outputs. The override now preserves pairs through that refresh;
preflight tests the refresh explicitly. Only rejected DuJo cases are repeated.

`audit_tcom_lct_count.py` independently verifies matching raw geometry/traffic,
ordered flow pairs across all counts and methods, nested candidate edges,
terminal availability, routes, directed link loads, source/target budgets, and
that the preceding DuJo matching remains feasible after adding terminals.
`audit_tcom_revision_inputs.py` checks archived ATP, delay, multi-shell and gateway
within-instance input fingerprints. Those comparisons pass. This does not certify
all historical raw states, which are unavailable for the original Fig. 8 sweep.

Corrected fixed-pair results passed all 15 cases and directed-link audits:
DuJo 147.17083544254766 / 254.13145325895312 / 209.75559954001437 Gbps.
Two-LCT results match the original run exactly for every method. The earlier
three-/four-LCT table is superseded. Previous DuJo matchings remain feasible
in the enlarged candidate sets, with the same ordered flow arrays and budgets.
Thus the remaining nonmonotonic result is an algorithm-output limitation, not
loss of feasibility. MRate leads at four LCTs (276.4659058180432 Gbps).

A diagnostic archive field initially applied Earth-radius scaling twice when
saving selected-link capacities; optimization itself used the correct solver
function. The archived field was recomputed from normalized link distances and
the driver corrected. The directed-capacity audit then passed without changing
any optimized rates/routes. See capacity_artifact_note.json and source hashes.

## Iteration diagnostic (2026-09-28)

User requested checking whether the 500-step budget explains the four-LCT drop.
`test_tcom_lct_iterations.py` runs the same fixed-pair, seed-0 DuJo trajectory
for 5000 updates in each 2/3/4-terminal configuration. Evaluate every 100 steps
on a separate deep copy of the solver (the simulation shell is shallow copied;
propagation objects are not picklable and are not used during conversion).
This prevents traffic refreshes and conversion scratch state from modifying
the optimization trajectory. Report 500/1000/2000/5000-step final values and
best feasible values among the every-100-step samples, not all iterates.
The driver checks matching, physical routes, directed capacity and endpoint
budgets at every sample, and asserts the original 500-step throughput exactly.
`audit_tcom_lct_iterations.py` separately checks all 150 saved route/rate samples
and exact 500-step routes, rates and matchings against the previous archives.
Remote diagnostic output is `lct-iterations/`; the existing table is unchanged
until the full sweep and audits are complete. No baseline reruns or DRL retraining.

Completed: all three 5000-step runs and all 150-sample independent audits passed. See `TCOM_LCT_ITERATIONS.md` for final and best-sampled results. Four-LCT drop persists; manuscript table unchanged.

## Current appendix: 90-degree full FOR (2026-09-28)

At the user's request, reran all 15 cases with `--for-half-angle 45` (full
FOR 90 degrees), keeping the original two-LCT/60-degree-half-angle reference
flow assignments and all other inputs fixed. The parameter is local to this
driver; other experiment defaults remain 60-degree half-angle. DuJo retains
500 fixed-traffic iterations and final-iterate conversion. The 60-degree runs
and iteration diagnostic remain archived, not overwritten.

All 15 independent audits passed. Additional comparison with the prior
archives confirms identical positions, mounting directions, gateways, traffic,
and ordered flow pairs. Candidate links are subsets of the previous FOR120
candidates, with 11760/17107/28367 terminal pairs for 2/3/4 LCTs. Each satellite
pair has only one candidate terminal assignment: no overlapping assignment
choices remain at the sampled geometry.

DuJo now achieves 121.90/218.77/263.07 Gbps (31.56/56.63/68.10 percent).
All five methods increase with terminal count; MRate leads at four LCTs with
296.04 Gbps. This does not prove overlap caused the previous decrease, because
changing FOR also removes candidate links. The appendix and Comment 3.6 now
report this 90-degree full-FOR study explicitly. Artifacts are stored locally
in `test_tcom_lct_count/comment-3-6-for90-20260928/` and remotely in `lct-for90/`.
