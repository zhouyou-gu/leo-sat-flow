# Comment 3.4 gateway-count sensitivity

The displayed counts are 50, 100, 150, and 200. Use one nested placement (seed 0),
no averaging, all five methods, and the saved 1000-satellite single-shell population
at 2025-07-16 16:00 UTC. The first 100 catalogue indices reproduce the original
seed-0 choice exactly. The first 50 form its subset; the extra 100 are sampled
without replacement from the remaining catalogue and prefixed for 150/200.
The 400-gateway extension appends a further 200 from the remaining indices using
the same RNG stream, preserving every original 50–200 selection and result.
Catalogue record indices and coordinates are archived. Preserve catalogue records
as in the original experiment; do not silently deduplicate or replace sites.

Only gateway coverage changes. Satellite geometry, candidate LISLs, active user
fraction 0.0001, traffic seed 0, and raw per-satellite user demand are fixed. Local
service changes with gateway coverage, so residual supplying capacity and residual
demand can both change. Each covered satellite has a 20 Gbps budget before local
service, regardless of how many gateways it sees. The served ratio is routed
throughput / residual demand, as in existing figures.

All methods use matched inputs for each gateway count. DuJo uses 500 fixed-traffic
updates and the final iterate, as in the multi-shell revision protocol. Unused
intermediate diagnostics are omitted; a five-step equality check against the
original loop verifies identical final rates, routes, matching, and objective.
DRL uses the frozen April 10 PG checkpoint, with NumPy/Torch seed 0. Physical,
terminal, gateway-range, and solver parameters are unchanged. These are fresh
measurements, not mixtures with earlier legacy-traffic results.

From the isolated 5090 experiment root, use the same virtual environment as the
other revision studies:

```
python sim_alg_v1_res/test_tcom_gateway_count.py --atp-root /home/zhouyou/tcom-comment11-ail --output /home/zhouyou/tcom-comment11-ail/gateway-results --seeds 0 --prepare
python sim_alg_v1_res/test_tcom_gateway_count.py --atp-root /home/zhouyou/tcom-comment11-ail --output /home/zhouyou/tcom-comment11-ail/gateway-results --seeds 0 --workers 4
python sim_alg_v1_res/figures/plot_test_tcom_gateway_count.py --results /home/zhouyou/tcom-comment11-ail/gateway-results
```

Set OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, NUMBA_NUM_THREADS=4. DuJo runs on the
workstation CPU; DRL inference uses CUDA. Multiprocessing uses spawn to avoid inheriting a CUDA context. No retraining occurs. Completed cases
are saved individually; reruns resume them. Inputs/manifest record the selections,
source/checkpoint/catalogue hashes, frozen population, settings, and preflight.
Per-case NPZ archives preserve gateways, raw/residual traffic, selected links,
routes and rates. The driver checks matched fingerprints, feasible terminals and
routes, finite rates, and throughput bounds. The final results/validation appear
only after the complete matrix passes. The plotter requires exactly 25 cases.

Local artifacts belong under ignored
`test_tcom_gateway_count/comment-3-4-20260926-ail/`; remote artifacts remain in
`/home/zhouyou/tcom-comment11-ail/gateway-results/`. The two-panel figure uses
throughput and residual served ratio, enlarged fonts, and no shading. This is one
placement and one snapshot, not a placement-robustness or temporal experiment.

Completed 2026-09-26: all 20 cases passed. Independent NPZ audit verified geometry, raw demand, nested sites, coverage, the single 20 Gbps per-satellite budget, residual supply/demand, and plotted metrics. DuJo leads at all four counts. A CUDA fork-start failure affected only the first DRL attempts; switching workers to spawn completed those four cases without rerunning completed methods. Both logs are retained remotely.

The gateway plot copies Fig. 10’s exact 350-by-285-pixel canvas at 100 DPI,
axis boxes [0.15, 0.55/0.15, 0.825, 0.35], legend box
(0.15, 0.915, 0.825, 0.12), fonts, line/marker sizes, and label spacing.
Axis units and data ranges remain appropriate to the gateway experiment.

400-gateway extension completed 2026-09-26: all 25 cases passed. The original
20 result rows are identical. DuJo remains highest at all five counts, but falls
from 172.71 Gbps / 46.54% at 200 to 150.60 Gbps / 42.46% at 400. +Grid also
declines at 400; this is not a monotonic-performance claim. Layout constants
were checked against Fig. 10 by parsing both plotters.

The user subsequently replaced the displayed 400-gateway case with 250.
The 250 sites are the first 250 indices of the same nested selection, so the
original 50–200 cases remain identical. The completed 400-gateway measurements
are retained in results_with_400.csv and the per-case archives, but excluded
from the active results.csv and figure. No layout settings were changed.

250-gateway replacement completed: all 25 active evaluations passed. DuJo
serves 150.98 Gbps and 41.35% at 250; the 50–200 rows remain identical.
The figure uses 50/100/150/200/250 and retains the exact Fig. 10 layout constants.

2026-09-27: user removed 250 from the displayed sweep. Plotter filters to
50/100/150/200 (20 evaluations), preserving 250/400 raw data and unchanged
Fig. 10 layout constants. Driver/results retain the completed extended runs.
