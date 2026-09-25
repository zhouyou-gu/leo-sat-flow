# Comment 3.1 propagation-delay evaluation

Run `test_flow_delay_metrics.py` for the synthetic distance, exclusion, weighted-statistic,
and invalid-route checks. Run `test_tcom_flow_delay.py --atp-root PATH --output PATH
--code-revision REV` on the existing RTX 5090 experiment host. The ATP root supplies
trusted DuJo pickle caches, population metadata, the frozen PG checkpoint, and saved
zero-ATP results. No DuJo optimization is performed. The other four methods are
reevaluated with NumPy/Torch seed 0 at every snapshot and traffic seed 0.

Use the saved single-shell population at all 37 offsets from 0 to 360 s, including
zero. Physical parameters, gateway inputs, traffic, and candidate topology match
the ATP run. Acquisition overhead is absent. DuJo caches retain the original 500
legacy-traffic dual updates and final evaluation at seed 0. DRL retains its April
10 checkpoint; no retraining occurs. Stochastic DRL decisions need not reproduce
the earlier ATP run, but all five methods must have identical input fingerprints.

For every served flow (rate > 1e-9 Gbps), sum Euclidean distances along its selected
satellite route. Coordinates are normalized by 6371 km; divide physical distance
by 299792.458 km/s and convert to milliseconds. `lengths` counts vertices, so it
is not the propagation distance or hop count. Exclude unserved/unreachable flows
from delay statistics; undefined delays are NaN in per-flow artifacts and blank
in summary CSV. No delay is imputed as zero for an unserved flow.

Pool all served flow-snapshot records, with allocated rate as the weight. Report
the weighted mean and the smallest observed delay whose cumulative weight reaches
95%. The snapshot spacing is uniform. Throughput and served-demand ratio are
arithmetic means over all 37 snapshots. These delay statistics describe each
method's served traffic, which can differ between methods.

Each method/offset NPZ stores positions, full padded paths, vertex counts, source
and destination IDs, allocated rates, selected links, and per-flow delay. JSON
sidecars record input/artifact hashes and statistics. The driver validates route
endpoints/edges, alignment, finite positions/rates, throughput bounds and agreement
with saved deterministic zero-ATP results. Summary and validation files appear
only after all 185 cases pass. A missing NPZ/JSON pair is recomputed on resume.
Keep artifacts in the ignored `test_tcom_flow_delay/comment-3-1-20260925-ail/`
directory locally; remote data are in `/home/zhouyou/tcom-comment11-ail/delay-results`.

Completed 2026-09-25: all 185 cases passed driver checks and an independent audit of input/artifact hashes, straight-line delay lower bounds, flow exclusions, throughput, weighted means, and weighted inverse-CDF percentiles. `validation.json`, `independent_audit.json`, and `source_manifest.json` record the evidence. The executed source archive accompanies the ignored results.
