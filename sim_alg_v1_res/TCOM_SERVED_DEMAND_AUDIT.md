# Comment 3.2 served-demand audit

`test_tcom_served_demand_audit.py --atp-root PATH --output PATH` reads the trusted
37 DuJo caches from Comment 1.1 and does not rerun optimization. The same snapshots
were used for the Comment 3.1 delay appendix. They are diagnostic observations,
not replacements for the original six-time Fig. 5 / load-stress measurements.

The plotted ratio is sum of routed rates divided by residual demand, after local
service. `compute_metrics` uses `forward_traffic_demand` for the denominator.
Local traffic is absent from both numerator and denominator. Local service is
reconstructed as 20 Gbps times the number of gateway-connected satellites minus
residual supplying capacity, using the original 800 km gateway range and 6371 km
Earth-radius normalization. Summing local service and residual demand reconstructs
total user demand. Including local service gives a different metric (mean 43.94%
versus the published residual-demand mean of 39.08% for these snapshots).

The selected-topology bound is a single-commodity maximum flow with source arcs
bounded by residual supplying capacities, sink arcs bounded by residual demands,
and both directions of each selected LISL bounded by its physical capacity.
Parallel selected terminal links are summed. It relaxes the M=5 source-pair
restriction and unsplittable routing. The candidate bound also enables every
candidate terminal pair simultaneously, relaxing terminal matching. Neither
bound establishes the best feasible terminal matching or the global optimality
of DuJo. Unordered terminal pairs are deduplicated so reverse listings are not
counted twice. `test_served_demand_bounds.py` tests capacity cuts, parallel route
relaxation, reversed-pair deduplication, and disconnected demand.

All snapshots satisfy achieved throughput <= reachability/flow/supply/demand
bounds. The existing 240-row load-scaling results also reproduce served ratio
as routed throughput / residual demand. No figure or historical result is changed.

Observed means in Gbps: residual demand 382.18; residual supply 1619.29;
local service 33.14; routed throughput 149.26; selected-topology relaxed bound
179.60; candidate-topology relaxed bound 382.18. Mean snapshot ratios are 39.08%
(achieved) and 47.03% (selected-topology relaxed bound); these are means of ratios,
not ratios of means. The latter bounds do not isolate a unique contribution from
matching, routing, and capacity restrictions or rule out optimization improvement.

Artifacts are stored under the ignored local
`test_tcom_served_demand_audit/comment-3-2-20260925-ail/` directory and remote
`/home/zhouyou/tcom-comment11-ail/demand-audit/`. Snapshot rows include source-cache
hashes; validation metadata contains the audit-source hash and physical constants.
