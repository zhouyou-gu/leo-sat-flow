# DuJo iteration diagnostic — 2026-09-28

Completed on the RTX 5090 workstation (DuJo uses CPUs). Same fixed-pair 1000-satellite snapshot and seed-0 inputs as the corrected LCT comparison; only iteration budget varies. Step size remains 0.1 / sqrt(iteration). No manuscript/table changes.

| Iterations | 2 LCTs (Gbps) | 3 LCTs (Gbps) | 4 LCTs (Gbps) |
|---|---:|---:|---:|
| 500 | 147.17 | 254.13 | 209.76 |
| 1000 | 150.67 | 260.55 | 208.91 |
| 2000 | 146.13 | 258.65 | 199.13 |
| 5000 | 153.62 | 267.33 | 220.24 |

These are final-iterate converted throughputs, matching the existing reporting protocol.

| LCTs | Best sampled throughput (Gbps) | Iteration |
|---|---:|---:|
| 2 | 157.23 | 4400 |
| 3 | 280.08 | 3600 |
| 4 | 238.79 | 3200 |

Best sampled means the best of the 50 evaluations at iterations 100, 200, ..., 5000, not the best of every optimization iterate. Conversion uses a deep-copied solver so it cannot affect subsequent multiplier updates.

All 150 saved feasible solutions passed an independent audit of matching, routes, nonnegative rates, directed-link capacities, endpoint budgets, throughput and best-so-far accounting. The 500-step rates, routes and terminal matchings exactly reproduce the previous archive for all three counts.

Conclusion: a larger budget can find better converted solutions, but final-iterate throughput fluctuates and 5000 updates do not remove the four-LCT decrease. These runs do not establish convergence or optimality. Retaining the best feasible solution would be a separate algorithm/reporting change and would need to be applied consistently; the existing manuscript has not been silently switched to that protocol.

Artifacts: `test_tcom_lct_count/comment-3-6-iterations-20260928/` locally and `/home/zhouyou/tcom-comment11-ail/lct-iterations/` remotely. Includes every sampled route/rate result, traces, 12-row summary, settings, source snapshots/hashes and independent audit.
