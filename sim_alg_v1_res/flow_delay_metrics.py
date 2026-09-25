"""Physical LISL propagation statistics; positions use Earth-radius units."""
import numpy as np
EARTH_RADIUS_KM = 6371.0
LIGHT_SPEED_KM_S = 299792.458
RATE_EPS = 1e-9


def route_delays(positions, paths, lengths, rates, sources, targets, connected):
    positions = np.asarray(positions)
    assert np.isfinite(positions).all()
    assert len(paths) == len(lengths) == len(rates) == len(sources) == len(targets)
    assert np.isfinite(rates).all() and np.min(rates, initial=0) >= -RATE_EPS
    edges = {tuple(sorted(map(int, e))) for e in connected}
    delays = np.full(len(rates), np.nan)
    for i, n in enumerate(lengths):
        n = int(n)
        if n <= 0:
            assert rates[i] <= RATE_EPS, 'Unreachable flow has positive rate'
            continue
        route = np.asarray(paths[i, :n], dtype=int)
        assert np.all((route >= 0) & (route < len(positions)))
        assert route[0] == sources[i] and route[-1] == targets[i]
        assert len(set(route)) == len(route), 'Route contains cycle'
        assert all(tuple(sorted((int(u), int(v)))) in edges for u, v in zip(route[:-1], route[1:]))
        km = np.linalg.norm(np.diff(positions[route], axis=0), axis=1).sum() * EARTH_RADIUS_KM
        if rates[i] > RATE_EPS:
            delays[i] = km / LIGHT_SPEED_KM_S * 1000
    return delays


def weighted_stats(delays, rates):
    keep = (np.asarray(rates) > RATE_EPS) & np.isfinite(delays)
    d, w = np.asarray(delays)[keep], np.asarray(rates)[keep]
    if not len(d):
        return {'mean_delay_ms': None, 'p95_delay_ms': None}
    order = np.argsort(d, kind='stable')
    p95 = d[order][np.searchsorted(np.cumsum(w[order]), 0.95 * w.sum(), side='left')]
    return {'mean_delay_ms': float(np.average(d, weights=w)), 'p95_delay_ms': float(p95)}
