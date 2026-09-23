#!/usr/bin/env python3
"""Plot the first predetermined two-shell population over time, as in Fig. 5(a)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from plot_test_tcom_multishell import METHODS, OFFSETS, COLORS, MARKERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(p.read_text()) for p in sorted((args.results / 'results').glob('*.json'))]
    rows = [r for r in rows if r['config'] == 'mixed' and r['selection'] == 0]
    expected = {(s, t, m) for s in [0] for t in OFFSETS for m in METHODS}
    assert len(rows) == 30 and {(r['selection'], r['offset_minutes'], r['method']) for r in rows} == expected, 'Incomplete or duplicate two-shell results'
    for s in [0]:
        for t in OFFSETS:
            same = [r for r in rows if (r['selection'], r['offset_minutes']) == (s, t)]
            assert all(r['fingerprint'] == same[0]['fingerprint'] for r in same)
            assert len({r['offered_demand_gbps'] for r in same}) == 1

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                         'font.size': 9, 'mathtext.fontset': 'cm', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.85))
    for method, color, marker in zip(METHODS, COLORS, MARKERS):
        for ax, metric in zip(axes, ['served_throughput_gbps', 'served_ratio']):
            values = [next(r[metric] for r in rows if r['method'] == method
                           and r['offset_minutes'] == t) for t in OFFSETS]
            ax.plot(OFFSETS, values, label=method, color=color,
                    marker=marker, linewidth=1.2, markersize=4)
    axes[0].set_ylabel('Throughput (Gbps)')
    axes[1].set_ylabel('Served Ratio')
    axes[1].set_xlabel('Time Offset from $T_0$ (min)')
    axes[0].tick_params(labelbottom=False)
    for ax, metric, margin_fraction in zip(axes, ['served_throughput_gbps', 'served_ratio'], [.08, .10]):
        ax.grid(True, zorder=0, alpha=.35, linewidth=.5)
        ax.set_xticks(OFFSETS)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis='y', labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(.8)
        low, high = min(r[metric] for r in rows), max(r[metric] for r in rows)
        margin = margin_fraction * (high - low)
        ax.set_ylim(max(0, low - margin), min(1, high + margin) if metric == 'served_ratio' else high + margin)
    axes[0].set_position([.15, .55, .825, .35])
    axes[1].set_position([.15, .15, .825, .35])
    legend = fig.legend(*axes[0].get_legend_handles_labels(), fontsize=9,
                        loc='lower left', bbox_to_anchor=(.15, .915, .825, .12),
                        mode='expand', ncol=5, borderaxespad=0, frameon=True,
                        fancybox=False, edgecolor='black', facecolor='white',
                        framealpha=1, borderpad=.3, labelspacing=.2,
                        handlelength=1, handleheight=.8, handletextpad=.2)
    legend.get_frame().set_linewidth(.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format='pdf', pad_inches=0)
    plt.close(fig)


if __name__ == '__main__':
    main()
