#!/usr/bin/env python3
"""Plot the complete Comment 2.1 matrix; never select a run by modification time."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ['DuJo', 'DRL', 'SaTE', 'MRate', '+Grid']
CONFIGS = ['shell53', 'shell43', 'mixed']
OFFSETS = [0, 30, 60, 90, 120, 180]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
MARKERS = ['o', 's', '^', 'D', 'v']
STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(p.read_text()) for p in sorted((args.results / 'results').glob('*.json'))]
    keys = {(r['config'], r['selection'], r['offset_minutes'], r['method']) for r in rows}
    expected = {(c, s, t, m) for c in CONFIGS for s in range(3) for t in OFFSETS for m in METHODS}
    assert keys == expected and len(rows) == 270, 'Incomplete or duplicate experiment matrix'
    for c in CONFIGS:
        for s in range(3):
            for t in OFFSETS:
                same = [r for r in rows if (r['config'], r['selection'], r['offset_minutes']) == (c,s,t)]
                assert all(r['fingerprint'] == same[0]['fingerprint'] for r in same)
                assert len({r['offered_demand_gbps'] for r in same}) == 1
    fields = [k for k in rows[0] if k != 'fingerprint']
    with (args.results / 'results.csv').open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['STIXGeneral'],
                         'mathtext.fontset': 'cm', 'font.size': 9, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.8), sharex=True)
    summaries = {}
    for j, config in enumerate(CONFIGS):
        subset = [r for r in rows if r['config'] == config]
        summaries[config] = {}
        for method, color, marker, style in zip(METHODS, COLORS, MARKERS, STYLES):
            data = [r for r in subset if r['method'] == method]
            summaries[config][method] = {
                metric: {'mean': float(np.mean([r[metric] for r in data])),
                         'min': float(min(r[metric] for r in data)),
                         'max': float(max(r[metric] for r in data))}
                for metric in ['served_throughput_gbps', 'served_ratio', 'selected_inter_shell_pairs', 'evaluation_seconds']}
            for i, metric in enumerate(['served_throughput_gbps', 'served_ratio']):
                values = np.array([[next(r[metric] for r in data if r['selection']==s and r['offset_minutes']==t)
                                    for s in range(3)] for t in OFFSETS])
                ax = axes[i,j]
                ax.fill_between(OFFSETS, values.min(axis=1), values.max(axis=1), color=color, alpha=.10, linewidth=0)
                ax.plot(OFFSETS, values.mean(axis=1), color=color, marker=marker,
                        linewidth=1.5, linestyle=style, markersize=3.4, label=method)
        wins, gains = 0, []
        for s in range(3):
            for t in OFFSETS:
                d = {r['method']:r['served_throughput_gbps'] for r in subset if r['selection']==s and r['offset_minutes']==t}
                best = max(d[m] for m in METHODS if m != 'DuJo')
                wins += int(d['DuJo'] > best + 1e-6)
                if best > 0: gains.append(100*(d['DuJo']/best-1))
        summaries[config]['paired_comparison'] = {'dujo_strict_wins':wins, 'instances':18,
            'relative_difference_defined_instances':len(gains),
            'gain_over_best_baseline_percent': {'min':min(gains) if gains else None,
                'max':max(gains) if gains else None, 'mean':float(np.mean(gains)) if gains else None}}
        axes[0,j].set_title(['(a) 53.22° cluster', '(b) 43.00° cluster', '(c) Two-cluster mixture'][j], fontsize=10)
        axes[1,j].set_xlabel('Time offset (min)')
        for ax in axes[:,j]:
            ax.grid(alpha=.25, linewidth=.5)
            ax.set_xticks(OFFSETS)
            ax.set_ylim(bottom=0)
            ax.tick_params(width=.8)
        axes[0,j].set_ylim(0, 1.08 * max(r['served_throughput_gbps'] for r in rows))
        axes[1,j].set_ylim(0, 1)
    axes[0,0].set_ylabel('Throughput (Gbps)')
    axes[1,0].set_ylabel('Served-demand ratio')
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc='upper center', ncol=5,
               bbox_to_anchor=(.52, 1), frameon=False, columnspacing=1.8)
    fig.subplots_adjust(left=.085, right=.99, bottom=.12, top=.85, wspace=.27, hspace=.20)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    (args.results / 'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
