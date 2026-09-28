"""Two-panel gateway-count figure; one curve per method, no averaging/shading."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from legacy_augmented_plot_common import apply_plot_style, METHOD_COLORS

METHODS=['DuJo','DRL','SaTE','MRate','+Grid']
COUNTS=[50,100,150,200]
MARKERS=['o','s','^','D','v']

def main(args):
    d=pd.read_csv(Path(args.results)/'results.csv')
    d=d[d.gateway_count.isin(COUNTS)].copy()
    assert len(d)==20 and set(d.seed)=={0}
    assert not d.duplicated(['gateway_count','method']).any()
    apply_plot_style()
    plt.rcParams.update({'font.size':10,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':9,'legend.fontsize':10})
    # Exact dimensions and axis/legend boxes from Fig. 10's ATP plotter.
    FIG_WIDTH_PX = 350
    FIG_HEIGHT_PX = 285
    DPI = 100
    TOP_AXIS_BOX = [0.15, 0.55, 0.825, 0.35]
    BOTTOM_AXIS_BOX = [0.15, 0.15, 0.825, 0.35]
    LEGEND_BOX = (0.15, 0.915, 0.825, 0.12)
    fig,axs=plt.subplots(2,1,figsize=(FIG_WIDTH_PX / DPI,FIG_HEIGHT_PX / DPI))
    for method,marker in zip(METHODS,MARKERS):
        rows=d[d.method==method].sort_values('gateway_count');assert rows.gateway_count.tolist()==COUNTS
        for ax,key in zip(axs,['served_throughput_gbps','served_ratio']):
            ax.plot(rows.gateway_count,rows[key],label=method,color=METHOD_COLORS[method],marker=marker,markersize=4,linewidth=1.2,markerfacecolor='none',markeredgecolor=METHOD_COLORS[method],markeredgewidth=1.0)
    axs[0].set_ylabel('Throughput (Gbps)');axs[1].set_ylabel('Served Ratio');axs[1].set_xlabel('Number of Gateways')
    for ax in axs:
        ax.set_xticks(COUNTS);ax.grid(True,alpha=.35,linewidth=.5);ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        for spine in ax.spines.values():spine.set_linewidth(.8)
    axs[0].set_ylim(0,d.served_throughput_gbps.max()*1.12)
    axs[1].set_ylim(0,min(1.,d.served_ratio.max()*1.12))
    axs[0].tick_params(labelbottom=False)
    axs[0].set_position(TOP_AXIS_BOX);axs[1].set_position(BOTTOM_AXIS_BOX)
    handles,labels=axs[0].get_legend_handles_labels()
    legend=fig.legend(handles,labels,loc='lower left',bbox_to_anchor=LEGEND_BOX,mode='expand',ncol=5,borderaxespad=0.,frameon=True,fancybox=False,edgecolor='black',facecolor='white',framealpha=1,borderpad=.3,labelspacing=.2,handlelength=1,handleheight=.8,handletextpad=.2)
    legend.get_frame().set_linewidth(.8)
    output=Path(__file__).with_suffix('.pdf');fig.savefig(output,format='pdf',pad_inches=0.);print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);main(p.parse_args())
