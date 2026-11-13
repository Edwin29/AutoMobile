"""Static diagnostic figure; curves stop at the actual termination time."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .recovery_diagnosis import CASES


def plot(root,out):
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),sharey=True)
    modes=[('student','Student','#d64545'),('teacher_burst','Teacher 0.72s, then student','#3465bd'),
           ('teacher','Teacher throughout','#36994b'),('left','Left candidate 0.72s','#b17a1a')]
    for ax,case in zip(axes,CASES):
        for mode,label,color in modes:
            rows=json.loads((root/(case['name']+'__'+mode)/'window.json').read_text())
            ax.plot([r['t']-case['start'] for r in rows],[r['margin'] for r in rows],label=label,color=color,lw=1.7)
        ax.axhline(0,color='black',lw=.8);ax.axvspan(0,.72,color='gray',alpha=.15)
        ax.set_title(case['name'].replace('_',' '));ax.set_xlabel('Seconds after intervention');ax.set_xlim(0,15)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Footprint boundary clearance (cm)')
    axes[0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Identical replay prefix, different actions; no state reset')
    fig.tight_layout();fig.savefig(out/'recovery_curves.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    plot(Path(a.root),Path(a.out))
