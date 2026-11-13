"""Standalone figures for the frozen-model robustness comparison."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(folder):
    s=json.loads((folder/'summary.json').read_text());pairs=json.loads((folder/'pairs.json').read_text())
    groups=list(s['groups']);labels=['Normal (4 vehicles)','Single disturbances','Combined mild + bright','Harsh camera factors','Harsh stress']
    colors=['#64748b','#087e8b'];arms=['a5b_v4','direct_v2'];names=['A5b + v4 allocator','direct_v2']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    x=np.arange(len(groups));w=.36
    for ax,metric,title in zip(axes,['laps','clean'],['Verified completion','Completion without boundary intrusion']):
        for j,arm in enumerate(arms):
            vals=[s['groups'][g][arm][metric]/s['groups'][g][arm]['n']*100 for g in groups]
            bars=ax.bar(x+(j-.5)*w,vals,w,label=names[j],color=colors[j])
            for b,g in zip(bars,groups):
                r=s['groups'][g][arm];ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f"{r[metric]}/{r['n']}",ha='center',fontsize=9)
        ax.set_xticks(x,labels,rotation=23,ha='right');ax.set_ylim(0,113);ax.set_ylabel('Runs (%)');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(loc='lower left');fig.suptitle('Frozen models | 3 seeds x 2 directions | 240 simulations',fontsize=14)
    fig.savefig(folder/'group_results.png',dpi=170);plt.close(fig)
    conditions=list(s['conditions']);cols=[(seed,d) for seed in [601,607,613] for d in ['cw','ccw']]
    matrix=np.array([[next(p['margin_delta_cm'] for p in pairs if p['condition']==c and (p['seed'],p['direction'])==sd) for sd in cols] for c in conditions])
    fig,ax=plt.subplots(figsize=(10,10),layout='constrained')
    im=ax.imshow(matrix,cmap='RdYlGn',vmin=-6,vmax=6,aspect='auto')
    for i in range(len(conditions)):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',fontsize=8,color='black')
    ax.set_yticks(range(len(conditions)),conditions);ax.set_xticks(range(6),[f'{sd} {d}' for sd,d in cols]);ax.set_title('Change in minimum boundary clearance: direct_v2 minus A5b (cm)\nPositive = better; colors clipped at +/-6 cm, numbers are not clipped')
    fig.colorbar(im,ax=ax,label='Clearance change (cm)',shrink=.7)
    fig.savefig(folder/'paired_clearance.png',dpi=170);plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder');a=ap.parse_args();plot(Path(a.folder))
