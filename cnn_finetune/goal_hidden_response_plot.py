"""Report figure from frozen response-identification outputs."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .goal_hidden_response_experiment import REPORT,OUT
from .goal_image_response import predict,NOMINAL
from .goal_hidden_response_score import true_values


def main():
    rows=json.loads((REPORT/'scores.json').read_text())
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    ax=axs[0,0];folder=OUT/'probe_08';data=json.loads((folder/'input.json').read_text());fits=json.loads((folder/'fits.json').read_text())
    intervals=[dict(start=float(t),end=float(t+.2)) for t in np.arange(.2,11.81,.2)]
    ts=np.array([r['end'] for r in intervals]);truth=np.load(folder/'scorer.npz')['truth']
    ax.plot(ts,true_values(truth,intervals)[:,0],color='black',label='Actual (scorer only)',lw=2)
    ax.plot(ts,predict(data['commands'],intervals,NOMINAL)[:,0],color='#d35e37',label='Fixed nominal model')
    ax.plot(ts,predict(data['commands'],intervals,fits[-1]['best'])[:,0],color='#168575',label='Image + PWM fit at 6s')
    ax.axvspan(0,6,color='grey',alpha=.12,label='History available for fit')
    ax.axvline(6,color='grey',ls='--');ax.set(xlabel='Time (s)',ylabel='Forward speed (cm/s)',title='A. Hidden sluggish vehicle: later response prediction')
    ax.legend(fontsize=8,loc='lower left')
    ax=axs[0,1];labels=['nominal','low_gain','delay_only','lag_only','sluggish','unequal_sides']
    for i,label in enumerate(labels):
        selected=[r for r in rows if r['group']=='probe' and r['vehicle']==label and r['condition']!='shove']
        p=selected[0]['parameters'];actual=p['actuation_delay_s']+p['time_constant_s']
        ax.plot([i-.32,i+.32],[actual,actual],color='black',lw=2,label='Actual delay + lag' if i==0 else None)
        for j,r in enumerate(selected):
            c='#168575' if r['condition']=='normal' else '#bd8240'
            x=i+(-.15 if r['condition']=='normal' else .15)+(.04 if r['seed']==24002 else -.04)
            ax.scatter(x,r['fits'][-1]['sum'],c=c,s=35,label=r['condition'] if i==0 and r['seed']==24001 else None)
    ax.set(xticks=range(6),xticklabels=['Nominal','Low gain','Delay','Lag','Both','Unequal'],ylabel='Delay + time constant (s)',title='B. Response difference inferred without vehicle labels')
    ax.legend(fontsize=8)
    ax=axs[1,0];colors={'field_estimate':'#356abd','sluggish':'#b34556','unequal_sides':'#168575'}
    for vehicle,c in colors.items():
        selected=[r for r in rows if r['group']=='natural' and r['vehicle']==vehicle]
        ax.scatter([r['nominal']['combined_rmse'] for r in selected],
                   [r['fitted']['combined_rmse'] for r in selected],c=c,label=vehicle,s=40,alpha=.8)
    ax.plot([.01,3],[.01,3],color='grey',ls='--',label='No improvement')
    ax.set(xscale='log',yscale='log',xlabel='Fixed nominal prediction error',ylabel='Image + PWM fitted prediction error',title='C. Existing driving: lower than diagonal is better')
    ax.legend(fontsize=8)
    ax=axs[1,1];groups=[('probe','normal'),('probe','light'),('probe','shove'),('natural','normal')]
    for i,(group,condition) in enumerate(groups):
        selected=[r for r in rows if r['group']==group and r['condition']==condition]
        vals=np.array([r['usable_intervals']/60*100 for r in selected]);jitter=np.linspace(-.15,.15,len(vals))
        ax.scatter(i+jitter,vals,color=['#168575','#bd8240','#9470ab','#356abd'][i],s=25,alpha=.7)
        ax.plot([i-.25,i+.25],[np.median(vals)]*2,color='black',lw=2)
    ax.axhline(90,color='grey',ls='--',label='Declared 90% gate')
    ax.set(xticks=range(4),xticklabels=['Probe normal','Probe light','Probe shove','Natural drive'],ylabel='Usable visual intervals (%)',ylim=(0,105),title='D. Observation reliability is still a bottleneck')
    ax.legend(fontsize=8)
    fig.suptitle('Hidden vehicle response feasibility | Simulation only | No neural training',fontsize=14)
    fig.savefig(REPORT/'response_diagnosis.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
