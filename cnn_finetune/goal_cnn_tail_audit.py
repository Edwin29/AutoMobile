"""Fixed-dataset fitting audit; not a new validation or checkpoint selector."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .goal_cnn_tail import OUT,REPORT,SOURCE,INITIAL,source_rows,bundle_load,sha
from .junction_learning import setup,model_at,Encoder
from .junction_stage02 import dump


def main():
    setup();rows=sum(source_rows(),[]);cache=json.loads((REPORT/'cache.json').read_text())['files']
    models={'initial':(model_at(INITIAL),None)}
    for arm in ['teacher_only','full_dagger']:
        models['frozen_'+arm]=(model_at(SOURCE/'models'/arm/'update_1200.pt'),None)
        model,enc=bundle_load(OUT/'models'/arm/'update_1200.pt')
        models['tail_'+arm]=(model,enc.net[6:])
    records=[]
    with torch.no_grad():
        for row in rows:
            with np.load(Path(row['artifact'])/'sequence.npz') as z:
                arrays={k:z[k] for k in ['f','aux','label','supervise','margin','capture']}
            features=torch.from_numpy(arrays['f']).cuda().float()[None]
            aux=torch.from_numpy(arrays['aux']).cuda().float()[None]
            labels=torch.from_numpy(arrays['label']).cuda().float()[None]
            prefix=torch.from_numpy(np.load(cache[row['artifact']])).cuda()
            valid=(arrays['supervise']>0)&(arrays['margin']>=-3)
            for name,(model,tail) in models.items():
                pred,_=model(features if tail is None else tail(prefix)[None],aux)
                ae=abs(pred-labels).mean(-1)[0].cpu().numpy()
                huber=F.smooth_l1_loss(pred,labels,beta=.05,reduction='none').mean(-1)[0].cpu().numpy()
                turn=1+2*(abs(arrays['label'][:,0]-arrays['label'][:,1])>.15)
                for segment,mask in [('all',valid),('entry_3_6s',valid&(arrays['capture']>=3)&(arrays['capture']<6))]:
                    if not mask.any():continue
                    records.append(dict(model=name,source=row['source'],key=row['key'],condition=row['condition'],
                        segment=segment,frames=int(mask.sum()),mae=float(ae[mask].mean()),
                        weighted_loss=float((huber[mask]*turn[mask]).sum()/turn[mask].sum())))
    summary=[]
    for name in models:
        for source in ['teacher','replay','student']:
            for segment in ['all','entry_3_6s']:
                a=[r for r in records if (r['model'],r['source'],r['segment'])==(name,source,segment)]
                summary.append(dict(model=name,source=source,segment=segment,episodes=len(a),
                    mean_episode_mae=float(np.mean([r['mae'] for r in a])),
                    mean_episode_weighted_loss=float(np.mean([r['weighted_loss'] for r in a]))))
    original=Encoder().state_dict();deltas={}
    for arm in ['teacher_only','full_dagger']:
        enc=torch.load(OUT/'models'/arm/'update_1200.pt',weights_only=True,map_location='cpu')['encoder']
        delta={k:float((v-original[k]).abs().max()) for k,v in enc.items()}
        assert all(v==0 for k,v in delta.items() if k.startswith(('net.0.','net.2.','net.4.')))
        assert any(v>0 for k,v in delta.items() if k.startswith(('net.6.','net.8.')))
        deltas[arm]=delta
    dump(REPORT/'fitting_audit.json',dict(source_sha256=sha(__file__),summary=summary,rows=records,cnn_deltas=deltas,
        interpretation='Training-source teacher-forced diagnostics only. No closed-loop or independent generalization claim.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
