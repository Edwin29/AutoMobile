"""Audit all shards and produce a contact sheet without loading the dataset together."""
import json,hashlib
from pathlib import Path
import numpy as np,cv2
root=Path('D:/autodrive_cnn/data/v1');rows=[];tiles=[]
for p in sorted(root.glob('seed_*.npz')):
    meta=json.loads(p.with_suffix('.json').read_text())
    with np.load(p) as z:
        assert z['x'].dtype==np.uint8 and z['x'].shape[1:]==(3,66,200)
        assert len(z['y'])==len(meta)
        assert np.isfinite(z['y']).all() and np.max(np.abs(z['y']))<=1
        assert np.allclose(z['y'][:,0],np.array([m['mean'] for m in meta])/.66)
        assert np.allclose(z['y'][:,1],-np.array([m['k'] for m in meta])/.095)
        rows.append(dict(seed=meta[0]['seed'],n=len(meta),corner=int(z['corner'].sum()),recovery=int(z['recovery'].sum()),inner=sum(m['road']=='inner' for m in meta),minimum_junction_distance=min(m['junction_distance'] for m in meta),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        if len(tiles)<12:
            i=int(np.argmax(z['corner'])) if len(tiles)%2 else 0
            tile=cv2.cvtColor(z['x'][i].transpose(1,2,0),cv2.COLOR_YUV2BGR)
            tile=cv2.resize(tile,(400,132));cv2.putText(tile,f"seed {meta[i]['seed']} {meta[i]['road']} k={meta[i]['k']:.3f}",(4,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);tiles.append(tile)
n=sum(r['n'] for r in rows);corner=sum(r['corner'] for r in rows)/max(n,1);recovery=sum(r['recovery'] for r in rows)/max(n,1)
result=dict(count=n,corner_fraction=corner,recovery_fraction=recovery,inner_count=sum(r['inner'] for r in rows),minimum_junction_distance=min(r['minimum_junction_distance'] for r in rows),shards=rows)
result['ok']=30000<=n<=80000 and corner>=.25 and recovery>=.15 and result['minimum_junction_distance']>=35
Path('cnn_finetune/results/DATA_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if len(tiles)==12:cv2.imwrite(str(root/'contact_sheet.jpg'),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,12,3)]))
print(json.dumps({k:v for k,v in result.items() if k!='shards'}))
