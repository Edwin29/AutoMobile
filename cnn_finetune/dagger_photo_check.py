"""Local saved-photo output comparison. No Pi connection or runtime imports."""
import json,csv,hashlib
from pathlib import Path
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .teacher import Teacher,MotorModel
from .model import A5B_ONNX,preprocess_bgr

def infer(net,image):net.setInput(preprocess_bgr(image)[None]);return net.forward()[0]
def nominal(motor,mean,k,adapter=None,source_pwm=None):
    values=[];duties=[]
    for _ in range(256):
        p=adapter._remap_cnn(source_pwm) if adapter is not None else (motor.mix(mean,k,.66)[0] if mean>=.05 else np.zeros(2))
        values.append(motor.velocity(p));duties.append(np.mean(p))
    v,w=np.mean(values,axis=0)
    return float(np.mean(duties)),float(w/v) if v>1e-6 else 0.
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);a=ap.parse_args()
    root=Path('D:/autodrive_cnn/pi_data/20261111/hybrid_mission_pi_runs');out=Path('cnn_finetune/results');rawout=Path('D:/autodrive_cnn/logs/dagger_photos');rawout.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1);base=cv2.dnn.readNetFromONNX(np.frombuffer(A5B_ONNX.read_bytes(),np.uint8));new=cv2.dnn.readNetFromONNX(np.frombuffer(Path(a.model).read_bytes(),np.uint8));teacher=Teacher('cw');summaries=[];total=0
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():continue
        rows=[];bm=MotorModel(dict(teacher.motor.c));nm=MotorModel(dict(teacher.motor.c));manifest=[];teacher.adapter.driver.motor=bm;teacher.adapter.cnn_alloc=bm
        for p in sorted(folder.glob('*.jpg'),key=lambda p:int(p.stem)):
            data=p.read_bytes();image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR);assert image is not None
            b=infer(base,image);n=infer(new,image);assert np.isfinite(b).all() and np.isfinite(n).all()
            pwm=teacher.team_pwm(b);mean=float(pwm.mean());raw=(pwm[0]-pwm[1])/13;k=raw*bm.yaw_gain(raw)/max(mean,1e-6)
            bd,bk=nominal(bm,mean,k,teacher.adapter,pwm);nd,nk=nominal(nm,.66*float(np.clip(n[0],0,1)),float(np.clip(-n[1]*.095,-.085,.085)))
            rows.append(dict(frame=int(p.stem),baseline_mean=bd,baseline_k=bk,new_mean=nd,new_k=nk,new_raw0=float(n[0]),new_raw1=float(n[1])))
            manifest.append(dict(name=p.name,sha256=hashlib.sha256(data).hexdigest()))
        total+=len(rows);x=np.array([r['frame'] for r in rows]);b=np.array([[r['baseline_mean'],r['baseline_k']] for r in rows]);n=np.array([[r['new_mean'],r['new_k']] for r in rows]);corner=np.abs(b[:,1])>.02
        summary=dict(run=folder.name,count=len(rows),corner_proxy_count=int(corner.sum()),mean_abs_difference=np.abs(n-b).mean(axis=0).tolist(),corner_k_difference=float(np.abs(n[corner,1]-b[corner,1]).mean()) if corner.any() else None,baseline_jump_p95=np.percentile(np.abs(np.diff(b,axis=0)),95,axis=0).tolist(),new_jump_p95=np.percentile(np.abs(np.diff(n,axis=0)),95,axis=0).tolist(),new_min=n.min(axis=0).tolist(),new_max=n.max(axis=0).tolist(),negative_raw_speed=sum(r['new_raw0']<0 for r in rows),new_large_k_jumps=int(np.sum(np.abs(np.diff(n[:,1]))>.02)))
        summaries.append(summary)
        with (rawout/(folder.name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (rawout/(folder.name+'_manifest.json')).write_text(json.dumps(manifest,indent=2))
        fig,ax=plt.subplots(2,1,figsize=(12,6),sharex=True)
        for j,label in enumerate(['Mean duty','Curvature (1/cm; right +)']):ax[j].plot(x,b[:,j],label='A5b + v4',alpha=.8);ax[j].plot(x,n[:,j],label='direct v2',alpha=.8);ax[j].set_ylabel(label);ax[j].grid(alpha=.2);ax[j].legend()
        ax[1].set_xlabel('Saved frame number (gaps are not uniform)');fig.suptitle(folder.name+' — offline nominal outputs, no ground truth');fig.tight_layout();fig.savefig(out/(folder.name+'_dagger.png'),dpi=130);plt.close(fig)
    assert total==1072
    result=dict(total=total,runs=summaries,model_sha256=hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),baseline_sha256=hashlib.sha256(A5B_ONNX.read_bytes()).hexdigest(),note='Already-upright saved images, no extra rotation. Per-frame held command averaged over 256 allocator steps. No ground truth; corner proxy uses A5b |k|>0.02. Differences across saved frames are not per-second rates.')
    (out/'DAGGER_PHOTO_CHECK.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
