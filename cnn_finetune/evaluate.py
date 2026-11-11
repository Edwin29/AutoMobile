"""Run paired models with explicit environment selection, never vehicle hardware."""
import argparse,subprocess,sys,os,json,concurrent.futures
from pathlib import Path

def job(spec,root,onnx):
    arm,v,d,fx=spec;dst=root/f'{arm}_{v}_{d}_{fx}'
    env=dict(os.environ,BENCH_CNN_MIXER='direct' if arm=='new' else 'v4',BENCH_CNN_SPEED='1.0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env.pop('BENCH_CNN_ONNX',None)
    if arm=='new':env['BENCH_CNN_ONNX']=str(Path(onnx).resolve())
    with (root/(dst.name+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,'-m','bench.run','--controller','hyb2','--vehicle',v,'--direction',d,'--effects',fx,'--seed','41','--out',str(dst)],env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(str(dst))
    s=json.loads((dst/'summary.json').read_text());print(dst.name,s['verified_lap'],s['min_margin_cm'],flush=True)
    return dict(arm=arm,vehicle_name=v,direction=d,effects_name=fx,path=str(dst),summary=s)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--onnx',default='');ap.add_argument('--out',required=True);ap.add_argument('--arms',default='baseline,new');a=ap.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    specs=[(arm,v,d,fx) for arm in a.arms.split(',') for v in ['field_estimate','weak_left_turn','sluggish','unequal_sides'] for d in ['cw','ccw'] for fx in ['realcam']]
    specs += [(arm,'field_estimate',d,'realcam_harsh') for arm in a.arms.split(',') for d in ['cw','ccw']]
    rows=json.loads((root/"matrix.json").read_text()) if (root/"matrix.json").exists() else []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(job,s,root,a.onnx) for s in specs]
        for f in concurrent.futures.as_completed(futures):
            rows.append(f.result());(root/'matrix.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
