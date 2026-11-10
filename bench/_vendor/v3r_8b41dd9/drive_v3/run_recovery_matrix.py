"""Run independent paired simulations in bounded worker processes, no hardware."""
import argparse,os,subprocess,sys,json
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--scenarios',nargs='+',default=['baseline','start_outside','slip','bias','dropout','occlusion'])
    p.add_argument('--workers',type=int,default=3);p.add_argument('--seconds',default='180')
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    jobs=[(policy,direction,scenario) for scenario in a.scenarios for direction in ['ccw','cw'] for policy in ['baseline','recovery']]
    def job(spec):
        name='_'.join(spec);env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        cmd=[sys.executable,'-m','drive_v3.compare_recovery','--output',str(root/name),'--policy',spec[0],
            '--direction',spec[1],'--scenario',spec[2],'--seconds',a.seconds]
        if spec[2] in ('baseline','slip'):cmd.append('--video')
        with (root/f'{name}.stdout.txt').open('w') as log:
            r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        if r.returncode:raise RuntimeError(f'{name}: exit {r.returncode}')
        result=json.loads((root/name/'summary.json').read_text());print(name,result['outcome'],round(result['progress_cm'],1),flush=True)
        return result
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(job,spec) for spec in jobs]):
            results.append(future.result());(root/'summary.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
