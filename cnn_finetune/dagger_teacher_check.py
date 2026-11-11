import subprocess,sys,os,concurrent.futures,json
from pathlib import Path
from cnn_finetune.compare_metrics import extra
root=Path('D:/autodrive_cnn/runs/dagger_teacher_initial')
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
def job(spec):
 v,fx=spec;out=root/(v+'_'+fx)
 p=subprocess.run([sys.executable,'-m','cnn_finetune.direct_run','--variant','3','--vehicles',v,'--effects',fx,'--out',str(out)],env=env)
 return p.returncode
if __name__=='__main__':
 specs=[(v,'realcam') for v in ['field_estimate','weak_left_turn','sluggish','unequal_sides']]+[('field_estimate','realcam_harsh')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:print(list(pool.map(job,specs)))
 rows=[]
 for p in root.glob('*/matrix.json'):
  for r in json.loads(p.read_text()):
   folder=p.parent/(r['vehicle_name'].replace(':','_')+'_'+r['effects'][0]+'_'+r['direction'])
   r['extra']=extra(folder,r['direction']);rows.append(r)
 (root/'teacher_metrics.json').write_text(json.dumps(rows,indent=2))
 import numpy as np
 print('RECOVERY',np.median([t for r in rows for t in r['extra']['post_corner_recovery_s']]))
