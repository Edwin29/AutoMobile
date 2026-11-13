"""Frozen legacy controls; three subprocess workers, no hardware."""
import concurrent.futures
import json
from cnn_finetune.junction_stage02 import OUT, REPORT, DEV, dump
from cnn_finetune.junction_stage02_checks import launch
from bench.junction import load_plan_cache

def main():
    tasks=[]
    for key in load_plan_cache():
        spec=dict(junction=key[0],came_from=key[1],exit=key[2],mixer='v4',seed=1303,stable=True,seconds=40)
        tasks.append((spec,OUT/'legacy'/('_'.join(key)+'_fixed_1303')))
    focus=[('south','N','left'),('south','N','right'),('north','S','left'),('north','S','right'),('center','N','right'),('center','E','right')]
    for key in focus:
        for level in ['mild','harsh']:
            for seed in DEV:
                spec=dict(junction=key[0],came_from=key[1],exit=key[2],mixer='v4',seed=seed,stable=True,seconds=40,
                          composite=level,variation=level,effects=['realcam_light' if level=='mild' else 'realcam_harsh'])
                tasks.append((spec,OUT/'legacy'/('_'.join(key)+f'_{level}_{seed}')))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for dst in pool.map(lambda item:launch(*item),tasks):
            r=json.loads((dst/'result.json').read_text());print(dst.name,r['outcome'],round(r['min_margin_cm'],2),flush=True)
    dump(REPORT/'legacy_results.json',[json.loads((dst/'result.json').read_text()) for spec,dst in tasks])

if __name__=='__main__':main()
