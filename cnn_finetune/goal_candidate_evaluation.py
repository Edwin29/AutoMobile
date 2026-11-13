"""Evaluate a named development candidate without changing frozen baseline runs."""
import argparse
import json
from pathlib import Path

from .goal_development import OUT, REPORT, ROOT, SEEDS, PRESETS, KEYS, CONDITIONS, LocalScene, sha
from .junction_learning import setup, Encoder, model_at
from .junction_stage02 import dump


def main(checkpoint, tag):
    if not tag.replace('_','').isalnum():raise ValueError('Simple artifact tag required')
    checkpoint=Path(checkpoint).resolve();setup()
    protocol=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),seeds=SEEDS,presets=PRESETS,
                  keys=KEYS,conditions=CONDITIONS,metric_protocol_sha256=sha(REPORT/'protocol.json'),
                  source_sha256=sha(__file__),purpose='development comparison; no final exam or model selection on final data')
    file=REPORT/(tag+'_protocol.json')
    if file.exists():assert json.loads(file.read_text())==protocol
    else:dump(file,protocol)
    model=model_at(checkpoint);encoder=Encoder().cuda().eval();rows=[]
    for seed in SEEDS:
        for preset in PRESETS:
            for key in KEYS:
                for condition in CONDITIONS:
                    dst=OUT/'rollouts'/tag/('_'.join(key)+f'_{condition}_{seed}_{preset}')
                    if (dst/'result.json').exists():r=dict(json.loads((dst/'result.json').read_text()),artifact=str(dst))
                    else:r=LocalScene(key,condition,seed,preset,encoder).finish(model).save(dst,save_images=False)
                    rows.append(r)
            dump(REPORT/(tag+'_progress.json'),rows)
            print(tag,seed,preset,'clean',sum(r['clean_stable'] for r in rows),'/',len(rows),flush=True)
    dump(REPORT/(tag+'.json'),rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--tag',required=True);a=p.parse_args()
    main(a.checkpoint,a.tag)
