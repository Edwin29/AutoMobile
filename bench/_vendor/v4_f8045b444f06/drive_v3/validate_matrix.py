"""Repeatable closed-loop stress checks, including failures; no threshold auto-tuning."""
import argparse
import json
from pathlib import Path
from .simulation import run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True)
    p.add_argument("--seconds",type=float,default=25.)
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    jobs=[(d,s,None) for d in ("cw","ccw") for s in ("baseline","speed_bias","yaw_loss","left_bias","delay","short_dropout","long_dropout","glare","occlusion")]
    jobs += [("ccw","baseline",j) for j in ("center:N:right","center:E:right","west:S:right","north:S:right")]
    summaries=[]
    for index,(direction,scenario,junction) in enumerate(jobs):
        summary=run(root/f"{index:02d}_{direction}_{scenario}",direction,scenario,a.seconds,junction=junction)
        summaries.append(summary)
        (root/"matrix.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    print(f"Completed {len(summaries)} cases. Inspect outcomes; a hold is not a completion.")


if __name__=="__main__":main()
