"""Collect executed v3 evidence without copying private photos/video into Git."""
import argparse
import collections
import json
from pathlib import Path
import platform
import numpy as np
import cv2
import scipy


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def describe(directory):
    directory=Path(directory);summary=read(directory/"summary.json")
    rows=[json.loads(line) for line in (directory/"steps.jsonl").read_text(encoding="utf-8").splitlines()]
    held=False;recoveries=0;start=0.;longest=0.
    for row in rows:
        hold=row["path_reason"]!="tracking"
        if hold and not held:start=row["now_s"]
        if held and not hold:recoveries+=1;longest=max(longest,row["now_s"]-start)
        held=hold
    if held:longest=max(longest,rows[-1]["now_s"]-start)
    summary.update(run=directory.name,planner_hold_frames=sum(r["path_reason"]!="tracking" for r in rows),
                   recoveries_after_hold=recoveries,longest_hold_s=longest,
                   last_covariance=rows[-1]["covariance"],last_planner=rows[-1]["plan"])
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix",required=True);p.add_argument("--loops",nargs="+",required=True)
    p.add_argument("--real",required=True);p.add_argument("--map",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();matrix=Path(a.matrix)
    cases=[describe(d) for d in sorted(matrix.iterdir()) if d.is_dir() and (d/"summary.json").exists()]
    counts=collections.Counter(c["outcome"] for c in cases)
    real=read(Path(a.real)/"review_summary.json")
    marks=collections.Counter()
    for log in Path(a.real).glob("*/observations.jsonl"):
        for line in log.read_text(encoding="utf-8").splitlines():
            for m in json.loads(line)["landmarks"]:marks[m["kind"]]+=1
    data=dict(schema=1,version="0.2.0",unit_tests=57,
              environment=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,scipy=scipy.__version__),
              matrix_outcomes=dict(counts),matrix=cases,loops=[describe(d) for d in a.loops],
              real_review=real,marker_candidate_counts=dict(marks),map_library=read(Path(a.map)/"manifest.json"),
              limits=["Simulated truth is evaluation only", "time_limit does not mean completed",
                      "Real image counts are not accuracy", "Concurrent laptop timings are not Pi benchmarks",
                      "See each run's code hash and source-change indicator; do not combine development revisions"])
    dest=Path(a.output)
    if dest.exists():raise FileExistsError(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"outcomes":dict(counts),"marker_candidates":dict(marks)},ensure_ascii=False))
    for c in cases:print(c["direction"],c["junction"] or c["scenario"],c["outcome"],round(c["progress_cm"],1),round(c["minimum_margin_cm"],2),c["planner_hold_frames"],c["recoveries_after_hold"])


if __name__=="__main__":main()
