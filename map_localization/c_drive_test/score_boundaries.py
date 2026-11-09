"""Split footprint clearance into the outside border and the four inner islands."""
import json
import numpy as np
from simulate_c import HERE, env, rim

def measure(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    result=json.loads(path.with_suffix(".json").read_text(encoding="utf8"))
    sdf=env["rrect_sdf"]
    outer_min,inner_min=[],[]
    for frames in np.array_split(data, max(1,len(data)//48)):
        heading=np.radians(frames["yaw"])[:,None]
        c,s=np.cos(heading),np.sin(heading)
        x=frames["x"][:,None]+c*rim[:,0]+s*rim[:,1]
        y=frames["y"][:,None]-s*rim[:,0]+c*rim[:,1]
        outer_min.extend((-sdf(x,y,*env["OUTER"],env["OUTER_R"])).min(axis=1))
        inner_min.extend(np.minimum.reduce([sdf(x,y,*b,env["BLOCK_R"]) for b in env["BLOCKS"]]).min(axis=1))
    worst=np.minimum(outer_min,inner_min)
    assert np.max(abs(worst-data["body_clear_cm"]))<.21
    result.update(outer_min_cm=min(outer_min),island_min_cm=min(inner_min),
                  outer_negative_pct=100*np.mean(np.array(outer_min)<0),
                  island_negative_pct=100*np.mean(np.array(inner_min)<0),
                  finish_offset_from_start_cm=float(np.hypot(data["x"][-1]-data["x"][0],data["y"][-1]-data["y"][0])))
    return result

if __name__=="__main__":
    result=[measure(p) for p in sorted(HERE.glob("*.csv"))]
    expected={(d,p) for d in ("cw","ccw") for p in ("base","no_bias","start2")}|{("ccw","extra")}
    assert {(r["direction"],r["profile"]) for r in result} == expected
    (HERE/"comparison.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    for r in result:
        print(r["direction"],r["profile"],*(round(r[k],2) for k in ("turns","outer_min_cm","island_min_cm","negative_frame_percent","elapsed_s","finish_offset_from_start_cm")))
