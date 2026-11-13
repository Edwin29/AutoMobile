"""Export checked geometric routes, landmarks and sampled clearance tables."""
from pathlib import Path
import argparse
import json
import numpy as np
from .map_model import TrackMap
from .settings import load_settings


def build(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    track=TrackMap();cfg,cfg_id=load_settings();routes=[]
    for direction in ("cw","ccw"):routes.append(track.outer_route(direction))
    for node,data in track.data["junctions"].items():
        for entry in data["arms"]:routes.extend(track.junction_routes(node,entry,cfg["vehicle"]).values())
    manifest={"map_hash":track.id,"config_hash":cfg_id,"source":track.data["provenance"],
              "limits":"Sampled footprint feasibility only; no actuator/slip/uncertainty guarantee", "routes":[]}
    for route in routes:
        margins=track.route_clearance(route,cfg["vehicle"])
        minimum=float(margins.min())
        data=dict(id=route.id,points_cm=route.points.tolist(),closed=route.closed,junction=route.junction,
                  generation=route.generation,
                  entry=route.entry,exit=route.exit,s_cm=route.s.tolist(),heading_rad=route.heading.tolist(),
                  curvature_cm_inv=route.curvature.tolist(),nominal_clearance_cm=margins.tolist(),
                  margin_table_columns=["s_cm","lateral_cm","yaw_rad","clearance_cm"],
                  margin_table=track.margin_table(route,cfg["vehicle"]))
        (output/f"{route.id}.json").write_text(json.dumps(data,separators=(",",":")),encoding="utf-8")
        manifest["routes"].append(dict(id=route.id,min_clearance_cm=minimum,
            geometric_feasible=minimum>=cfg["planner"]["clearance_cm"],
            curvature_within_planner_limit=bool(np.max(np.abs(route.curvature))<=cfg["planner"]["max_curvature_cm_inv"]),
            max_curvature_cm_inv=float(np.max(np.abs(route.curvature)))))
    (output/"landmarks.json").write_text(json.dumps(track.landmarks,indent=2),encoding="utf-8")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(manifest,indent=2,ensure_ascii=False));return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True)
    build(p.parse_args().output)
