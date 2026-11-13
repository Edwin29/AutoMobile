"""Offline real-video shadow run. New commands NEVER move the recorded vehicle.

Without command CSV, prediction is zero with growing uncertainty; output is
a diagnostic, not a closed-loop test. CSV: time_s,left_pwm,right_pwm in video time.
"""
from pathlib import Path
import argparse
import csv
import json
import cv2
import numpy as np
from .runtime import Runtime
from .simulation import json_default
from .camera import sha256_file
from .map_model import TrackMap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input");p.add_argument("--output",required=True)
    p.add_argument("--pose",nargs=3,type=float,required=True,metavar=("X_CM","Y_CM","YAW_RAD"))
    p.add_argument("--direction",choices=["cw","ccw"],default="ccw")
    p.add_argument("--rectified",action="store_true");p.add_argument("--rotate180",action="store_true")
    p.add_argument("--commands");p.add_argument("--seconds",type=float,default=10.)
    p.add_argument("--every",type=int,default=3)
    args=p.parse_args()
    if args.every<1 or args.seconds<=0:raise ValueError("Invalid replay duration or stride")
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    cap=cv2.VideoCapture(args.input);fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not 0<fps<1000:raise ValueError("Unreadable video or missing frame rate")
    commands=[]
    if args.commands:
        with open(args.commands,newline="",encoding="utf-8-sig") as f:
            commands=[(float(r["time_s"]),float(r["left_pwm"]),float(r["right_pwm"])) for r in csv.DictReader(f)]
        if any(b[0]<a[0] for a,b in zip(commands,commands[1:])):raise ValueError("Commands must be ordered")
    route=TrackMap().outer_route(args.direction)
    frames=0;index=0;last=0.;command_index=0;current=np.zeros(2)
    try:
        with Runtime(args.pose,route,start_s=0.,clock_id="video_pts") as rt,(output/"shadow.jsonl").open("w",encoding="utf-8") as log:
            while True:
                ok,image=cap.read()
                if not ok:break
                index+=1
                # Nominal frame timestamp fallback is disclosed in the manifest.
                at=index/fps
                if at>args.seconds:break
                if (index-1)%args.every:continue
                while command_index<len(commands) and commands[command_index][0]<=at:
                    time_s,left,right=commands[command_index];command_index+=1
                    if time_s>=rt.timeline.now:
                        rt.timeline.step(time_s)
                        current=np.array([left,right]);v,w=rt.model.forward_model(current)
                        rt.timeline.command(time_s,v,w)
                # Never feed previously proposed shadow control back into a recorded stream.
                record,observation,debug,path=rt.step(image,at,at,rectified=args.rectified,rotate180=args.rotate180,shadow_pwm=current)
                record["mode"]="shadow_with_recorded_commands" if args.commands else "shadow_no_motion_telemetry"
                log.write(json.dumps(record,default=json_default)+"\n");frames+=1
                if frames==1 or frames%20==0:cv2.imwrite(str(output/f"frame_{index:06d}.jpg"),debug["rectified"])
    finally:cap.release()
    manifest=dict(input_sha256=sha256_file(args.input),command_sha256=sha256_file(args.commands) if args.commands else None,
                  frames=frames,pose_origin="user_supplied_not_ground_truth",timestamps="frame_index / nominal_fps",
                  limitation="Offline shadow cannot prove a new policy would pass this recorded scene")
    (output/"summary.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8");print(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
