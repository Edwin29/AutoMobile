"""Causal learner audit on all saved images; safety stops evaluated separately in the driver."""
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np
from controller import Driver,HERE,WHEEL_SPEED,TRACK_CM
from yaw_gain import GainTracker,PaintDetector

def audit():
    cv2.setNumThreads(1)
    run=HERE.parents[1]/"junction_logic/visibility_review/run_160024"
    with (run/"log.csv").open(encoding="utf8") as f:log=list(csv.DictReader(f))
    with np.load(HERE/"track.npz") as data:detector=PaintDetector(data)
    tracker=GainTracker();events=[];detections={};times=[]; views=[]
    driver=Driver("ccw",adaptive_yaw=True)
    driver.pose[0]+=json.loads((HERE/"heldout_crosswalk.json").read_text())["start_shift_cm"]
    driver.last_index=driver.nearest()
    stopped=None
    previous=0
    for p in sorted(run.glob("frame_*.jpg")):
        i=int(p.stem.split("_")[-1]);motion=[]
        for j in range(previous,i):
            dt=float(log[j+1]["t"])-float(log[j]["t"])
            l,r=[float(log[j][k]) for k in ("left","right")]
            motion.append([dt,np.degrees((l-r)*WHEEL_SPEED/TRACK_CM)*dt,(l+r)*WHEEL_SPEED/2*dt])
        dt,raw,ds=np.sum(motion,axis=0) if motion else (0.,0.,0.)
        frame=cv2.imread(str(p));start=time.perf_counter()
        marks=detector.detect(frame)
        times.append((time.perf_counter()-start)*1000)
        if i in (145,155,165,370):
            bird=cv2.remap(frame,detector.maps[0],None,cv2.INTER_LINEAR)
            for mark in marks:
                center=mark.center*np.array([5,-5])+[150,210]
                angle=np.radians(mark.angle)
                v=25*np.array([np.cos(angle),-np.sin(angle)])
                cv2.line(bird,tuple(np.rint(center-v).astype(int)),tuple(np.rint(center+v).astype(int)),(0,225,255),2)
            cv2.putText(bird,f"#{i} "+",".join(m.kind for m in marks),(5,18),0,.47,(255,255,255),1,cv2.LINE_AA)
            views.append(bird)
        if stopped is None:
            for j in range(previous,i-1):
                driver.predict(tuple(float(log[j][k]) for k in ("left","right")),float(log[j+1]["t"])-float(log[j]["t"]))
            cmd=tuple(float(log[i-1][k]) for k in ("left","right")) if i else (0.,0.)
            driver.step(frame,float(log[i]["t"])-(float(log[i-1]["t"]) if i else 0.),cmd)
            if driver.reason:
                stopped=dict(frame=i,time_s=float(log[i]["t"]),reason=driver.reason,left=driver.yaw_tracker.left)
        if any(a[0]>.35 for a in motion) or (any(a[1]<0 for a in motion) and any(a[1]>0 for a in motion)):
            tracker.reset()
        tracker.update(marks,raw,dt,ds)
        detections[i]=[(m.kind,round(m.angle,1),m.count) for m in marks]
        if tracker.event:events.append(dict(frame=i,**tracker.event))
        previous=i
    # Independent manual measurement on the same red stripe, not an input to the learner.
    test=json.loads((HERE/"second_corner_yaw.json").read_text())[-1]
    prior=[e for e in events if e["frame"]<test["start"] and e["raw_deg"]<0]
    frozen=prior[-1]["after"] if prior else 13/18
    out=dict(frames=len(times),frames_by_type={k:sum(any(x[0]==k for x in m) for m in detections.values()) for k in ("dash","cross")},
             events=events,final_left=tracker.left,final_right=tracker.right,operational_stop=stopped,
             heldout=dict(start=test["start"],end=test["end"],measured=test["measured_left_deg"],gain=frozen,
                          fixed_prediction=test["corrected_left_deg"],adaptive_prediction=test["original_left_deg"]*frozen),
             ms_percentiles=np.percentile(times,[50,95,100]).tolist())
    (HERE/"adaptive_audit.json").write_text(json.dumps(out,indent=2))
    cv2.imwrite(str(HERE/"adaptive_marks.jpg"),np.hstack(views))
    print(json.dumps(out,indent=1))
    print({i:m for i,m in detections.items() if 100<=i<=180})
if __name__=="__main__":audit()
