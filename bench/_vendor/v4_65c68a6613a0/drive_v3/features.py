"""Marker grouping, optional YOLO interface and short-term image tracks."""
from dataclasses import dataclass, field
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from .contracts import LandmarkCandidate


def axis_shape(points):
    p=np.asarray(points,float)
    center=p.mean(0)
    _,_,vt=np.linalg.svd(p-center,full_matrices=False)
    axis=vt[0]
    along=(p-center)@axis
    across=(p-center)@np.array([-axis[1],axis[0]])
    length,width=float(np.ptp(along)),float(np.ptp(across))
    return center,axis,length,max(width,.1)


def group_marks(patches,cfg):
    """Geometric candidates only; no mission event or persistent map association."""
    out=[]
    # At low resolution adjacent pink dashes merge. Retain transverse line
    # geometry for localization without claiming the dash pattern was resolved.
    for patch in patches:
        if patch.kind=="pink_patch" and "view_or_roi_clipped" not in patch.flags:
            center,axis,length,width=axis_shape(patch.contour_cm)
            if length>=8. and length/width>=8.:
                out.append(LandmarkCandidate(f"pink_line_{len(out):03d}","pink_line_candidate",center.tolist(),
                    patch.contour_cm,{"length_cm":length,"line_angle_rad":float(np.arctan2(axis[1],axis[0]))},
                    ["dash_pattern_unresolved","localization_only"]))
    # Tiny dashes cannot supply a stable per-dash angle. Fit the arrangement of
    # distinct component centers instead of demanding every 2-pixel blob be a line.
    pink=[p for p in patches if p.kind=="pink_patch" and "view_or_roi_clipped" not in p.flags][:64]
    while len(pink)>=cfg["dash_min_count"]:
        centers=np.array([p.center_cm for p in pink]);best=None
        for i in range(len(pink)):
            for j in range(i+1,len(pink)):
                delta=centers[j]-centers[i];length=np.linalg.norm(delta)
                if length<cfg["pink_min_span_cm"]:continue
                axis=delta/length;normal=np.array([-axis[1],axis[0]])
                mask=np.abs((centers-centers[i])@normal)<=cfg["pink_center_band_cm"]
                if mask.sum()<cfg["dash_min_count"]:continue
                along=np.sort(centers[mask]@axis);gaps=np.diff(along)
                if len(gaps) and gaps.max()>cfg["group_gap_cm"]:continue
                score=(int(mask.sum()),float(np.ptp(along)))
                if best is None or score>best[0]:best=(score,mask)
        if best is None:break
        selected=[p for p,keep in zip(pink,best[1]) if keep]
        c=centers[best[1]];center=c.mean(0)
        _,_,vt=np.linalg.svd(c-center,full_matrices=False);axis=vt[0]
        normal=np.array([-axis[1],axis[0]])
        out.append(LandmarkCandidate(f"dash_{len(out):03d}","dashline",center.tolist(),
            np.concatenate([p.contour_cm for p in selected]).tolist(),
            {"pieces":len(selected),"line_rms_cm":float(np.sqrt(np.mean(((c-center)@normal)**2))),
             "line_angle_rad":float(np.arctan2(axis[1],axis[0])),"span_cm":best[0][1]},
            ["geometric_candidate_not_mission_trigger"]))
        pink=[p for p,keep in zip(pink,best[1]) if not keep]
    for kind,target,min_count in (("white_patch","crosswalk",cfg["cross_min_count"]),
                                   ("red_patch","red_line",1)):
        candidates=[]
        for patch in patches:
            if patch.kind != kind or "view_or_roi_clipped" in patch.flags:
                continue
            center,axis,length,width=axis_shape(patch.contour_cm)
            ratio=length/width
            if ratio < (cfg["red_min_aspect"] if kind=="red_patch" else cfg["stripe_min_aspect"]):
                continue
            if kind=="red_patch" and length < cfg["red_min_length_cm"]:
                continue
            candidates.append((patch,center,axis,length,width))
        graph={i:set() for i in range(len(candidates))}
        for i,a in enumerate(candidates):
            for j in range(i+1,len(candidates)):
                b=candidates[j]
                if abs(a[2]@b[2]) < np.cos(np.deg2rad(cfg["parallel_angle_deg"])):
                    continue
                delta=b[1]-a[1]
                along=abs(delta@a[2]); across=abs(delta@np.array([-a[2][1],a[2][0]]))
                if target=="dashline":
                    linked=across<=max(a[4],b[4],.6) and along<=(a[3]+b[3])/2+cfg["group_gap_cm"]
                elif target=="crosswalk":
                    linked=along<=max(a[3],b[3])*.3 and across<=(a[4]+b[4])/2+cfg["group_gap_cm"]
                else:
                    linked=False
                if linked:
                    graph[i].add(j);graph[j].add(i)
        pending=set(graph)
        while pending:
            seed=min(pending); group={seed}; todo=[seed];pending.remove(seed)
            while todo:
                for j in graph[todo.pop()] & pending:
                    pending.remove(j);group.add(j);todo.append(j)
            if len(group)<min_count:
                continue
            pieces=[candidates[i] for i in sorted(group)]
            contour=np.concatenate([np.asarray(p[0].contour_cm) for p in pieces])
            center=contour.mean(0)
            axes=np.array([p[2]*np.sign(p[2]@pieces[0][2]) for p in pieces])
            axis=axes.mean(0);axis/=max(np.linalg.norm(axis),1e-8)
            centers=np.array([p[1] for p in pieces])
            across=np.array([-axis[1],axis[0]])
            order_axis=axis if target=="dashline" else across
            gaps=np.diff(np.sort(centers@order_axis))
            spacing_cv=float(np.std(gaps)/max(np.mean(gaps),1e-6)) if len(gaps)>1 else 0.
            if spacing_cv>.6:
                continue
            direction=axis if target!="crosswalk" else across  # direction of transverse marking
            out.append(LandmarkCandidate(f"mark_{len(out):03d}",target,center.tolist(),contour.tolist(),
                       {"pieces":len(group),"spacing_cv":spacing_cv,
                        "line_angle_rad":float(np.arctan2(direction[1],direction[0]))},
                       ["geometric_candidate_not_mission_trigger"]))
    return out


@dataclass
class Detection:
    kind:str
    bbox:list[float]
    confidence:float
    timestamp:float
    track_id:int|None=None


class YoloAdapter:
    """Explicit opt-in dependency; no model loading at module import."""
    aliases={"right":"right","left":"left","straight":"straight","horn":"horn",
             "slow":"slow","stop":"stop","red":"signal_red","green":"signal_green"}
    def __init__(self,model_path,roi=(.5,0.,1.,1.)):
        from ultralytics import YOLO
        self.model=YOLO(model_path)
        self.roi=roi
        self.supported={self.aliases[n] for n in self.model.names.values() if n in self.aliases}

    def detect(self,image,timestamp):
        h,w=image.shape[:2]
        x0,y0,x1,y1=[round(v*d) for v,d in zip(self.roi,(w,h,w,h))]
        result=self.model.predict(image[y0:y1,x0:x1],verbose=False)[0]
        out=[]
        for box in result.boxes:
            name=result.names[int(box.cls.item())]
            if name not in self.aliases:
                continue
            b=box.xyxy[0].cpu().numpy()+[x0,y0,x0,y0]
            out.append(Detection(self.aliases[name],b.tolist(),float(box.conf.item()),timestamp))
        return out


def iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    size=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    inter=np.prod(size)
    return float(inter/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter,1e-9))


class DetectionTracker:
    """SORT-style constant-velocity Kalman + Hungarian, not persistent landmark IDs."""
    def __init__(self,cfg):
        self.cfg=cfg;self.tracks={};self.counter=0;self.last=None

    def update(self,detections,now):
        if self.last is not None and now<=self.last:
            raise ValueError("Tracker timestamps must increase")
        dt=0. if self.last is None else now-self.last;self.last=now
        F=np.eye(8);F[:4,4:]=np.eye(4)*dt
        for t in self.tracks.values():
            t["x"]=F@t["x"];t["P"]=F@t["P"]@F.T+np.eye(8)*max(dt,.001)
        ids=list(self.tracks)
        costs=np.array([[1-iou(self.tracks[k]["x"][:4],d.bbox) for d in detections] for k in ids])
        assigned=set();result=[]
        if ids and detections:
            rows,cols=linear_sum_assignment(costs)
            for r,c in zip(rows,cols):
                if costs[r,c]>1-self.cfg["association_iou_min"]:
                    continue
                key=ids[r];t=self.tracks[key];d=detections[c]
                S=t["P"][:4,:4]+np.eye(4)*4
                K=t["P"][:,:4]@np.linalg.inv(S)
                t["x"]+=K@(np.array(d.bbox)-t["x"][:4])
                H=np.c_[np.eye(4),np.zeros((4,4))]
                A=np.eye(8)-K@H;t["P"]=A@t["P"]@A.T+K@K.T*4
                t["last"]=now;t["hits"]+=1;d.track_id=key;assigned.add(c);result.append(d)
        for c,d in enumerate(detections):
            if c not in assigned:
                self.counter+=1;d.track_id=self.counter
                self.tracks[self.counter]={"x":np.r_[d.bbox,np.zeros(4)],"P":np.eye(8)*10,"last":now,"hits":1}
                result.append(d)
        self.tracks={k:t for k,t in self.tracks.items() if now-t["last"]<=self.cfg["track_expire_s"]}
        return result
