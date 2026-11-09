"""T-junction alternative: approach a swept-footprint-checked pivot, stop, rotate, confirm exit, depart."""
import argparse,csv,json,math
from functools import lru_cache
import cv2
import numpy as np
from trials import HERE,axes,env,SITES,Plant,FixedGain,ShortGapDriver,calibrate,OUT,DT,CENTER_AHEAD,FRAME_ON,FRAME_OFF,surface,rim,clearance
from controller import LENS_AHEAD,advance

def offset_at(s,offset):
    t=np.clip((s+35)/30,0,1)
    return offset*t*t*(3-2*t)

@lru_cache(maxsize=2)
def choose_offset(site):
    center,yaw=SITES[site];rot=axes(yaw); scores=[]
    z=np.linspace(-49,0,197);a=np.linspace(0,np.pi/2,181)
    for offset in np.arange(0,5.1,.25):
        entry=np.c_[offset_at(z,offset),z]
        path=np.vstack([entry,np.tile([offset,0],(len(a),1)),np.c_[np.arange(offset,46,.25),np.zeros(len(np.arange(offset,46,.25)))]])
        fw=np.gradient(entry,axis=0);fw/=np.linalg.norm(fw,axis=1)[:,None]
        fw=np.vstack([fw,np.c_[np.sin(a),np.cos(a)],np.tile([1.,0.],(len(path)-len(a)-len(entry),1))])
        right=np.c_[fw[:,1],-fw[:,0]]
        pts=(path[:,None,:]+right[:,None,:]*rim[:,0,None]+fw[:,None,:]*(rim[:,1,None]+CENTER_AHEAD))@rot+center
        scores.append((float(clearance(*pts.transpose(2,0,1)).min()),float(offset)))
    return max(scores)

def exit_edges(frame,d,site,turn):
    # Independent of simulator truth. Assign observed edges to the two sides of the specified outlet in world coordinates.
    center,yaw=SITES[site];hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    c=cv2.inRange(hsv,(20,100,130),(87,255,255))
    road=((hsv[:,:,2]<130)&(c==0)).astype(np.uint8)*255
    dark=cv2.warpPerspective(road,d.M,d.bev)>=128
    edge=(cv2.warpPerspective(c,d.M,d.bev)>=128)&(cv2.dilate(dark.astype(np.uint8),np.ones((3,3),np.uint8))>0)
    yy,xx=np.nonzero(edge);pts=np.c_[xx/5-40,50-yy/5]
    pts=pts[(pts[:,1]>=6)&(pts[:,1]<=35)]
    world=(pts+[0,LENS_AHEAD])@axes(d.pose[2])+d.pose[:2]
    local=(world-center)@axes(yaw).T
    along,lat=(local[:,0],local[:,1]) if turn=="right" else (local[:,1],local[:,0])
    hw=(10.5 if site=="wide" else 9.25) if turn=="right" else 13.
    for sign in (-1,1):
        hits=along[(along>23)&(along<68)&(abs(lat-sign*hw)<1.2)]
        if len(hits)<10 or np.ptp(hits)<4:return False
    return True

class SlidingPlant(Plant):
    """Inject unmeasured translation along the entry road during counter-rotation."""
    def observe(self):
        change = math.remainder(self.yaw - self.previous_yaw, 2 * math.pi)
        self.previous_yaw = self.yaw
        if self.out[0] * self.out[1] < 0:
            self.pos += self.slide_vector * change / (math.pi / 2)
        return super().observe()


def floor_edges(image,driver):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    green=cv2.inRange(hsv,(20,100,130),(87,255,255))
    dark=((hsv[:,:,2]<130)&(green==0)).astype(np.uint8)*255
    g=cv2.warpPerspective(green,driver.M,driver.bev)>=128
    r=cv2.warpPerspective(dark,driver.M,driver.bev)>=128
    return (g & (cv2.dilate(r.astype(np.uint8),np.ones((3,3),np.uint8))>0))[75:220].astype(np.uint8)


def same_layout(expected,seen):
    # Bidirectional coverage at 5 pixels/cm: a short surviving edge cannot
    # stand in for the entire predicted junction view.
    if min(expected.sum(),seen.sum())<45:return False
    def covered(a,b):
        distance=cv2.distanceTransform(1-b,cv2.DIST_L2,5)
        return np.mean(distance[a!=0]<=5)
    return covered(expected,seen)>=.8 and covered(seen,expected)>=.8


def run(site,condition,late=False,recover=False,slide=0.,permanent=False,comparison=None):
    cv2.setNumThreads(1); center,yaw=SITES[site];rot=axes(yaw);geom,offset=choose_offset(site)
    left=calibrate()["left_gain"];first,second=json.loads((OUT/"cw_calibration.json").read_text())
    p=SlidingPlant("cw",pivot_behind_center_cm=CENTER_AHEAD,left_yaw_response=left,right_yaw_response=second["gain"])
    p.previous_yaw=yaw;p.slide_vector=slide*rot[1]
    p.yaw=yaw;p.pos=center+np.array([0.,-49.])@rot+CENTER_AHEAD*rot[1]
    d=ShortGapDriver();d.pose=np.r_[p.pos-CENTER_AHEAD*rot[1],yaw];d.yaw_tracker=FixedGain();d.yaw_tracker.right=first["gain"]
    clean=p.world.img;bad=surface(p.world,site,condition);phase="approach";reason="timeout";stable=0;rows=[];phases={}
    anchor=None;resumed=None;moving_while_waiting=0
    reference_world=env["World"]() if recover else None
    if comparison is not None:
        comparison.initialize(d,first["gain"],second["gain"])
    for tick in range(1,841):
        start,end=(245,335) if late else (FRAME_ON,FRAME_OFF)
        if permanent:end=841
        p.world.img=bad if start<=tick<end else clean
        if comparison is not None:
            comparison.set_disturbance(p,tick,rot)
        frame=p.capture_array()
        if comparison is not None and comparison.blind:frame[:]=255
        old_yaw=d.pose[2]
        d.predict(p.out,DT)
        loc=(d.pose[:2]-center)@rot.T
        err=math.remainder(yaw+math.pi/2-d.pose[2],2*math.pi)
        target=np.zeros(2)
        observed=d.observe(frame,True)
        if comparison is not None:
            loc=(d.pose[:2]-center)@rot.T
            err=math.remainder(yaw+math.pi/2-d.pose[2],2*math.pi)
            comparison.observe(d,frame,p.out,old_yaw,phase)
        waiting=False
        if phase=="wait":
            waiting=True
            close=(np.linalg.norm(d.pose[:2]-anchor[:2])<=2 and
                   abs(math.remainder(d.pose[2]-anchor[2],2*math.pi))<=math.radians(8))
            qualified=(d.matched and close and d.residual<.8 and
                       same_layout(reference_edges,floor_edges(frame,d)))
            stable=stable+1 if qualified else 0
            if stable>=3:
                resumed=tick;phase="rotate";stable=0
            elif tick-phases["wait"]>=180:
                reason="stationary verification timed out"
        elif not observed and recover and phase=="rotate":
            phase="wait";anchor=d.pose.copy();stable=0;waiting=True
            camera=anchor[:2]+LENS_AHEAD*axes(anchor[2])[1]
            reference_edges=floor_edges(reference_world.render((*camera,anchor[2])),d)
        elif not observed:
            reason="unmatched road edges"
        elif phase=="approach" or phase=="exit":
            if phase=="approach" and loc[1]>=-.3:
                phase="brake";stable=0
            elif phase=="exit" and loc[0]>=40.:
                phase="done";reason="estimated exit gate"
            else:
                if phase=="approach":
                    f=loc[1]+9.;q=np.array([offset_at(f,offset),f])
                    speed=min(.23,max(.055,-loc[1]*.045))
                else:
                    q=np.array([loc[0]+9,0.]);speed=.23
                local=(q@rot+center-d.pose[:2])@axes(d.pose[2]).T
                k=2*local[0]/max(1e-6,local@local)
                target=speed*(1+np.array([1.,-1.])*6.5*k/(first["gain"] if k>0 else 13/18))
        elif phase=="brake":
            if np.max(abs(d.out))<.002:
                stable=stable+1 if d.matched else 0
                if stable>=3:phase="rotate";stable=0
        elif phase=="rotate":
            if abs(math.degrees(err))>1.4:
                target=np.array([1.,-1.])*np.sign(err)*min(.26,max(.045,abs(err)*.65))
                if comparison is not None:
                    target=np.clip(target*comparison.initial_gain/d.right_mix,-.26,.26)
            elif np.max(abs(d.out))<.002:
                phase="confirm";stable=0
        elif phase=="confirm":
            phases.setdefault("first_confirm",tick)
            stable=stable+1 if d.matched and exit_edges(frame,d,site,"right") else 0
            if stable>=3:phase="exit"
            elif tick-phases["first_confirm"]>22:reason="exit edges unconfirmed"
        cmd=d.out+np.clip(target-d.out,-.6*DT,.6*DT)
        if waiting:cmd=np.zeros(2)
        margins=[]
        for h in (0.,.15,.30):
            for gain in (1.,d.yaw_tracker.gain(cmd[0]-cmd[1])):
                q=advance(d.pose,cmd,h,gain)
                margins.append(d.sample(d.sdf,d.rim@axes(q[2])+q[:2]).min())
        d.margin=float(min(margins))
        if d.margin<.25 and reason=="timeout":reason="footprint limit"
        if reason!="timeout":cmd=np.zeros(2)
        if waiting and np.any(cmd):moving_while_waiting+=1
        d.out=cmd;p.set(*cmd)
        truth=p.pos-CENTER_AHEAD*axes(p.yaw)[1]
        pos_err=np.linalg.norm(d.pose[:2]-truth)
        yaw_err=abs(math.degrees(math.remainder(p.yaw-d.pose[2],2*np.pi)))
        rows.append([tick,tick*DT,*p.pos,p.yaw,p.rows[-1][4],*d.pose,pos_err,yaw_err,int(d.matched),*cmd])
        if comparison is not None:
            comparison.record(tick,phase,d,p.yaw,yaw)
        phases.setdefault(phase,tick)
        if reason!="timeout":break
    a=np.array(rows);final=(truth-center)@rot.T;heading=abs(math.degrees(math.remainder(p.yaw-yaw-np.pi/2,2*np.pi)))
    if reason=="estimated exit gate":reason="pass" if final[0]>39 and abs(final[1])<2 and heading<5 else "incorrect exit or alignment"
    prefix=HERE/(f"{site}_pivot_{condition}"+("_late" if late else ""))
    if comparison is not None:
        prefix=comparison.prefix
    elif recover:
        folder=HERE/"recovery";folder.mkdir(exist_ok=True)
        prefix=folder/f"{site}_{condition}_{slide:+g}_{'persistent' if permanent else 'temporary'}"
    np.savetxt(str(prefix)+".csv",a,delimiter=",",header="frame,t,x,y,yaw,margin,est_x,est_y,est_yaw,pos_error,angle_error,matched,L,R",comments="")
    data=dict(site=site,condition=condition,outcome=reason,elapsed_s=a[-1,1],min_cm=a[:,5].min(),crossed_frames=int(sum(a[:,5]<0)),phase_ticks=phases,
              fault_start=start,fault_end=end,geom_cm=geom,offset_cm=offset,heading_error_deg=heading,truth_exit_progress_cm=final[0],truth_exit_offset_cm=final[1])
    if recover:
        data.update(slide_cm_per_90deg=slide,resume_frame=resumed,wait_nonzero_commands=moving_while_waiting)
        if data["crossed_frames"] and reason=="pass":data["outcome"]="boundary crossed"
    if comparison is not None:
        data.update(comparison.summary(a))
    (prefix.parent/(prefix.name+".json")).write_text(json.dumps(data,indent=2))
    print(site,condition,reason,round(data["min_cm"],2),phases,flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("site",choices=SITES);ap.add_argument("condition",choices=("normal","hidden","fake"))
    ap.add_argument("--late",action="store_true")
    ap.add_argument("--recover",action="store_true");ap.add_argument("--slide",type=float,default=0.)
    ap.add_argument("--permanent",action="store_true");args=ap.parse_args()
    if args.slide and not args.recover:ap.error("--slide requires --recover")
    if args.permanent and not args.recover:ap.error("--permanent requires --recover")
    run(args.site,args.condition,args.late,args.recover,args.slide,args.permanent)
