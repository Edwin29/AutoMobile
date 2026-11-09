"""Portable output-feedback controller shared by simulation and the Pi."""
import math
import numpy as np
from controller import Driver, WHEEL_SPEED, TRACK_CM
from yaw_gain import GainTracker

class FixedGain(GainTracker):
    def update(self,marks,raw_deg,dt,forward_cm):
        pass  # identical initialization, no online adaptation


class CompensatedDriver(Driver):
    """Simulation-only actuator comparison; feedforward uses prior 13/18, not fitted plant truth."""
    left_mix=13./18.
    right_mix=1.
    def wheel_targets(self,speed,curvature):
        effective=curvature/(self.left_mix if curvature<0 else self.right_mix)
        return super().wheel_targets(speed,effective)


class Feedback(CompensatedDriver):
    def __init__(self,direction="ccw"):
        super().__init__(direction,adaptive_yaw=True)
        self.window=[]
        self.last_update=None

    def step(self,frame,dt,applied,rectified=False):
        old_yaw=self.pose[2]
        command=super().step(frame,dt,applied,rectified)
        self.update_output(old_yaw,dt,applied)
        return command  # new multiplier only affects the NEXT command

    def update_output(self,old_yaw,dt,applied):
        self.last_update=None
        raw=math.degrees((applied[0]-applied[1])*WHEEL_SPEED/TRACK_CM*dt)
        # Never mix left and right evidence in the same update window.
        if self.window and self.window[-1][0]*raw<=0:
            self.window.clear()
        if self.reason or self.residual is None or self.residual>.65 or abs(raw)<.12:
            self.window.clear()
            return
        measured=math.degrees(self.pose[2]-old_yaw)
        self.window.append((raw,measured,dt))
        predicted,observed,elapsed=np.sum(self.window,axis=0)
        if abs(predicted)<10:
            if elapsed>2:self.window.clear()
            return
        values=np.asarray(self.window)
        self.window.clear()
        ratio=observed/predicted
        fit_error=np.max(abs(np.cumsum(values[:,1]-ratio*values[:,0])))
        if not .5<ratio<1.05 or fit_error>.65 or elapsed>2 or len(values)<5:
            return
        old=self.left_mix if predicted<0 else self.right_mix
        new=float(np.clip(old+.5*(ratio-old),max(.55,old-.04),min(1.,old+.04)))
        if predicted<0:self.left_mix=new
        else:self.right_mix=new
        self.last_update=dict(side="left" if predicted<0 else "right",raw_deg=predicted,observed_deg=observed,ratio=ratio,output=1/new)
