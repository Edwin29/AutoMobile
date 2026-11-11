"""Direct teacher: mean/0.66 and LEFT-positive curvature/0.095."""
import numpy as np
from .teacher import Teacher

class DirectTeacher(Teacher):
    curvature_cap = 0.085
    def __init__(self,direction,lookahead=18,variant=1):
        super().__init__(direction,lookahead,variant=8)
        self.direct_variant=variant
        self.corner_bias = 0.0 if variant == 1 else 2.0
    def step(self,pose):
        k,mean=self.target(pose)
        label=np.array([mean/0.66,-k/0.095])
        pwm,note=self.motor.mix(mean,k,0.66)
        self.last_pwm=np.array(pwm)
        return self.last_pwm,dict(k=k,requested_mean=mean,used_mean=mean,label=label.tolist(),allocator=note)
