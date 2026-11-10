"""Do not confuse prohibiting NEW deep departure with forbidding an escape."""
import numpy as np


def depth_rejection(profile,limit=6.,allow_existing=False,tolerance=.1):
    p=np.asarray(profile,float)
    if p[0]<-limit and allow_existing:
        if p.min()<p[0]-tolerance:return 'existing_deep_worsens'
        if p[-1]<0:return 'existing_deep_not_back_inside'
        return None
    return 'deep_departure' if p.min()<-limit else None
