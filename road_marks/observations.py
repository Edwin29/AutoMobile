"""Associate boxes with the current approach before passing them to mission_rules."""
from collections import deque
from math import hypot
import cv2
import numpy as np


class Observations:
    """Create a fresh instance at each approach. Inputs are upright image pixels.
    Polygons come from lane/map registration. An absent polygon gives no observation.
    The freshness and motion thresholds are initial replay parameters.
    """
    def __init__(self, max_age=.4):
        self.history = deque(maxlen=2)
        self.max_age = max_age

    def update(self, timestamp, detections, image_shape, lane, right_verge):
        if self.history and timestamp <= self.history[-1][0]:
            raise ValueError("Use unique frames in exposure-time order")
        h,w = image_shape[:2]
        current = {}
        for key, names, polygon in (
            ("right_sign", ("slow","horn","stop","right"), right_verge),
            ("right_light", ("red","green"), right_verge),
            ("crossing", ("cross",), lane),
            ("pink", ("dashline",), lane),
            ("start_line", ("line",), lane),
        ):
            candidates = []
            if polygon is not None and len(polygon) >= 3:
                region = np.asarray(polygon,dtype=np.float32)
                for item in detections:
                    if item["kind"] not in names:
                        continue
                    x1,y1,x2,y2 = item["box"]
                    foot = (float((x1+x2)/2),float(y2))
                    if cv2.pointPolygonTest(region,foot,False) >= 0:
                        candidates.append((item["kind"],foot[0]/w,foot[1]/h))
            # Multiple different instructions in the same region: withhold the decision.
            if candidates and len({c[0] for c in candidates}) == 1:
                current[key] = max(candidates,key=lambda c:c[2])
        confirmed = {}
        for key,(kind,x,y) in current.items():
            for t,previous in self.history:
                old = previous.get(key)
                if timestamp-t <= self.max_age and old and old[0]==kind and hypot(x-old[1],y-old[2]) <= .15:
                    confirmed[key] = kind
                    break
        self.history.append((timestamp,current))
        return confirmed
