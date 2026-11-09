"""Camera boundary candidate extraction for offline inspection, not road truth.

Input must already have lens correction. H maps these exact image coordinates
to the vehicle-centered ground plane in metres. A separately calibrated floor
ROI excludes walls/signs. Yellow HSV thresholds must be checked on the track.
"""
import cv2
import numpy as np
from guard import ground_points


def yellow_boundary_candidates(image_bgr, floor_roi, H, *,
                               hsv_low=(15,70,70), hsv_high=(45,255,255),
                               max_range_m=1.0, min_component_pixels=20):
    if image_bgr is None or image_bgr.ndim!=3 or image_bgr.shape[2]!=3:
        raise ValueError('Expected a BGR image')
    if floor_roi is None or np.shape(floor_roi)!=image_bgr.shape[:2] or H is None:
        raise ValueError('Measured ground homography and floor ROI are mandatory')
    if max_range_m<=0:
        raise ValueError('Range must be positive')
    hsv=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array(hsv_low,np.uint8),np.array(hsv_high,np.uint8))
    mask[np.asarray(floor_roi)==0]=0
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    clean=np.zeros_like(mask)
    for label in range(1,count):
        if stats[label,cv2.CC_STAT_AREA]>=min_component_pixels:
            clean[labels==label]=255
    y,x=np.nonzero(clean)
    # Deterministic bounded sampling; points are candidates, never hard free space.
    step=max(1,len(x)//500)
    pixels=np.column_stack([x[::step],y[::step]])
    if len(pixels)==0:
        return np.empty((0,2)),clean
    pts=ground_points(pixels,H)
    pts=pts[np.linalg.norm(pts,axis=1)<=max_range_m]
    return pts,clean
