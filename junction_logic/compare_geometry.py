"""Synthetic geometry only: no track fit, simulation of dynamics, camera or motors."""
import json
from pathlib import Path
import numpy as np


def bezier_metrics(radius, handle):
    t=np.linspace(0,1,10001)[:,None]
    p=np.array([[0,0],[0,handle],[radius-handle,radius],[radius,radius]],float)
    d=3*((1-t)**2*(p[1]-p[0])+2*(1-t)*t*(p[2]-p[1])+t*t*(p[3]-p[2]))
    dd=6*((1-t)*(p[2]-2*p[1]+p[0])+t*(p[3]-2*p[2]+p[1]))
    k=(d[:,1]*dd[:,0]-d[:,0]*dd[:,1])/np.linalg.norm(d,axis=1)**3
    angle=np.degrees(np.arctan2(d[-1,0],d[-1,1])-np.arctan2(d[0,0],d[0,1]))
    assert abs(angle-90)<1e-9 and np.isfinite(k).all()
    return dict(handle_cm=handle,heading_change_deg=float(angle),
                endpoint_radius_cm=float(1/k[0]),minimum_radius_cm=float(1/np.max(np.abs(k))))


if __name__=='__main__':
    result={'scope':'Synthetic endpoints (0,0) to (20,20) cm; not measured track geometry',
            'bezier':[bezier_metrics(20,a) for a in (8,20*4/3*np.tan(np.pi/8),16)],
            'width_assumption_error_cm':(26-18)/2,
            'straight_clearance_per_side_cm':(18-15.5)/2,
            'circle_geometric_center_radius_cm':20,
            'outer_corner_radius_cm':float(np.hypot(20+15.5/2,26/2))}
    (Path(__file__).parent/'geometry_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
