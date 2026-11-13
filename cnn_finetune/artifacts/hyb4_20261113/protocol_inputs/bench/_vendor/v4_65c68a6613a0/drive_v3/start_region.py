"""Approximate pose prior from a physical placement region, not a measured pose."""
import math
import numpy as np


def right_red_prior(track,vehicle,direction):
    region=track.data['start_regions']['right_red']
    yaw=region['heading_by_direction'][direction]
    ys=sorted(float(np.mean(track.data['red_lines'][i],axis=0)[1]) for i in region['red_indices'])
    length=vehicle['front_cm']+vehicle['rear_cm']
    body_y=[ys[0]+length/2,ys[1]-length/2]
    x0,x1=region['road_x_cm'];half=vehicle['half_width_cm']
    body_x=[x0+half,x1-half]
    if body_y[0]>body_y[1] or body_x[0]>body_x[1]:raise ValueError('Vehicle does not fit start region')
    # Body midpoint is ahead of the axle midpoint by (front-rear)/2.
    shift=(vehicle['front_cm']-vehicle['rear_cm'])/2
    mean=np.array([np.mean(body_x),np.mean(body_y),yaw])
    mean[:2]-=shift*np.array([math.sin(yaw),math.cos(yaw)])
    # Engineering approximation: uniform placement ranges plus existing map
    # uncertainty. This is not an empirically calibrated probability or hard bound.
    map_sigma=track.data.get('position_uncertainty_cm',1.)
    sigma=[math.hypot((b[1]-b[0])/math.sqrt(12),map_sigma) for b in (body_x,body_y)]
    sigma.append(math.radians(8))
    return mean,sigma,dict(region='right_red',body_midpoint_x_cm=body_x,
        body_midpoint_y_cm=body_y,red_y_cm=ys,nominal_heading_rad=yaw,
        note='Ranges assume body parallel and wholly inside lines. Yaw spread is provisional; rotated footprint couples x/y/yaw. Visual fit must refine the pose; centre is not asserted as ground truth.')
