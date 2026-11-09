"""Single-photo boundary-fit illustration. No vehicle control or ground-truth pose."""
import ast
import cv2
import numpy as np
from replay import SRC, RUN, ROOT, lane


def rounded_box(x, y, box, radius):
    x0,x1,y0,y1=box
    a=np.abs(x-(x0+x1)/2)-(x1-x0)/2+radius
    b=np.abs(y-(y0+y1)/2)-(y1-y0)/2+radius
    return np.hypot(np.maximum(a,0),np.maximum(b,0))+np.minimum(np.maximum(a,b),0)-radius


def main():
    # Read geometry literals without importing Pi-specific modules.
    config={}
    for n in ast.parse((SRC/'sim_drive.py').read_text(encoding='utf-8')).body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
            if n.targets[0].id in ('OUTER','OUTER_R','BLOCKS','BLOCK_R'):
                config[n.targets[0].id]=ast.literal_eval(n.value)
    def sdf(x,y):
        d=-rounded_box(x,y,config['OUTER'],config['OUTER_R'])
        for box in config['BLOCKS']:
            d=np.minimum(d,rounded_box(x,y,box,config['BLOCK_R']))
        return d
    c=np.load(SRC/'calib_checkerboard.npz')
    size=tuple(map(int,c['size']))
    K=cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(c['K'],c['D'],size,np.eye(3),balance=0)
    maps=cv2.fisheye.initUndistortRectifyMap(c['K'],c['D'],np.eye(3),K,size,cv2.CV_16SC2)
    g=dict(np.load(SRC/'ground.npz')); M,bev,_=lane.bev_transform(g,K,size[1])
    und=cv2.remap(cv2.imread(str(RUN/'frame_00515.jpg')),*maps,cv2.INTER_LINEAR)
    edge,_=lane.edge_masks(und,M,bev)
    pts=lane.to_cm(np.argwhere(edge>0)[:,::-1])
    pts=pts[(pts[:,1]>=6)&(pts[:,1]<=35)][::5]
    # Camera-ground origin, +yaw from north toward east. Half-cm world lookup.
    axis=np.arange(-160,160.5,.5); xx,yy=np.meshgrid(axis,axis)
    distances=np.abs(sdf(xx,yy))
    grid=np.arange(-98,100,2); x,y=np.meshgrid(grid,grid)
    ok=(np.abs(x)>45)&(np.abs(y)>45)&(sdf(x,y)>0)
    xy=np.column_stack((x[ok],y[ok]))
    best={}
    for yaw in np.arange(0,360,4):
        s,b=np.sin(np.radians(yaw)),np.cos(np.radians(yaw))
        X=xy[:,0,None]+pts[:,0]*b+pts[:,1]*s
        Y=xy[:,1,None]-pts[:,0]*s+pts[:,1]*b
        ds=distances[np.rint((Y+160)*2).astype(int),np.rint((X+160)*2).astype(int)]
        costs=np.minimum(ds,5).mean(axis=1)
        for sx,sy in [(1,1),(1,-1),(-1,-1),(-1,1)]:
            quadrant=(xy[:,0]*sx>0)&(xy[:,1]*sy>0)
            idx=np.flatnonzero(quadrant)[np.argmin(costs[quadrant])]
            entry=(float(costs[idx]),*map(float,xy[idx]),float(yaw))
            best[sx,sy]=min(best.get((sx,sy),entry),entry)
    print(sorted(best.values()))
    Hg=lane.ground_to_image(g,K)
    ix,iy=np.meshgrid(np.arange(size[0]),np.arange(size[1]))
    ray=np.linalg.inv(Hg)@np.stack([ix.ravel(),iy.ravel(),np.ones(ix.size)])
    lat,fwd=ray[:2]/np.where(abs(ray[2])>1.e-9,ray[2],np.nan)
    visible=(ray[2]>0)&(fwd>=6)&(fwd<=35)
    tiles=[]
    for k,(cost,x,y,yaw) in enumerate(sorted(best.values())[:2]):
        s,b=np.sin(np.radians(yaw)),np.cos(np.radians(yaw))
        dist=sdf(x+lat*b+fwd*s,y-lat*s+fwd*b)
        im=und.copy()
        im[(visible&(abs(dist)<.2)).reshape(iy.shape)]=(235,40,255)
        im=cv2.resize(im,(480,360))
        cv2.rectangle(im,(0,0),(480,27),(20,20,20),-1)
        cv2.putText(im,f'Candidate {k+1}   ({x:.0f}, {y:.0f}) cm   yaw {yaw:.0f} deg   fit {cost:.2f} cm',
                    (6,19),0,.47,(255,255,255),1,cv2.LINE_AA)
        tiles.append(im)
    cv2.imwrite(str(ROOT/'map_pose_example.jpg'),np.hstack(tiles))


if __name__=='__main__':
    main()
