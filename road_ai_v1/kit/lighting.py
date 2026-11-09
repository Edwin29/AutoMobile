"""Paired photometric probes in camera RGB. Position and mask geometry stay aligned."""
import cv2
import numpy as np

PROBES=("clean","warm_bright","shadow","seam","hard_mix")

def lighting(rgb,labels,condition,rng):
    if condition=="clean":return rgb,labels
    im=rgb.astype(np.float32)/255
    h,w=labels.shape
    y,x=np.mgrid[:h,:w].astype(np.float32)
    hard=condition=="hard_mix"
    if condition=="seam" or hard:
        # A floor join inside the road is still traversable. Never paint over an occluder.
        line=np.zeros((h,w),np.uint8)
        v=int(rng.uniform(.47,.8)*h)
        cv2.line(line,(0,v),(w-1,v+int(rng.uniform(-.14,.14)*h)),255,int(rng.integers(1,4)))
        im[(line>0)&(labels==1)]*=rng.uniform(.40,.75)
    if condition=="warm_bright" or hard:
        exposure=rng.uniform(1.08,1.32) if not hard else rng.uniform(1.35,1.75)
        im*=exposure*np.array([1.06,1.,rng.uniform(.62,.85)],np.float32)
    if condition=="shadow" or hard:
        theta=rng.uniform(0,np.pi)
        axis=(x-w*rng.uniform(.2,.8))*np.cos(theta)+(y-h*rng.uniform(.3,.8))*np.sin(theta)
        width=rng.uniform(12,40)
        filt=np.clip((width-np.abs(axis))/rng.uniform(3,12),0,1)
        im*=1-(filt*rng.uniform(.30,.70))[:,:,None]
    if hard:
        highlight=np.exp(-.5*(((x-rng.uniform(.15,.85)*w)/rng.uniform(14,45))**2+
                                ((y-rng.uniform(.40,.86)*h)/rng.uniform(4,13))**2))
        im+=highlight[:,:,None]*rng.uniform(.55,.95)
        im=cv2.GaussianBlur(im,(5,5),rng.uniform(.6,1.4))
    # Fully clipped regions contain no usable detail; do not supervise hidden geometry there.
    target=labels.copy()
    target[np.min(im,axis=2)>=1.]=255
    return np.uint8(np.clip(im,0,1)*255),target
