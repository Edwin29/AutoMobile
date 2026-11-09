"""Preserve recent camera inputs and the exact edge samples used by the controller."""
import csv
from collections import deque

import cv2
import numpy as np

from controller import LENS_AHEAD


def overlay(driver, frame, points, residuals):
    image=cv2.remap(frame,*driver.remap,cv2.INTER_LINEAR)
    if len(points):
        bird=(points-[0,LENS_AHEAD])*[5,-5]+[200,250]
        pixels=cv2.perspectiveTransform(bird.astype(np.float32)[:,None],np.linalg.inv(driver.M))[:,0]
        for uv,error in zip(pixels,residuals):
            color=(220,220,20) if abs(error)<2.5 else (20,40,250)
            cv2.circle(image,tuple(uv.round().astype(int)),3,color,-1)
    return image


class EdgeBuffer:
    def __init__(self,driver,destination):
        self.frames=deque(maxlen=24)
        self.driver=driver
        self.directory=destination.parent/(destination.stem+"_edges")
        driver.record_edges=True

    def append(self,frame,t,metadata,observation,reason):
        self.frames.append((frame,t,metadata.get("ExposureTime"),metadata.get("AnalogueGain"),
                            observation,reason))

    def save(self):
        self.directory.mkdir(exist_ok=False)
        with (self.directory/"index.csv").open("x",newline="") as stream:
            writer=csv.writer(stream)
            writer.writerow(("file","t","exposure_us","analogue_gain","samples","inlier_fraction",
                             "median_abs_cm","prior_x","prior_y","prior_yaw_rad","reason"))
            for i,(frame,t,exposure,gain,observation,reason) in enumerate(self.frames):
                name=f"{i:02d}"
                points,residuals,pose=(observation if observation is not None else
                                       (np.empty((0,2)),np.empty(0),[float("nan")]*3))
                if not cv2.imwrite(str(self.directory/(name+".png")),frame):
                    raise OSError("Could not write camera frame")
                if not cv2.imwrite(str(self.directory/(name+"_edges.png")),overlay(self.driver,frame,points,residuals)):
                    raise OSError("Could not write overlay")
                np.savez(self.directory/(name+".npz"),points=points,residual_cm=residuals,prior=pose)
                writer.writerow((name,t,exposure,gain,len(points),
                                 (abs(residuals)<2.5).mean() if len(points) else "",
                                 np.median(abs(residuals)) if len(points) else "",*pose,reason))
