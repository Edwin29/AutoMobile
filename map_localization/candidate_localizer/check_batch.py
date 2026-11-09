"""Check batching against scalar evaluation and time an isolated local search."""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from matcher import grid
from validate_candidates import setup,scenes,axes,LENS_AHEAD

ROOT=Path(__file__).resolve().parent
cv2.setNumThreads(1)
world,matcher=setup()
truth=scenes()["mid_turn"]
prior=truth+np.array([2.3,-3.7,np.radians(8.3)])
camera=truth[:2]+LENS_AHEAD*axes(truth[2])[1]
frame=world.render((*camera,truth[2]))
matcher.prepare(frame,prior)
poses=grid(prior)[::7]
t=time.perf_counter()
batch=matcher.forward(poses)[0]
batch_seconds=time.perf_counter()-t
t=time.perf_counter()
scalar=np.array([matcher.forward(p[None])[0][0] for p in poses])
scalar_seconds=time.perf_counter()-t
assert np.allclose(batch,scalar,rtol=1e-5,atol=1e-5,equal_nan=True)
result,_=matcher.search(frame,prior)
test_point=np.array([[0.,25.]],np.float32)
pose=np.array([[-83.,-.3,.7]],np.float32)
projected,_,_=matcher.project(pose,test_point,np.array([[1.,0.]],np.float32))
# Independent explicit coordinate transformation must match the batch transform.
delta=test_point[0]-pose[0,:2]
local=delta@axes(pose[0,2]).T-[0,LENS_AHEAD]
q=matcher.H@np.r_[local,1.]
assert np.allclose(projected[0,0],q[:2]/q[2],atol=.001)
report=dict(candidates=len(poses),batch_seconds=batch_seconds,scalar_seconds=scalar_seconds,
            full_search_seconds=result["seconds"],full_search_candidates=result["candidates"],
            max_score_difference=float(np.max(np.abs(batch-scalar))),
            opencv_threads=cv2.getNumThreads(),platform="Windows laptop, not Raspberry Pi")
(ROOT/"benchmark.json").write_text(json.dumps(report,indent=2),encoding="utf8")
print(json.dumps(report,indent=2))
