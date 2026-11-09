"""Camera/Claude analysis workload only. Never import lane_drive or open GPIO."""
import sys
import threading
import time


class VisionLoad(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.last_frame = 0.
        self.frames = 0
        self.error = None

    def run(self):
        cam = None
        try:
            import cv2
            import numpy as np
            from picamera2 import Picamera2
            sys.path.insert(0, '/home/pi/TEST')
            import lane_pipeline as L
            with np.load('/home/pi/TEST/calib_checkerboard.npz') as c:
                size_img = tuple(int(v) for v in c['size'])
                newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    c['K'], c['D'], size_img, np.eye(3), balance=0.)
                m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                    c['K'], c['D'], np.eye(3), newK, size_img, cv2.CV_16SC2)
            with np.load('/home/pi/TEST/ground.npz') as g:
                M, size, _ = L.bev_transform(dict(g), newK, size_img[1])
            cam = Picamera2()
            cfg = cam.create_video_configuration(main={'format':'RGB888', 'size':size_img},
                                                 controls={'FrameRate':15})
            cam.align_configuration(cfg)
            cam.configure(cfg)
            cam.start()
            hint = L.DEFAULT_ROAD
            while not self.stop_event.is_set():
                img = cv2.flip(cam.capture_array(), -1)
                und = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
                result, _ = L.analyze(und, M, size, None, 'straight', hint)
                if result is not None:
                    hint = result['road']
                self.frames += 1
                self.last_frame = time.monotonic()
        except BaseException as exc:
            self.error = repr(exc)
        finally:
            if cam is not None:
                cam.close()
