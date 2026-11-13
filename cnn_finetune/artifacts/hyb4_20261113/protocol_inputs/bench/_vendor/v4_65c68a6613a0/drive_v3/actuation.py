"""Single-writer output adapter with an independent command-expiry watchdog.

Callbacks may wrap an existing GPIO implementation. No board/pin guesses here.
The four callbacks accept signed duty; calibration is per logical wheel.
"""
import threading
import time
import numpy as np


class WatchdogSink:
    def __init__(self, wheels, horn=lambda active:None, timeout_s=.35, clock=time.monotonic):
        required={"front_left","rear_left","front_right","rear_right"}
        if set(wheels)!=required:raise ValueError("Supply all four wheel callbacks explicitly")
        if timeout_s<=0:raise ValueError("Positive command timeout required")
        self.wheels=wheels;self.horn=horn;self.timeout=timeout_s;self.clock=clock
        self.lock=threading.RLock();self.done=threading.Event();self.last=-1e9;self.error=None
        self.stop()
        self.worker=threading.Thread(target=self._watch,name="drive-v3-watchdog",daemon=True);self.worker.start()

    def _zero(self):
        failures=[]
        for callback in self.wheels.values():
            try:callback(0.)
            except Exception as e:failures.append(repr(e))
        try:self.horn(False)
        except Exception as e:failures.append(repr(e))
        if failures:self.error="; ".join(failures)

    def write(self,left,right,horn=False):
        with self.lock:
            if self.done.is_set() or self.error:raise RuntimeError(self.error or "Sink closed")
            if not np.isfinite([left,right]).all() or max(abs(left),abs(right))>1:
                self._zero();raise ValueError("Invalid PWM")
            try:
                for name,callback in self.wheels.items():callback(float(left if name.endswith("left") else right))
                self.horn(bool(horn));self.last=self.clock()
            except BaseException:
                self._zero();raise

    def stop(self):
        with self.lock:self._zero();self.last=-1e9

    def check_expiry(self):
        with self.lock:
            if self.clock()-self.last>self.timeout:self._zero();return True
        return False

    def _watch(self):
        while not self.done.wait(min(.02,self.timeout/4)):self.check_expiry()

    def close(self):
        self.done.set();self.stop();self.worker.join(timeout=1.)

