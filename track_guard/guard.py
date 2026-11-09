"""Offline map constraints; never writes GPIO or sends motor commands.

World: x east/right, y north/up, yaw counter-clockwise from east, metres.
Image mask: row 0 = world y=0 (a plotting choice, not a camera convention).
Pose refers to the geometric center of the full vehicle footprint.
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class TianGraph:
    """Nine nodes, twelve undirected segments. Geometry requires survey data."""
    def __init__(self):
        self.nodes = {f'{x}{y}': (x, y) for x in range(3) for y in range(3)}
        self.adj = {n: set() for n in self.nodes}
        for n, (x, y) in self.nodes.items():
            for other, (xx, yy) in self.nodes.items():
                if abs(x-xx) + abs(y-yy) == 1:
                    self.adj[n].add(other)

    def exits(self, previous, junction):
        if previous not in self.adj.get(junction, set()):
            raise ValueError('Incoming edge is not in map')
        a = np.array(self.nodes[junction]) - self.nodes[previous]
        result = {}
        for nxt in self.adj[junction] - {previous}:
            b = np.array(self.nodes[nxt]) - self.nodes[junction]
            cross = int(a[0]*b[1] - a[1]*b[0])
            result['left' if cross > 0 else 'right' if cross < 0 else 'straight'] = nxt
        return result

    def common_turns(self, hypotheses):
        """Intersection over plausible directed arrivals: ambiguity never adds an exit."""
        if not hypotheses:
            return set()
        allowed = {'left', 'right', 'straight'}
        for previous, junction in hypotheses:
            try:
                allowed &= set(self.exits(previous, junction))
            except ValueError:
                return set()
        return allowed


class PoseFilter:
    """Local EKF. Requires known start or an unambiguous camera measurement.

    Commanded speed is only a noisy prediction, NOT encoder odometry.
    A single Gaussian cannot resolve the symmetric track's multiple locations.
    The caller must keep topological hypotheses separately and stop if unresolved.
    """
    def __init__(self, pose, covariance):
        self.x = np.asarray(pose, dtype=float).copy()
        self.P = np.asarray(covariance, dtype=float).copy()
        self.last_camera_time = None
        self.time = None
        if self.x.shape != (3,) or self.P.shape != (3, 3):
            raise ValueError('Pose must be 3-vector, covariance 3x3')
        if not np.isfinite(self.x).all() or not np.isfinite(self.P).all() or not np.allclose(self.P,self.P.T) or np.min(np.linalg.eigvalsh(self.P))<0:
            raise ValueError('Invalid initial state/covariance')

    def predict(self, v, omega, dt, now, speed_std=.05, yaw_rate_std=.2):
        if dt <= 0 or dt > 1 or not np.isfinite([v, omega, dt, now]).all():
            raise ValueError('Invalid prediction interval/control')
        if self.time is not None and now <= self.time:
            raise ValueError('Non-monotonic timestamp')
        theta = self.x[2] + omega*dt/2
        F = np.eye(3)
        F[0, 2], F[1, 2] = -v*dt*math.sin(theta), v*dt*math.cos(theta)
        self.x += [v*dt*math.cos(theta), v*dt*math.sin(theta), omega*dt]
        self.x[2] = wrap(self.x[2])
        # Conservative isotropic translation uncertainty; tune against real motion.
        Q = np.diag([(speed_std*dt)**2]*2 + [(yaw_rate_std*dt)**2])
        self.P = F @ self.P @ F.T + Q
        self.time = now

    def update_camera(self, pose, covariance, captured_at, now, max_age=.25):
        z, R = np.asarray(pose, float), np.asarray(covariance, float)
        if z.shape != (3,) or R.shape != (3,3):
            return False
        if not np.isfinite(z).all() or not np.isfinite(R).all():
            return False
        if now < captured_at or now-captured_at > max_age:
            return False
        if self.last_camera_time is not None and captured_at <= self.last_camera_time:
            return False
        if np.min(np.linalg.eigvalsh(R)) <= 0:
            return False
        delta = z-self.x; delta[2] = wrap(delta[2])
        S = self.P+R
        if float(delta @ np.linalg.solve(S, delta)) > 11.345:
            return False  # 99% chi-square gate, three measured dimensions.
        K = np.linalg.solve(S.T, self.P.T).T
        self.x += K@delta; self.x[2] = wrap(self.x[2])
        I = np.eye(3)-K
        self.P = I@self.P@I.T + K@R@K.T
        self.last_camera_time = captured_at
        return True


def ground_points(undistorted_pixels, H):
    """Ground-only image points -> vehicle-centered metric ground plane.

    H is NOT the lens matrix. Survey at least four ground correspondences;
    it must include camera-to-vehicle position/orientation. No standing signs.
    """
    points = np.asarray(undistorted_pixels, float)
    H = np.asarray(H, float)
    if points.ndim != 2 or points.shape[1] != 2 or H.shape != (3,3):
        raise ValueError('Expected Nx2 pixels and 3x3 homography')
    q = np.column_stack([points, np.ones(len(points))]) @ H.T
    if not np.isfinite(q).all() or np.any(abs(q[:,2]) < 1e-8):
        raise ValueError('Invalid ground projection / near horizon')
    return q[:,:2]/q[:,2,None]


def pose_from_landmarks(vehicle_points, world_points, max_rms=.01):
    """Estimate SE(2) pose from >=3 uniquely identified ground landmarks.

    ID association is the caller's responsibility. Repeated zebra stripes are
    not unique landmarks. Rejects gross disagreement, not symmetric aliases.
    """
    a, b = np.asarray(vehicle_points,float), np.asarray(world_points,float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or len(a)<3:
        raise ValueError('At least three paired landmarks required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite landmarks')
    ac, bc = a-a.mean(0), b-b.mean(0)
    if np.linalg.svd(ac,compute_uv=False)[-1] < .005:
        raise ValueError('Landmarks must span two dimensions')
    U, _, Vt = np.linalg.svd(ac.T@bc)
    R = Vt.T@U.T
    if np.linalg.det(R)<0:
        Vt[-1]*=-1; R=Vt.T@U.T
    t = b.mean(0)-R@a.mean(0)
    rms = float(np.sqrt(np.mean(np.sum((a@R.T+t-b)**2,axis=1))))
    if rms>max_rms:
        raise ValueError('Camera/map landmarks disagree')
    return np.array([t[0],t[1],math.atan2(R[1,0],R[0,0])]), rms


@dataclass
class Vehicle:
    width: float
    length: float
    margin: float


@dataclass
class Sample:
    t: float
    x: float
    y: float
    yaw: float
    speed: float


class RoadMask:
    def __init__(self, mask, resolution, measured=False):
        self.mask=np.asarray(mask,dtype=bool)
        self.resolution=float(resolution)
        self.measured=measured
        if self.mask.ndim!=2 or not np.isfinite(resolution) or resolution<=0:
            raise ValueError('Invalid road mask')

    def footprint_fits(self, sample, vehicle, uncertainty, sweep_padding=0):
        # Disc erosion includes pose uncertainty + between-sample movement bound.
        r=self.resolution
        pad=vehicle.margin+uncertainty+sweep_padding+math.sqrt(2)*r
        radius=int(math.ceil(pad/r))
        yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
        kernel=(xx*xx+yy*yy<=radius*radius).astype(np.uint8)
        safe=cv2.erode(self.mask.astype(np.uint8),kernel,
                       borderType=cv2.BORDER_CONSTANT,borderValue=0)
        corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]],float)*[vehicle.length/2,vehicle.width/2]
        c,s=math.cos(sample.yaw),math.sin(sample.yaw)
        points=corners@np.array([[c,s],[-s,c]])+[sample.x,sample.y]
        pix=np.rint(points/r).astype(np.int32)
        h,w=self.mask.shape
        if (pix<0).any() or (pix[:,0]>=w).any() or (pix[:,1]>=h).any():
            return False
        # Check the entire filled footprint: corners alone miss holes/islands.
        car=np.zeros_like(safe);cv2.fillConvexPoly(car,pix,1)
        return not np.any((car!=0)&(safe==0))


@dataclass
class Decision:
    allow: bool
    reason: str


class Supervisor:
    def __init__(self, road, vehicle, max_camera_age=.25):
        self.road,self.vehicle,self.max_camera_age=road,vehicle,max_camera_age
        if not np.isfinite([vehicle.width,vehicle.length,vehicle.margin]).all() or min(vehicle.width,vehicle.length)<=0 or vehicle.margin<0:
            raise ValueError('Vehicle measurements required')

    def evaluate(self, trajectory, covariance, now, last_camera, turn, hypotheses,
                 *, min_deceleration, latency, tracking_error_bound, simulation=False):
        """Trajectory must include reaction latency AND full braking to rest.

        An allow result is conditional on the supplied model/uncertainty bounds.
        3-sigma is probabilistic, NOT an absolute physical safety guarantee.
        Independent freshness checks must run at the motor command boundary.
        """
        reject=lambda why:Decision(False,why)
        if not np.isfinite(now) or (last_camera is not None and not np.isfinite(last_camera)):
            return reject('invalid_timestamp')
        if not simulation and not self.road.measured:
            return reject('unmeasured_map')
        if last_camera is None or now<last_camera or now-last_camera>self.max_camera_age:
            return reject('stale_or_missing_camera')
        # No unique directed edge => no autonomous motion, even in a symmetric map.
        if len(set(hypotheses))!=1:
            return reject('ambiguous_location')
        if turn not in TianGraph().common_turns(hypotheses):
            return reject('forbidden_turn')
        P=np.asarray(covariance,float)
        if P.shape!=(3,3) or not np.isfinite(P).all() or not np.allclose(P,P.T) or np.min(np.linalg.eigvalsh(P))<0:
            return reject('invalid_covariance')
        if min_deceleration is None or latency is None or not np.isfinite([min_deceleration,latency]).all() or min_deceleration<=0 or latency<0:
            return reject('missing_braking_measurements')
        if tracking_error_bound is None or not np.isfinite(tracking_error_bound) or tracking_error_bound<0:
            return reject('missing_tracking_error_bound')
        if len(trajectory)<2:
            return reject('missing_stopping_trajectory')
        data=np.array([[p.t,p.x,p.y,p.yaw,p.speed] for p in trajectory])
        if not np.isfinite(data).all() or abs(data[0,0])>1e-9 or np.any(np.diff(data[:,0])<=0):
            return reject('invalid_trajectory')
        if np.any(data[:,4]<0) or abs(data[-1,4])>1e-6:
            return reject('trajectory_must_end_stopped')
        if data[-1,0] < latency+max(data[:,4])/min_deceleration:
            return reject('insufficient_stopping_horizon')
        radius=math.hypot(self.vehicle.length,self.vehicle.width)/2
        uncertainty=3*math.sqrt(max(np.linalg.eigvalsh(P[:2,:2])))
        uncertainty+=radius*min(2,3*math.sqrt(P[2,2]))+tracking_error_bound
        for i,p in enumerate(trajectory):
            sweep=0.
            if i+1<len(trajectory):
                q=trajectory[i+1]
                # Conservative envelope for linear position/yaw interpolation.
                sweep=math.hypot(q.x-p.x,q.y-p.y)+radius*abs(wrap(q.yaw-p.yaw))
            if not self.road.footprint_fits(p,self.vehicle,uncertainty,sweep):
                return reject('footprint_or_stopping_path_outside_road')
        return Decision(True,'within_supplied_model_bounds')


def demo_road(width=1.8,height=1.8,resolution=.002):
    """FICTIONAL square geometry, not measured from the user's photograph."""
    y,x=np.mgrid[0:height+resolution:resolution,0:width+resolution:resolution]
    road=(x<=.26)|(x>=width-.26)|(y<=.26)|(y>=height-.26)
    road|=(abs(x-width/2)<=.09)|(abs(y-height/2)<=.09)
    return RoadMask(road,resolution,measured=False)
