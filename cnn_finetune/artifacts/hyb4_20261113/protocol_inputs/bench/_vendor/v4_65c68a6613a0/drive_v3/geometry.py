"""Planar math: x right/east, y forward/north; positive yaw clockwise."""
import math
import numpy as np


def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def axes(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def to_world(points, pose):
    return np.asarray(points) @ axes(pose[2]) + np.asarray(pose[:2])


def to_body(points, pose):
    return (np.asarray(points)-np.asarray(pose[:2])) @ axes(pose[2]).T


def advance(pose, v, omega, dt):
    a = omega*dt
    p = np.asarray(pose, float).copy()
    p[:2] += v*dt*np.sinc(a/(2*np.pi))*np.array([np.sin(p[2]+a/2), np.cos(p[2]+a/2)])
    p[2] = wrap(p[2]+a)
    return p


def path_geometry(points):
    p = np.asarray(points, float)
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if len(p) < 3 or np.any(np.diff(s) < 1e-8):
        raise ValueError("Path needs three distinct ordered samples")
    d = np.gradient(p, s, axis=0)
    theta = np.unwrap(np.arctan2(d[:, 0], d[:, 1]))
    k = np.gradient(theta, s)
    return s, theta, k


def footprint(cfg):
    w, f, r = cfg["half_width_cm"], cfg["front_cm"], cfg["rear_cm"]
    corners = np.array([[-w, -r], [w, -r], [w, f], [-w, f]])
    rim = np.concatenate([np.linspace(a, b, max(2, math.ceil(np.linalg.norm(b-a)/cfg["rim_step_cm"])), endpoint=False)
                          for a, b in zip(corners, np.roll(corners, -1, axis=0))])
    # Interior samples also catch a small excluded region entirely inside the footprint.
    xx, yy = np.meshgrid(np.arange(-w, w+.1, 2.), np.arange(-r, f+.1, 2.))
    return np.vstack((rim, np.c_[xx.ravel(), yy.ravel()]))


def sigma_poses(mean, covariance, scale):
    p = np.asarray(mean)
    eig, vec = np.linalg.eigh(np.asarray(covariance))
    d = vec * np.sqrt(np.maximum(eig, 0))*scale
    return np.vstack((p, p+d.T, p-d.T))
