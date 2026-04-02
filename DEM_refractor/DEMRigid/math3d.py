from __future__ import annotations
import math
import numpy as np


EPS = 1e-12


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def safe_norm(v: np.ndarray, eps: float = EPS) -> float:
    n = float(np.linalg.norm(v))
    return n if n > eps else eps


def unit(v: np.ndarray, eps: float = EPS) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = safe_norm(v, eps)
    return v / n


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0],dtype=float)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = safe_norm(q)
    return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = map(float, q1)
    w2, x2, y2, z2 = map(float, q2)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )

def quat_from_rotvec(rotvec: np.ndarray, eps: float = EPS) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=float)
    ang = float(np.linalg.norm(rv))
    if ang < eps:
        half = 0.5 * ang
        if ang < eps:
            return quat_identity()
        axis = rv / ang
        return quat_normalize(np.array(
            [math.cos(half), *(axis * math.sin(half))], dtype = float
        ))
        
    half = 0.5 * ang
    axis = rv / ang
    return np.array([math.cos(half), *(axis * math.sin(half))], dtype = float)

def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = map(float, q)
    
    ww = w*w
    xx = x*x
    yy = y*y
    zz = z*z
    
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    
    return np.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ], dtype = float,
    )

def apply_rot_mat(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    return R @ np.asarray(v, dtype = float)

def apply_rot_mat_batch(R: np.ndarray, V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype= float)
    return V @ R.T