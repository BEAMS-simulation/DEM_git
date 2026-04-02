from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .math3d import quat_identity, quat_normalize, quat_to_mat, apply_rot_mat_batch, apply_rot_mat


@dataclass(frozen=True)
class Sphere:
    r: float
    m: float
    pos_local: np.ndarray
    
    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError("Sphere radius must be positive.")
        if self.m <= 0:
            raise ValueError("Sphere mass must be positive.")
        p = np.asarray(self.pos_local, dtype=float)
        if p.shape != (3,):
            raise ValueError(f"pos_local must have shape (3,), got {p.shape}")

@dataclass
class Aggregate:
    spheres: list[Sphere]
    
    n: int = field(init=False)
    mass: float = field(init=False)
    radii: np.ndarray = field(init=False) # (n,)
    masses: np.ndarray = field(init=False) # (n,)
    pos_local: np.ndarray = field(init=False) # (n,3) (CoM 기준)
    Ib: np.ndarray = field(init=False)
    Ib_inv: np.ndarray = field(init=False)
    size: float = field(init=False) # bounding-sphere radius
    
    def __post_init__(self) -> None:
        if len(self.spheres) == 0:
            raise ValueError("Aggregate must contain at least 1 sphere.")
        self.n = len(self.spheres)
        
        self.radii = np.array([s.r for s in self.spheres], dtype=float)
        self.masses = np.array([s.m for s in self.spheres], dtype=float)
        self.mass = float(np.sum(self.masses))
        
        pos = np.array([np.asarray(s.pos_local, dtype=float) for s in self.spheres], dtype=float)
        
        # CoM 정렬
        com = np.average(pos, axis=0, weights=self.masses)
        self.pos_local = pos - com
        
        self.Ib = self._compute_body_inertia()
        self.Ib_inv = np.linalg.inv(self.Ib)
        
        # bounding sphere (CoM 기준)
        self.size = float(np.max(np.linalg.norm(self.pos_local, axis=1) + self.radii))

    def _compute_body_inertia(self) -> np.ndarray:
        I = np.zeros((3, 3), dtype=float)
        E = np.eye(3, dtype=float)
        for r, m, s in zip(self.radii, self.masses, self.pos_local):
            s = np.asarray(s, dtype=float)
            I += (2.0 / 5.0) * m * (r ** 2) * E
            I += m * ((np.dot(s, s) * E) - np.outer(s, s))
        return I

@dataclass
class SleepState:
    is_sleep: bool = False
    sleepy_steps: int = 0
    
    def reset(self) -> None:
        self.sleepy_steps = 0
    
    def sleep(self) -> None:
        self.is_sleep = True
    
    def wake(self) -> None:
        self.is_sleep = False
        self.sleepy_steps = 0

@dataclass
class Rigidbody:
    body: Aggregate
    id: int
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    quat: np.ndarray = field(default_factory=quat_identity) # body->world
    omega_body: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    
    sleep: SleepState = field(default_factory=SleepState)
    
    _rot_mat: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float), init=False)
    _omega_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float),
    init=False)
    _sphere_pos_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3),
    dtype=float), init=False)
    _sphere_vel_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3),
    dtype=float), init=False)
    _dirty: bool = field(default=True, init=False)
    
    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=float).copy()
        self.vel = np.asarray(self.vel, dtype=float).copy()
        self.quat = quat_normalize(np.asarray(self.quat, dtype=float))
        self.omega_body = np.asarray(self.omega_body, dtype=float).copy()
        if self.pos.shape != (3,) or self.vel.shape != (3,) or self.omega_body.shape != (3,):
            raise ValueError("pos/vel/omega_body must have shape (3,).")
        if self.quat.shape != (4,):
            raise ValueError("quat must have shape (4,).")

        self._sphere_pos_world = np.zeros((self.body.n, 3), dtype=float)
        self._sphere_vel_world = np.zeros((self.body.n, 3), dtype=float)
        self._dirty = True

    def mark_dirty(self) -> None:
        self._dirty = True

    def rot_mat(self) -> np.ndarray:
        if self._dirty:
            self.update_cache()
        return self._rot_mat

    def omega_world(self) -> np.ndarray:
        if self._dirty:
            self.update_cache()
        return self._omega_world

    def sphere_pos_world(self) -> np.ndarray:
        if self._dirty:
            self.update_cache()
        return self._sphere_pos_world

    def sphere_vel_world(self) -> np.ndarray:
        if self._dirty:
            self.update_cache()
        return self._sphere_vel_world
    
    def update_cache(self) -> None:
        """pos/vel/quat/omega_body 변경 후, world 좌표계 상태를 일괄 갱신."""
        self.quat = quat_normalize(self.quat)
        self._rot_mat = quat_to_mat(self.quat)
        self._omega_world = apply_rot_mat(self._rot_mat, self.omega_body)
        
        # sphere offsets in world
        offsets = apply_rot_mat_batch(self._rot_mat, self.body.pos_local) # (n,3)
        self._sphere_pos_world[:, :] = self.pos[None, :] + offsets
        
        # v = v_com + omega x r
        self._sphere_vel_world[:, :] = self.vel[None, :] + np.cross(self._omega_world[None, :], offsets)
        self._dirty = False
    
    def translational_ke(self) -> float:
        return 0.5 * self.body.mass * float(np.dot(self.vel, self.vel))

    def rotational_ke(self) -> float:
        return 0.5 * float(np.dot(self.omega_body, self.body.Ib @ self.omega_body))

    def linear_momentum(self) -> np.ndarray:
        return self.body.mass * self.vel
    
    def angular_momentum_world(self) -> np.ndarray:
        # L_world = R * (I_body * omega_body)
        L_body = self.body.Ib @ self.omega_body
        return apply_rot_mat(self.rot_mat(), L_body)