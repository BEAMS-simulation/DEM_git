from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import numpy as np

from .config import SimConfig
from .math3d import quat_from_rotvec, quat_mul, quat_normalize
from .world import World

@dataclass
class Simulator:
    world: World
    
    force: np.ndarray = field(init=False)
    torque: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.force = np.zeros((self.world.n, 3), dtype=float)
        self.torque = np.zeros((self.world.n, 3), dtype=float)
    
    def initialize(self) -> None:
        dt = float(self.world.config.time.dt)
        self.force, self.torque = self.world.compute_forces_and_torques(dt)
    
    def _kick(self, i: int, dt: float, F: np.ndarray, T: np.ndarray) -> None:
        b = self.world.bodies[i]
        if b.sleep.is_sleep and self.world.config.sleep.enable:
            return
        
        b.vel += 0.5 * (F / b.body.mass) * dt
        
        R = b.rot_mat()
        tau_body = R.T @ T
        
        Ib = b.body.Ib
        w = b.omega_body
        dw = b.body.Ib_inv @ (tau_body - np.cross(w, Ib @ w))
        b.omega_body = w + 0.5 * dw * dt
        
        b.mark_dirty()
    
    def _drift(self, i: int, dt: float) -> None:
        b = self.world.bodies[i]
        if b.sleep.is_sleep and self.world.config.sleep.enable:
            return
        
        b.pos += b.vel * dt
        
        dq = quat_from_rotvec(b.omega_body * dt)
        b.quat = quat_normalize(quat_mul(dq, b.quat))
        
        b.mark_dirty()
    
    def _maybe_sleep_wake(self, i: int, F: np.ndarray, T: np.ndarray) -> None:
        cfg = self.world.config
        if not cfg.sleep.enable:
            return
        
        b = self.world.bodies[i]
        s = b.sleep
        
        if s.is_sleep:
            v2 = float(np.dot(b.vel, b.vel))
            w2 = float(np.dot(b.omega_body, b.omega_body))
            if v2 > cfg.sleep.wake_speed ** 2 or w2 > cfg.sleep.wake_ang_speed ** 2:
                s.wake()
                return
            
            a = F / b.body.mass
            if float(np.dot(a, a)) >= cfg.sleep.wake_acc ** 2:
                s.wake()
                return
            
            R = b.rot_mat()
            tau_body = R.T @ T
            alpha = b.body.Ib_inv @ tau_body
            if float(np.dot(alpha, alpha)) >= cfg.sleep.wake_ang_acc ** 2:
                s.wake()
                return
            
            return
        
        v2 = float(np.dot(b.vel, b.vel))
        w2 = float(np.dot(b.omega_body, b.omega_body))
        low = (v2 < cfg.sleep.sleep_speed ** 2) and (w2 < cfg.sleep.sleep_ang_speed ** 2)

        had_contact = bool(self.world.contact.contact_flag[i]) if self.world.contact.contact_flag is not None else False
        
        if low and had_contact:
            s.sleepy_steps += 1
            if s.sleepy_steps >= cfg.sleep.sleep_dur_threshold:
                s.sleep()
                b.vel[:] = 0.0
                b.omega_body[:] = 0.0
                b.mark_dirty()
        else:
            s.reset()
    
    def step(self) -> None:
        cfg = self.world.config
        dt = float(cfg.time.dt)
        
        for i in range(self.world.n):
            self._kick(i, dt, self.force[i], self.torque[i])
        
        for i in range(self.world.n):
            self._drift(i, dt)
        
        self.world.wrap_positions()
        
        newF, newT = self.world.compute_forces_and_torques(dt)
        
        for i in range(self.world.n):
            self._maybe_sleep_wake(i, newF[i], newT[i])
        
        for i in range(self.world.n):
            self._kick(i, dt, newF[i], newT[i])
        
        self.force, self.torque = newF, newT
    
    def run(self) -> dict[str, np.ndarray]:
        cfg = self.world.config
        dt = float(cfg.time.dt)
        max_steps = int(np.ceil(cfg.time.max_time / dt))
        
        ts: list[float] = []
        ke_hist: list[float] = []
        p_hist: list[np.ndarray] = []
        l_hist: list[np.ndarray] = []
        
        stable_dur = 0.0
        t = 0.0
        
        import time as _time
        start = _time.time()
        
        for step in range(1, max_steps + 1):
            self.step()
            t += dt
            
            if step % cfg.time.record_stride == 0:
                ts.append(t)
                ke_hist.append(self.world.total_kinetic())
                p_hist.append(self.world.total_linear_momentu().copy())
                l_hist.append(self.world.total_angular_momentum().copy())
            
            if self.world.total_kinetic() < cfg.kinetic_tol:
                stable_dur += dt
            else:
                stable_dur = 0.0
            
            if cfg.time.stable_time > 0 and stable_dur >= cfg.time.stable_time:
                break
            
            if step % cfg.time.log_stride == 0:
                now = _time.time()
                dur = int(now - start)
                print(
                    f"Step {step}/{max_steps} t={t:.4f}s wall={datetime.timedelta(seconds=dur)} "
                    f"KE={self.world.total_kinetic():.6e} sleep={sum(b.sleep.is_sleep for b in self.world.bodies)}/{self.world.n}"
                )
        
        return {
            "t": np.array(ts, dtype=float),
            "ke": np.array(ke_hist, dtype=float),
            "p": np.array(p_hist, dtype=float) if p_hist else np.zeros((0, 3), dtype=float),
            "l": np.array(l_hist, dtype=float) if l_hist else np.zeros((0, 3), dtype=float),
        }