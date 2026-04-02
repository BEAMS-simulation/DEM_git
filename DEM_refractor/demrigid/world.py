from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimConfig
from .broadphase import GridBroadphase
from .model import Rigidbody


@dataclass
class World:
    config: SimConfig
    bodies: list[Rigidbody]

    def __post_init__(self) -> None:
        if len(self.bodies) == 0:
            raise ValueError("World must contain at least 1 body.")
        self.n = len(self.bodies)

        max_size = max(b.body.size for b in self.bodies)
        cell_size = 2.0 * float(max_size) + float(self.config.neighbor.skin)
        self.broadphase = GridBroadphase(self.config.box, cell_size=cell_size)
        
        from .contact import ContactSolver
        self.contact = ContactSolver(self.config.box, pp=self.config.contact_pp, pw=self.config.contact_pw)
        
    def positions_array(self) -> np.ndarray:
        return np.array([b.pos for b in self.bodies], dtype=float)
    
    def wrap_positions(self) -> None:
        if self.config.box.boxtype != "per":
            return
        Lx, Ly = self.config.box.Lx, self.config.box.Ly
        for b in self.bodies:
            b.pos[0] %= Lx
            b.pos[1] %= Ly
            b.mark_dirty()
    
    def disp(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        if self.config.box.boxtype == "per":
            Lx, Ly = self.config.box.Lx, self.config.box.Ly
            d[0] -= Lx * np.round(d[0] / Lx)
            d[1] -= Ly * np.round(d[1] / Ly)
        return d
    
    def compute_forces_and_torques(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        F = np.zeros((n, 3), dtype=float)
        T = np.zeros((n, 3), dtype=float)

        for b in self.bodies:
            b.update_cache()
        
        self.contact.begin_step(n_bodies=n)
        
        pairs = self.broadphase.candidate_pairs(self.positions_array())
        
        for i, j in pairs:
            bi = self.bodies[i]
            bj = self.bodies[j]
            d = self.disp(bj.pos, bi.pos)
            cutoff = bi.body.size + bj.body.size + self.config.neighbor.skin
            if np.dot(d, d) > cutoff * cutoff:
                continue
            pi_all = bi.sphere_pos_world()
            pj_all = bj.sphere_pos_world()
            vi_all = bi.sphere_vel_world()
            vj_all = bj.sphere_vel_world()
            wi = bi.omega_world()
            wj = bj.omega_world()
            
            for si in range(bi.body.n):
                pi = pi_all[si]
                vi = vi_all[si]
                ri = float(bi.body.radii[si])
                for sj in range(bj.body.n):
                    pj = pj_all[sj]
                    vj = vj_all[sj]
                    rj = float(bj.body.radii[sj])

                    d_s = self.disp(pj, pi)
                    dist2 = float(np.dot(d_s, d_s))
                    rsum = ri + rj
                    if dist2 >= rsum * rsum:
                        continue
                    
                    pj_img = pi + d_s
                    
                    F_on_j, cp, tau_i_ex, tau_j_ex = self.contact.sphere_sphere(
                        i=i, si=si, pi=pi, vi=vi, ri=ri, omega_i_world=wi,
                        j=j, sj=sj, pj=pj_img, vj=vj, rj=rj, omega_j_world=wj,
                        dt = dt,
                    )

                    F[j] += F_on_j
                    F[i] -= F_on_j
                    
                    r_i = cp - bi.pos
                    bj_pos_img = bi.pos + self.disp(bj.pos, bi.pos)
                    r_j = cp - bj_pos_img
                    T[i] += np.cross(r_i, -F_on_j) + tau_i_ex
                    T[j] += np.cross(r_j, F_on_j) + tau_j_ex
            
        for i, b in enumerate(self.bodies):
            p_all = b.sphere_pos_world()
            v_all = b.sphere_vel_world()
            w = b.omega_world()
            for si in range(b.body.n):
                p = p_all[si]
                v = v_all[si]
                r = float(b.body.radii[si])
                for wall in self.config.box.walls:
                    Fw, cp, tau_ex = self.contact.sphere_wall(i, si, p, v, r, w, wall, dt)
                    if np.allclose(Fw, 0.0):
                        continue
                    F[i] += Fw
                    T[i] += np.cross(cp - b.pos, Fw) + tau_ex
        
        g = np.asarray(self.config.gravity.g, dtype=float)
        for i, b in enumerate(self.bodies):
            F[i] += b.body.mass * g
        
        self.contact.end_step()
        return F, T
    
    def total_kinetic(self) -> float:
        return float(sum(b.translational_ke() + b.rotational_ke() for b in self.bodies))

    def total_linear_momentu(self) -> np.ndarray:
        p = np.zeros(3, dtype = float)
        for b in self.bodies:
            p += b.linear_momentum()
        return p
    
    def total_angular_momentum(self) -> np.ndarray:
        L = np.zeros(3, dtype=float)
        for b in self.bodies:
            L += b.angular_momentum_world()
        return L