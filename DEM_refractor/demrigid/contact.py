from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Set
import numpy as np

from .config import ContactParams, BoxParams
from .math3d import safe_norm, clamp


@dataclass
class ContactState:
    u_t: np.ndarray
    n_prev: np.ndarray

class ContactCache:
    def __init__(self) -> None:
        self._pp: Dict[Tuple[int, int, int, int], ContactState] = {}
        self._pw: Dict[Tuple[int, int, str], ContactState] = {}
        
    def get_pp(self, key: Tuple[int, int, int, int]) -> ContactState | None:
        return self._pp.get(key)
    
    def set_pp(self, key: Tuple[int, int, int, int], state: ContactState) -> None:
        self._pp[key] = state
        
    def get_pw(self, key: Tuple[int, int, str]) -> ContactState | None:
        return self._pw.get(key)
    
    def set_pw(self, key: Tuple[int, int, str], state: ContactState) -> None:
        self._pw[key] = state
        
    def prune(self, active_pp: Set[Tuple[int, int, int, int]], active_pw: Set[Tuple[int, int, str]]) -> None:
        for k in list(self._pp.keys()):
            if k not in active_pp:
                del self._pp[k]
        for k in list(self._pw.keys()):
            if k not in active_pw:
                del self._pw[k]
    

def _tangential_component(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - np.dot(v, n) * n


class ContactSolver:
    def __init__(self, box: BoxParams, pp: ContactParams, pw: ContactParams) -> None:
        self.box = box
        self.pp = pp
        self.pw = pw
        self.cache = ContactCache()
        
        self._active_pp: Set[Tuple[int, int, int, int]] = set()
        self._active_pw: Set[Tuple[int, int, str]] = set()
        
        # 바디 단위로 "이번 스텝에 접촉이 있었는지" 플래그
        self.contact_flag: np.ndarray | None = None
    
    def begin_step(self, n_bodies: int) -> None:
        self._active_pp.clear()
        self._active_pw.clear()
        self.contact_flag = np.zeros(n_bodies, dtype=bool)

    def end_step(self) -> None:
        self.cache.prune(self._active_pp, self._active_pw)
    
    #=====================
    # sphere-sphere contact
    #=====================
    def sphere_sphere(
        self,
        i: int, si: int, pi: np.ndarray, vi: np.ndarray, ri: float, omega_i_world: np.ndarray,
        j: int, sj: int, pj: np.ndarray, vj: np.ndarray, rj: float, omega_j_world: np.ndarray,
        dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """두 구(sphere) 간 접촉력/추가토크(세계좌표)를 계산.
        
        반환:
        F_on_j, contact_point, tau_i_extra, tau_j_extra
        
        - F_on_j: j에 작용하는 힘
        - tau_*_extra: 롤링 저항 등 '힘의 작용점 토크'에 추가되는 회전저항 토크
        """
        
        if i < j:
            key = (i, si, j, sj)
            sign = +1.0
            p_i, p_j = pi, pj
            v_i, v_j = vi, vj
            a_i, a_j = ri, rj
            w_i, w_j = omega_i_world, omega_j_world
        else:
            key = (j, sj, i, si)
            sign = -1.0
            p_i, p_j = pj, pi
            v_i, v_j = vj, vi
            a_i, a_j = rj, ri
            w_i, w_j = omega_j_world, omega_i_world
        
        d = p_j - p_i
        dist = float(np.linalg.norm(d))
        
        if dist < 1e-12:
            prev = self.cache.get_pp(key)
            n = prev.n_prev if prev is not None else np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            n = d / dist
        
        delta = a_i + a_j - dist
        if delta <= 0:
            z = np.zeros(3, dtype=float)
            return z, p_j, z, z
        
        self._active_pp.add(key)
        
        r_ci =  ri * n
        r_cj = -rj * n

        v_ci = v_i + np.cross(w_i, r_ci)
        v_cj = v_j + np.cross(w_j, r_cj)

        v_rel = v_cj - v_ci
        v_n = float(np.dot(v_rel, n))
        v_t = v_rel - v_n * n
        
        pars = self.pp
        F_n = pars.k_n * delta - pars.c_n * v_n
        F_n = max(0.0, float(F_n))
        
        st = self.cache.get_pp(key)
        if st is None:
            st = ContactState(u_t=np.zeros(3, dtype=float), n_prev=n.copy())
        
        st.u_t = _tangential_component(st.u_t, n)
        st.u_t = st.u_t + v_t * dt
        st.u_t = _tangential_component(st.u_t, n)
        
        F_t = -pars.k_t * st.u_t - pars.c_t * v_t
        
        Ft_norm = float(np.linalg.norm(F_t))
        Ft_max = pars.mu * F_n
        
        if Ft_norm > Ft_max and Ft_norm > 0.0:
            # 슬립: 동마찰 방향으로 클램프
            F_t = -Ft_max * (v_t / safe_norm(v_t, pars.v_eps))
            st.u_t = -(F_t + pars.c_t * v_t) / max(pars.k_t, 1e-12)
        
        st.n_prev = n.copy()
        self.cache.set_pp(key, st)
        
        if dist > 1e-12:
            a = (dist * dist - a_j * a_j + a_i * a_i) / (2.0 * dist)
            a = clamp(a, 0.0, a_i)
            contact_point = p_i + a * n
        else:
            contact_point = p_i + a_i * n
        
        F = (F_n * n + F_t)

        tau_i_extra = np.zeros(3, dtype=float)
        tau_j_extra = np.zeros(3, dtype=float)
        if pars.mu_roll > 0.0 and F_n > 0.0:
            r_eff = (a_i * a_j) / max(a_i + a_j, 1e-12)
            w_rel = w_i - w_j
            w_rel_t = _tangential_component(w_rel, n)
            wrel_norm = float(np.linalg.norm(w_rel_t))
            if wrel_norm > 0.0:
                M_mag = pars.mu_roll * F_n * r_eff
                T_roll = -M_mag * (w_rel_t / wrel_norm) # i에 작용
                tau_i_extra += T_roll
                tau_j_extra -= T_roll
        
        if self.contact_flag is not None:
            self.contact_flag[i] = True
            self.contact_flag[j] = True
        
        return sign * F, contact_point, sign * tau_i_extra, sign * tau_j_extra
    
    #=====================
    # sphere-wall contact
    #=====================
    def sphere_wall(
        self,
        body_i: int,
        si: int,
        p: np.ndarray,
        v: np.ndarray,
        r: float,
        omega_world: np.ndarray,
        wall: str,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        key = (body_i, si, wall)
        Lx, Ly, Lz = self.box.Lx, self.box.Ly, self.box.Lz
        
        if wall == "west":
            dist = float(p[0])
            n = np.array([1.0, 0.0, 0.0], dtype=float)
        elif wall == "east":
            dist = float(Lx - p[0])
            n = np.array([-1.0, 0.0, 0.0], dtype=float)
        elif wall == "south":
            dist = float(p[1])
            n = np.array([0.0, 1.0, 0.0], dtype=float)
        elif wall == "north":
            dist = float(Ly - p[1])
            n = np.array([0.0, -1.0, 0.0], dtype=float)
        elif wall == "down":
            dist = float(p[2])
            n = np.array([0.0, 0.0, 1.0], dtype=float)
        elif wall == "up":
            dist = float(Lz - p[2])
            n = np.array([0.0, 0.0, -1.0], dtype=float)
        else:
            raise ValueError(f"Unknown wall {wall!r}")
        
        delta = r - dist
        if delta <= 0:
            z = np.zeros(3, dtype=float)
            return z, p, z
        
        self._active_pw.add(key)
        
        pars = self.pw
        
        r_c = -r * n
        v_contact = v + np.cross(omega_world, r_c)
        v_n = float(np.dot(v_contact, n))
        v_t = v_contact - v_n * n
        
        F_n = pars.k_n * delta - pars.c_n * v_n
        F_n = max(0.0, float(F_n))
        
        st = self.cache.get_pw(key)
        if st is None:
            st = ContactState(u_t = np.zeros(3, dtype=float), n_prev=n.copy())
        
        st.u_t = _tangential_component(st.u_t, n)
        st.u_t = st.u_t + v_t * dt
        st.u_t = _tangential_component(st.u_t, n)
        F_t = -pars.k_t * st.u_t - pars.c_t * v_t
        Ft_norm = float(np.linalg.norm(F_t))
        Ft_max = pars.mu * F_n
        if Ft_norm > Ft_max and Ft_norm > 0.0:
            F_t = -Ft_max * (v_t / safe_norm(v_t, pars.v_eps))
            st.u_t = -(F_t + pars.c_t * v_t) / max(pars.k_t, 1e-12)
            
        st.n_prev = n.copy()
        self.cache.set_pw(key, st)
        
        contact_point = p - n * (r - 0.5 * delta)
        
        tau_extra = np.zeros(3, dtype=float)
        if pars.mu_roll > 0.0 and F_n > 0.0:
            w_rel_t = _tangential_component(omega_world, n)
            wrel_norm = float(np.linalg.norm(w_rel_t))
            if wrel_norm > 0.0:
                M_mag = pars.mu_roll * F_n * r
                tau_extra += -M_mag * (w_rel_t / wrel_norm)
        
        if self.contact_flag is not None:
            self.contact_flag[body_i] = True
        
        return (F_n * n + F_t), contact_point, tau_extra