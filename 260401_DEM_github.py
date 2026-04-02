import datetime
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from time import time
import csv
from scipy.spatial.transform import Rotation as R
from typing import Any, Iterable
from collections import defaultdict

rng = np.random.default_rng()


SPRING_PARTICLE = 400.0
DAMPING_PARTICLE = 15.0
MU_PARTICLE = 0.3
SPRING_WALL = 10000.0
DAMPING_WALL = 15.0 
MU_WALL = 0.3 
ROLLING_DAMPING = 0.6

DIST_TOL = 1e-15

TIME_STEP = 1e-4
MAX_TIME  = 30.0
RECORD_STEP = 1000
LOG_STEP    = 1000
STABLE_TIME = 0.3

KINETIC_TOL = 1e-3
TMOM_TOL    = 1e-3
RMOM_TOL    = 1e-3

SLEEP_DUR_THRESHOLD = 30
SLEEP_SPD = 1e-4
SLEEP_ANG_SPD = 1e-4
WAKE_SPD = 5e-4
WAKE_ANG_SPD = 5e-4
WAKE_ACC = 1e-2
WAKE_ANG_ACC = 1e-2

WALL_X = 4.0
WALL_Y = 4.0
WALL_Z = 10000.0

GRAVITY_ACCEL = 9.8

BATCH_NUMBER = 3
MAX_BATCH_TRIALS = 1000

SKIN = 0.01


CSV_HEADER = ["body id", "particle id", "x", "y", "z", "r", "m"]



#------------------------------------------------------------------------
#region class

#region sphere
@dataclass
class Sphere:
    r:      float
    m:      float
    pos:    np.ndarray      # body-frame local position wrt aggregate CoM
#endregion sphere

#region aggregate
@dataclass
class Aggregate:
    spheres:    list[Sphere]
    n:          int             = field(init=False)
    mass:       float           = field(init=False) # total mass
    size:       float           = field(init=False) # size
    Ib:         np.ndarray      = field(init=False) # body frame moment of inertia
    Ib_inv:     np.ndarray      = field(init=False)

    
    def __post_init__(self):
        self.n      = len(self.spheres)
        self.mass   = np.sum(self.mass_array())
        self.set_pos_com()
        self.Ib     = self.body_inertia()
        self.Ib_inv = np.linalg.inv(self.Ib)
        self.size   = self.get_size()
    
    def mass_array(self):
        return np.array([q.m for q in self.spheres])
    
    def radii_array(self):
        return np.array([q.r for q in self.spheres])
    
    def position_array(self):
        return np.array([q.pos for q in self.spheres])
    
    def size_array(self):
        return np.array([np.linalg.norm(q.pos) + q.r for q in self.spheres])
    
    def set_pos_com(self):
        scom = np.average(self.position_array(), axis = 0, weights = self.mass_array())
        for q in self.spheres:
            q.pos -= scom
    
    def body_inertia(self):
        I = np.zeros((3, 3), dtype=float)
        E = np.eye(3)
        for q in self.spheres:
            s = q.pos
            I += 2.0 / 5.0 * q.m * (q.r ** 2) * E
            I += q.m * (np.dot(s, s)*E - np.outer(s, s))
        return I
    
    def get_size(self):
        return np.max(self.size_array())

@dataclass
class Sleepstate:
    is_sleep:     bool = False  # True (sleeping) False (awake)
    sleepy_dur:   int = 0       # duration for sleepy steps
    
    def sleep(self):
        self.is_sleep = True
    def wakeup(self):
        self.is_sleep = False
        self.sleepy_dur = 0
    def nodoff(self) -> bool:
        self.sleepy_dur += 1
        if self.sleepy_dur >= SLEEP_DUR_THRESHOLD:
            self.sleep()
            return True
        return False
#endregion aggregate

#region rigidbody
@dataclass
class Rigidbody:
    body:       Aggregate
    id:         int                     # id of rigid body
    pos:        np.ndarray              # world position of Center of Mass
    
    vel:        np.ndarray  = field(default_factory=lambda: np.zeros(3, dtype= float)) # world velocity of Center of Mass
    
    rot:        R           = field(default_factory=R.identity)
    w:          np.ndarray  = field(default_factory=lambda: np.zeros(3, dtype= float)) # angular velocity of rigid body (body frame)

    sleep_state:Sleepstate  = field(default_factory=Sleepstate)
    neighbors:  set[int]    = field(default_factory=set)
    ref_pos:    np.ndarray | None = None

    def world_pos(self, j: int):
        return self.pos + self.rot.apply(self.body.spheres[j].pos)
    
    def world_vel(self, j: int):
        return self.vel + np.cross(self.world_w(), self.rot.apply(self.body.spheres[j].pos))
    
    def world_w(self):
        return self.rot.apply(self.w)
    
    def add_pos(self, diff: np.ndarray):
        self.pos += diff
    
    def add_vel(self, diff: np.ndarray):
        self.vel += diff

    def add_angular_vel(self, diff: np.ndarray):
        self.w += diff
    
    def update_orientation(self, dt: float):
        self.rot *= R.from_rotvec(self.w * dt)
    
    def set_orientation(self, ini: R):
        self.rot = ini
    
    
    def greet_neighbor(self, id: int):
        self.neighbors.add(id)
    def expel_neighbor(self, id: int):
        self.neighbors.discard(id)
    def clear_neighbor(self):
        self.neighbors.clear()
    
    
    # KDK, translational
    def kick(self, force: np.ndarray, dt: float):
        self.vel += 0.5 * force / self.body.mass * dt
    def drift(self, dt: float):
        self.pos += self.vel * dt
    
    # KDK, rotational
    def rot_kick(self, torque: np.ndarray, dt: float):
        tau = self.rot.inv().apply(torque)
        dw  = self.body.Ib_inv @ (tau - np.cross(self.w, self.body.Ib @ self.w))
        self.w += 0.5 * dw * dt
    def rot_drift(self, dt: float):
        self.update_orientation(dt)
    
    
    def lullaby(self):
        if self.sleep_state.is_sleep:
            return
        if np.sum(self.vel ** 2) < (SLEEP_SPD ** 2) and np.sum(self.w ** 2) < (SLEEP_ANG_SPD ** 2):
            if self.sleep_state.nodoff():
                self.vel[:] = 0.0
                self.w[:] = 0.0
        else:
            self.sleep_state.sleepy_dur = 0
    def stimulate(self, force: np.ndarray | None, torque: np.ndarray | None):
        if not self.sleep_state.is_sleep:
            return
        wake = False
        # 속도 기반 wake
        if (np.sum(self.vel ** 2) > (WAKE_SPD ** 2) or 
            np.sum(self.w ** 2) > (WAKE_ANG_SPD ** 2)):
            wake = True
        
        #가속도 기반 wake
        if force is not None:
            a = force / self.body.mass
            if np.dot(a, a) >= WAKE_ACC ** 2:
                wake = True
        if torque is not None:
            alpha = self.body.Ib_inv @ self.rot.inv().apply(torque)
            if np.dot(alpha, alpha) >= WAKE_ANG_ACC ** 2:
                wake = True
        
        if wake:
            self.sleep_state.wakeup()
            
        
    
    
    
    def translation_energy(self): #world frame
        m = self.body.mass
        v = self.vel
        te = 0.5 * m * np.dot(v, v)
        return te
    
    def translation_momentum(self): #world frame
        return self.body.mass * self.vel
    
    def rotation_energy(self): 
        return 0.5 * np.dot(self.w, self.body.Ib @ self.w)
    
    def rotation_momentum(self):
        return self.rot.apply(self.body.Ib @ self.w)
    
    def total_energy(self):
        return self.translation_energy() + self.rotation_energy()
#endregion rigidbody

#region box

@dataclass
class Box:
    boxtype: str # "per" : periodic, "imp" : impermeable
    
    def __post_init__(self):
        if self.boxtype == "per":
            self.disp = self.disp_periodic
        elif self.boxtype == "imp":
            self.disp = self.disp_impermeable
        else:
            raise RuntimeError("Boxtype can only be per or imp.")
    
    def disp_periodic(self, pos_i: np.ndarray, pos_j: np.ndarray):
        diff = pos_i - pos_j
        dx, dy, dz = diff[0], diff[1], diff[2]
        
        dx -= WALL_X * np.round(dx / WALL_X)
        dy -= WALL_Y * np.round(dy / WALL_Y)

        return np.array([dx, dy, dz], dtype= float)

    def disp_impermeable(self, pos_i: np.ndarray, pos_j: np.ndarray):
        return (pos_i - pos_j)
    
    def disp_dist_wall(self, pos: np.ndarray, wall: str):
        # wall : north, west, south, east, up, down
        w = np.array([WALL_X, WALL_Y, WALL_Z])
        u = np.array([[1,0,0],[0,1,0],[0,0,1]])
        disp = np.zeros(3, dtype= float)
        dist = 0.0
        match wall:
            case "west":
                disp = pos * u[0]
                dist = disp[0]
            case "south":
                disp = pos * u[1]
                dist = disp[1]
            case "east":
                disp = (pos - w) * u[0]
                dist = disp[0]
            case "north":
                disp = (pos - w) * u[1]
                dist = disp[1]
            case "down":
                disp = pos * u[2]
                dist = disp[2]
            case "up":
                disp = (pos - w) * u[2]
                dist = disp[2]
            case _:
                raise RuntimeError("wall type can only be in [north, west, south, east, up, down]")
        return (disp, np.abs(dist))
#endregion box

#region world

@dataclass
class World(Box):
    bodies: list[Rigidbody]
    
    def __post_init__(self):
        super().__post_init__()
        self.n = len(self.bodies)
    
    def add_body(self, body: Rigidbody):
        self.bodies.append(body)
        self.n += 1
    
    def delete_body(self, j: int):
        self.bodies.pop(j)
        self.n -= 1
    
    def wrap_positions(self):
        if self.boxtype == "per":
            for b in self.bodies:
                b.pos[0] %= WALL_X
                b.pos[1] %= WALL_Y
        
    def is_overlap_bodies(self, i: int, j: int) -> bool:
        body_i = self.bodies[i]
        body_j = self.bodies[j]
        d_ij = self.disp(body_i.pos, body_j.pos)
        
        return (body_i.body.size + body_j.body.size) ** 2 > np.dot(d_ij, d_ij)
    
    def is_overlap_body_wall(self, i: int, wall: str) -> bool:
        body_i = self.bodies[i]
        disp, dist = self.disp_dist_wall(body_i.pos, wall)
        return dist < body_i.body.size
    
    def is_neighbor_bodies(self, i: int, j: int) -> bool:
        body_i = self.bodies[i]
        body_j = self.bodies[j]
        d_ij = self.disp(body_i.pos, body_j.pos)
        cutoff = body_i.body.size + body_j.body.size + SKIN
        
        return cutoff * cutoff > np.dot(d_ij, d_ij)
    
    def is_overlap_particles(self, i: int, x: int, j: int, y: int) -> bool:
        body_i = self.bodies[i]
        body_j = self.bodies[j]
        part_x = body_i.body.spheres[x]
        part_y = body_j.body.spheres[y]
        
        pos_x = body_i.world_pos(x)
        pos_y = body_j.world_pos(y)
        r_x = part_x.r
        r_y = part_y.r
        
        d_xy = self.disp(pos_x, pos_y)
        
        return (r_x + r_y) ** 2 > np.dot(d_xy, d_xy)
    
    def is_overlap_particle_wall(self, i: int, x: int, wall: str) -> bool:
        body_i = self.bodies[i]
        part_x = body_i.body.spheres[x]
        
        pos_x = body_i.world_pos(x)
        
        disp, dist = self.disp_dist_wall(pos_x, wall)
        return dist < part_x.r
    
    def needs_rebuild_verlet(self) -> bool:
        if self.n < 2:
            return False
        max1 = 0.0
        max2 = 0.0
        for body in self.bodies:
            if body.ref_pos is None:
                return True
            d = self.disp(body.pos, body.ref_pos)
            d2 = np.dot(d, d)
            if d2 >= max1:
                max2 = max1
                max1 = d2
            elif d2 > max2:
                max2 = d2
        return np.sqrt(max1) + np.sqrt(max2) >= SKIN
    
    def rebuild_verlet(self):
        for body in self.bodies:
            body.clear_neighbor()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.is_neighbor_bodies(i, j):
                    self.bodies[i].greet_neighbor(j)
                    self.bodies[j].greet_neighbor(i)
        for body in self.bodies:
            body.ref_pos = body.pos.copy()
    
    def update_verlet(self):
        if self.needs_rebuild_verlet():
            self.rebuild_verlet()
            
    
    def compute_contact_force_torque(self):
        F_tot = np.zeros((self.n, 3), dtype= float)
        T_tot = np.zeros((self.n, 3), dtype= float)
        
        for i in range(self.n):
            for j in self.bodies[i].neighbors:
                if j <= i:
                    continue
                
                bi = self.bodies[i]
                bj = self.bodies[j]
                
                if bi.sleep_state.is_sleep and bj.sleep_state.is_sleep:
                    continue
                
                f, tau_i, tau_j = self.compute_contact_body_force_torque(i, j)
                F_tot[i] -= f
                F_tot[j] += f
                T_tot[i] += tau_i
                T_tot[j] += tau_j
        return F_tot, T_tot
    
    def compute_contact_body_force_torque(self, i: int, j: int):
        f = np.zeros(3, dtype= float)       # force on i
        tau_i = np.zeros(3, dtype= float)   # torque on i
        tau_j = np.zeros(3, dtype= float)   # torque on j
        
        if self.is_overlap_bodies(i, j):
            for x in range(self.bodies[i].body.n):
                for y in range(self.bodies[j].body.n):
                    df, dti, dtj = self.compute_contact_particle_force_torque(i,x,j,y)
                    f += df
                    tau_i += dti
                    tau_j += dtj
        return f, tau_i, tau_j

    def compute_contact_particle_force_torque(self, i: int, x: int, j: int, y: int):
        F = np.zeros(3, dtype= float)       # force on j
        tau_i = np.zeros(3, dtype= float)   # torque on i
        tau_j = np.zeros(3, dtype= float)   # torque on j
        
        b, c = self.bodies[i], self.bodies[j]
        p, q = b.body.spheres[x], c.body.spheres[y]
        
        if self.is_overlap_particles(i, x, j, y):
            d_xy    = self.disp(c.world_pos(y), b.world_pos(x))
            dist_xy = np.linalg.norm(d_xy)
            if dist_xy <= DIST_TOL:
                return F, tau_i, tau_j
            n_xy    = d_xy / dist_xy
            v_xy    = c.world_vel(y) - b.world_vel(x)
            v_n     = np.dot(v_xy, n_xy)
            v_t     = v_xy - v_n * n_xy
            v_t2    = np.linalg.norm(v_t)
            
            ov_xy   = p.r + q.r - dist_xy
            
            F_mag   = SPRING_PARTICLE * ov_xy
            if v_n < 0.0:
                F_mag -= DAMPING_PARTICLE * v_n
            F_mag   = max(F_mag, 0.0)
            F += F_mag * n_xy
            if v_t2 > 0.0:
                F -= MU_PARTICLE * F_mag * v_t / v_t2
        
            xc = b.world_pos(x) + (p.r - 0.5 * ov_xy) * n_xy
            tau_i += np.cross((xc - b.pos), -F)
            tau_j += np.cross((xc - c.pos), F)
            
            r_eff = p.r * q.r / (p.r + q.r)
            w_rel = b.world_w() - c.world_w()
            w_rel_2 = np.linalg.norm(w_rel)
            if w_rel_2 > 0.0:
                T_roll = -ROLLING_DAMPING * F_mag * r_eff * w_rel / w_rel_2
                tau_i += T_roll
                tau_j -= T_roll
        return F, tau_i, tau_j
    
    def compute_wall_force_torque(self):
        F_tot = np.zeros((self.n, 3), dtype= float)
        T_tot = np.zeros((self.n, 3), dtype= float)
        
        for i in range(self.n):
            b = self.bodies[i]
            if b.sleep_state.is_sleep:
                continue
            f, tau = self.compute_wall_body_force_torque(i)
            F_tot[i] += f
            T_tot[i] += tau
        return F_tot, T_tot
    
    def compute_wall_body_force_torque(self, i: int):
        f = np.zeros(3, dtype= float)
        tau = np.zeros(3, dtype= float)
        walls = ["west", "south", "east", "north", "down", "up"]
        for wall in walls:
            if self.is_overlap_body_wall(i, wall):
                for x in range(self.bodies[i].body.n):
                    df, dtau = self.compute_wall_particle_force_torque(i, x, wall)
                    f += df
                    tau += dtau
        return f, tau
    
    def compute_wall_particle_force_torque(self, i: int, x: int, wall: str):
        f = np.zeros(3, dtype= float)
        tau = np.zeros(3, dtype= float)
        if self.is_overlap_particle_wall(i, x, wall):
            b = self.bodies[i]
            p = b.body.spheres[x]
            
            disp, dist = self.disp_dist_wall(b.world_pos(x), wall)
            ov = max(0.0, p.r - dist)
            
            v = b.world_vel(x)
            if dist <= DIST_TOL:
                return f, tau
            n = disp / dist
            vn = np.dot(v, n)
            vt = v - vn * n
            vt2 = np.linalg.norm(vt)
            
            f_mag = SPRING_WALL * ov
            if vn < 0.0:
                f_mag -= DAMPING_WALL * vn
            f_mag = max(f_mag, 0.0)
            f += f_mag * n
            if vt2 > 0.0:
                f -= MU_WALL * f_mag * vt / vt2
            
            xc = b.world_pos(x) + (p.r - 0.5 * ov) * n
            tau += np.cross((xc - b.pos), f)
            
            w_rel = b.world_w()
            w_rel_2 = np.linalg.norm(w_rel)
            
            if w_rel_2 > 0.0:
                T_roll = - ROLLING_DAMPING * f_mag * p.r * w_rel / w_rel_2
                tau += T_roll
        return f, tau

    def compute_gravity_force(self):
        F_tot = np.zeros((self.n, 3), dtype = float)
        
        for i in range(self.n):
            b = self.bodies[i]
            f = np.array([0, 0, 0])
            if not b.sleep_state.is_sleep:
                f[2] += -GRAVITY_ACCEL * b.body.mass
            F_tot[i] += f
        return F_tot
    
    def compute_total_force_torque(self):
        F_c, T_c    = self.compute_contact_force_torque()
        F_w, T_w    = self.compute_wall_force_torque()
        F_g         = self.compute_gravity_force()
        
        Force = F_c + F_w + F_g
        Torque = T_c + T_w
        
        return Force, Torque
    
    def total_trans_energy(self):
        te = 0.0
        for b in self.bodies:
            te += b.translation_energy()
        return te
    
    def total_rot_energy(self):
        re = 0.0
        for b in self.bodies:
            re += b.rotation_energy()
        return re
    
    def total_trans_mom(self):
        tmom = np.zeros(3, dtype = float)
        for b in self.bodies:
            tmom += b.translation_momentum()
        return tmom

    def total_rot_mom(self):
        rmom = np.zeros(3, dtype = float)
        for b in self.bodies:
            rmom += b.rotation_momentum()
        return rmom
#endregion world

#region simulator
@dataclass
class Simulator:
    world: World
    
    force: np.ndarray   = field(init= False)
    torque: np.ndarray  = field(init= False)
    
    def __post_init__(self):
        self.force = np.zeros((self.world.n, 3), dtype= float)
        self.torque = np.zeros((self.world.n, 3), dtype= float)
    
    def initialize(self):
        self.world.rebuild_verlet()
        self.force, self.torque = self.world.compute_total_force_torque()
    
    def integrator_kdk(self):
        dt = TIME_STEP
        
        #first kick
        for i, body in enumerate(self.world.bodies):
            if body.sleep_state.is_sleep:
                continue
            body.kick(self.force[i], dt)
            body.rot_kick(self.torque[i], dt)

        #drift
        for body in self.world.bodies:
            if body.sleep_state.is_sleep:
                continue
            body.drift(dt)
            body.rot_drift(dt)
        
        self.world.wrap_positions()
        self.world.update_verlet()
        
        force, torque = self.world.compute_total_force_torque()
        
        for i, body in enumerate(self.world.bodies):
            body.stimulate(force[i], torque[i])
        
        
        #second kick
        for i, body in enumerate(self.world.bodies):
            if body.sleep_state.is_sleep:
                continue
            body.kick(force[i], dt)
            body.rot_kick(torque[i], dt)
        
        self.force, self.torque = force, torque
        
        for body in self.world.bodies:
            body.lullaby()
    
    def is_stable(self):
        return (
            (self.world.total_trans_energy() + self.world.total_rot_energy() < KINETIC_TOL)
            and
            np.linalg.norm(self.world.total_trans_mom()) < TMOM_TOL
            and
            np.linalg.norm(self.world.total_rot_mom()) < RMOM_TOL
        )
    
    def simulation(self):
        ts              = []
        te_hist         = []
        re_hist         = []
        p_hist          = []
        l_hist          = []
        
        t               = 0.0
        stable_duration = 0.0
        
        step            = 0
        max_step        = int(np.ceil(MAX_TIME / TIME_STEP))
        
        start = time()
        
        while step < max_step:
            self.integrator_kdk()
            t += TIME_STEP
            step += 1
            
            te      = self.world.total_trans_energy()
            re      = self.world.total_rot_energy()
            tote    = te + re
            p       = self.world.total_trans_mom()
            l       = self.world.total_rot_mom()
            p_norm  = np.linalg.norm(p)
            l_norm  = np.linalg.norm(l)
            
            if step % RECORD_STEP == 0:
                ts.append(t)
                te_hist.append(te)
                re_hist.append(re)
                p_hist.append(p)
                l_hist.append(l)
                
            if self.is_stable():
                stable_duration += TIME_STEP
            else:
                stable_duration = 0.0
            if stable_duration >= STABLE_TIME:
                print(f"stabilized at t = {t:.4f}")
                break
            
            now = time()
            if step % LOG_STEP == 0:
                dur = int(now - start)
                print(
                    f"Step {step}/{max_step} : Time Elapsed={t:.6f} sec : "
                    f"Real Time Elapsed={datetime.timedelta(seconds = dur)}"
                )
                print(f"total K = {tote:.6f}, tl K = {te:.6f}, rot K = {re:.6f}")
                print(f"total p = {p_norm:.6f}, total l = {l_norm:.6f}, sleeping : {self.sleep_counter()}/{self.world.n}")
        end = time()
        
        if t >= MAX_TIME:
            print("WARNING : MAX_TIME reached before full stabilization.")
        print(f"simulation finished in {datetime.timedelta(seconds = end-start)}")
        
        return (
            self.world,
            np.array(ts),
            np.array(te_hist),
            np.array(re_hist),
            np.array(p_hist),
            np.array(l_hist)
        )
    
    def sleep_counter(self):
        cnt = 0
        for b in self.world.bodies:
            if b.sleep_state.is_sleep:
                cnt += 1
        return cnt
#endregion simulator

#region storage
class Storage:
    """read and write CSV files. Builder will have a Storage."""

    def save_world_csv(self, world: World, filename: str = "final_state.csv") -> None:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

            for body in world.bodies:
                for particle_id, p in enumerate(body.body.spheres):
                    pos = body.world_pos(particle_id)
                    writer.writerow([
                        body.id,
                        particle_id,
                        pos[0],
                        pos[1],
                        pos[2],
                        p.r,
                        p.m,
                    ])

    @staticmethod
    def _group_csv_rows(rows: Iterable[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
        grouped: dict[int, list[dict[str, str]]] = defaultdict(list)

        for row in rows:
            try:
                body_id = int(row["body id"])
            except KeyError as exc:
                raise ValueError(
                    "CSV header must contain: 'body id', 'particle id', 'x', 'y', 'z', 'r', 'm'"
                ) from exc

            grouped[body_id].append(row)

        return grouped

    @staticmethod
    def _make_body_from_group(body_id: int, items: list[dict[str, str]]) -> Rigidbody:
        # particle id 순서 복원
        items = sorted(items, key=lambda row: int(row["particle id"]))

        world_positions: list[np.ndarray] = []
        masses: list[float] = []
        spheres: list[Sphere] = []

        for row in items:
            pos = np.array(
                [
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                ],
                dtype=float,
            )
            r = float(row["r"])
            m = float(row["m"])

            world_positions.append(pos)
            masses.append(m)

            # Aggregate가 __post_init__에서 CoM 기준 local pos로 바꿔줌
            spheres.append(
                Sphere(
                    r=r,
                    m=m,
                    pos=pos.copy(),
                )
            )

        com = np.average(
            np.array(world_positions, dtype=float),
            axis=0,
            weights=np.array(masses, dtype=float),
        )

        agg = Aggregate(spheres=spheres)

        body = Rigidbody(
            body=agg,
            id=body_id,
            pos=np.array(com, dtype=float),
            vel=np.zeros(3, dtype=float),
            rot=R.identity(),
            w=np.zeros(3, dtype=float),
        )
        return body

    def load_world_csv(self, filename: str, boxtype: str) -> World:
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames != CSV_HEADER:
                raise ValueError(
                    f"Unexpected CSV header: {reader.fieldnames}. Expected {CSV_HEADER}."
                )

            grouped = self._group_csv_rows(reader)

        bodies: list[Rigidbody] = []

        for body_id, items in sorted(grouped.items(), key=lambda x: x[0]):
            body = self._make_body_from_group(body_id, items)
            bodies.append(body)

        return World(boxtype=boxtype, bodies=bodies)
#endregion storage

#region builder
@dataclass
class Builder:
    storage: Storage
    
    def make_sphere(self, pos: np.ndarray, r: float, m: float) -> Sphere:
        return Sphere(
            r = r, m = m, pos = pos
        )
    
    def make_aggregate(self, spheres: list[Sphere]) -> Aggregate:
        return Aggregate(
            spheres
        )
    
    def make_rigid_body(self, aggregate: Aggregate, id: int, pos: np.ndarray) -> Rigidbody:
        return Rigidbody(
            body= aggregate, id= id, pos= pos
        )
    
    def random_rotation(self) -> R:
        return R.random(random_state= rng)
    
    def random_position(self, xmin, ymin, zmin, xmax, ymax, zmax) -> np.ndarray:
        low = np.array([xmin, ymin, zmin])
        high = np.array([xmax, ymax, zmax])
        return rng.uniform(low = low, high = high, size = 3)
    
    def make_init_world(self, bodies: list[Rigidbody], boxtype: str) -> World:
        world = World(boxtype=boxtype, bodies=[])
        indicator = 0.0
        queue = bodies.copy()

        # 한 층 BATCH_NUMBER 개씩 배치
        while queue:
            temp, queue = queue[:BATCH_NUMBER], queue[BATCH_NUMBER:]
            sz = np.max([b.body.size for b in temp])

            # 층 높이 범위
            zmin = indicator
            zmax = indicator + sz * 2.2

            while temp:
                proto = temp[0]
                trials = 0

                while True:
                    trials += 1
                    if trials > MAX_BATCH_TRIALS:
                        raise RuntimeError("Failed to initialize batch placement")

                    margin = proto.body.size

                    # boxtype에 따라 spawn 범위 설정
                    if boxtype == "per":
                        # periodic이면 x, y는 박스 전체 사용 가능
                        xmin = 0.0
                        xmax = WALL_X
                        ymin = 0.0
                        ymax = WALL_Y
                    else:
                        # impermeable이면 벽 overlap 피하려고 margin 확보
                        xmin = margin
                        xmax = WALL_X - margin
                        ymin = margin
                        ymax = WALL_Y - margin

                    # z도 아래 벽과 초기 overlap 피하도록 margin 확보
                    zlow = zmin + margin
                    zhigh = zmax - margin

                    if xmax <= xmin or ymax <= ymin or zhigh <= zlow:
                        raise RuntimeError("Spawn region too small for current body size")

                    pos = self.random_position(xmin, ymin, zlow, xmax, ymax, zhigh)

                    cand = Rigidbody(
                        body=proto.body,
                        id=proto.id,
                        pos=pos.copy(),
                        vel=proto.vel.copy(),
                        rot=self.random_rotation(),
                        w=proto.w.copy(),
                    )

                    world.add_body(cand)
                    new_idx = world.n - 1

                    is_overlap = False
                    for i in range(new_idx):
                        if world.is_overlap_bodies(i, new_idx):
                            is_overlap = True
                            break

                    if is_overlap:
                        world.delete_body(new_idx)
                        continue

                    temp.pop(0)
                    break

            indicator += sz * 2.2

        return world
#endregion builder

#region distributioner
@dataclass
class Distributioner:
    @staticmethod
    def lognormal_array(mu: float, sigma: float, n: int = 1):
        rad = rng.lognormal(mu, sigma, size = n)
        return rad[0] if n == 1 else rad
    
    @staticmethod
    def uniform_array(a: float, b: float, n: int = 1):
        rad = rng.uniform(a, b, size = n)
        return rad[0] if n == 1 else rad
    
    @staticmethod
    def get_random_dist(dist_type: str | int, *args):
        match dist_type:
            case "logn" | "lognormal" | 0:
                return Distributioner.lognormal_array(*args)
            case "unif" | "uniform" | 1:
                return Distributioner.uniform_array(*args)
            case _:
                return RuntimeError(f"Unsupported distribution: {dist_type!r}")
#endregion distributioner


#region main
def plot_history(
    ts: np.ndarray,
    te_hist: np.ndarray,
    re_hist: np.ndarray,
    p_hist: np.ndarray,
    l_hist: np.ndarray,
):
    if len(ts) == 0:
        raise ValueError("No recorded history to plot. Increase RECORD_STEP or run longer.")

    te_hist = np.asarray(te_hist, dtype=float)
    re_hist = np.asarray(re_hist, dtype=float)
    p_hist = np.asarray(p_hist, dtype=float)
    l_hist = np.asarray(l_hist, dtype=float)

    if p_hist.ndim != 2 or p_hist.shape[1] != 3:
        raise ValueError(f"p_hist must have shape (N, 3), got {p_hist.shape}")
    if l_hist.ndim != 2 or l_hist.shape[1] != 3:
        raise ValueError(f"l_hist must have shape (N, 3), got {l_hist.shape}")

    tote_hist = te_hist + re_hist
    p_norm = np.linalg.norm(p_hist, axis=1)
    l_norm = np.linalg.norm(l_hist, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1) 에너지
    ax = axes[0]
    ax.plot(ts, te_hist, label="Translational KE")
    ax.plot(ts, re_hist, label="Rotational KE")
    ax.plot(ts, tote_hist, label="Total KE")
    ax.set_ylabel("Energy")
    ax.set_title("Energy History")
    ax.grid(True)
    ax.legend()

    # 2) 선운동량
    ax = axes[1]
    ax.plot(ts, p_hist[:, 0], label="px")
    ax.plot(ts, p_hist[:, 1], label="py")
    ax.plot(ts, p_hist[:, 2], label="pz")
    ax.plot(ts, p_norm, label="|p|")
    ax.set_ylabel("Linear Momentum")
    ax.set_title("Linear Momentum History")
    ax.grid(True)
    ax.legend()

    # 3) 각운동량
    ax = axes[2]
    ax.plot(ts, l_hist[:, 0], label="lx")
    ax.plot(ts, l_hist[:, 1], label="ly")
    ax.plot(ts, l_hist[:, 2], label="lz")
    ax.plot(ts, l_norm, label="|l|")
    ax.set_xlabel("Time")
    ax.set_ylabel("Angular Momentum")
    ax.set_title("Angular Momentum History")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    plt.show()

    return fig, axes

if __name__ == "__main__":
    storage = Storage()
    builder = Builder(storage=storage)

    n_balls = 10
    radius = 0.5
    mass = 1.0

    sphere = builder.make_sphere(
        pos=np.zeros(3, dtype=float),
        r=radius,
        m=mass,
    )
    aggregate = builder.make_aggregate([sphere])
    
    bodies = [
        builder.make_rigid_body(
            aggregate=aggregate,
            id=i,
            pos=np.zeros(3, dtype=float),
        )
        for i in range(n_balls)
    ]

    tic = time.time()
    # 초기 랜덤 배치
    world = builder.make_init_world(
        bodies=bodies,
        boxtype="imp",
    )
    toc = time.time()
    print(f"initialized in {toc-tic:.3f} sec")

    z_offset = 5.0
    for body in world.bodies:
        body.add_pos(np.array([0.0, 0.0, z_offset], dtype=float))

    sim = Simulator(world=world)
    sim.initialize()

    world_final, ts, te_hist, re_hist, p_hist, l_hist = sim.simulation()

    storage.save_world_csv(world_final, "final_state.csv")
    plot_history(ts, te_hist, re_hist, p_hist, l_hist)

#endregion main