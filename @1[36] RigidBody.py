import datetime
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from time import time
import csv
from scipy.spatial.transform import Rotation as R
from typing import Any

rng = np.random.default_rng()


#-----------------------------------------------------------
#region Parameter
# Simulation Parameters
CONTACT_SPRING_CONST    = 400.0
CONTACT_DAMPING         = 15.0
FLOOR_SPRING_CONST      = 10000.0
FLOOR_DAMPING           = 15.0
WALL_SPRING_CONST       = 2000.0
WALL_DAMPING            = 15.0
GRAVITY_ACCEL           = 9.81
FRICTION_COEFF          = 0.3
WALL_FRICTION           = 0.3
FLOOR_FRICTION          = 0.3
OVERLAP_COEFF           = 0.4
ROLLING_DAMPING         = 0.3

TIME_STEP               = 2e-5
MAX_TIME                = 5.0

# Simulation Box
WALL_LENGTH_X           = 4.00
WALL_LENGTH_Y           = 4.00

# Initializating Type

# Particle Number
N                       = 10

# Stabilized Condition
STABLE_TIME             = 0.50
KINETIC_TOL             = 1e-3
MOM_TL_TOL              = 1e-3
MOM_ROT_TOL             = 1e-3

VMAX_TOL                = 1e-3
Z_TOL                   = 1e-4
Z_WINDOW                = 200

# Can't divide by 0
DIST_TOL                = 1e-15
#endregion Parameter

#-----------------------------------------------------------
#region Classes
# Particle

@dataclass
class Particle:
    pos:    np.ndarray  # relative position vector
    r:      float
    m:      float

#-----------------------------------------------------------
#

#-----------------------------------------------------------
# Particles Utility Functions

def position_array(particles: list[Particle]):
    return np.array([q.pos for q in particles], dtype = float)

def radii_array(particles: list[Particle]):
    return np.array([q.r for q in particles], dtype = float)

def mass_array(particles: list[Particle]):
    return np.array([q.m for q in particles], dtype = float)

def rel_pos(p: Particle, q: Particle):
    return q.pos - p.pos

#-----------------------------------------------------------
# Rigid Body

@dataclass
class RigidBody:
    id:     int         # id of rigid body
    particles:  list[Particle]  # list of particles
    pos:    np.ndarray  # initial position of COM
    vel:    np.ndarray  # initial velocity of COM

    rot:    R           = field(default_factory=R.identity)  # initial rotation of rigid body
    w:      np.ndarray  = field(default_factory=lambda : np.zeros(3))
    mass:   float       = field(init=False)
    Ib:     np.ndarray  = field(init=False)
    Ib_inv: np.ndarray  = field(init=False)
    size:   float       = field(init=False)

    def __post_init__(self):
        self.mass = np.sum(mass_array(self.particles))
        self.Ib   = self.inertia_body()
        self.Ib_inv = np.linalg.inv(self.Ib)
        self.size = self.get_size()

    def get_size(self):
        return max(np.linalg.norm(q.pos) + q.r for q in self.particles)

    def inertia_body(self):
        I = np.zeros((3, 3), dtype = float)
        E = np.eye(3)
        for q in self.particles:
            s = q.pos
            I += 2.0 / 5.0 * q.m * (q.r ** 2) * E
            I += q.m*(np.dot(s, s)*E - np.outer(s, s))
        return I

    def world_pos(self, q: Particle):
        return self.pos + self.rot.apply(q.pos)

    def world_vel(self, q: Particle):
        world_w = self.rot.apply(self.w)
        world_s = self.rot.apply(q.pos)
        return self.vel + np.cross(world_w, world_s)
    
    def world_w(self):
        return self.rot.apply(self.w)

    def update_orientation(self, dt: float):
        dr = R.from_rotvec(self.w * dt)
        self.rot = self.rot * dr

#-----------------------------------------------------------
# Bodies Utility Functions
def world_pos_array(bodies: list[RigidBody]):
    return np.array([[body.world_pos(p) for p in body.particles] for body in bodies])

def body_vel_array(bodies: list[RigidBody]):
    return np.array([b.vel for b in bodies])

def body_mass_array(bodies: list[RigidBody]):
    return np.array([b.mass for b in bodies])

#-----------------------------------------------------------
# Radius Distribution
def get_lognormal_array(mu: float, sigma: float, n: int = 1):
    rad = rng.lognormal(mu, sigma, size = n)
    return rad[0] if n == 1 else rad

def get_normal_array(mu: float, sigma: float, n: int = 1):
    rad = rng.normal(mu, sigma, size = n)
    return rad[0] if n == 1 else rad

def get_uniform_array(a: float, b: float, n: int = 1):
    rad = rng.uniform(a, b, size = n)
    return rad[0] if n == 1 else rad

def get_random_dist(type: str | int, *args):
    match type:
        case "logn" | 0:
            return get_lognormal_array(*args)
        case "norm" | 1:
            return get_normal_array(*args)
        case "unif" | 2:
            return get_uniform_array(*args)
        case _:
            raise RuntimeError("no matched distribution!")
#endregion Classes

#-----------------------------------------------------------
#region Contact
# Contact Check
def contact_bodies(bodies: list[RigidBody]):
    n = len(bodies)
    ov_pairs = []

    for i in range(n):
        for j in range(i+1, n):
            d_ij = np.linalg.norm(bodies[i].pos - bodies[j].pos)
            ov_ij = bodies[i].size + bodies[j].size - d_ij
            if ov_ij > 0:
                ov_pairs.append((i,j))
    return ov_pairs

def contact_particles(bodies: list[RigidBody])->list[dict[str, Any]]:
    ov_bodies = contact_bodies(bodies)
    ov_pairs = []

    world_positions = world_pos_array(bodies)

    for (i, j) in ov_bodies:
        for x, p in enumerate(bodies[i].particles):
            ppos = world_positions[i][x]
            rp   = p.r
            for y, q in enumerate(bodies[j].particles):
                qpos = world_positions[j][y]
                rq   = q.r

                dvec = qpos - ppos
                dxy  = np.linalg.norm(dvec)
                if dxy <= DIST_TOL:
                    dvec = bodies[j].pos - bodies[i].pos
                    dnorm = np.linalg.norm(dvec)
                    if dnorm <= DIST_TOL:
                        dvec = np.array([1.0, 0.0, 0.0])
                        dnorm = 1.0
                    nij = dvec / dnorm
                    dxy = DIST_TOL
                else:
                    nij = dvec / dxy

                ov_xy= rp+rq-dxy
                if ov_xy > 0:
                    ov_pairs.append(
                        {
                            "body_i": i,
                            "part_i": x,
                            "pos_i" : ppos,
                            "body_j": j,
                            "part_j": y,
                            "pos_j" : qpos,
                            "ov": ov_xy,
                            "nij": nij
                        }
                    )
    return ov_pairs

def floor_contact_particles(bodies: list[RigidBody])->list[dict[str, Any]]:
    ov_particles = []
    world_positions = world_pos_array(bodies)

    for i, b in enumerate(bodies):
        for x, p in enumerate(b.particles):
            pz = world_positions[i][x][2]
            pr = p.r
            ov = pr - pz
            if ov > 0:
                ov_particles.append({
                    "body" : i,
                    "part" : x,
                    "ov"   : ov
                })
    return ov_particles

def wall_contact_particles(bodies: list[RigidBody]) -> list[dict[str, int | float]]:
    ov_particles = []
    world_positions = world_pos_array(bodies)
    
    for i, b in enumerate(bodies):
        for x, p in enumerate(b.particles):
            ov_dict = {}
            ppos = world_positions[i][x]
            pr = p.r
            
            px, py = ppos[0], ppos[1]
            ov_W = pr - px
            if ov_W > 0:
                ov_dict["ov_W"] = ov_W
            ov_S = pr - py
            if ov_S > 0:
                ov_dict["ov_S"] = ov_S
            ov_E = px + pr - WALL_LENGTH_X
            if ov_E > 0:
                ov_dict["ov_E"] = ov_E
            ov_N = py + pr - WALL_LENGTH_Y
            if ov_N > 0:
                ov_dict["ov_N"] = ov_N
            if ov_dict:
                ov_dict["body"] = i
                ov_dict["part"] = x
                ov_dict["pos"] = ppos
                ov_particles.append(ov_dict)
    return ov_particles
#endregion Contact

#-----------------------------------------------------------
#region Force
# Forces
def contact_force_torque(bodies: list[RigidBody]):
    n = len(bodies)
    ov_pairs = contact_particles(bodies)

    Force = np.zeros((n, 3), dtype = float)
    Torque = np.zeros((n, 3), dtype = float)

    for targ in ov_pairs:
        (i, j) = (targ["body_i"], targ["body_j"])
        (x, y) = (targ["part_i"], targ["part_j"])
        (ppos, qpos) = (targ["pos_i"], targ["pos_j"])
        ov = targ["ov"]

        b, d = bodies[i], bodies[j]
        p, q = b.particles[x], d.particles[y]
        rp, rq = p.r, q.r

        nij = targ["nij"]
        vij = d.world_vel(q) - b.world_vel(p)
        vn  = np.dot(vij, nij)
        vt  = vij - vn*nij

        F_n = CONTACT_SPRING_CONST * ov * nij - CONTACT_DAMPING * vn * nij
        if np.dot(F_n, nij) < 0:
            F_n = np.zeros(3)
        F_t = np.zeros(3)
        vt_norm = np.linalg.norm(vt)
        if vt_norm > DIST_TOL:
            fn = np.linalg.norm(F_n)
            F_t = - FRICTION_COEFF * fn * vt / vt_norm

        F = F_n + F_t
        Force[i] -= F
        Force[j] += F

        xc = 0.5 * ((ppos + p.r*nij) + (qpos - q.r*nij))
        Torque[i] += np.cross((xc - b.pos), -F)
        Torque[j] += np.cross((xc - d.pos), F)

        Reff = rp * rq / (rp + rq)
        wrel = b.world_w() - d.world_w()
        wrel_norm = np.linalg.norm(wrel)
        if wrel_norm > 0:
            Trol = -ROLLING_DAMPING * np.linalg.norm(F_n) * Reff * wrel / wrel_norm
            Torque[i] += Trol
            Torque[j] -= Trol
    return Force, Torque

def floor_force_torque(bodies: list[RigidBody]):
    n = len(bodies)
    ov_particles = floor_contact_particles(bodies)

    Force = np.zeros((n, 3), dtype = float)
    Torque = np.zeros((n, 3), dtype = float)

    for targ in ov_particles:
        (i, x) = targ["body"], targ["part"]
        ov = targ["ov"]

        b = bodies[i]
        p = b.particles[x]
        ppos = b.world_pos(p)
        ni = np.array([0, 0, 1], dtype = float)
        vi = b.world_vel(p)
        vn = np.dot(vi, ni)
        vt = vi - vn*ni
        
        F_n = FLOOR_SPRING_CONST * ov * ni - FLOOR_DAMPING * vn * ni
        if np.dot(F_n, ni) < 0:
            F_n = np.zeros(3)
        F_t = np.zeros(3)
        vt_norm = np.linalg.norm(vt)
        if vt_norm > DIST_TOL:
            fn = np.linalg.norm(F_n)
            F_t = -FLOOR_FRICTION * fn * vt / vt_norm
        
        F = F_n + F_t
        Force[i] += F
        xc = ppos - p.r * ni
        Torque[i] += np.cross((xc - bodies[i].pos), F)

        wrel = b.world_w()
        wrel_norm = np.linalg.norm(wrel)
        if wrel_norm > 0:
            Trol = - ROLLING_DAMPING * np.linalg.norm(F_n) * p.r * wrel / wrel_norm
            Torque[i] += Trol
    return Force, Torque

def wall_force_torque(bodies: list[RigidBody]):
    n = len(bodies)
    ov_particles = wall_contact_particles(bodies)
    
    searches = ["ov_W", "ov_S", "ov_E", "ov_N"]
    normals = [np.array([1, 0, 0], dtype = float), np.array([0, 1, 0], dtype = float),
               np.array([-1, 0, 0], dtype = float), np.array([0, -1, 0], dtype = float)]
    
    Force = np.zeros((n, 3), dtype = float)
    Torque = np.zeros((n, 3), dtype = float)
    
    for targ in ov_particles:
        (i, x) = targ["body"], targ["part"]
        b = bodies[i]
        p = b.particles[x]
        ppos = targ["pos"]
        vi = b.world_vel(p)
        wrel = b.world_w()
        wrel_norm = np.linalg.norm(wrel)

        for j in range(4):
            search = searches[j]
            if search in targ:
                nvec = normals[j]
                ov = targ[search]
                vn = np.dot(vi, nvec)
                vt = vi - vn * nvec
                
                F_n = WALL_SPRING_CONST * ov * nvec - WALL_DAMPING * vn * nvec
                if np.dot(F_n, nvec) < 0:
                    F_n = np.zeros(3)
                F_t = np.zeros(3)
                vt_norm = np.linalg.norm(vt)
                if vt_norm > DIST_TOL:
                    fn = np.linalg.norm(F_n)
                    F_t = - WALL_FRICTION * fn * vt / vt_norm
                F = F_n + F_t
                Force[i] += F

                xc = ppos - p.r * nvec
                Torque[i] += np.cross(xc - b.pos, F)
                if wrel_norm > 0:
                    Trol = - ROLLING_DAMPING * np.linalg.norm(F_n) * p.r * wrel/wrel_norm
                    Torque[i] += Trol
    return Force, Torque        
    
def gravity_force(bodies: list[RigidBody]):
    n = len(bodies)
    Force = np.zeros((n, 3), dtype = float)

    for i, b in enumerate(bodies):
        m = b.mass
        Force[i] += np.array([0, 0, -GRAVITY_ACCEL * m], dtype = float)
    return Force

def total_force_torque(bodies: list[RigidBody]):
    Fc, Tc = contact_force_torque(bodies)
    Ff, Tf = floor_force_torque(bodies)
    Fw, Tw = wall_force_torque(bodies)
    Fg = gravity_force(bodies)
    
    Force = Fc + Ff + Fw + Fg
    Torque = Tc + Tf + Tw
    
    return Force, Torque
#endregion Force

#-----------------------------------------------------------
#region Initialize
# Initialization
def random_bodies_states(bodies: list[RigidBody]):
    outputs = []
    current_top = 0.0

    for b in bodies:
        sz = b.size

        xpos = rng.uniform(sz, WALL_LENGTH_X - sz)
        ypos = rng.uniform(sz, WALL_LENGTH_Y - sz)
        zpos = current_top + sz

        b.pos = np.array([xpos, ypos, zpos], dtype=float)
        b.rot = R.random()

        current_top = zpos + sz
        outputs.append(b)

    return outputs
#endregion Initialize

#-----------------------------------------------------------
#region Physical States
def trans_kinetic_energy(bodies: list[RigidBody]):
    ke = 0.0
    for b in bodies:
        m = b.mass
        v = b.vel
        ke += 0.5 * m * np.dot(v, v)
    return ke

def rot_kinetic_energy(bodies : list[RigidBody]):
    ke = 0.0
    for b in bodies:
        wb = b.w
        Ib = b.Ib
        ke += 0.5 * np.dot(wb, Ib @ wb)
    return ke

def tot_kinetic_energy(bodies: list[RigidBody]):
    trans_ke    = trans_kinetic_energy(bodies)
    rot_ke      = rot_kinetic_energy(bodies)
    return trans_ke + rot_ke

def max_speed(bodies: list[RigidBody]):
    vels = body_vel_array(bodies)
    return np.max(np.linalg.norm(vels, axis = 1))

def box_trans_momentum(bodies: list[RigidBody]):
    vels = body_vel_array(bodies)
    mass = body_mass_array(bodies)
    return np.sum(mass[:, None] * vels, axis = 0)

def box_rot_momentum(bodies: list[RigidBody]):
    L_tot = np.zeros(3, dtype = float)
    for b in bodies:
        L_orbital = np.cross(b.pos, b.mass * b.vel)
        L_spin = b.rot.apply(b.Ib @ b.w)
        L_tot += L_orbital + L_spin
    return L_tot

#-----------------------------------------------------------
#region Visualization
# Plotter
def plot_results(ts, tke_hist, rke_hist, totke_hist, px_hist, py_hist, pz_hist, lx_hist, ly_hist, lz_hist):
    fig = plt.figure(figsize = (16, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(ts, tke_hist, label = "Trans KE")
    ax1.plot(ts, rke_hist, label = "Rotat KE")
    ax1.plot(ts, totke_hist, label = "Total KE")
    ax1.set_title("Kinetic Energies")
    ax1.set_xlabel("time")
    ax1.set_ylabel("KE")
    ax1.grid(True)
    ax1.legend()
        
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(ts, px_hist, label = "px")
    ax2.plot(ts, py_hist, label = "py")
    ax2.plot(ts, pz_hist, label = "pz")
    ax2.plot(ts, lx_hist, label = "lx")
    ax2.plot(ts, ly_hist, label = "ly")
    ax2.plot(ts, lz_hist, label = "lz")
    ax2.set_title("Momentums")
    ax2.set_xlabel("time")
    ax2.set_ylabel("momentum")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()



#endregion Visualization

#-----------------------------------------------------------
#region Simulation
# Integrator
def leapfrog(bodies: list[RigidBody]):
    Force, Torque = total_force_torque(bodies)
    for i, b in enumerate(bodies):
        b.vel   += 0.5 * Force[i] / b.mass * TIME_STEP
        b.pos   += b.vel * TIME_STEP
        tau     = b.rot.inv().apply(Torque[i])
        dw      = b.Ib_inv @ (tau - np.cross(b.w, b.Ib @ b.w))
        b.w     += 0.5 * dw * TIME_STEP
        b.update_orientation(TIME_STEP)
        
    Force, Torque = total_force_torque(bodies)
    for i, b in enumerate(bodies):
        b.vel   += 0.5 * Force[i] / b.mass * TIME_STEP
        tau     = b.rot.inv().apply(Torque[i])
        dw      = b.Ib_inv @ (tau - np.cross(b.w, b.Ib @ b.w))
        b.w     += 0.5 * dw * TIME_STEP

def simulate(bodies: list[RigidBody]):
    ts              = []
    tke_hist        = []
    rke_hist        = []
    totke_hist      = []
    px_hist, py_hist, pz_hist = [], [], []
    lx_hist, ly_hist, lz_hist = [], [], []
    
    t               = 0.0
    stable_duration = 0.0
    
    start           = time()
    
    step            = 0
    max_step        = MAX_TIME // TIME_STEP + 1.0
    
    while t < MAX_TIME:
        leapfrog(bodies)
        t += TIME_STEP
        step += 1
        
        tke         = trans_kinetic_energy(bodies)
        rke         = rot_kinetic_energy(bodies)
        totke       = tke + rke
        p           = box_trans_momentum(bodies)
        p_norm      = np.linalg.norm(p)
        px, py, pz  = p[0], p[1], p[2]
        l           = box_rot_momentum(bodies)
        l_norm      = np.linalg.norm(l)
        lx, ly, lz  = l[0], l[1], l[2]
        
        ts.append(t)
        tke_hist.append(tke)
        rke_hist.append(rke)
        totke_hist.append(totke)
        px_hist.append(px), py_hist.append(py), pz_hist.append(pz)
        lx_hist.append(lx), ly_hist.append(ly), lz_hist.append(lz)
        
        if (totke < KINETIC_TOL) and (p_norm < MOM_TL_TOL) and (l_norm < MOM_ROT_TOL):
            stable_duration += TIME_STEP
        else:
            stable_duration = 0.0
        
        if stable_duration >= STABLE_TIME:
            print(f"stabilized at t = {t:.4f}")
            print(f"K = {totke:.6e}")
            break
        
        flag        = time()
        if step % 1000 == 0:
            dur = int(flag-start)
            print(f"Step {step}/{max_step} : Simulation Time={t:.6f}sec : Real Time Elapsed={datetime.timedelta(seconds=dur)}sec")
            print(f"total K = {totke:.6f}, tl K = {tke:.6f}, rot K = {rke:.6f}")
            print(f"total p = {p_norm:.6f}, total l = {l_norm:.6f}")
            if stable_duration > 0.0:
                print(f"Stabilzed...{stable_duration}/{STABLE_TIME}sec")
        if step % 10000 == 0:
            save_bodies_csv(bodies, filename = f"bodies_step_{step}.csv")
            print(f"Saved bodies at step {step} to bodies_step_{step}.csv")
    end = time()
    
    if t >= MAX_TIME:
        print("WARNING : max_time reached before full stabilization.")
    print(f"simulation finished in {end-start:.3f}sec")
    
    return (
        bodies,
        np.array(ts),
        np.array(tke_hist),
        np.array(rke_hist),
        np.array(totke_hist),
        np.array(px_hist), np.array(py_hist), np.array(pz_hist),
        np.array(lx_hist), np.array(ly_hist), np.array(lz_hist)
    )
        
#endregion Simulation

#-----------------------------------------------------------
#region csv
def save_bodies_csv(bodies: list[RigidBody], filename = "final_bodies.csv"):
    with open(filename, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["body id", "particle id", "x", "y", "z", "r", "m"])
        
        for i, b in enumerate(bodies):
            for x, p in enumerate(b.particles):
                pos = b.world_pos(p)
                writer.writerow([
                    i, x, pos[0], pos[1], pos[2], p.r, p.m
                ])
#endregion csv


def make_body_from_specs(
    body_id: int,
    specs: list[tuple[tuple[float, float, float], float, float]],
    pos: np.ndarray | None = None,
    vel: np.ndarray | None = None,
) -> RigidBody:
    if pos is None:
        pos = np.zeros(3, dtype=float)
    else:
        pos = np.asarray(pos, dtype=float)

    if vel is None:
        vel = np.zeros(3, dtype=float)
    else:
        vel = np.asarray(vel, dtype=float)

    particles = [
        Particle(pos=np.array(xyz, dtype=float), r=float(rad), m=float(mass))
        for xyz, rad, mass in specs
    ]

    return RigidBody(
        id=body_id,
        particles=particles,
        pos=pos,
        vel=vel,
    )

def make_body_from_csv(
        filename,
        body_id: int = 0,
        pos: np.ndarray | None = None,
        vel: np.ndarray | None = None
):
    if pos is None:
        pos = np.zeros(3, dtype=float)
    else:
        pos = np.asarray(pos, dtype=float)

    if vel is None:
        vel = np.zeros(3, dtype=float)
    else:
        vel = np.asarray(vel, dtype=float)
    
    f = open(filename , 'r' , encoding='utf-8')
    data = csv.reader(f,delimiter = ',')
    header = next(data)

    particles = []

    for row in data:
        bid, x, y, z, r, m = map(float, row)
        bid = int(bid)
        particles.append(
            Particle(pos = np.array([x, y, z]), r = float(r), m = float(m))
        )
    return RigidBody(
        id = body_id,
        particles=particles,
        pos = pos,
        vel = vel
    )

def make_balls(
        n: int,
        radius_dist: str | int,
):
    bodies = []
    for i in range(n):
        r = get_random_dist(radius_dist, -0.5, 0.1)
        m = 4/3 * np.pi * r**3
        particles = [Particle(pos=np.zeros(3), r=r, m=m)]
        body = RigidBody(id=i, particles=particles, pos=np.zeros(3), vel=np.zeros(3))
        bodies.append(body)
    return bodies
    

#-----------------------------------------------------------
#region Main
# Main
if __name__ == "__main__":
    # triangle_specs = [
    # ((0.0, 0.0, 0.0), 0.2, 1.0),
    # ((0.35, 0.0, 0.0), 0.2, 1.0),
    # ((0.0, 0.35, 0.0), 0.2, 1.0),
    # ]

    # temp = []
    # for i in range(4):
    #     temp.append(make_body_from_specs(i, triangle_specs))
    
    # bodies = random_bodies_states(temp)
    # res_bodies, ts, tke_hist, rke_hist, totke_hist, px_hist, py_hist, pz_hist, lx_hist, ly_hist, lz_hist = simulate(bodies)
    # plot_results(ts, tke_hist, rke_hist, totke_hist, px_hist, py_hist, pz_hist, lx_hist, ly_hist, lz_hist)
    # save_bodies_csv(res_bodies, filename = "final_bodies.csv")
    # body = make_body_from_csv(filename = "aggregate_particles.csv")

    # temp = [body]
    # bodies = random_bodies_states(temp)
    bodies = make_balls(N, radius_dist = "logn")
    bodies = random_bodies_states(bodies)
    res_bodies, ts, tke_hist, rke_hist, totke_hist, px_hist, py_hist, pz_hist, lx_hist, ly_hist, lz_hist = simulate(bodies)
    plot_results(ts, tke_hist, rke_hist, totke_hist, px_hist, py_hist, pz_hist, lx_hist, ly_hist, lz_hist)
    save_bodies_csv(res_bodies, filename = "final_bodies.csv")
    # save_bodies_csv([body], filename = "test.csv")