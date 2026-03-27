import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from time import time
import csv
from matplotlib.patches import Circle

rng = np.random.default_rng()

#-----------------------------------------------------
# Parameters

# Physical_properties
CONTACT_SPRING_CONST    = 200.0
PARTICLE_MASS           = 2.0
PARTICLE_RADIUS         = 0.5
NORMAL_DAMPING          = 15.0
GRAVITY_ACCEL           = 9.8
FLOOR_SPRING_CONTST     = 3000.0
TIME_STEP               = 5e-4

# Simulation_box
WALL_LENGTH_X           = 2.0
WALL_LENGTH_Y           = 2.0

# Initial_simulating_box
INIT_Z_MIN              = 2.0
INIT_Z_MAX              = 20.0
INIT_MIN_DIST_FACTOR    = 1.05

INIT_TYPE               = "EACH"    # "FLOOR", "RANDOM", "EACH"
INIT_MAX_TRIALS         = 10000

# Particle_number
N                       = 15

# Stabilized_condition
MAX_TIME                = 20.0
STABLE_TIME             = 0.3
KINETIC_TOL             = 1e-3
VMAX_TOL                = 1e-3
Z_TOL                   = 1e-4
Z_WINDOW                = 200

# Contact_tolerance
CONTACT_TOL             = 5e-2

#-----------------------------------------------------

@dataclass
class Sphere:
    pos:    np.ndarray
    vel:    np.ndarray
    r:      float       = PARTICLE_RADIUS
    m:      float       = PARTICLE_MASS


#-----------------------------------------------------
# Array Utility Functions

def position_array(particles: list[Sphere]):
    return np.array([q.pos for q in particles], dtype = float)

def velocity_array(particles: list[Sphere]):
    return np.array([q.vel for q in particles], dtype = float)

def radii_array(particles: list[Sphere]):
    return np.array([q.r for q in particles], dtype = float)

def mass_array(particles: list[Sphere]):
    return np.array([q.m for q in particles], dtype = float)

#-----------------------------------------------------
# Radius_Distribution
def get_random_radius():
    rad = rng.lognormal(-0.701, 0.401)
    rad = np.clip(rad, 0.2, 0.8)
    rad = 0.5
    return rad


#-----------------------------------------------------
# Simulation Box Functions

def dist_periodic(p1: Sphere, p2: Sphere):
    dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    
    dx -= WALL_LENGTH_X * np.round(dx / WALL_LENGTH_X)
    dy -= WALL_LENGTH_Y * np.round(dy / WALL_LENGTH_Y)
    
    return np.array([dx, dy, dz], dtype = float)

def normal_vectors(particles: list[Sphere]):
    poss = position_array(particles)
    
    dR = poss[None, :, :] - poss[:, None, :]    # dR[i, j] = r_j - r_i
    
    dR[:, :, 0] -= WALL_LENGTH_X * np.round(dR[:, :, 0] / WALL_LENGTH_X)
    dR[:, :, 1] -= WALL_LENGTH_Y * np.round(dR[:, :, 1] / WALL_LENGTH_Y)
    
    dist2   = np.sum(dR**2, axis = 2)
    dist    = np.sqrt(dist2)
    
    nvec    = np.zeros_like(dR)
    mask    = dist > 1e-14
    nvec[mask] = dR[mask] / dist[mask, None]
    
    return dist, nvec

#-----------------------------------------------------
# Force Functions

def contact_forces(particles: list[Sphere]):
    n           = len(particles)
    dist, nvec  = normal_vectors(particles)
    vels        = velocity_array(particles)
    
    dV          = vels[:, None, :] - vels[None, :, :]   # dV[i, j] = v_i - v_j
    
    Force       = np.zeros((n, n, 3), dtype=float)
    
    for i in range(n):
        ri = particles[i].r
        for j in range(n):
            if i == j:
                continue
            
            rj = particles[j].r
            ov = ri + rj - dist[i, j]
            
            if ov <= 0:
                continue
            
            nij = nvec[i, j]                # from i to j
            vij_n = np.dot(dV[i, j], nij)   # (v_i - v_j) \cdot nij
            
            Force[i, j] -= CONTACT_SPRING_CONST * ov * nij
            
            if vij_n > 0:
                Force[i, j] -= NORMAL_DAMPING * vij_n * nij
    
    return Force.sum(axis = 1)

def gravity_forces(particles: list[Sphere]):
    n           = len(particles)
    Force       = np.zeros((n, 3), dtype = float)
    
    for i, q in enumerate(particles):
        Force[i, 2] -= q.m * GRAVITY_ACCEL
    return Force

def floor_forces(particles: list[Sphere]):
    n           = len(particles)
    Force       = np.zeros((n, 3), dtype = float)
    
    for i, q in enumerate(particles):
        ov      = max(q.r - q.pos[2], 0.0)
        if ov > 0:
            Force[i, 2] += FLOOR_SPRING_CONTST * ov
            
            if q.vel[2] < 0:
                Force[i, 2] -= NORMAL_DAMPING * q.vel[2]
    return Force

def total_forces(particles: list[Sphere]):
    return (
        contact_forces(particles)
        + gravity_forces(particles)
        + floor_forces(particles)
    )


#-----------------------------------------------------
# Integrator
def inbox(particles: list[Sphere]):
    for q in particles:
        q.pos[0]    %= WALL_LENGTH_X
        q.pos[1]    %= WALL_LENGTH_Y

def leapfrog(particles: list[Sphere]):
    Force = total_forces(particles)
    
    for i, q in enumerate(particles):
        q.vel += 0.5 * Force[i] / q.m * TIME_STEP
        q.pos += q.vel * TIME_STEP
    
    inbox(particles)
    
    Force = total_forces(particles)
    
    for i, q in enumerate(particles):
        q.vel += 0.5 * Force[i] / q.m * TIME_STEP

#-----------------------------------------------------
# Initialization
def random_position(zmin = INIT_Z_MIN, zmax = INIT_Z_MAX):
    x = rng.uniform(0.0, WALL_LENGTH_X)
    y = rng.uniform(0.0, WALL_LENGTH_Y)
    z = rng.uniform(zmin, zmax)
    return np.array([x, y, z], dtype = float)

def is_valid_position(new_pos, particles, min_dist):
    for p in particles:
        dR = dist_periodic(new_pos, p.pos)
        dist = np.linalg.norm(dR)
        if dist < min_dist:
            return False
    return True

def initialize_particles_eachfloor(n = N):
    particles = []
    
    for i in range(n):
        zpos = INIT_MIN_DIST_FACTOR * 2 * PARTICLE_RADIUS * (i + 1)
        xpos = rng.uniform(0.0, WALL_LENGTH_X)
        ypos = rng.uniform(0.0, WALL_LENGTH_Y)
        pos = np.array([xpos, ypos, zpos])
        rad = get_random_radius()
        m   = (2 * rad) ** 3
        particles.append(Sphere(pos = pos, vel = np.zeros(3, dtype = float), r = rad, m = m))
    print("Initializing complete")
    print("size distribution :")
    print(radii_array(particles))
    return particles
    
    
def initialize_particles_random(n = N):
    particles = []
    
    min_dist = INIT_MIN_DIST_FACTOR * 2 * PARTICLE_RADIUS
    
    for k in range(n):
        placed = False
        for _ in range(INIT_MAX_TRIALS):
            pos = random_position()
            
            if is_valid_position(pos, particles, min_dist):
                particles.append(Sphere(pos = pos, vel = np.zeros(3, dtype = float)))
                placed = True
                break
        if not placed:
            raise RuntimeError(f"{k}번째 입자 배치 실패.")
    print("Initializing complete")
    return particles

def initialize_particles_floor(n = N):
    particles = []
    
    dx = 2.2 * PARTICLE_RADIUS
    dy = 2.2 * PARTICLE_RADIUS
    dz = 2.4 * PARTICLE_RADIUS
    
    xs = np.arange(PARTICLE_RADIUS, WALL_LENGTH_X, dx)
    ys = np.arange(PARTICLE_RADIUS, WALL_LENGTH_Y, dy)
    
    cnt = 0
    layer = 0
    z = INIT_Z_MIN
    
    while cnt < n:
        for x in xs:
            for y in ys:
                if cnt >= n:
                    break
                pos = np.array([
                    x + rng.uniform(-0.1, 0.1) * PARTICLE_RADIUS,
                    y + rng.uniform(-0.1, 0.1) * PARTICLE_RADIUS,
                    z + rng.uniform(-0.1, 0.1) * PARTICLE_RADIUS
                ], dtype = float)
                
                pos[0] %= WALL_LENGTH_X
                pos[1] %= WALL_LENGTH_Y
                
                particles.append(Sphere(pos=pos, vel=np.zeros(3, dtype=float)))
                cnt += 1
            
            if cnt >= n:
                break
        
        layer += 1
        z = INIT_Z_MIN + layer * dz
    print("Initializing complete")
    return particles

#-----------------------------------------------------
# Physical States
def kinetic_energy(particles):
    vels = velocity_array(particles)
    mass = mass_array(particles)
    return 0.5 * np.sum(mass[:, None] * vels ** 2)

def max_speed(particles):
    vels = velocity_array(particles)
    return np.max(np.linalg.norm(vels, axis = 1))

def box_momentum(particles):
    vels = velocity_array(particles)
    mass = mass_array(particles)
    return np.sum(mass[:, None] * vels, axis = 0)

#-----------------------------------------------------
# Diagnostics

#-----------------------------------------------------
# Simulating
def simulate_until_stable():
    match INIT_TYPE: #"FLOOR", "RANDOM", "EACH"
        case "FLOOR":
            particles = initialize_particles_floor(N)
        case "RANDOM":
            particles = initialize_particles_random(N)
        case "EACH":
            particles = initialize_particles_eachfloor(N)
    
    ts              = []
    z_mean_hist     = []
    kinetic_hist    = []
    px_hist, py_hist, pz_hist = [], [], []
    
    t               = 0.0
    stable_duration = 0.0

    start           = time()
    
    step            = 0
    max_step        = MAX_TIME//TIME_STEP
    
    while t < MAX_TIME:
        leapfrog(particles)
        t += TIME_STEP
        step += 1
        
        poss        = position_array(particles)
        kin_erg     = kinetic_energy(particles)
        vmax        = max_speed(particles)
        z_mean      = np.mean(poss[:, 2])
        momentum    = box_momentum(particles)
        px, py, pz  = momentum[0], momentum[1], momentum[2]
        
        ts.append(t)
        z_mean_hist.append(z_mean)
        kinetic_hist.append(kin_erg)
        px_hist.append(px)
        py_hist.append(py)
        pz_hist.append(pz)
        
        if len(z_mean_hist) > Z_WINDOW:
            dz = abs(z_mean_hist[-1] - z_mean_hist[-Z_WINDOW])
        else:
            dz = np.inf
        
        if (kin_erg < KINETIC_TOL) and (vmax < VMAX_TOL) and (dz < Z_TOL):
            stable_duration += TIME_STEP
        else:
            stable_duration = 0.0
        
        if stable_duration >= STABLE_TIME:
            print(f"stabilized at t = {t:.4f}")
            print(f"K = {kin_erg:.6e}, vmax = {vmax:.6e}, dz = {dz:.6e}")
            break
        
        flag        = time()
        if step % 1000 == 0:
            print(f"Step {step}/{max_step} : Time Elapsed={int(flag-start)}sec")
            print(f"KE = {kin_erg:.6f}, dz = {dz:.6f}")
    end = time()
    
    if t >= MAX_TIME:
        print("WARNING : max_time reached before full stabilization.")
    print(f"simulation finished in {end-start:.3f}sec")

    return (
        particles,
        np.array(ts),
        np.array(z_mean_hist),
        np.array(kinetic_hist),
        np.array(px_hist),
        np.array(py_hist),
        np.array(pz_hist)
    )

# -------------------------------------------------------
# Plotting
    # def draw_sphere(ax, center, radius, n_theta=36, n_phi=20):
    #     x0, y0, z0 = center

    #     theta = np.linspace(0, 2*np.pi, n_theta)
    #     phi = np.linspace(0, np.pi, n_phi)
    #     theta, phi = np.meshgrid(theta, phi)

    #     x = x0 + radius * np.sin(phi) * np.cos(theta)
    #     y = y0 + radius * np.sin(phi) * np.sin(theta)
    #     z = z0 + radius * np.cos(phi)

    #     ax.plot_surface(
    #         x, y, z,
    #         linewidth=0,
    #         antialiased=True,
    #         shade=True,
    #         alpha=0.95
    #     )


    # def plot_results(particles, ts, z_mean_hist, kinetic_hist, px_hist, py_hist, pz_hist):
    #     poss    = position_array(particles)
    #     radii   = radii_array(particles)

    #     fig = plt.figure(figsize=(14, 5))

    #     # -----------------------------
    #     # final configuration
    #     ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    #     for p in particles:
    #         draw_sphere(ax1, p.pos, p.r)

    #     # 중심점 표시
    #     ax1.scatter(poss[:, 0], poss[:, 1], poss[:, 2], s=8, depthshade=False)

    #     ax1.set_title("Final particle positions")
    #     ax1.set_xlabel("x")
    #     ax1.set_ylabel("y")
    #     ax1.set_zlabel("z")

    #     zmax = np.max(poss[:, 2] + radii) + 0.5
    #     ax1.set_xlim(0, WALL_LENGTH_X)
    #     ax1.set_ylim(0, WALL_LENGTH_Y)
    #     ax1.set_zlim(0, zmax)

    #     # 박스 비율 고정
    #     ax1.set_box_aspect((WALL_LENGTH_X, WALL_LENGTH_Y, zmax))
    #     ax1.view_init(elev=20, azim=-60)

    #     # floor
    #     xx = np.array([[0, WALL_LENGTH_X], [0, WALL_LENGTH_X]])
    #     yy = np.array([[0, 0], [WALL_LENGTH_Y, WALL_LENGTH_Y]])
    #     zz = np.zeros_like(xx)
    #     ax1.plot_surface(xx, yy, zz, alpha=0.08)

    #     # -----------------------------
    #     # height evolution
    #     ax2 = fig.add_subplot(1, 2, 2)
    #     ax2.plot(ts, z_mean_hist, label="mean z")
    #     ax2.set_title("Height evolution")
    #     ax2.set_xlabel("time")
    #     ax2.set_ylabel("z")
    #     ax2.grid(True)
    #     ax2.legend()

    #     plt.tight_layout()
    #     plt.show()

    #     # -----------------------------
    #     # kinetic energy and momentum
    #     fig2 = plt.figure(figsize = (14, 5))
    #     fig2_ax1 = fig2.add_subplot(1, 2, 1)
    #     fig2_ax1.plot(ts, kinetic_hist, label = "Kinetic Energy")
    #     fig2_ax1.set_title("Kinetic Energy")
    #     fig2_ax1.set_xlabel("time")
    #     fig2_ax1.set_ylabel("KE")
        
    #     fig2_ax2 = fig2.add_subplot(1, 2, 2)
    #     fig2_ax2.set_title("Momentum")
    #     fig2_ax2.plot(ts, px_hist, label = "px")
    #     fig2_ax2.plot(ts, py_hist, label = "py")
    #     fig2_ax2.plot(ts, pz_hist, label = "pz")
    #     fig2_ax2.set_xlabel("time")
    #     fig2_ax2.set_ylabel("kg m/s")
    #     fig2_ax2.legend()
    #     plt.tight_layout()
    #     plt.show()

def plot_results(particles, ts, z_mean_hist, kinetic_hist, px_hist, py_hist, pz_hist):
    poss = position_array(particles)   # shape (N, 3)
    radii = radii_array(particles)     # shape (N,)

    zmax = np.max(poss[:, 2] + radii) + 0.5

    # =================================================
    # Figure 1: final configuration
    fig1 = plt.figure(figsize=(18, 5))

    # -------------------------------------------------
    # (1) 3D scatter: 빠르게 전체 구조 보기
    ax1 = fig1.add_subplot(1, 3, 1, projection='3d')

    # scatter의 s는 물리 반지름이 아니라 화면상 크기이므로
    # "대충 반지름 비례"만 반영
    s = 600 * (radii / np.max(radii))**2

    ax1.scatter(
        poss[:, 0], poss[:, 1], poss[:, 2],
        s=s,
        alpha=0.75,
        depthshade=True
    )

    ax1.set_title("Final particle centers (3D)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax1.set_xlim(0, WALL_LENGTH_X)
    ax1.set_ylim(0, WALL_LENGTH_Y)
    ax1.set_zlim(0, zmax)
    ax1.set_box_aspect((WALL_LENGTH_X, WALL_LENGTH_Y, zmax))
    ax1.view_init(elev=20, azim=-60)

    # floor
    xx = np.array([[0, WALL_LENGTH_X], [0, WALL_LENGTH_X]])
    yy = np.array([[0, 0], [WALL_LENGTH_Y, WALL_LENGTH_Y]])
    zz = np.zeros_like(xx)
    ax1.plot_surface(xx, yy, zz, alpha=0.08)

    # -------------------------------------------------
    # (2) x-z projection: 반지름 정확히 표현
    ax2 = fig1.add_subplot(1, 3, 2)

    for p in particles:
        circ = Circle(
            (p.pos[0], p.pos[2]),
            p.r,
            alpha=0.5,
            linewidth=1,
            fill=False
        )
        ax2.add_patch(circ)

    ax2.set_xlim(0, WALL_LENGTH_X)
    ax2.set_ylim(0, zmax)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.set_title("XZ projection (true radius)")
    ax2.grid(True)

    # -------------------------------------------------
    # (3) y-z projection: 반지름 정확히 표현
    ax3 = fig1.add_subplot(1, 3, 3)

    for p in particles:
        circ = Circle(
            (p.pos[1], p.pos[2]),
            p.r,
            alpha=0.5,
            linewidth=1,
            fill=True
        )
        ax3.add_patch(circ)

    ax3.set_xlim(0, WALL_LENGTH_Y)
    ax3.set_ylim(0, zmax)
    ax3.set_aspect("equal")
    ax3.set_xlabel("y")
    ax3.set_ylabel("z")
    ax3.set_title("YZ projection (true radius)")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    # =================================================
    # Figure 2: height + kinetic energy
    fig2 = plt.figure(figsize=(14, 5))

    ax4 = fig2.add_subplot(1, 2, 1)
    ax4.plot(ts, z_mean_hist, label="mean z")
    ax4.set_title("Height evolution")
    ax4.set_xlabel("time")
    ax4.set_ylabel("z")
    ax4.grid(True)
    ax4.legend()

    ax5 = fig2.add_subplot(1, 2, 2)
    ax5.plot(ts, kinetic_hist, label="Kinetic Energy")
    ax5.set_title("Kinetic Energy")
    ax5.set_xlabel("time")
    ax5.set_ylabel("KE")
    ax5.grid(True)
    ax5.legend()

    plt.tight_layout()
    plt.show()

    # =================================================
    # Figure 3: momentum
    fig3 = plt.figure(figsize=(8, 5))
    ax6 = fig3.add_subplot(111)

    ax6.plot(ts, px_hist, label="px")
    ax6.plot(ts, py_hist, label="py")
    ax6.plot(ts, pz_hist, label="pz")

    ax6.set_title("Total momentum")
    ax6.set_xlabel("time")
    ax6.set_ylabel("kg·m/s")
    ax6.grid(True)
    ax6.legend()

    plt.tight_layout()
    plt.show()

def save_particles_csv(particles, filename="final_particles.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "z", "r", "m"])

        for i, p in enumerate(particles):
            writer.writerow([i, p.pos[0], p.pos[1], p.pos[2], p.r, p.m])

    print(f"saved to {filename}")

# -------------------------------------------------------
# Main
if __name__ == "__main__":
    particles, ts, z_mean_hist, kin_hist, px_hist, py_hist, pz_hist = simulate_until_stable()
    save_particles_csv(particles, filename="final_particles_test_test.csv")
    plot_results(particles, ts, z_mean_hist, kin_hist, px_hist, py_hist, pz_hist)