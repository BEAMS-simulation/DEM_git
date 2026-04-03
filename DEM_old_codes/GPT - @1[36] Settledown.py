import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from time import time
import csv

rng = np.random.default_rng(0)

# -------------------------------------------------------
# Parameters
CONTACT_SPRING_CONST = 10.0
PARTICLE_MASS        = 2.0
PARTICLE_RADIUS      = 0.75
NORMAL_DAMPING       = 15.0
GRAVITY_ACCEL        = 9.8
FLOOR_SPRING_CONST   = 3000.0
TIME_STEP            = 1e-3

WALL_LENGTH_X        = 2
WALL_LENGTH_Y        = 2

N = 30

# 초기 배치 높이
INIT_Z_MIN = 2.0
INIT_Z_MAX = 20.0

# 초기 입자 간 최소 중심거리 계수
INIT_MIN_DIST_FACTOR = 1.05

# 안정화 종료 조건
MAX_TIME    = 20.0
STABLE_TIME = 0.5
KINETIC_TOL = 1e-3
VMAX_TOL    = 5e-2
Z_TOL       = 1e-4
Z_WINDOW    = 200

# contact 판정용 tolerance
CONTACT_TOL = 5e-2
# -------------------------------------------------------


@dataclass
class Sphere:
    pos: np.ndarray   # shape (3,)
    vel: np.ndarray   # shape (3,)
    r: float = PARTICLE_RADIUS
    m: float = PARTICLE_MASS


# -------------------------------------------------------
# Basic array helpers
def positions_array(particles):
    return np.array([p.pos for p in particles], dtype=float)

def velocities_array(particles):
    return np.array([p.vel for p in particles], dtype=float)

def radii_array(particles):
    return np.array([p.r for p in particles], dtype=float)

def masses_array(particles):
    return np.array([p.m for p in particles], dtype=float)


# -------------------------------------------------------
# Periodic geometry
def periodic_displacement(p1, p2):
    """
    returns displacement vector from p1 to p2 with minimum image in x,y
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]

    dx -= WALL_LENGTH_X * np.round(dx / WALL_LENGTH_X)
    dy -= WALL_LENGTH_Y * np.round(dy / WALL_LENGTH_Y)

    return np.array([dx, dy, dz], dtype=float)


def normal_vectors(particles):
    poss = positions_array(particles)

    dR = poss[None, :, :] - poss[:, None, :]   # dR[i,j] = r_j - r_i

    dR[:, :, 0] -= WALL_LENGTH_X * np.round(dR[:, :, 0] / WALL_LENGTH_X)
    dR[:, :, 1] -= WALL_LENGTH_Y * np.round(dR[:, :, 1] / WALL_LENGTH_Y)

    dist2 = np.sum(dR**2, axis=2)
    dist = np.sqrt(dist2)

    nvec = np.zeros_like(dR)
    mask = dist > 1e-14
    nvec[mask] = dR[mask] / dist[mask, None]

    return dist, nvec


# -------------------------------------------------------
# Forces
def contact_forces(particles):
    n = len(particles)
    dists, nvecs = normal_vectors(particles)
    vels = velocities_array(particles)

    # relative velocity: v_i - v_j
    dvels = vels[:, None, :] - vels[None, :, :]

    Force = np.zeros((n, n, 3), dtype=float)

    for i in range(n):
        ri = particles[i].r
        for j in range(n):
            if i == j:
                continue

            rj = particles[j].r
            overlap = ri + rj - dists[i, j]

            if overlap <= 0:
                continue

            nij = nvecs[i, j]                 # points from i to j
            vij_n = np.dot(dvels[i, j], nij) # (v_i - v_j)·n_ij

            # elastic repulsion: push i away from j
            Force[i, j] -= CONTACT_SPRING_CONST * overlap * nij

            # normal damping: only when approaching
            if vij_n > 0:
                Force[i, j] -= NORMAL_DAMPING * vij_n * nij

    return Force.sum(axis=1)


def gravity_forces(particles):
    n = len(particles)
    Force = np.zeros((n, 3), dtype=float)
    for i, q in enumerate(particles):
        Force[i, 2] -= q.m * GRAVITY_ACCEL
    return Force


def floor_forces(particles):
    n = len(particles)
    Force = np.zeros((n, 3), dtype=float)

    for i, q in enumerate(particles):
        overlap = max(q.r - q.pos[2], 0.0)
        if overlap > 0:
            Force[i, 2] += FLOOR_SPRING_CONST * overlap

            # downward velocity damping on floor contact
            if q.vel[2] < 0:
                Force[i, 2] -= NORMAL_DAMPING * q.vel[2]

    return Force


def total_forces(particles):
    return (
        contact_forces(particles)
        + gravity_forces(particles)
        + floor_forces(particles)
    )


# -------------------------------------------------------
# Integrator
def inbox(particles):
    for q in particles:
        q.pos[0] %= WALL_LENGTH_X
        q.pos[1] %= WALL_LENGTH_Y


def leapfrog(particles):
    Force = total_forces(particles)

    # half kick + drift
    for i, q in enumerate(particles):
        q.vel += 0.5 * Force[i] / q.m * TIME_STEP
        q.pos += q.vel * TIME_STEP

    inbox(particles)

    Force = total_forces(particles)

    # half kick
    for i, q in enumerate(particles):
        q.vel += 0.5 * Force[i] / q.m * TIME_STEP


# -------------------------------------------------------
# Initialization
def random_position(zmin=INIT_Z_MIN, zmax=INIT_Z_MAX):
    x = rng.uniform(0.0, WALL_LENGTH_X)
    y = rng.uniform(0.0, WALL_LENGTH_Y)
    z = rng.uniform(zmin, zmax)
    return np.array([x, y, z], dtype=float)


def is_valid_position(new_pos, particles, min_dist):
    for p in particles:
        dR = periodic_displacement(new_pos, p.pos)
        dist = np.linalg.norm(dR)
        if dist < min_dist:
            return False
    return True


def initialize_particles(n=N):
    particles = []

    dx = 2.2 * PARTICLE_RADIUS
    dy = 2.2 * PARTICLE_RADIUS
    dz = 2.4 * PARTICLE_RADIUS

    xs = np.arange(PARTICLE_RADIUS, WALL_LENGTH_X, dx)
    ys = np.arange(PARTICLE_RADIUS, WALL_LENGTH_Y, dy)

    count = 0
    layer = 0
    z = INIT_Z_MIN

    while count < n:
        for x in xs:
            for y in ys:
                if count >= n:
                    break

                # 약간 랜덤 perturbation만 추가
                pos = np.array([
                    x + rng.uniform(-0.1, 0.1) * PARTICLE_RADIUS,
                    y + rng.uniform(-0.1, 0.1) * PARTICLE_RADIUS,
                    z + rng.uniform(-0.1, 0.1) * PARTICLE_RADIUS
                ], dtype=float)

                pos[0] %= WALL_LENGTH_X
                pos[1] %= WALL_LENGTH_Y

                particles.append(Sphere(pos=pos, vel=np.zeros(3, dtype=float)))
                count += 1

            if count >= n:
                break

        layer += 1
        z = INIT_Z_MIN + layer * dz

    return particles


# -------------------------------------------------------
# Diagnostics
def kinetic_energy(particles):
    vels = velocities_array(particles)
    masses = masses_array(particles)
    return 0.5 * np.sum(masses[:, None] * vels**2)


def max_speed(particles):
    vels = velocities_array(particles)
    return np.max(np.linalg.norm(vels, axis=1))


def contact_pairs(particles, tol=CONTACT_TOL):
    poss = positions_array(particles)
    radii = radii_array(particles)
    n = len(particles)

    pairs = []
    min_gap = np.inf

    for i in range(n):
        for j in range(i + 1, n):
            dR = poss[j] - poss[i]
            dR[0] -= WALL_LENGTH_X * np.round(dR[0] / WALL_LENGTH_X)
            dR[1] -= WALL_LENGTH_Y * np.round(dR[1] / WALL_LENGTH_Y)

            dist = np.linalg.norm(dR)
            gap = dist - (radii[i] + radii[j])

            if gap < min_gap:
                min_gap = gap

            if gap <= tol:
                pairs.append((i, j, dR, gap))

    return pairs, min_gap


def check_contacts(particles, tol=CONTACT_TOL):
    pairs, min_gap = contact_pairs(particles, tol=tol)
    print(f"minimum gap = {min_gap:.6f}")
    print(f"number of near contacts (gap <= {tol}) = {len(pairs)}")


# -------------------------------------------------------
# Simulation until stable
def simulate_until_stable():
    particles = initialize_particles(N)

    ts = []
    z_mean_hist = []
    z_min_hist = []
    kinetic_hist = []

    t = 0.0
    quiet_duration = 0.0

    tic = time()

    while t < MAX_TIME:
        leapfrog(particles)
        t += TIME_STEP

        poss = positions_array(particles)
        K = kinetic_energy(particles)
        vmax = max_speed(particles)
        z_mean = np.mean(poss[:, 2])
        z_min = np.min(poss[:, 2])

        ts.append(t)
        z_mean_hist.append(z_mean)
        z_min_hist.append(z_min)
        kinetic_hist.append(K)

        if len(z_mean_hist) > Z_WINDOW:
            dz = abs(z_mean_hist[-1] - z_mean_hist[-Z_WINDOW])
        else:
            dz = np.inf

        if (K < KINETIC_TOL) and (vmax < VMAX_TOL) and (dz < Z_TOL):
            quiet_duration += TIME_STEP
        else:
            quiet_duration = 0.0

        if quiet_duration >= STABLE_TIME:
            print(f"stabilized at t = {t:.4f}")
            print(f"K = {K:.6e}, vmax = {vmax:.6e}, dz = {dz:.6e}")
            break

    toc = time()

    if t >= MAX_TIME:
        print("max_time reached before full stabilization")

    print(f"simulation finished in {toc - tic:.3f} s")

    return (
        particles,
        np.array(ts),
        np.array(z_mean_hist),
        np.array(z_min_hist),
        np.array(kinetic_hist),
    )


# -------------------------------------------------------
# Plotting
def draw_sphere(ax, center, radius, n_theta=36, n_phi=20):
    x0, y0, z0 = center

    theta = np.linspace(0, 2*np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi)

    x = x0 + radius * np.sin(phi) * np.cos(theta)
    y = y0 + radius * np.sin(phi) * np.sin(theta)
    z = z0 + radius * np.cos(phi)

    ax.plot_surface(
        x, y, z,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.95
    )


def plot_results(particles, ts, z_mean_hist, z_min_hist, kinetic_hist):
    poss = positions_array(particles)
    radii = radii_array(particles)

    fig = plt.figure(figsize=(14, 5))

    # -----------------------------
    # final configuration
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    for p in particles:
        draw_sphere(ax1, p.pos, p.r)

    # 중심점 표시
    ax1.scatter(poss[:, 0], poss[:, 1], poss[:, 2], s=8, depthshade=False)

    # contact line 표시
    pairs, _ = contact_pairs(particles, tol=CONTACT_TOL)
    for i, j, dR, gap in pairs:
        p1 = poss[i]
        p2 = p1 + dR
        ax1.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            linewidth=1.0
        )

    ax1.set_title("Final particle positions")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    zmax = np.max(poss[:, 2] + radii) + 0.5
    ax1.set_xlim(0, WALL_LENGTH_X)
    ax1.set_ylim(0, WALL_LENGTH_Y)
    ax1.set_zlim(0, zmax)

    # 박스 비율 고정
    ax1.set_box_aspect((WALL_LENGTH_X, WALL_LENGTH_Y, zmax))
    ax1.view_init(elev=20, azim=-60)

    # floor
    xx = np.array([[0, WALL_LENGTH_X], [0, WALL_LENGTH_X]])
    yy = np.array([[0, 0], [WALL_LENGTH_Y, WALL_LENGTH_Y]])
    zz = np.zeros_like(xx)
    ax1.plot_surface(xx, yy, zz, alpha=0.08)

    # -----------------------------
    # height evolution
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(ts, z_mean_hist, label="mean z")
    ax2.plot(ts, z_min_hist, label="min z")
    ax2.set_title("Height evolution")
    ax2.set_xlabel("time")
    ax2.set_ylabel("z")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # kinetic energy
    plt.figure(figsize=(6, 4))
    plt.plot(ts, kinetic_hist)
    plt.title("Kinetic energy")
    plt.xlabel("time")
    plt.ylabel("K")
    plt.grid(True)
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
    particles, ts, z_mean_hist, z_min_hist, kinetic_hist = simulate_until_stable()
    check_contacts(particles, tol=CONTACT_TOL)
    save_particles_csv(particles, filename="final_particles.csv")
    plot_results(particles, ts, z_mean_hist, z_min_hist, kinetic_hist)