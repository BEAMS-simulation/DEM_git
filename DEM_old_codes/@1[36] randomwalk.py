import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from time import time
import csv

rng = np.random.default_rng()


# -----------------------------
# Parameters

PARTICLE_RADIUS     = 0.5
KILL_RADIUS         = 30.0
MAX_AGGREGATE_SIZE  = 10.0
SPAWN_MARGIN        = 3.0
SHOVEL_DAMPING      = 0.5
OVERLAPPING_DAMPING = 0.4
MOVING_DISTANCE     = 0.8
MAX_SHOVEL_ITER     = 100
MAX_PARTICLES       = 3000
MAX_WALKER_STEP     = 20000
LPS_MEAN            = -0.5
LPS_SIGMA           = 0.201

# -----------------------------

@dataclass
class Particle:
    pos: np.ndarray     # [x, y, z]
    r: float            # Radius of particle
    status: str         # "Initial" / "Attached" / "Walker"

    def __post_init__(self):
        self.m = 4.0 / 3.0 * np.pi * self.r**2

def random_unit_vector():
    rng = np.random.default_rng()
    v = rng.standard_normal(3)
    return v/np.linalg.norm(v)

def random_radius(mu: float = LPS_MEAN, sigma: float = LPS_SIGMA):
    radius = max(min(0.8, rng.lognormal(mu, sigma)), 0.2)
    # radius = PARTICLE_RADIUS
    return radius

def step_random_walk(particle: Particle):
    particle.pos += MOVING_DISTANCE * random_unit_vector()

def aggregate_size(aggregate: list[Particle]):
    if aggregate == []:
        return 0.0
    ls = [np.linalg.norm(par.pos) + par.r for par in aggregate]
    return max(ls)

def aggregate_center(aggregate: list[Particle]):
    particles = np.array([q.pos for q in aggregate])
    return particles.mean(axis = 0)

def spawn_walker(aggregate: list[Particle], radius: float):
    agg_size = aggregate_size(aggregate)
    center = aggregate_center(aggregate)
    spawn_size = max(agg_size + SPAWN_MARGIN, 3.0)
    
    pos = center + spawn_size * random_unit_vector()
    return Particle(pos=pos.copy(), r = radius, status = "Walker")

def far_walker(walker: Particle, center: np.ndarray, agg_radius: float):
    farland = max(agg_radius + KILL_RADIUS, 10.0)
    return np.linalg.norm(walker.pos - center) > farland

def is_touched(walker: Particle, aggregate: list[Particle]):
    wpos, wr = walker.pos, walker.r
    for idx, q in enumerate(aggregate):
        dist = np.linalg.norm(q.pos - wpos)
        if dist <= (wr + q.r):
            return idx
    return None

def attach_particle(walker: Particle):
    walker.status = "Attached"
    
def tol_attach(p: Particle, q: Particle):
    return OVERLAPPING_DAMPING * p.r * q.r / (p.r + q.r)

def shovel(new_particle: Particle, aggregate: list[Particle]):
    for _ in range(MAX_SHOVEL_ITER):
        moved = False
        total_push = np.zeros(3)
        
        for q in aggregate:
            if q is new_particle:
                continue
            
            diff = new_particle.pos - q.pos
            dist = np.linalg.norm(diff)
            min_dist = new_particle.r + q.r
            tol_dist = tol_attach(new_particle, q)
            
            if dist < 1e-12:
                diff = random_unit_vector()
                dist = 1e-13
            if dist < min_dist - tol_dist:
                overlap = min_dist - dist
                total_push += (overlap * diff / dist)
                moved = True
        
        if not moved:
            break
        
        new_particle.pos += SHOVEL_DAMPING * total_push

def max_overlap(particles):
    """Diagnostic only."""
    m = 0.0
    n = len(particles)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(particles[i].pos - particles[j].pos)
            ov = particles[i].r + particles[j].r - d
            if ov > m:
                m = ov
    return max(m, 0.0)


def simulation():
    start = time()
    aggregate = [Particle(pos = np.array([0.0, 0.0, 0.0]), r = random_radius(), status = "Initial")]
    
    n_attempts = 0
    n_attached = 1
    
    while True:
        agg_size    = aggregate_size(aggregate)
        center      = aggregate_center(aggregate)
        
        if agg_size > MAX_AGGREGATE_SIZE:
            print(f"Reached target aggregate size : {agg_size:.3f}")
            break
        
        if len(aggregate) >= MAX_PARTICLES:
            print(f"Reached maximum particle number: {len(aggregate)}")
            break
        
        walker = spawn_walker(aggregate, random_radius())
        n_attempts += 1
        
        attached = False
        for _ in range(MAX_WALKER_STEP):
            step_random_walk(walker)
            
            hit_idx = is_touched(walker, aggregate)
            if hit_idx is not None:
                attach_particle(walker)
                aggregate.append(walker)
                shovel(walker, aggregate)
                attached = True
                n_attached += 1
                break
            
            if far_walker(walker, center, agg_size):
                break
        
        if n_attempts % 100 == 0:
            buzzer = time()
            print(
                f"Attempts={n_attempts:5d}, "
                f"Attached={n_attached:5d}, "
                f"AggRadius={aggregate_size(aggregate):6.2f}, "
                f"MaxOverlap={max_overlap(aggregate):.4f}, "
                f"Time Elapsed={int(buzzer-start):>4d}s"
            )

    return aggregate


# ----------------------------------------------------------------------------
# Visualization
def plot_aggregate(aggregate):
    pts = np.array([p.pos for p in aggregate])
    rs = np.array([p.r for p in aggregate])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=(rs * 80)**2, alpha=0.2)
    ax.plot(pts[:, 0], pts[:,1], pts[:,2], 'o', c='red')

    # Equal aspect ratio
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    mid_x = (x_max + x_min) / 2.0
    mid_y = (y_max + y_min) / 2.0
    mid_z = (z_max + z_min) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title("3D DLA-like Aggregate (Annealing Step)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.show()
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Save CSV file
def save_aggregate_csv(aggregate, filename="aggregate_particles.csv"):
    with open(filename, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["body id", "id", "x", "y", "z", "r", "m"])

        for i, p in enumerate(aggregate):
            pos = p.pos
            writer.writerow(
                [0, i, pos[0], pos[1], pos[2], p.r, p.m]
            )
    print(f"Saved particle data to {filename}")
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Main
if __name__ == "__main__":
    aggregate = simulation()
    print(f"Final number of particles: {len(aggregate)}")
    print(f"Final aggregate radius: {aggregate_size(aggregate):.3f}")
    print(f"Final max overlap: {max_overlap(aggregate):.6f}")

    save_aggregate_csv(aggregate, "aggregate_particles.csv")
    plot_aggregate(aggregate)
# ----------------------------------------------------------------------------
