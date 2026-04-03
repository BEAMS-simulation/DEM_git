import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =========================
# Parameters
# =========================
PARTICLE_RADIUS = 0.5          # particle radius
TARGET_AGG_RADIUS = 10.0       # stop when aggregate max extent reaches this
STEP_SIZE = 0.6                # random walk step length
SPAWN_MARGIN = 5.0             # walker spawn radius = current agg radius + margin
KILL_MARGIN = 20.0             # if too far from center, discard walker
TOUCH_TOL = 0.05               # extra tolerance for contact
MAX_WALK_STEPS = 20000         # max steps per walker
MAX_SHOVE_ITERS = 30           # overlap relaxation iterations
SHOVE_DAMPING = 0.7            # relaxation damping
MAX_PARTICLES = 3000           # hard cap for safety

np.random.seed(0)


# =========================
# Particle definition
# =========================
@dataclass
class Particle:
    pos: np.ndarray
    r: float
    state: str   # "seed", "aggregate", "walker"


# =========================
# Utility functions
# =========================
def random_unit_vector():
    v = np.random.normal(size=3)
    n = np.linalg.norm(v)
    while n < 1e-12:
        v = np.random.normal(size=3)
        n = np.linalg.norm(v)
    return v / n


def aggregate_radius(particles):
    """Max distance from origin to particle surface."""
    if len(particles) == 0:
        return 0.0
    vals = [np.linalg.norm(p.pos) + p.r for p in particles]
    return max(vals)


def current_center(particles):
    """Center of mass approximation."""
    pts = np.array([p.pos for p in particles])
    return pts.mean(axis=0)


def spawn_walker(agg_radius, center, particle_radius):
    spawn_radius = max(agg_radius + SPAWN_MARGIN, 3.0)
    pos = center + spawn_radius * random_unit_vector()
    return Particle(pos=pos.copy(), r=particle_radius, state="walker")


def too_far(walker, center, agg_radius):
    kill_radius = max(agg_radius + KILL_MARGIN, 10.0)
    return np.linalg.norm(walker.pos - center) > kill_radius


def step_walker(walker, step_size):
    walker.pos += step_size * random_unit_vector()


def touches_any(walker, aggregate, tol=0.0):
    """Return index of touched particle, else None."""
    wpos = walker.pos
    wr = walker.r
    for i, q in enumerate(aggregate):
        d = np.linalg.norm(wpos - q.pos)
        if d <= wr + q.r + tol:
            return i
    return None


def attach_to_particle(walker, target_particle):
    """Project walker to exact contact position with target particle."""
    direction = walker.pos - target_particle.pos
    norm = np.linalg.norm(direction)

    if norm < 1e-12:
        direction = random_unit_vector()
    else:
        direction = direction / norm

    walker.pos = target_particle.pos + direction * (walker.r + target_particle.r)
    walker.state = "aggregate"


def shove_particle(new_particle, aggregate):
    """
    Resolve overlaps by pushing only the new particle away from neighbors.
    Simple and stable enough for this use.
    """
    for _ in range(MAX_SHOVE_ITERS):
        moved = False
        total_push = np.zeros(3)

        for q in aggregate:
            if q is new_particle:
                continue

            diff = new_particle.pos - q.pos
            dist = np.linalg.norm(diff)
            min_dist = new_particle.r + q.r

            if dist < 1e-12:
                diff = random_unit_vector()
                dist = 1e-12

            if dist < min_dist:
                overlap = min_dist - dist
                total_push += (overlap * diff / dist)
                moved = True

        if not moved:
            break

        new_particle.pos += SHOVE_DAMPING * total_push


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


# =========================
# Main DLA-like annealing simulation
# =========================
def run_annealing():
    aggregate = [Particle(pos=np.array([0.0, 0.0, 0.0]), r=PARTICLE_RADIUS, state="seed")]

    n_attempts = 0
    n_attached = 1

    while True:
        agg_radius = aggregate_radius(aggregate)
        center = current_center(aggregate)

        if agg_radius >= TARGET_AGG_RADIUS:
            print(f"Reached target aggregate radius: {agg_radius:.2f}")
            break

        if len(aggregate) >= MAX_PARTICLES:
            print(f"Reached particle limit: {len(aggregate)}")
            break

        walker = spawn_walker(agg_radius, center, PARTICLE_RADIUS)
        n_attempts += 1

        attached = False
        for _ in range(MAX_WALK_STEPS):
            step_walker(walker, STEP_SIZE)

            hit_idx = touches_any(walker, aggregate, tol=TOUCH_TOL)
            if hit_idx is not None:
                attach_to_particle(walker, aggregate[hit_idx])
                aggregate.append(walker)
                shove_particle(walker, aggregate)
                attached = True
                n_attached += 1
                break

            if too_far(walker, center, agg_radius):
                break

        if n_attempts % 100 == 0:
            print(
                f"Attempts={n_attempts:5d}, "
                f"Attached={n_attached:5d}, "
                f"AggRadius={aggregate_radius(aggregate):6.2f}, "
                f"MaxOverlap={max_overlap(aggregate):.4f}"
            )

    return aggregate


# =========================
# Visualization
# =========================
def plot_aggregate(aggregate):
    pts = np.array([p.pos for p in aggregate])
    rs = np.array([p.r for p in aggregate])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=(rs * 80)**2, alpha=0.8)

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


# =========================
# Save result
# =========================
def save_aggregate_csv(aggregate, filename="aggregate_particles.csv"):
    pts = np.array([p.pos for p in aggregate])
    rs = np.array([p.r for p in aggregate])
    data = np.column_stack([pts, rs])
    header = "x,y,z,r"
    np.savetxt(filename, data, delimiter=",", header=header, comments="")
    print(f"Saved particle data to {filename}")


# =========================
# Run
# =========================
if __name__ == "__main__":
    aggregate = run_annealing()
    print(f"Final number of particles: {len(aggregate)}")
    print(f"Final aggregate radius: {aggregate_radius(aggregate):.3f}")
    print(f"Final max overlap: {max_overlap(aggregate):.6f}")

    save_aggregate_csv(aggregate, "aggregate_particles.csv")
    plot_aggregate(aggregate)