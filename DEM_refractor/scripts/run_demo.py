from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import random

from demrigid.config import SimConfig, BoxParams, GravityParams, ContactParams, TimeParams
from demrigid.io_csv import CsvStorage
from demrigid.model import Sphere, Aggregate, Rigidbody
from demrigid.world import World
from demrigid.simulator import Simulator


def plot_history(hist: dict[str, np.ndarray]) -> None:
    t = hist["t"]
    ke = hist["ke"]
    p = hist["p"]
    l = hist["l"]

    if len(t) == 0:
        print("No history recorded. Increase record_stride.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(t, ke)
    axes[0].set_ylabel("Kinetic Energy")
    axes[0].grid(True)

    pnorm = np.linalg.norm(p, axis=1)
    axes[1].plot(t, p[:, 0], label="px")
    axes[1].plot(t, p[:, 1], label="py")
    axes[1].plot(t, p[:, 2], label="pz")
    axes[1].plot(t, pnorm, label="|p|")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylabel("Linear Momentum")

    lnorm = np.linalg.norm(l, axis=1)
    axes[2].plot(t, l[:, 0], label="lx")
    axes[2].plot(t, l[:, 1], label="ly")
    axes[2].plot(t, l[:, 2], label="lz")
    axes[2].plot(t, lnorm, label="|l|")
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_ylabel("Angular Momentum")
    axes[2].set_xlabel("Time [s]")

    plt.tight_layout()
    plt.show()


def build_world(cfg: SimConfig,
                n_1: int = 20, r_1: float = 1.2, d_1: float = 2.0,
                n_2: int = 120, r_2: float = 0.3, d_2: float = 2.0) -> World:
    bodies: list[Rigidbody] = []
    rng = np.random.default_rng()
    poses = []
    agges = []
    
    n_tot = n_1 + n_2
    dz = 0
    for i in range(n_1):
        m = d_1 * (4.0 / 3.0) * math.pi * (r_1 ** 3)
        agg = Aggregate([Sphere(r = r_1, m = m, pos_local = np.zeros(3, dtype = float))])
        pos = np.array([
            rng.uniform(r_1, cfg.box.Lx - r_1),
            rng.uniform(r_1, cfg.box.Ly - r_1),
            1.0 + 2.01 * r_1 * i + dz
        ], dtype = float)
        poses.append(pos)
        agges.append(agg)
    dz = 1.0 + 2.01 * r_1 * (n_1 - 1) + r_1
    for i in range(n_2):
        m = d_2 * (4.0 / 3.0) * math.pi * (r_2 ** 3)
        agg = Aggregate([Sphere(r = r_2, m = m, pos_local = np.zeros(3, dtype = float))])
        pos = np.array([
            rng.uniform(r_2, cfg.box.Lx - r_2),
            rng.uniform(r_2, cfg.box.Ly - r_2),
            2.2 * r_2 * i + dz
        ], dtype = float)
        poses.append(pos)
        agges.append(agg)
    # random.shuffle(poses)
    fix_pos = poses[:n_1//2] + poses[-(n_2//3):]
    shuffle_pos = poses[n_1//2:- (n_2//3)]
    random.shuffle(shuffle_pos)
    poses = fix_pos + shuffle_pos


    for j in range(n_tot):
        bodies.append(Rigidbody(body = agges[j], id = j, pos = poses[j]))
    
    return World(cfg, bodies)

def build_bodies(cfg: SimConfig, n_balls: int, density: float = 2.0, r: float = 0.5, dz: float = 0.0, di: int = 0) -> list[np.ndarray]:
    # bodies : list[Rigidbody] = []
    rng = np.random.default_rng(321)
    positions = []
    
    for i in range(n_balls):
        m = density * (4.0 / 3.0) * math.pi * (r ** 3)
        agg = Aggregate([Sphere(r=r, m=m, pos_local=np.zeros(3, dtype=float))])
        pos = np.array([
            rng.uniform(0.5, cfg.box.Lx - 0.5),
            rng.uniform(0.5, cfg.box.Ly - 0.5),
            0.5 + 2.1 * r * i + dz,
        ], dtype = float)
        positions.append(pos)
    
    return positions


def save_outputs(world: World, hist: dict[str, np.ndarray], out_dir: str | Path) -> None:
    
    out_path = Path(out_dir)
    storage = CsvStorage()

    history_csv = out_path / "history.csv"
    particles_csv = out_path / "final_particles_2.csv"

    storage.save_history(hist, history_csv)
    storage.save_particles(world.bodies, particles_csv)

    print(f"Saved history CSV: {history_csv.resolve()}")
    print(f"Saved particle CSV: {particles_csv.resolve()}")


def main() -> None:
    cfg = SimConfig(
        box=BoxParams(boxtype="per", Lx=5.0, Ly=5.0, Lz=10000.0),
        gravity=GravityParams.standard(9.81),
        contact_pp=ContactParams(k_n=4000.0, c_n=35.0, k_t=2000.0, c_t=20.0, mu=0.3, mu_roll=0.1),
        contact_pw=ContactParams(k_n=20000.0, c_n=80.0, k_t=10000.0, c_t=35.0, mu=0.3, mu_roll=0.1),
        time=TimeParams(dt=2e-4, max_time=20.0, record_stride=500, log_stride=500, stable_time=0.1),
    )

    
    
    world = build_world(cfg, n_1=27, r_1=1.0, d_1=2.5, n_2=800, r_2=0.25, d_2=1.6)
    sim = Simulator(world)
    sim.initialize()
    # print(sim.world.config.box.Lx, sim.world.config.box.Ly, sim.world.config.box.Lz)

    hist = sim.run()
    save_outputs(world, hist, Path("output"))
    plot_history(hist)


if __name__ == "__main__":
    main()
