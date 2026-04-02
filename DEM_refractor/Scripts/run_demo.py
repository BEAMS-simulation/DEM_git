from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt

from demrigid.config import SimConfig, BoxParams, GravityParams, ContactParams, TimeParams
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

def main() -> None:
    cfg = SimConfig(
        box=BoxParams(boxtype="imp", Lx=2.2, Ly=2.2, Lz=10000.0),
        gravity=GravityParams.standard(9.81),
        contact_pp=ContactParams(k_n=400.0, c_n=20.0, k_t=200.0, c_t=5.0, mu=0.3, mu_roll=0.3),
        contact_pw=ContactParams(k_n=4000.0, c_n=60.0, k_t=2000.0, c_t=10.0, mu=0.3, mu_roll=0.3),
        time=TimeParams(dt=2e-4, max_time=10.0, record_stride=500, log_stride=500, stable_time=0.2),
    )
    n_balls = 30
    density = 2.0
    r = 0.5

    bodies: list[Rigidbody] = []
    
    rng = np.random.default_rng(123)
    for i in range(n_balls):
        m = density * (4.0 / 3.0) * math.pi * (r ** 3)
        agg = Aggregate([Sphere(r=r, m=m, pos_local=np.zeros(3, dtype=float))])
        # 초기 위치: 충돌 방지 위해 z를 층별로 올림
        pos = np.array([
            rng.uniform(0.5, cfg.box.Lx - 0.5),
            rng.uniform(0.5, cfg.box.Ly - 0.5),
            1.0 + 2.2 * r * i,
        ], dtype=float)
        
        bodies.append(Rigidbody(body=agg, id=i, pos=pos))

    world = World(cfg, bodies)
    sim = Simulator(world)
    sim.initialize()
    
    hist = sim.run()
    plot_history(hist)

if __name__ == "__main__":
    main()