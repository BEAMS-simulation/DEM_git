from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
from collections import defaultdict

from .model import Sphere, Aggregate, Rigidbody
from .math3d import quat_identity

CSV_HEADER = ["body id", "particle id", "x", "y", "z", "r", "m"]
HISTORY_HEADER = [
    "t",
    "ke",
    "px",
    "py",
    "pz",
    "pnorm",
    "lx",
    "ly",
    "lz",
    "lnorm",
]


@dataclass
class CsvStorage:
    def save_particles(self, bodies: list[Rigidbody], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            for b in bodies:
                p_all = b.sphere_pos_world()
                for pid in range(b.body.n):
                    w.writerow([
                        int(b.id),
                        int(pid),
                        float(p_all[pid, 0]),
                        float(p_all[pid, 1]),
                        float(p_all[pid, 2]),
                        float(b.body.radii[pid]),
                        float(b.body.masses[pid]),
                    ])

    def save_history(self, hist: dict[str, np.ndarray], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        t = np.asarray(hist.get("t", np.zeros(0, dtype=float)), dtype=float)
        ke = np.asarray(hist.get("ke", np.zeros(0, dtype=float)), dtype=float)
        p = np.asarray(hist.get("p", np.zeros((0, 3), dtype=float)), dtype=float)
        l = np.asarray(hist.get("l", np.zeros((0, 3), dtype=float)), dtype=float)

        if t.size == 0:
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(HISTORY_HEADER)
            return

        pnorm = np.linalg.norm(p, axis=1)
        lnorm = np.linalg.norm(l, axis=1)

        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(HISTORY_HEADER)
            for i in range(len(t)):
                w.writerow([
                    float(t[i]),
                    float(ke[i]),
                    float(p[i, 0]),
                    float(p[i, 1]),
                    float(p[i, 2]),
                    float(pnorm[i]),
                    float(l[i, 0]),
                    float(l[i, 1]),
                    float(l[i, 2]),
                    float(lnorm[i]),
                ])

    def load_particles(self, path: str | Path) -> list[Rigidbody]:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header.")
            if list(reader.fieldnames) != CSV_HEADER:
                raise ValueError(f"Unexpected header:{reader.fieldnames}, expected:{CSV_HEADER}")
            grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
            for row in reader:
                body_id = int(row["body id"])
                grouped[body_id].append(row)

        bodies: list[Rigidbody] = []

        for body_id, rows in sorted(grouped.items(), key=lambda x: x[0]):
            rows = sorted(rows, key=lambda r: int(r["particle id"]))

            spheres: list[Sphere] = []
            masses: list[float] = []
            world_pos: list[np.ndarray] = []

            for r in rows:
                p = np.array([float(r["x"]), float(r["y"]), float(r["z"])], dtype=float)
                rad = float(r["r"])
                m = float(r["m"])
                world_pos.append(p)
                masses.append(m)
                spheres.append(Sphere(r=rad, m=m, pos_local=p.copy()))

            com = np.average(np.array(world_pos, dtype=float), axis=0, weights=np.array(masses, dtype=float))
            agg = Aggregate(spheres=spheres)

            rb = Rigidbody(
                body=agg,
                id=int(body_id),
                pos=np.array(com, dtype=float),
                vel=np.zeros(3, dtype=float),
                quat=quat_identity(),
                omega_body=np.zeros(3, dtype=float),
            )
            bodies.append(rb)

        return bodies
