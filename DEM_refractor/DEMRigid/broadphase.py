from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from .config import BoxParams


@dataclass
class GridBroadphase:
    box: BoxParams
    cell_size: float

    def __post_init__(self) -> None:
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive.")
        
        self.inv = 1.0 / float(self.cell_size)
        self.nx = max(1, int(np.floor(self.box.Lx * self.inv))) if self.box.boxtype == "per" else max(1, int(np.floor(self.box.Lx * self.inv)))
        self.ny = max(1, int(np.floor(self.box.Ly * self.inv))) if self.box.boxtype == "per" else max(1, int(np.floor(self.box.Ly * self.inv)))
        self.nz = max(1, int(np.floor(self.box.Lz * self.inv)))

    def _cell_index(self, pos: np.ndarray) -> tuple[int, int, int]:
        x, y, z = map(float, pos)
        ix = int(np.floor(x * self.inv))
        iy = int(np.floor(y * self.inv))
        iz = int(np.floor(z * self.inv))

        if self.box.boxtype == "per":
            ix %= self.nx
            iy %= self.ny
        else:
            ix = min(max(ix, 0), self.nx - 1)
            iy = min(max(iy, 0), self.ny - 1)
        
        iz = min(max(iz, 0), self.nz - 1)
        return ix, iy, iz
    
    def candidate_pairs(self, positions: np.ndarray) -> list[tuple[int, int]]:
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (N,3), got {positions.shape}")
        
        cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for i, p in enumerate(positions):
            cells[self._cell_index(p)].append(i)
            
        pairs: set[tuple[int, int]] = set()
        
        for (ix, iy, iz), ids in cells.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        jx = ix + dx
                        jy = iy + dy
                        jz = iz + dz
                        
                        if self.box.boxtype == "per":
                            jx %= self.nx
                            jy %= self.ny
                        else:
                            if jx < 0 or jx >= self.nx or jy < 0 or jy >= self.ny:
                                continue
                        if jz < 0 or jz >= self.nz:
                            continue
                        
                        other = cells.get((jx, jy, jz))
                        if not other:
                            continue
                        
                        if (jx, jy, jz) == (ix, iy, iz):
                            for a in range(len(ids)):
                                ia = ids[a]
                                for b in range(a+1, len(ids)):
                                    ib = ids[b]
                                    pairs.add((ia, ib))
                        else:
                            if (jx, jy, jz) < (ix, iy, iz):
                                continue
                            for ia in ids:
                                for ib in other:
                                    if ia < ib:
                                        pairs.add((ia, ib))
                                    elif ib < ia:
                                        pairs.add((ib, ia))
        return sorted(pairs)