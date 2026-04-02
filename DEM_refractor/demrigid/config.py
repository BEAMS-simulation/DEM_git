from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class BoxParams:
    boxtype: str = "imp"
    Lx: float = 2.2
    Ly: float = 2.2
    Lz: float = 10000.0
    
    def __post_init__(self)->None:
        if self.boxtype not in ("imp", "per"):
            raise ValueError(f"boxtype must be 'imp' or 'per', got {self.boxtype!r}")
        if self.Lx <= 0 or self.Ly <=0 or self.Lz <= 0:
            raise ValueError("Box dimensions must be positive.")
    
    @property
    def walls(self) -> tuple[str, ...]:
        if self.boxtype == "per":
            return ("down", "up")
        return ("west", "south", "east", "north", "down", "up")
    
@dataclass(frozen=True)
class GravityParams:
    g: np.ndarray # shape (3,)
    
    @staticmethod
    def standard(g0: float = 9.8) -> "GravityParams":
        return GravityParams(g = np.array([0.0, 0.0, -float(g0)], dtype=float))
    
    def __post_init__(self) -> None:
        g = np.asarray(self.g, dtype=float)
        if g.shape != (3,):
            raise ValueError(f"Gravity vector must have shape (3,), got {g.shape}")

@dataclass(frozen=True)
class ContactParams:
    k_n: float = 400.0
    c_n: float = 20.0
    k_t: float = 200.0
    c_t: float = 5.0
    
    mu: float = 0.3         # Coulomb 마찰계수
    mu_roll: float = 0.0    # Rolling 저항
    
    v_eps: float = 1e-9     # can't divide by zero
    
    def __post_init__(self) -> None:
        for name in ("k_n", "c_n", "k_t", "c_t"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.mu < 0:
            raise ValueError("mu must be non-negative.")
        if self.mu_roll < 0:
            raise ValueError("mu_roll must be non-negative.")
        if self.v_eps <= 0:
            raise ValueError("v_eps must be positive.")
    
    @staticmethod
    def damping_from_restitution(k_n: float, m_eff: float, e: float) -> float:
        if not (0.0 < e < 1.0):
            raise ValueError("restitution coefficient e must satisfy 0 < e < 1.")
        if k_n <= 0 or m_eff <= 0:
            raise ValueError("k_n and m_eff must be positive.")
        ln_e = math.log(e)
        return math.sqrt(4.0 * m_eff * k_n / (1.0 + (math.pi / ln_e) ** 2))
    
@dataclass(frozen=True)
class TimeParams:
    dt: float = 2e-4
    max_time: float = 10.0
    record_stride: int = 1000
    log_stride: int = 1000
    stable_time: float = 0.15
    
    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.max_time <= 0:
            raise ValueError("max_time must be positive.")
        if self.record_stride <= 0 or self.log_stride <= 0:
            raise ValueError("record_stride and log_stride must be positive.")
        if self.stable_time < 0:
            raise ValueError("stable_time must be non-negative.")

@dataclass(frozen=True)
class SleepParams:
    enable: bool = True
    
    sleep_dur_threshold: int = 1000
    sleep_speed: float = 5e-3
    sleep_ang_speed: float = 5e-2
    
    wake_speed: float = 2e-2
    wake_ang_speed: float = 2e-1
    wake_acc: float = 3.0
    wake_ang_acc: float = 10.0
    
    def __post_init__(self) -> None:
        if self.sleep_dur_threshold <= 0:
            raise ValueError("sleep_dur_threshold must be positive.")
        for name in (
            "sleep_speed",
            "sleep_ang_speed",
            "wake_speed",
            "wake_ang_speed",
            "wake_acc",
            "wake_ang_acc",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")

@dataclass(frozen=True)
class NeighborParams:
    skin: float = 0.03
    
    def __post_init__(self) -> None:
        if self.skin <= 0:
            raise ValueError("skin must be positive.")

@dataclass(frozen=True)
class SimConfig:
    box: BoxParams = BoxParams()
    gravity: GravityParams = GravityParams.standard()

    contact_pp: ContactParams = ContactParams() # particle-particle
    contact_pw: ContactParams = ContactParams(k_n=4000.0, c_n=60.0, k_t=2000.0, c_t=10.0, mu=0.3)

    time: TimeParams = TimeParams()
    sleep: SleepParams = SleepParams()
    neighbor: NeighborParams = NeighborParams()
    
    kinetic_tol: float = 5e-5
    
    def __post_init__(self) -> None:
        if self.kinetic_tol < 0:
            raise ValueError("kinetic_tol must be non-negative.")