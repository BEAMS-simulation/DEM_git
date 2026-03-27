import numpy as np

rng = np.random.default_rng()

LPS_MEAN    = 1.62
LPS_SIGMA   = 0.678

def get_radius(mu: float = LPS_MEAN, sigma: float = LPS_SIGMA) -> float:
    return rng.lognormal(mu, sigma)